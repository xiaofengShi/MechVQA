import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .io_utils import load_jsonl, read_json, write_json, write_jsonl
from .providers import build_clients


ANSWER_SUFFIX = (
    "\n\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process and then provides the answer. "
    "Use <think>...</think><answer>...</answer> tags."
)


def get_user_question(record: Dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "")
    return record.get("question") or record.get("check_results", {}).get("question", "")


def get_correct_answer(record: Dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant" and message.get("content"):
            return extract_answer(str(message.get("content", "")))
    metadata = record.get("metadata", {})
    check_results = record.get("check_results", {})
    return str(metadata.get("correct_answer") or check_results.get("correct_answer") or record.get("answer", ""))


def get_images(record: Dict[str, Any]) -> List[str]:
    images = record.get("images", [])
    if isinstance(images, str):
        return [images] if images else []
    return [str(path) for path in images if path]


def resolve_images(image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    image_root = config.get("image_root")
    input_root = Path(config["input_file"]).parent
    resolved = []
    for image_path in image_paths:
        path = Path(image_path)
        if path.is_absolute():
            resolved.append(str(path))
            continue

        candidates = []
        if image_root:
            candidates.append(Path(image_root) / path)
        candidates.append(input_root / path)
        candidates.append(Path.cwd() / path)

        resolved_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        resolved.append(str(resolved_path))
    return resolved


def extract_answer(response: str) -> str:
    if not response:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"<answer>(.*)", response, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"</think>(.*)", response, flags=re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return response.strip()


def build_judge_prompt(question: str, correct_answer: str, model_answer: str) -> str:
    return f"""Please judge whether the model answer is semantically correct for this VQA item.

Question:
{question}

Reference answer:
{correct_answer}

Model answer:
{model_answer}

Scoring rules:
- Return 1 if the model answer is correct or captures all key information.
- Return 0 if it is wrong, irrelevant, or missing key information.
- Only use score 0 or 1.

Return only a JSON object:
{{"score": 0 or 1, "reason": "short explanation"}}"""


def parse_judge_response(response: str) -> Tuple[float, str]:
    if not response:
        return 0.0, "empty judge response"
    match = re.search(r"\{[^{}]*\}", response, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            score = max(0.0, min(1.0, float(payload.get("score", 0))))
            return score, str(payload.get("reason", ""))
        except Exception:
            pass
    if "1" in response and "0" not in response:
        return 1.0, "parsed as 1"
    if "0" in response:
        return 0.0, "parsed as 0"
    return 0.0, f"parse failed: {response[:120]}"


def phase1(config: Dict[str, Any], max_samples: Optional[int] = None) -> None:
    records = load_jsonl(config["input_file"])
    if max_samples:
        records = records[:max_samples]

    clients = build_clients(config["target_models"])
    output_file = config["phase1_output"]
    max_workers = int(config.get("max_workers", 8))

    def run_one(record_idx: int, record: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        question = get_user_question(record)
        prompt = question + (config.get("answer_suffix", ANSWER_SUFFIX) if config.get("append_answer_suffix", True) else "")
        try:
            response = clients[model_name].chat(prompt, resolve_images(get_images(record), config))
            return {
                "record_idx": record_idx,
                "target_model": model_name,
                "model_response": response,
                "model_answer": extract_answer(response),
                "response_error": None,
            }
        except Exception as exc:
            return {
                "record_idx": record_idx,
                "target_model": model_name,
                "model_response": "",
                "model_answer": "",
                "response_error": str(exc),
            }

    tasks = [(idx, record, name) for idx, record in enumerate(records) for name in clients]
    responses: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, *task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="phase1"):
            responses.append(future.result())

    by_record: Dict[int, Dict[str, Any]] = {
        idx: {"record_idx": idx, "original_record": record, "responses": {}} for idx, record in enumerate(records)
    }
    for response in responses:
        by_record[response["record_idx"]]["responses"][response["target_model"]] = {
            "model_response": response["model_response"],
            "model_answer": response["model_answer"],
            "response_error": response["response_error"],
        }

    write_jsonl((by_record[idx] for idx in sorted(by_record)), output_file)
    print(f"phase1 saved to {output_file}")


def phase2(config: Dict[str, Any]) -> None:
    phase1_records = load_jsonl(config["phase1_output"])
    judge_clients = build_clients(config["judge_models"])
    max_workers = int(config.get("max_workers", 8))

    def run_one(record: Dict[str, Any], target_model: str, response_data: Dict[str, Any], judge_name: str) -> Dict[str, Any]:
        original = record["original_record"]
        if response_data.get("response_error") or not response_data.get("model_answer"):
            return {"score": 0.0, "reason": "missing model answer", "error": response_data.get("response_error") or "empty answer"}
        prompt = build_judge_prompt(get_user_question(original), get_correct_answer(original), response_data["model_answer"])
        try:
            raw = judge_clients[judge_name].chat(prompt)
            score, reason = parse_judge_response(raw)
            return {"score": score, "reason": reason, "error": None, "raw_response": raw}
        except Exception as exc:
            return {"score": 0.0, "reason": "judge error", "error": str(exc), "raw_response": ""}

    futures = []
    evaluated = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for record in phase1_records:
            for target_model, response_data in record.get("responses", {}).items():
                for judge_name in judge_clients:
                    futures.append((record, target_model, judge_name, executor.submit(run_one, record, target_model, response_data, judge_name)))
        for record, target_model, judge_name, future in tqdm(futures, desc="phase2"):
            result = future.result()
            record.setdefault("evaluation_result", {}).setdefault(target_model, {"judge_results": {}})
            record["evaluation_result"][target_model]["judge_results"][judge_name] = result

    for record in phase1_records:
        for target_model, response_data in record.get("responses", {}).items():
            judge_results = record.get("evaluation_result", {}).get(target_model, {}).get("judge_results", {})
            valid_scores = [round(v.get("score", 0), 1) for v in judge_results.values() if not v.get("error")]
            voted_score = Counter(valid_scores).most_common(1)[0][0] if valid_scores else 0.0
            record["evaluation_result"][target_model].update(
                {
                    "model_response": response_data.get("model_response", ""),
                    "model_answer": response_data.get("model_answer", ""),
                    "correct_answer": get_correct_answer(record["original_record"]),
                    "voted_score": voted_score,
                    "error": response_data.get("response_error"),
                }
            )
        evaluated.append(record)

    write_jsonl(evaluated, config["evaluated_output"])
    write_json(build_stats(evaluated, config), config["stats_output"])
    print(f"evaluated records saved to {config['evaluated_output']}")
    print(f"stats saved to {config['stats_output']}")


def build_stats(records: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    model_stats: Dict[str, Any] = {}
    for record in records:
        for model_name, result in record.get("evaluation_result", {}).items():
            bucket = model_stats.setdefault(model_name, {"total": 0, "sum_score": 0.0, "full_score_count": 0, "error_count": 0})
            score = float(result.get("voted_score", 0))
            bucket["total"] += 1
            bucket["sum_score"] += score
            bucket["full_score_count"] += int(score == 1.0)
            bucket["error_count"] += int(bool(result.get("error")))
    for bucket in model_stats.values():
        total = bucket["total"] or 1
        bucket["overall_voted_score"] = bucket.pop("sum_score") / total
        bucket["overall_full_score_rate"] = bucket["full_score_count"] / total
    return {
        "evaluation_time": datetime.now().isoformat(),
        "input_file": config["input_file"],
        "phase1_output": config["phase1_output"],
        "evaluated_output": config["evaluated_output"],
        "model_stats": model_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MechVQA evaluation.")
    parser.add_argument("--config", required=True, help="Path to evaluation config JSON.")
    parser.add_argument("--phase", choices=["phase1", "phase2", "all"], default="all")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = read_json(args.config)
    if args.phase in {"phase1", "all"}:
        phase1(config, args.max_samples)
    if args.phase in {"phase2", "all"}:
        phase2(config)


if __name__ == "__main__":
    main()
