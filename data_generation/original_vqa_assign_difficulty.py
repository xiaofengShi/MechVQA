#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assign difficulty labels to VQA message-format JSONL records."""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from .model_client import call_openai_compatible
    from .taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory
except ImportError:
    from model_client import call_openai_compatible
    from taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory


DIFFICULTY_PROMPT = """你是一名机械工程师兼教学教研人员，需要根据一道机械图纸问答题，评估其综合难度等级。

【总体原则】
- 综合难度由三部分组成：图纸复杂度、问题难度、答案复杂度。
- 不要把绝大多数题都判成 Medium；对明显复杂、专业性强或需要多步推理的题，要敢于评为 Hard。
- 难度标签必须使用英文：Easy、Medium、Hard。

【输入信息】

问题：
{question}

答案：
{answer}

元数据：
- 能力类别: {capability}
- 子类别: {subcategory}

【输出要求】
请仅返回以下 JSON 格式，不要添加多余文字：

{{
    "drawing_complexity": "Easy/Medium/Hard",
    "question_difficulty": "Easy/Medium/Hard",
    "answer_complexity": "Easy/Medium/Hard",
    "difficulty": "Easy/Medium/Hard",
    "reason": "用 2-3 句话简要说明理由，分别说明图纸、问题和答案难度的判断依据。"
}}
"""


def require_model(model: Optional[str]) -> str:
    if model:
        return model
    env_model = os.environ.get("DEFAULT_MODEL")
    if env_model:
        return env_model
    raise ValueError("Model is required. Pass --model or set DEFAULT_MODEL.")


def extract_question(user_content: str) -> str:
    content = user_content.replace("<image>", "").strip()
    if "\n\nA conversation between" in content:
        content = content.split("\n\nA conversation between")[0].strip()
    return content


def extract_answer(assistant_content: str) -> str:
    return assistant_content.strip()


def parse_difficulty_result(result_text: str) -> Dict[str, Any]:
    try:
        return normalize_difficulty_result(json.loads(result_text))
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", result_text, re.DOTALL)
    if json_match:
        try:
            return normalize_difficulty_result(json.loads(json_match.group(1)))
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
    if json_match:
        try:
            return normalize_difficulty_result(json.loads(json_match.group(0)))
        except json.JSONDecodeError:
            pass

    return {
        "drawing_complexity": "Unknown",
        "question_difficulty": "Unknown",
        "answer_complexity": "Unknown",
        "difficulty": "Unknown",
        "reason": f"解析失败: {result_text[:100]}",
    }


def normalize_difficulty_result(result: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(result)
    alias_pairs = (
        ("复杂度", "drawing_complexity"),
        ("问题难度", "question_difficulty"),
        ("答案复杂度", "answer_complexity"),
    )
    for old_key, new_key in alias_pairs:
        value = normalized.pop(old_key, normalized.get(new_key))
        if value is not None:
            normalized[new_key] = normalize_difficulty(value)

    if "difficulty" in normalized:
        normalized["difficulty"] = normalize_difficulty(normalized.get("difficulty"))
    return normalized


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print("Skip invalid JSON line.")
    return records


def assign_difficulty_to_record(record: Dict[str, Any], model: str) -> Dict[str, Any]:
    messages = record.get("messages") or []
    question = ""
    answer = ""

    for msg in messages:
        if msg.get("role") == "user":
            question = extract_question(msg.get("content", ""))
        elif msg.get("role") == "assistant":
            answer = extract_answer(msg.get("content", ""))

    metadata = record.get("metadata") or {}
    prompt = DIFFICULTY_PROMPT.format(
        question=question,
        answer=answer,
        capability=normalize_capability(metadata.get("capability", "Unknown")),
        subcategory=normalize_subcategory(metadata.get("subcategory", "Unknown")),
    )

    try:
        result_text = call_openai_compatible(
            prompt=prompt,
            model=model,
            image_paths=record.get("images", []),
            temperature=0.3,
        )
        difficulty_result = parse_difficulty_result(result_text)
    except Exception as exc:
        difficulty_result = {
            "drawing_complexity": "Unknown",
            "question_difficulty": "Unknown",
            "answer_complexity": "Unknown",
            "difficulty": "Unknown",
            "reason": f"LLM 调用失败: {exc}",
        }

    new_record = record.copy()
    new_record["metadata"] = metadata.copy()
    new_record["metadata"]["assign_diff"] = difficulty_result
    return new_record


def assign_difficulty(
    input_jsonl: str,
    output_jsonl: str,
    model: str,
    max_workers: int,
    start_idx: int,
    end_idx: Optional[int],
) -> None:
    records = read_jsonl(input_jsonl)
    end_idx = len(records) if end_idx is None else min(end_idx, len(records))
    selected = list(enumerate(records[start_idx:end_idx], start_idx))

    results: Dict[int, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(assign_difficulty_to_record, record, model): idx
            for idx, record in selected
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Assign difficulty"):
            idx = futures[future]
            results[idx] = future.result()

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for idx in sorted(results):
            f.write(json.dumps(results[idx], ensure_ascii=False) + "\n")

    print(f"Output: {output_jsonl}")


def apply_difficulty(input_jsonl: str, output_jsonl: str) -> None:
    records = read_jsonl(input_jsonl)
    updated = 0

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            metadata = record.setdefault("metadata", {})
            assign_diff = metadata.get("assign_diff")
            if isinstance(assign_diff, dict) and assign_diff.get("difficulty"):
                metadata["difficulty"] = normalize_difficulty(assign_diff["difficulty"])
                updated += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Updated records: {updated}")
    print(f"Output: {output_jsonl}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign or apply VQA difficulty labels.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assign = subparsers.add_parser("assign", help="Call a model to assign difficulty labels.")
    assign.add_argument("--input-jsonl", required=True)
    assign.add_argument("--output-jsonl", required=True)
    assign.add_argument("--model", default=os.environ.get("DEFAULT_MODEL"))
    assign.add_argument("--max-workers", type=int, default=5)
    assign.add_argument("--start-idx", type=int, default=0)
    assign.add_argument("--end-idx", type=int, default=None)

    apply = subparsers.add_parser("apply", help="Copy assign_diff.difficulty into metadata.difficulty.")
    apply.add_argument("--input-jsonl", required=True)
    apply.add_argument("--output-jsonl", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "assign":
        assign_difficulty(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            model=require_model(args.model),
            max_workers=args.max_workers,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )
    elif args.command == "apply":
        apply_difficulty(args.input_jsonl, args.output_jsonl)


if __name__ == "__main__":
    main()
