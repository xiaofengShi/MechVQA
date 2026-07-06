#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build and parse generated-question quality checks for VQA data.

This script is the question-check stage between question generation and answer
generation. It supports two sources:

1. VQA-free model outputs whose response contains a "问题列表" array.
2. Already parsed question JSONL records, such as VQA temp non-GT questions.

The output records can be passed to a model runner, then parsed with `analyze`
to add `fix_result`. `generate_fixed_questions.py` keeps rows whose
`fix_result.verdict` is `correct`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

try:
    from .language import check_language_instruction, normalize_language
    from .taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory
except ImportError:
    from language import check_language_instruction, normalize_language
    from taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory


DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL")


QUESTION_CHECK_PROMPT_TEMPLATE = """你是一名熟练掌握机械制图标准的工程师，任务是：
只检查问题本身是否和图纸事实一致、是否可回答，而不是去真正回答尺寸或技术要求。

输入包括：
1. 一张机械零件图或装配图或组合图（作为图像输入）；
2. 一句话形式的问题（question），该问题是针对这张图纸提出的。
3. 问题对应的难度等级（difficulty）、子类别（subcategory）。
4. 统一 extract metadata（已抽取结构化信息）。

{question_line}
难度等级：{difficulty}
子类别：{subcategory}
统一 extract metadata：
```json
{metadata_block}
```

在推理时要以实际图像内容为最终依据：
- 如果 metadata 与图像内容矛盾，以图像为准；
- 如果是多张图组合，问题中提到哪个子图，就观察哪个子图；
- 不要凭空假设图中存在未看到的视图、标注或尺寸。

请检查问题中的隐含假设是否真实存在，常见错误包括但不限于：
- 提到某个不存在的标记或视图；
- 把基准符号误当成剖视图或剖切线标记；
- 把视图内部的局部剖视区域误称为独立剖视图；
- 视图引用错误；
- 问的是图中没有标出、也不能由唯一尺寸链推出的尺寸；
- 把孔和轴、内径和外径、长度和深度等搞混；
- 问的是图中不存在的几何要素或对象；
- 误读粗糙度、形位公差、配合代号等；
- 自然语言无法唯一定位到一个几何要素或一条尺寸标注。

输出结论类型（verdict）必须从以下三种中选一个：
- "correct"：问题和图纸事实完全一致，且可以在图中唯一回答；
- "ill-posed"：问题包含事实错误、引用了不存在的元素，或从图中无法作答；
- "ambiguous"：图能提供部分信息，但问题表述不够明确，存在多种合理理解。

问题类型标签（problem_types）从下面标签中选一个或多个：
- "nonexistent_symbol"
- "wrong_view_reference"
- "missing_dimension"
- "datum_misinterpreted_as_section"
- "misread_tolerance"
- "ambiguous_reference"
- "other"

fixed_query 的要求：
- 当 verdict = "ill-posed" 且稍作修改就能成为合理问题时，给出修改后的更合理提问；
- 当问题完全无法挽救时，写 null；
- 当 verdict = "correct" 或 "ambiguous" 时，如果不需要修改，写 null；如果需要补充定位信息让它更清晰，也可以给出改写版。

{language_instruction}

只输出 JSON，不要额外文字。格式如下：
```json
{{
  "verdict": "correct | ill-posed | ambiguous",
  "problem_types": ["nonexistent_symbol | wrong_view_reference | missing_dimension | datum_misinterpreted_as_section | misread_tolerance | ambiguous_reference | other"],
  "explanation": "Short explanation.",
  "fixed_query": "A corrected question if needed; otherwise null."
}}
```
"""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_files(paths: Iterable[str], ignore_errors: bool = False) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in paths:
        if not os.path.exists(path):
            if ignore_errors:
                continue
            raise FileNotFoundError(path)
        results.extend(read_jsonl(path))
    return results


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response.")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    raise ValueError("Unbalanced JSON braces in response.")


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = json.loads(_extract_json_block(text))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")
    return parsed


def strip_think(text: str) -> str:
    marker = "</think>"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def extract_response_json(response: str) -> Optional[Dict[str, Any]]:
    if not response:
        return None
    try:
        return safe_json_loads(strip_think(response))
    except Exception:
        return None


def parse_fix_result(response: str) -> Dict[str, Any]:
    parsed = extract_response_json(response)
    if not isinstance(parsed, dict):
        return {"parse_error": "JSONDecodeError"}
    return {
        "verdict": parsed.get("verdict", ""),
        "problem_types": parsed.get("problem_types") or [],
        "explanation": parsed.get("explanation", parsed.get("explanation_zh", "")),
        "fixed_query": parsed.get("fixed_query", parsed.get("fixed_query_zh")),
        "raw": parsed,
    }


def build_check_prompt(
    question: str,
    difficulty: str,
    subcategory: str,
    params: Dict[str, Any],
    language: str = "中文",
) -> str:
    language = normalize_language(language)
    metadata_block = json.dumps(params or {}, ensure_ascii=False, indent=2)
    question_line = f"Question: {question}" if language == "英文" else f"问题：{question}"
    return QUESTION_CHECK_PROMPT_TEMPLATE.format(
        question_line=question_line,
        difficulty=difficulty or "",
        subcategory=subcategory or "",
        metadata_block=metadata_block,
        language_instruction=check_language_instruction(language),
    )


def first_present(record: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return default


def question_record_to_check_record(record: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    question = first_present(record, ("question", "prompt"))
    if not question.strip():
        return None

    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    result = copy.deepcopy(record)
    result["capability"] = normalize_capability(record.get("capability", ""))
    result["subcategory"] = normalize_subcategory(record.get("subcategory", ""))
    result["difficulty"] = normalize_difficulty(record.get("difficulty", ""))
    result["language"] = normalize_language(
        record.get("language") or (record.get("metadata") or {}).get("language") or "中文"
    )
    result.setdefault("id", f"question_{idx:06d}")
    result["question"] = question
    result["prompt"] = build_check_prompt(
        question=question,
        difficulty=result["difficulty"],
        subcategory=result["subcategory"],
        params=params,
        language=result["language"],
    )
    result["response"] = ""
    result["_question_check_source"] = "question_record"
    return result


def extract_questions_from_free_response(response: str) -> Optional[Dict[str, Any]]:
    return extract_response_json(response)


def free_output_to_check_records(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    parsed = extract_questions_from_free_response(str(record.get("response", "")))
    if not parsed:
        return []

    question_list = parsed.get("问题列表") or []
    if not isinstance(question_list, list):
        return []

    output: List[Dict[str, Any]] = []
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    for q_idx, q_item in enumerate(question_list):
        if not isinstance(q_item, list) or len(q_item) < 5:
            continue

        question = str(q_item[0] or "").strip()
        if not question:
            continue

        capability = normalize_capability(q_item[1])
        subcategory = normalize_subcategory(q_item[2])
        difficulty = normalize_difficulty(q_item[3])
        language = normalize_language(
            q_item[4] or record.get("language") or (record.get("metadata") or {}).get("language") or "中文"
        )
        source_id = str(record.get("id") or record.get("source_id") or "sample")

        item = {
            "id": f"{source_id}_q{q_idx:03d}",
            "prompt": build_check_prompt(question, difficulty, subcategory, params, language=language),
            "response": "",
            "question": question,
            "image_path": record.get("image_path", ""),
            "original_image_path": record.get("original_image_path", record.get("image_path", "")),
            "capability": capability,
            "subcategory": subcategory,
            "difficulty": difficulty,
            "language": language,
            "source_id": source_id,
            "params": params,
            "metadata": {
                "extract_metadata_summary": (record.get("metadata") or {}).get("extract_metadata_summary", {}),
            },
            "_question_check_source": "vqa_free_output",
        }
        output.append(item)
    return output


def build_from_free_outputs(input_jsonls: List[str], output_jsonl: str) -> None:
    records = load_jsonl_files(input_jsonls, ignore_errors=True)
    results: List[Dict[str, Any]] = []
    failed = 0

    for record in tqdm(records, desc="Build free question checks"):
        items = free_output_to_check_records(record)
        if not items:
            failed += 1
        results.extend(items)

    write_jsonl(results, output_jsonl)
    print(f"Input records: {len(records)}")
    print(f"Failed records: {failed}")
    print(f"Question-check prompts: {len(results)}")
    print(f"Output: {output_jsonl}")


def build_from_question_records(input_jsonls: List[str], output_jsonl: str) -> None:
    records = load_jsonl_files(input_jsonls, ignore_errors=True)
    results: List[Dict[str, Any]] = []

    for idx, record in enumerate(tqdm(records, desc="Build question checks"), start=1):
        item = question_record_to_check_record(record, idx)
        if item is not None:
            results.append(item)

    write_jsonl(results, output_jsonl)
    print(f"Input records: {len(records)}")
    print(f"Question-check prompts: {len(results)}")
    print(f"Output: {output_jsonl}")


def import_model_client():
    try:
        from .model_client import call_openai_compatible
    except ImportError:
        from model_client import call_openai_compatible

    return call_openai_compatible


def image_paths_for_record(record: Dict[str, Any]) -> Optional[List[str]]:
    images = record.get("images")
    if isinstance(images, list) and images:
        return [str(path) for path in images if path]
    image_path = first_present(record, ("image_path", "original_image_path"))
    return [image_path] if image_path else None


def run_one_check(record: Dict[str, Any], model: str, temperature: float, timeout: int, max_retries: int) -> Dict[str, Any]:
    call_model = import_model_client()
    result = copy.deepcopy(record)
    try:
        response = call_model(
            result.get("prompt", ""),
            model=model,
            image_paths=image_paths_for_record(result),
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        result["response"] = response or ""
        result["error"] = None
    except Exception as exc:
        result["response"] = ""
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_checks(
    input_jsonl: str,
    output_jsonl: str,
    model: str,
    workers: int,
    temperature: float,
    timeout: int,
    max_retries: int,
    overwrite: bool,
) -> None:
    input_records = read_jsonl(input_jsonl)
    output_records = read_jsonl(output_jsonl) if os.path.exists(output_jsonl) and not overwrite else []

    if output_records and len(output_records) == len(input_records):
        records = output_records
    else:
        records = copy.deepcopy(input_records)

    pending = [
        idx for idx, record in enumerate(records)
        if overwrite or not record.get("response")
    ]

    print(f"Input records: {len(input_records)}")
    print(f"Pending checks: {len(pending)}")
    print(f"Model: {model}; workers: {workers}")
    if not pending:
        write_jsonl(records, output_jsonl)
        print(f"Output is up to date: {output_jsonl}")
        return

    lock = threading.Lock()
    completed_since_save = 0

    def save_checkpoint(force: bool = False) -> None:
        nonlocal completed_since_save
        if force or completed_since_save >= 20:
            write_jsonl(records, output_jsonl)
            completed_since_save = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one_check, records[idx], model, temperature, timeout, max_retries): idx
            for idx in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Run question checks"):
            idx = futures[future]
            checked = future.result()
            with lock:
                records[idx] = checked
                completed_since_save += 1
                save_checkpoint()

    save_checkpoint(force=True)
    errors = sum(1 for record in records if record.get("error"))
    print(f"Output: {output_jsonl}")
    print(f"Errors: {errors}")


def analyze_check_outputs(input_jsonls: List[str], output_jsonl: str) -> None:
    records = load_jsonl_files(input_jsonls, ignore_errors=True)
    results: List[Dict[str, Any]] = []
    verdict_counts: Dict[str, int] = {}
    problem_type_counts: Dict[str, int] = {}
    parse_errors: Dict[str, int] = {}

    for record in tqdm(records, desc="Analyze question checks"):
        item = copy.deepcopy(record)
        response = str(item.get("response", ""))
        fix_result = parse_fix_result(response)
        if fix_result.get("parse_error"):
            reason = str(fix_result.get("parse_error"))
            parse_errors[reason] = parse_errors.get(reason, 0) + 1
            item["fix_result"] = {
                "parse_error": reason,
                "raw_answer": strip_think(response)[:500],
            }
            results.append(item)
            continue

        verdict = str(fix_result.get("verdict", "unknown"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        problem_types = fix_result.get("problem_types") or []
        if isinstance(problem_types, list):
            for problem_type in problem_types:
                key = str(problem_type)
                problem_type_counts[key] = problem_type_counts.get(key, 0) + 1

        item["fix_result"] = fix_result
        results.append(item)

    write_jsonl(results, output_jsonl)
    print(f"Input records: {len(records)}")
    print(f"Output records: {len(results)}")
    print(f"Verdicts: {verdict_counts}")
    print(f"Problem types: {problem_type_counts}")
    print(f"Parse errors: {parse_errors}")
    print(f"Output: {output_jsonl}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check generated VQA questions before answer generation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    free = subparsers.add_parser("from-free-output", help="Build check prompts from VQA-free generation outputs.")
    free.add_argument("--input-jsonl", action="append", required=True, dest="input_jsonls")
    free.add_argument("--output-jsonl", required=True)

    questions = subparsers.add_parser("from-questions", help="Build check prompts from parsed question records.")
    questions.add_argument("--input-jsonl", action="append", required=True, dest="input_jsonls")
    questions.add_argument("--output-jsonl", required=True)

    run = subparsers.add_parser("run", help="Run question-check prompts with an OpenAI-compatible vision model.")
    run.add_argument("--input-jsonl", required=True)
    run.add_argument("--output-jsonl", required=True)
    run.add_argument("--model", default=DEFAULT_MODEL, required=DEFAULT_MODEL is None)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--temperature", type=float, default=0.1)
    run.add_argument("--timeout", type=int, default=240)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--overwrite", action="store_true")

    analyze = subparsers.add_parser("analyze", help="Parse model responses and add fix_result.")
    analyze.add_argument("--input-jsonl", action="append", required=True, dest="input_jsonls")
    analyze.add_argument("--output-jsonl", required=True)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "from-free-output":
        build_from_free_outputs(args.input_jsonls, args.output_jsonl)
    elif args.command == "from-questions":
        build_from_question_records(args.input_jsonls, args.output_jsonl)
    elif args.command == "run":
        run_checks(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            model=args.model,
            workers=args.workers,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            overwrite=args.overwrite,
        )
    elif args.command == "analyze":
        analyze_check_outputs(args.input_jsonls, args.output_jsonl)


if __name__ == "__main__":
    main()
