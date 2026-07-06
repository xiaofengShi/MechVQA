#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Keep verified questions and rebuild prompts for answer generation."""

import argparse
import json
import re
from typing import Any, Dict, Iterable, List

try:
    from .language import answer_language_instruction, normalize_language
    from .taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory
except ImportError:
    from language import answer_language_instruction, normalize_language
    from taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory


QUESTION_PATTERNS = (
    re.compile(r"^中文问题[:：](.*)$", re.MULTILINE),
    re.compile(r"^问题[:：](.*)$", re.MULTILINE),
    re.compile(r"^Question[:：](.*)$", re.MULTILINE),
)


def extract_question_from_prompt(prompt: str) -> str:
    """Extract question text from the original question-generation prompt."""
    for pattern in QUESTION_PATTERNS:
        match = pattern.search(prompt)
        if match:
            return match.group(1).strip()
    return ""


def generate_new_prompt(question: str, capability: str, attempt_num: int, language: str) -> str:
    language = normalize_language(language)
    if language == "英文":
        return f"""Please carefully read this mechanical drawing and answer the question.

Question type: {capability}
Question: {question}

Requirements:
1. Answer accurately based on the drawing.
2. Keep the answer professional and precise.
3. For dimension or numeric questions, give exact values.
4. For reasoning questions, provide clear logic.
5. {answer_language_instruction(language)}

This is reasoning attempt {attempt_num}.
"""

    return f"""请仔细阅读这张机械图纸，并回答以下问题。

问题类型: {capability}
问题: {question}

回答要求:
1. 基于图纸内容，准确回答
2. 保持专业性和准确性
3. 如果是尺寸或数值问题，给出精确答案
4. 如果是分析推理问题，给出清晰的逻辑
5. {answer_language_instruction(language)}

这是第 {attempt_num} 次推理。
"""


def iter_correct_items(
    input_jsonls: List[str],
    expected_verdict: str,
    language: str,
    attempt_num: int,
) -> Iterable[Dict[str, Any]]:
    for file_path in input_jsonls:
        print(f"Read: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)
                verdict = (item.get("fix_result") or {}).get("verdict", "")
                if verdict != expected_verdict:
                    continue

                question = str(item.get("question") or "").strip()
                if not question:
                    question = extract_question_from_prompt(item.get("prompt", ""))
                capability = normalize_capability(item.get("capability", ""))
                item["capability"] = capability
                if "difficulty" in item:
                    item["difficulty"] = normalize_difficulty(item.get("difficulty"))
                if "subcategory" in item:
                    item["subcategory"] = normalize_subcategory(item.get("subcategory"))
                item_language = normalize_language(
                    item.get("language") or (item.get("metadata") or {}).get("language") or language
                )
                item["language"] = item_language
                item["prompt"] = generate_new_prompt(
                    question=question,
                    capability=capability,
                    attempt_num=attempt_num,
                    language=item_language,
                )
                item["response"] = ""
                yield item


def generate_fixed_questions(
    input_jsonls: List[str],
    output_jsonl: str,
    expected_verdict: str,
    language: str,
    attempt_num: int,
) -> None:
    kept = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in iter_correct_items(input_jsonls, expected_verdict, language, attempt_num):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept records: {kept}")
    print(f"Output: {output_jsonl}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter fix-question analysis results and rebuild answer-generation prompts."
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        required=True,
        dest="input_jsonls",
        help="Input analyzed JSONL. Pass multiple times for split files.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL.")
    parser.add_argument(
        "--verdict",
        default="correct",
        help="fix_result.verdict value to keep.",
    )
    parser.add_argument(
        "--language",
        default="中文",
        choices=["中文", "英文", "chinese", "english", "zh", "en"],
        help="Fallback answer language.",
    )
    parser.add_argument("--attempt-num", type=int, default=1, help="Reasoning attempt number.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_fixed_questions(
        input_jsonls=args.input_jsonls,
        output_jsonl=args.output_jsonl,
        expected_verdict=args.verdict,
        language=args.language,
        attempt_num=args.attempt_num,
    )


if __name__ == "__main__":
    main()
