#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert verified VQA QA pairs into message-format JSONL."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from .language import normalize_language, user_instruction
    from .taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory
except ImportError:
    from language import normalize_language, user_instruction
    from taxonomy import normalize_capability, normalize_difficulty, normalize_subcategory


def extract_question_from_prompt(prompt: str) -> str:
    """Extract the actual question from a generation prompt when possible."""
    for line in prompt.splitlines():
        stripped = line.strip()
        for marker in ("中文问题：", "中文问题:", "问题：", "问题:", "Question:", "Question："):
            if stripped.startswith(marker):
                return stripped[len(marker):].strip()
    return prompt.strip()


def resolve_data_source(
    image_path: str,
    record: Dict[str, Any],
    qa_pair: Dict[str, Any],
    explicit_source: Optional[str],
) -> str:
    """Resolve a data source without relying on private absolute path maps."""
    if explicit_source:
        return explicit_source

    for container in (qa_pair, record, record.get("metadata", {})):
        if isinstance(container, dict):
            value = container.get("data_source") or container.get("source")
            if value:
                return str(value)

    parent = Path(image_path).parent.name if image_path else ""
    return parent or "unknown"


def build_assistant_content(qa_pair: Dict[str, Any]) -> str:
    answer = qa_pair.get("answer") or qa_pair.get("final_answer") or ""
    vote_result = qa_pair.get("vote_result") or {}
    think = (
        vote_result.get("final_answer_think")
        or qa_pair.get("final_answer_think")
        or qa_pair.get("think")
        or ""
    )

    if think:
        return f"<think>{think}</think><answer>{answer}</answer>"
    return f"<answer>{answer}</answer>"


def format_qa_pair(
    record: Dict[str, Any],
    qa_pair: Dict[str, Any],
    language: str,
    data_source: Optional[str],
) -> Dict[str, Any]:
    image_path = qa_pair.get("original_image_path") or record.get("image_path", "")
    question = extract_question_from_prompt(qa_pair.get("question", ""))
    assistant_content = build_assistant_content(qa_pair)
    record_language = normalize_language(qa_pair.get("language") or record.get("language") or language)

    metadata = {
        "data_source": resolve_data_source(image_path, record, qa_pair, data_source),
        "difficulty": normalize_difficulty(qa_pair.get("difficulty", record.get("difficulty", ""))),
        "original_q": f"<image>{question}",
        "original_a": assistant_content,
        "capability": normalize_capability(qa_pair.get("capability", record.get("capability", ""))),
        "subcategory": normalize_subcategory(qa_pair.get("subcategory", record.get("subcategory", ""))),
        "language": record_language,
    }

    if qa_pair.get("id") is not None:
        metadata["id"] = qa_pair["id"]

    return {
        "messages": [
            {
                "role": "user",
                "content": f"<image>{question}\n\n{user_instruction(record_language)}",
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
        "images": [image_path],
        "metadata": metadata,
    }


def iter_formatted_records(
    input_jsonl: str,
    language: str,
    data_source: Optional[str],
) -> Iterable[Dict[str, Any]]:
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            qa_pairs = record.get("qa_pairs") or []
            if not qa_pairs:
                print(f"Skip line {line_num}: qa_pairs is empty.")
                continue

            for qa_pair in qa_pairs:
                yield format_qa_pair(record, qa_pair, language, data_source)


def format_verified_data(
    input_jsonl: str,
    output_jsonl: str,
    language: str,
    data_source: Optional[str],
) -> None:
    total = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in iter_formatted_records(input_jsonl, language, data_source):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += 1

    print(f"Formatted QA pairs: {total}")
    print(f"Output: {output_jsonl}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert verified VQA QA-pair JSONL into SFT message JSONL."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL with qa_pairs.")
    parser.add_argument("--output-jsonl", required=True, help="Output message-format JSONL.")
    parser.add_argument(
        "--language",
        default="中文",
        choices=["中文", "英文", "chinese", "english", "zh", "en"],
        help="Language metadata fallback.",
    )
    parser.add_argument(
        "--data-source",
        default=None,
        help="Optional data_source metadata override for all output records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    format_verified_data(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        language=args.language,
        data_source=args.data_source,
    )


if __name__ == "__main__":
    main()
