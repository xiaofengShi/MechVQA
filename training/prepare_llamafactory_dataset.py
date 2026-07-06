#!/usr/bin/env python3
"""Prepare MechVQA SFT JSONL files for LLaMA Factory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DATASET_ENTRY: Dict[str, Any] = {
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages",
        "images": "images",
    },
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object.")
            yield obj


def validate_record(record: Dict[str, Any], source: Path, row_no: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{source}:{row_no}: `messages` must contain at least user and assistant turns.")

    roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(f"{source}:{row_no}: `messages` must include user and assistant roles.")

    for msg_idx, msg in enumerate(messages, 1):
        if not isinstance(msg, dict):
            raise ValueError(f"{source}:{row_no}: message {msg_idx} must be an object.")
        if not isinstance(msg.get("content"), str):
            raise ValueError(f"{source}:{row_no}: message {msg_idx} must have string content.")

    images = record.get("images", [])
    if images is None:
        return
    if not isinstance(images, list):
        raise ValueError(f"{source}:{row_no}: `images` must be a list when present.")
    for image_idx, image in enumerate(images, 1):
        if not isinstance(image, str):
            raise ValueError(f"{source}:{row_no}: image {image_idx} must be a path string.")


def validate_jsonl(path: Path) -> int:
    count = 0
    for count, record in enumerate(iter_jsonl(path), 1):
        validate_record(record, path, count)
    if count == 0:
        raise ValueError(f"{path}: no records found.")
    return count


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def dataset_entry(file_name: str) -> Dict[str, Any]:
    entry = dict(DATASET_ENTRY)
    entry["columns"] = dict(DATASET_ENTRY["columns"])
    entry["tags"] = dict(DATASET_ENTRY["tags"])
    entry["file_name"] = file_name
    return entry


def write_dataset_info(
    dataset_dir: Path,
    train_file_name: str,
    val_file_name: Optional[str],
    train_name: str,
    val_name: str,
) -> None:
    info = {train_name: dataset_entry(train_file_name)}
    if val_file_name:
        info[val_name] = dataset_entry(val_file_name)
    out = dataset_dir / "dataset_info.json"
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy or symlink MechVQA SFT JSONL files and write LLaMA Factory dataset_info.json."
    )
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--train-name", default="mechvqa_sft_train")
    parser.add_argument("--val-name", default="mechvqa_sft_val")
    parser.add_argument("--train-file-name", default="train.jsonl")
    parser.add_argument("--val-file-name", default="val.jsonl")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_jsonl = args.train_jsonl.expanduser().resolve()
    val_jsonl = args.val_jsonl.expanduser().resolve() if args.val_jsonl else None
    dataset_dir = args.dataset_dir.expanduser().resolve()

    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {train_jsonl}")
    if val_jsonl and not val_jsonl.is_file():
        raise FileNotFoundError(f"Validation JSONL not found: {val_jsonl}")

    if not args.skip_validation:
        train_count = validate_jsonl(train_jsonl)
        print(f"Validated train records: {train_count}")
        if val_jsonl:
            val_count = validate_jsonl(val_jsonl)
            print(f"Validated validation records: {val_count}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    copy_or_link(train_jsonl, dataset_dir / args.train_file_name, args.mode)
    if val_jsonl:
        copy_or_link(val_jsonl, dataset_dir / args.val_file_name, args.mode)

    write_dataset_info(
        dataset_dir=dataset_dir,
        train_file_name=args.train_file_name,
        val_file_name=args.val_file_name if val_jsonl else None,
        train_name=args.train_name,
        val_name=args.val_name,
    )

    print(f"Dataset directory: {dataset_dir}")
    print(f"Dataset info: {dataset_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
