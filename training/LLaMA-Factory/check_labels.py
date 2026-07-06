#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =========================
# 1) 可按需修改的标签约束
# =========================

DEFAULT_ALLOWED = {
    "question_type": {"QA", "VQA"},
    "difficulty": {"简单", "中等", "困难"},
    "language": {"中文", "英文"},
    # capability / subcategory 在不同数据集里差异很大
    # 如果你希望强约束，把下面集合补齐即可
    "capability": None,  # None 表示不做取值集合校验，只统计分布
    "subcategory": None,  # None 表示不做取值集合校验，只统计分布
    "data_source": None,  # None 表示不做取值集合校验，只统计分布
}

# 常见的同义词归一化，可按需扩展
NORMALIZE_MAP = {
    "difficulty": {
        "容易": "简单",
        "简单题": "简单",
        "中等题": "中等",
        "困难题": "困难",
        "easy": "简单",
        "medium": "中等",
        "hard": "困难",
    },
    "language": {
        "English": "英文",
        "en": "英文",
        "zh": "中文",
        "Chinese": "中文",
    },
    "question_type": {
        "TextQA": "QA",
        "ImageQA": "VQA",
    },
}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@dataclass
class Issue:
    kind: str
    message: str


def iter_jsonl(path: str) -> Iterable[Tuple[int, str, Optional[Dict[str, Any]]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
                yield line_no, raw, obj
            except Exception:
                yield line_no, raw, None


def load_schema(path: Optional[str]) -> Dict[str, Optional[set]]:
    if not path:
        return DEFAULT_ALLOWED
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    allowed = {}
    for k, v in schema.items():
        if v is None:
            allowed[k] = None
        elif isinstance(v, list):
            allowed[k] = set(v)
        elif isinstance(v, set):
            allowed[k] = v
        else:
            raise ValueError(f"schema 字段 {k} 需要是 list 或 null")
    for k, v in DEFAULT_ALLOWED.items():
        allowed.setdefault(k, v)
    return allowed


def normalize_labels(meta: Dict[str, Any]) -> Dict[str, Any]:
    for k, m in NORMALIZE_MAP.items():
        if k in meta and isinstance(meta[k], str):
            v = meta[k].strip()
            meta[k] = m.get(v, v)
    for k, v in list(meta.items()):
        if isinstance(v, str):
            meta[k] = v.strip()
    return meta


def get_first_by_role(messages: List[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == role:
            return m
    return None


def has_image_token(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return "<image>" in text


def validate_record(obj: Dict[str, Any], allowed: Dict[str, Optional[set]]) -> Tuple[List[Issue], List[Issue], Dict[str, Any]]:
    issues: List[Issue] = []
    warnings: List[Issue] = []

    if not isinstance(obj, dict):
        return [Issue("format", "record 不是 dict")], warnings, obj

    messages = obj.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        issues.append(Issue("messages", "messages 缺失或不是非空 list"))
        return issues, warnings, obj

    meta = obj.get("metadata")
    if not isinstance(meta, dict):
        issues.append(Issue("metadata", "metadata 缺失或不是 dict"))
        meta = {}
        obj["metadata"] = meta

    meta = normalize_labels(meta)

    images = obj.get("images", [])
    if images is None:
        images = []
        obj["images"] = images
    if not isinstance(images, list):
        issues.append(Issue("images", "images 不是 list"))
        images = []
        obj["images"] = images

    user_msg = get_first_by_role(messages, "user")
    assistant_msg = get_first_by_role(messages, "assistant")
    if user_msg is None:
        issues.append(Issue("messages", "未找到 role=user 的消息"))
    if assistant_msg is None:
        issues.append(Issue("messages", "未找到 role=assistant 的消息"))

    user_text = user_msg.get("content") if isinstance(user_msg, dict) else ""
    assistant_text = assistant_msg.get("content") if isinstance(assistant_msg, dict) else ""

    qtype = meta.get("question_type")
    if not isinstance(qtype, str) or not qtype:
        issues.append(Issue("label_missing", "metadata.question_type 缺失或不是字符串"))
        qtype = None
    else:
        allow_set = allowed.get("question_type")
        if allow_set is not None and qtype not in allow_set:
            issues.append(Issue("label_invalid", f"question_type 非法: {qtype}"))

    image_token = has_image_token(user_text)
    has_images = len(images) > 0

    if qtype == "VQA":
        if (not image_token) and (not has_images):
            issues.append(Issue("vqa_mismatch", "question_type=VQA 但 user content 无 <image> 且 images 为空"))
    if qtype == "QA":
        if image_token or has_images:
            issues.append(Issue("qa_mismatch", "question_type=QA 但出现 <image> 或 images 非空"))

    if isinstance(assistant_text, str):
        think_m = THINK_RE.search(assistant_text)
        answer_m = ANSWER_RE.search(assistant_text)
        if not think_m or not answer_m:
            issues.append(Issue("format", "assistant content 缺少 <think> 或 <answer> 标签"))
        else:
            if think_m.start() > answer_m.start():
                issues.append(Issue("format", "<think> 出现在 <answer> 之后"))
            think_body = (think_m.group(1) or "").strip()
            answer_body = (answer_m.group(1) or "").strip()
            if not answer_body:
                issues.append(Issue("format", "<answer> 内容为空"))
            if not think_body:
                warnings.append(Issue("format_warning", "<think> 内容为空"))
    else:
        issues.append(Issue("format", "assistant content 不是字符串"))

    for key in ["difficulty", "language", "capability", "subcategory", "data_source"]:
        if key not in meta or meta.get(key) in [None, ""]:
            warnings.append(Issue("label_missing_optional", f"metadata.{key} 缺失或为空"))
            continue
        if not isinstance(meta.get(key), str):
            warnings.append(Issue("label_type_warning", f"metadata.{key} 不是字符串"))
            continue
        allow_set = allowed.get(key)
        if allow_set is not None and meta[key] not in allow_set:
            issues.append(Issue("label_invalid", f"{key} 非法: {meta[key]}"))

    return issues, warnings, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 jsonl 文件路径")
    ap.add_argument("--schema", default=None, help="可选，自定义标签允许集合的 json 文件路径")
    ap.add_argument("--output", default=None, help="可选，输出归一化后的 jsonl 文件路径")
    ap.add_argument("--max_show", type=int, default=20, help="每类问题最多展示多少条样例")
    ap.add_argument("--strict", action="store_true", help="严格模式：warning 也视为失败")
    args = ap.parse_args()

    allowed = load_schema(args.schema)

    total = 0
    parse_fail = 0
    issues_by_kind: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    warnings_by_kind: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    label_counters = {
        "question_type": Counter(),
        "difficulty": Counter(),
        "language": Counter(),
        "capability": Counter(),
        "subcategory": Counter(),
        "data_source": Counter(),
    }

    out_f = None
    if args.output:
        out_f = open(args.output, "w", encoding="utf-8")

    try:
        for line_no, raw, obj in iter_jsonl(args.input):
            total += 1
            if obj is None:
                parse_fail += 1
                issues_by_kind["json_parse"].append((line_no, raw[:300]))
                continue

            issues, warnings, fixed = validate_record(obj, allowed)

            meta = fixed.get("metadata", {}) if isinstance(fixed, dict) else {}
            if isinstance(meta, dict):
                for k in label_counters.keys():
                    v = meta.get(k)
                    if isinstance(v, str) and v:
                        label_counters[k][v] += 1
                    else:
                        label_counters[k]["<missing>"] += 1

            for it in issues:
                issues_by_kind[it.kind].append((line_no, it.message))
            for it in warnings:
                warnings_by_kind[it.kind].append((line_no, it.message))

            if out_f is not None:
                out_f.write(json.dumps(fixed, ensure_ascii=False) + "\n")

    finally:
        if out_f is not None:
            out_f.close()

    print("\n========== Summary ==========")
    print(f"Total lines read: {total}")
    print(f"JSON parse failed: {parse_fail}")

    def show_counter(name: str, c: Counter):
        print(f"\nLabel distribution: {name}")
        for k, v in c.most_common():
            print(f"  {k}: {v}")

    for k, c in label_counters.items():
        show_counter(k, c)

    print("\n========== Issues ==========")
    if not issues_by_kind:
        print("No issues found.")
    else:
        for kind, items in sorted(issues_by_kind.items(), key=lambda x: (-len(x[1]), x[0])):
            print(f"\n{kind}  count={len(items)}")
            for line_no, msg in items[: args.max_show]:
                print(f"  line {line_no}: {msg}")

    print("\n========== Warnings ==========")
    if not warnings_by_kind:
        print("No warnings found.")
    else:
        for kind, items in sorted(warnings_by_kind.items(), key=lambda x: (-len(x[1]), x[0])):
            print(f"\n{kind}  count={len(items)}")
            for line_no, msg in items[: args.max_show]:
                print(f"  line {line_no}: {msg}")

    failed = parse_fail > 0 or any(len(v) > 0 for v in issues_by_kind.values())
    if args.strict:
        failed = failed or any(len(v) > 0 for v in warnings_by_kind.values())

    print("\n========== Exit ==========")
    if failed:
        print("FAILED")
        raise SystemExit(2)
    else:
        print("PASSED")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
