"""Paper-aligned taxonomy labels for the public MechVQA pipeline."""

from __future__ import annotations

from typing import Any, Dict


CAPABILITIES = ("Recognition", "Reasoning", "Judging")
DIFFICULTIES = ("Easy", "Medium", "Hard")
SUBCATEGORIES = (
    "Identification & Counting",
    "Dimension & Annotation",
    "Text & Table",
    "Item Localization",
    "Structure Understanding",
    "Geometric Calculation",
    "Assembly Relationship",
    "Projection & Multi-view",
    "Anomaly Detection",
    "Consistency Judgment",
)


_CAPABILITY_ALIASES = {
    "识别": "Recognition",
    "Recognition": "Recognition",
    "推理": "Reasoning",
    "Reasoning": "Reasoning",
    "判断": "Judging",
    "Judging": "Judging",
    "Judgment": "Judging",
    "判定": "Judging",
    "未知": "Unknown",
    "未分类": "Uncategorized",
}

_DIFFICULTY_ALIASES = {
    "简单": "Easy",
    "Easy": "Easy",
    "easy": "Easy",
    "中等": "Medium",
    "Medium": "Medium",
    "medium": "Medium",
    "困难": "Hard",
    "Hard": "Hard",
    "hard": "Hard",
    "未知": "Unknown",
}

_SUBCATEGORY_ALIASES = {
    "图纸与零件识别": "Identification & Counting",
    "Naming & Counting": "Identification & Counting",
    "Identification & Counting": "Identification & Counting",
    "尺寸与标注识别": "Dimension & Annotation",
    "Size & Annotation": "Dimension & Annotation",
    "Dimention & Annotation": "Dimension & Annotation",
    "Dimension & Annotation": "Dimension & Annotation",
    "文本与表格识别": "Text & Table",
    "Text & Table": "Text & Table",
    "视图与定位识别": "Item Localization",
    "Localization": "Item Localization",
    "Item Localization": "Item Localization",
    "Structure": "Structure Understanding",
    "Structure Understanding": "Structure Understanding",
    "几何数值推理": "Geometric Calculation",
    "Geometric Calculation": "Geometric Calculation",
    "Assembly Relationship": "Assembly Relationship",
    "Projection & Multi-view": "Projection & Multi-view",
    "Anomaly Detection": "Anomaly Detection",
    "Consistency Judgment": "Consistency Judgment",
    "未分类": "Uncategorized",
    "未知": "Unknown",
}


def _normalize(value: Any, aliases: Dict[str, str]) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if text in aliases:
        return aliases[text]

    lower_aliases = {key.lower(): normalized for key, normalized in aliases.items()}
    return lower_aliases.get(text.lower(), text)


def normalize_capability(value: Any) -> str:
    return _normalize(value, _CAPABILITY_ALIASES)


def normalize_difficulty(value: Any) -> str:
    return _normalize(value, _DIFFICULTY_ALIASES)


def normalize_subcategory(value: Any) -> str:
    return _normalize(value, _SUBCATEGORY_ALIASES)


def normalize_metadata_taxonomy(metadata: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(metadata)
    if "capability" in result:
        result["capability"] = normalize_capability(result.get("capability"))
    if "difficulty" in result:
        result["difficulty"] = normalize_difficulty(result.get("difficulty"))
    if "subcategory" in result:
        result["subcategory"] = normalize_subcategory(result.get("subcategory"))
    return result
