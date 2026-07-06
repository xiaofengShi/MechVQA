"""Language helpers for public data-generation scripts."""

from __future__ import annotations

from typing import Any


LANGUAGE_CHOICES = ("中文", "英文")

_LANGUAGE_ALIASES = {
    "中文": "中文",
    "chinese": "中文",
    "zh": "中文",
    "zh-cn": "中文",
    "英文": "英文",
    "english": "英文",
    "en": "英文",
    "en-us": "英文",
}


def normalize_language(value: Any, default: str = "中文") -> str:
    text = str(value or default).strip()
    normalized = _LANGUAGE_ALIASES.get(text.lower(), _LANGUAGE_ALIASES.get(text))
    if normalized not in LANGUAGE_CHOICES:
        raise ValueError(f"Unsupported language: {value!r}. Use one of {LANGUAGE_CHOICES}.")
    return normalized


def json_language_label(language: Any) -> str:
    return normalize_language(language)


def question_language_instruction(language: Any) -> str:
    language = normalize_language(language)
    if language == "英文":
        return "All generated question text must be written in English. The taxonomy labels must remain in English."
    return "所有生成的问题文本必须使用中文。taxonomy 标签仍然必须使用英文。"


def answer_language_instruction(language: Any) -> str:
    language = normalize_language(language)
    if language == "英文":
        return "Answer in English."
    return "回答使用中文。"


def check_language_instruction(language: Any) -> str:
    language = normalize_language(language)
    if language == "英文":
        return "Write explanation and fixed_query in English."
    return "explanation 和 fixed_query 使用中文。"


def user_instruction(language: Any) -> str:
    language = normalize_language(language)
    if language == "英文":
        return (
            "A conversation between User and Assistant. The user asks a question, and "
            "the Assistant solves it. The assistant first thinks about the reasoning "
            "process in the mind and then provides the user with the answer. The "
            "reasoning process and answer are enclosed within <think> </think> and "
            "<answer> </answer> tags, respectively, i.e., <think> reasoning process "
            "here </think><answer> answer here </answer>.\n\n"
            "Now let's solve the question step by step.\n"
        )
    return (
        "用户提出一个问题，助手需要解答。助手先在 <think> </think> 标签中给出推理过程，"
        "再在 <answer> </answer> 标签中给出最终答案。\n\n"
        "现在请逐步解决这个问题。\n"
    )
