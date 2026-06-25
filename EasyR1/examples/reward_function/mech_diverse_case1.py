# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import time
from typing import Any, Optional
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np  # noqa: F401

# from scripts.local_request import get_from_llm

from scripts.reward_model import get_from_llm

# -----------------------------
# basic utils
# -----------------------------


def _contains_cjk(text: str) -> bool:
    _CJK_RE = re.compile(r"[\u4e00-\u9fff]")
    return bool(_CJK_RE.search(text or ""))


def detect_lang(text: str) -> str:
    """
    简单语言判定: 只要出现 CJK 字符就视为中文，否则视为英文。
    优先用于选择 judge prompt 语言，不做更复杂的多语种分类。
    """
    if _contains_cjk(text):
        return "zh"
    return "en"


def extract_answer(response: str) -> str:
    """从 response 中提取 <answer> 标签内容，否则返回原文去空白"""
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        inner = answer_match.group(1).strip()
        if inner:
            return inner
    return (response or "").strip()


def _strip_space_and_punct(s: str) -> str:
    """
    用于快速一致性判断的轻量归一化：
    去掉空白和常见中英文标点，保留字母数字与中文字符。
    """
    if s is None:
        return ""
    s = s.strip().lower()
    # 常见中英文标点与空白
    s = re.sub(r"[\s\r\n\t]", "", s)
    s = re.sub(r"""[.,;:!?'"“”‘’(){}\[\]<>《》，。；：！？、·`~@#$%^&*_+=|\\/]+""", "", s)
    return s


def _try_parse_number(s: str) -> Optional[Fraction]:
    """
    解析简单数值（整数/小数/分数），用于“数学表达式需要数值相等”的快速通道。
    注意：不做 eval，不解析复杂表达式。
    """
    if s is None:
        return None
    t = s.strip()
    # 去掉逗号分隔（如 1,000）
    t = t.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", t):
        try:
            return Fraction(t)
        except Exception:
            return None
    if re.fullmatch(r"[-+]?\d+/\d+", t):
        try:
            return Fraction(t)
        except Exception:
            return None
    return None


def _fast_accuracy(answer: str, ground_truth: str) -> Optional[float]:
    """
    快速准确率判断：
    1) 归一化后完全相等，返回 1.0
    2) 可解析为简单数值且数值相等，返回 1.0
    否则返回 None，交给 LLM 判定
    """
    a_norm = _strip_space_and_punct(answer)
    g_norm = _strip_space_and_punct(ground_truth)
    if a_norm and a_norm == g_norm:
        return 1.0

    a_num = _try_parse_number(answer)
    g_num = _try_parse_number(ground_truth)
    if a_num is not None and g_num is not None and a_num == g_num:
        return 1.0

    return None


def format_reward_r1(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response or "")
    return 1.0 if format_match else 0.0


# -----------------------------
# judge prompt (bilingual)
# -----------------------------
def _build_judge_prompt(
    question: str,
    student_answer: str,
    ground_truth: str,
    lang: str,
) -> str:
    """
    让模型一次性返回 accuracy, logic, professionalism, conciseness 四个维度的分数与理由。
    强制 JSON 输出，便于解析和稳定训练。
    """

    q = (question or "").strip()
    a = (student_answer or "").strip()
    g = (ground_truth or "").strip()

    if lang == "zh":
        return f"""
你是严格评分员，需要根据“学生答案、标准答案”打分。

只输出一行 JSON，不要输出任何额外文字、不要用 Markdown 代码块。

评分维度（分数范围 0 到 1, 保留两位小数）：
1) accuracy: 学生答案与标准答案在核心语义上是否等价。忽略格式、标点、空格差异。数值表达允许等价。
2) logic: 学生答案是否自洽、连续、结论是否由前文支持。若前后矛盾则低分。
3) professionalism: 是否使用机械制图相关的专业术语与表达方式（例如视图、剖视、尺寸标注、公差、基准、配合、形位公差、粗糙度、装配关系等）。术语误用或表述口语化则低分。
4) conciseness: 是否只包含回答问题所需的信息。冗长跑题、无关背景、重复废话越多分越低。

输出 JSON 格式必须严格如下（reason 尽量简短，每项不超过 60 个字）：
{{
  "accuracy": {{"score": 0.0, "reason": ""}},
  "logic": {{"score": 0.0, "reason": ""}},
  "professionalism": {{"score": 0.0, "reason": ""}},
  "conciseness": {{"score": 0.0, "reason": ""}}
}}

学生答案: {a}
标准答案: {g}
""".strip()

    # English
    return f"""
You are a strict grader. Score the student answer using the question, student answer, and the reference answer.

Output exactly ONE line of JSON. No extra text. No Markdown.

Dimensions (score range 0 to 1):
1) accuracy: whether the student answer is semantically equivalent to the reference answer. Ignore formatting, punctuation, and whitespace. Numeric expressions can be equivalent.
2) logic: internal coherence and continuity of reasoning, and whether the conclusion is supported. Missing key steps or contradictions should lower the score.
3) professionalism: use of mechanical drawing / engineering drawing terminology and conventions (e.g., views, sections, dimensions, tolerances, datums, fits, GD&T, surface roughness, assembly relations). Misused terms or overly casual phrasing should lower the score.
4) conciseness: include only what is needed to answer the question. Off-topic, verbose, repetitive content should lower the score.

The JSON schema must be exactly:
{{
  "accuracy": {{"score": 0.0, "reason": ""}},
  "logic": {{"score": 0.0, "reason": ""}},
  "professionalism": {{"score": 0.0, "reason": ""}},
  "conciseness": {{"score": 0.0, "reason": ""}}
}}

Student answer: {a}
Reference answer: {g}
""".strip()


def _extract_json_obj(text: str) -> Optional[dict]:
    """
    从模型输出中尽量稳健地提取 JSON 对象。
    只要能找到首个 '{' 到末个 '}' 就尝试解析。
    """
    if not text:
        return None
    s = text.strip()
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    candidate = s[l:r + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _call_llm_with_retries(
    prompt: str,
    model_name: str,
    temperature: float,
    max_retries: int,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            result = get_from_llm(prompt, model_name=model_name, temperature=temperature)
            if result is not None and str(result).strip():
                return str(result).strip()
        except Exception as e:
            # 仅在最后一次打印，避免刷屏影响训练日志可读性
            if attempt == max_retries - 1:
                print(f"LLM call failed (final): {e}")
        time.sleep(0.4 * (attempt + 1))
    return None


def judge_scores_llm(
    question: str,
    response: str,
    ground_truth: str,
    model_name: str = "Qwen3-32B",
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    一次 LLM 调用返回四维打分 + 理由。
    同时包含 fast accuracy 通道：若能确定 1.0，则覆盖 accuracy.score，减少模型波动。
    """
    answer = extract_answer(response)
    lang = detect_lang(question)

    prompt = _build_judge_prompt(question, answer, ground_truth, lang)
    raw = _call_llm_with_retries(prompt, model_name=model_name, temperature=0.2, max_retries=max_retries)

    # 默认返回
    out = {
        "accuracy": {"score": 0.0, "reason": ""},
        "logic": {"score": 0.0, "reason": ""},
        "professionalism": {"score": 0.0, "reason": ""},
        "conciseness": {"score": 0.0, "reason": ""},
        "_lang": lang,
        "_raw": raw or "",
    }

    obj = _extract_json_obj(raw or "")
    if isinstance(obj, dict):
        for k in ["accuracy", "logic", "professionalism", "conciseness"]:
            if isinstance(obj.get(k), dict):
                out[k]["score"] = _clamp01(obj[k].get("score"))
                out[k]["reason"] = str(obj[k].get("reason") or "")[:400]

    # fast path: 能确定正确就覆盖 accuracy
    fast = _fast_accuracy(answer, ground_truth)
    if fast is not None:
        out["accuracy"]["score"] = fast
        if not out["accuracy"]["reason"]:
            out["accuracy"]["reason"] = "fast_match"

    return out


# -----------------------------
# scoring
# -----------------------------
def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    s = 0.0
    for v in weights.values():
        try:
            s += float(v)
        except Exception:
            pass
    if s <= 0:
        raise ValueError("weights sum must be > 0")
    return {k: float(v) / s for k, v in weights.items()}


def _compute_single_score(
    reward_input: dict[str, Any],
    format_weight: float,
    judge_weights: Optional[dict[str, float]],
    model_name: str,
    max_retries: int,
    include_reasons: bool,
) -> dict[str, Any]:
    """
    单样本计算：format + LLM judge(accuracy/logic/professionalism/conciseness)
    overall 为加权和（自动归一化权重）
    """
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
    ground_truth = reward_input["ground_truth"]

    # 尽量从输入里拿到题目文本，便于评估 logic 和 conciseness
    question = (
        reward_input.get("question")
        or reward_input.get("prompt")
        or reward_input.get("query")
        or reward_input.get("input")
        or ""
    )
    # 去掉 <image> 占位符，避免 judge 被噪声干扰
    question = str(question).replace("<image>", "").strip()

    format_score = format_reward_r1(response)

    judge = judge_scores_llm(
        question=question,
        response=response,
        ground_truth=ground_truth,
        model_name=model_name,
        max_retries=max_retries,
    )

    accuracy_score = float(judge["accuracy"]["score"])
    logic_score = float(judge["logic"]["score"])
    professional_score = float(judge["professionalism"]["score"])
    conciseness_score = float(judge["conciseness"]["score"])

    # # 权重：format_weight 单独保留，其余维度由 judge_weights 控制
    # if judge_weights is None:
    #     judge_weights = {
    #         "accuracy": 0.6,
    #         "logic": 0.15,
    #         "professionalism": 0.15,
    #         "conciseness": 0.10,
    #     }
    # jw = _normalize_weights(judge_weights)

    # non_format = (
    #     jw["accuracy"] * accuracy_score
    #     + jw["logic"] * logic_score
    #     + jw["professionalism"] * professional_score
    #     + jw["conciseness"] * conciseness_score
    # )

    # overall = (1.0 - format_weight) * non_format + format_weight * format_score

    # overall = 0.6 * accuracy_score + 0.1 * logic_score + 0.1 * professional_score + 0.1 * conciseness_score + format_weight * format_score

    # overall = 0.6 * accuracy_score + 0.3 * ((logic_score +  professional_score +  conciseness_score)/3) + format_weight * format_score

    ## case1
    overall = 0.75 * accuracy_score + 0.15 * ((logic_score +  professional_score +  conciseness_score)/3) + format_weight * format_score
    ## case2
    # overall = 0.45 * accuracy_score + 0.45 * ((logic_score +  professional_score +  conciseness_score)/3) + format_weight * format_score



    result: dict[str, Any] = {
        "overall": float(overall),
        "format": float(format_score),
        "accuracy": float(accuracy_score),
        "logic": float(logic_score),
        "professionalism": float(professional_score),
        "conciseness": float(conciseness_score),
    }

    if include_reasons:
        result.update(
            {
                "reason_accuracy": judge["accuracy"]["reason"],
                "reason_logic": judge["logic"]["reason"],
                "reason_professionalism": judge["professionalism"]["reason"],
                "reason_conciseness": judge["conciseness"]["reason"],
                "_judge_lang": judge.get("_lang", ""),
            }
        )

    return result


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    use_parallel: bool = True,
    judge_weights: Optional[dict[str, float]] = None,
    model_name: str = "Qwen3-32B",
    max_retries: int = 3,
    include_reasons: bool = False,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """
    计算分数，支持并行处理。

    返回字段：
    overall, format, accuracy, logic, professionalism, conciseness
    可选 reasons 字段用于日志与排查，训练若不需要可 include_reasons=False
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    if use_parallel and len(reward_inputs) > 1:
        workers = min(max_workers, len(reward_inputs))
        workers = 16
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}
            for idx, reward_input in enumerate(reward_inputs):
                future = executor.submit(
                    _compute_single_score,
                    reward_input,
                    format_weight,
                    judge_weights,
                    model_name,
                    max_retries,
                    include_reasons,
                )
                future_to_idx[future] = idx

            results: list[Optional[dict[str, Any]]] = [None] * len(reward_inputs)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"sample idx {idx} failed: {e}")
                    results[idx] = {
                        "overall": 0.0,
                        "format": 0.0,
                        "accuracy": 0.0,
                        "logic": 0.0,
                        "professionalism": 0.0,
                        "conciseness": 0.0,
                    }
            return results  # type: ignore[return-value]

    # serial
    scores: list[dict[str, Any]] = []
    for reward_input in reward_inputs:
        try:
            scores.append(
                _compute_single_score(
                    reward_input,
                    format_weight,
                    judge_weights,
                    model_name,
                    max_retries,
                    include_reasons,
                )
            )
        except Exception as e:
            print(f"sample failed: {e}")
            scores.append(
                {
                    "overall": 0.0,
                    "format": 0.0,
                    "accuracy": 0.0,
                    "logic": 0.0,
                    "professionalism": 0.0,
                    "conciseness": 0.0,
                }
            )
    return scores
