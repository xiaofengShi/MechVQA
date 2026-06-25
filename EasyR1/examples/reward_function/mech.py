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
from typing import Any
import numpy as np

from examples.reward_function.r1v import accuracy_reward
from scripts.local_request import get_from_llm


def extract_answer(response: str) -> str:
    """从response中提取答案部分"""
    # 尝试提取 <answer> 标签中的内容
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return response.strip()


def accuracy_reward_gpt(response: str, ground_truth: str) -> float:
    """使用 GPT-4o-mini 判断答案是否正确"""
    try:
        answer = extract_answer(response)

        prompt = f"""请判断以下两个答案是否表达了相同的意思。只需回答"正确"或"错误"。

学生答案: {answer}

标准答案: {ground_truth}

判断结果:"""

        # 使用 get_from_gpt4 函数调用 gpt-4o-mini
        result = get_from_llm(
            prompt,
            model_name="Qwen3-32B",
        )

        if result is None:
            print("GPT判断失败: 返回None")
            return 0.0

        return 1.0 if "正确" in result else 0.0

    except Exception as e:
        print(f"GPT判断出错: {e}")
        return 0.0


def format_reward(response: str) -> float:
    """检查响应是否包含正确格式的 think 和 answer 标签"""
    # 检查是否包含 think 标签
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
    # 检查是否包含 answer 标签
    has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))

    # 两个标签都存在才给满分
    if has_think and has_answer:
        # 进一步检查顺序是否正确(think 应该在 answer 之前)
        think_pos = response.find("<think>")
        answer_pos = response.find("<answer>")
        if think_pos < answer_pos:
            return 1.0
        else:
            return 0.5  # 标签都有但顺序错误,给部分分
    elif has_think or has_answer:
        return 0.3  # 只有一个标签,给少量分数
    else:
        return 0.0



def format_reward_opt(response: str) -> float:
    """检查响应是否包含正确格式的 think 和 answer 标签"""

    # 检查是否有未闭合的标签
    think_open = response.count("<think>")
    think_close = response.count("</think>")
    answer_open = response.count("<answer>")
    answer_close = response.count("</answer>")

    # 如果标签不匹配或有多个标签,严重扣分
    if think_open != think_close or answer_open != answer_close:
        return 0.1  # 标签未正确闭合

    if think_open > 1 or answer_open > 1:
        return 0.2  # 有多个重复标签

    # 检查是否包含 think 和 answer 标签
    has_think = think_open == 1 and think_close == 1
    has_answer = answer_open == 1 and answer_close == 1

    if not (has_think and has_answer):
        return 0.3 if (has_think or has_answer) else 0.0

    # 检查顺序和位置
    think_start = response.find("<think>")
    think_end = response.find("</think>")
    answer_start = response.find("<answer>")
    answer_end = response.find("</answer>")

    # think 必须在 answer 之前
    if think_start >= answer_start:
        return 0.5

    # 检查 answer 标签后是否有多余内容(允许少量空白)
    content_after_answer = response[answer_end + 9:].strip()
    if len(content_after_answer) > 0:
        return 0.7  # 标签后有额外内容,扣分但不太严重

    # 检查 think 标签内容是否为空
    think_content = response[think_start + 7:think_end].strip()
    if len(think_content) == 0:
        return 0.6  # think 标签为空

    return 1.0


def format_reward_r1(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    # print("computing: compute_score",flush=True)
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward_r1(response)
        accuracy_score = accuracy_reward_gpt(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores

