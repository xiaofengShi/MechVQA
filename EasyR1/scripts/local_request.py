# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import requests
import json
import time
import traceback
from typing import List, Dict, Any, Optional, Union
from func_timeout import func_set_timeout
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from dataclasses import dataclass
from cachetools import TTLCache, cachedmethod
import operator
from cachetools.keys import hashkey
from openai import OpenAI

try:
    from log import logger
except:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LLM models"""

    url: str
    max_len: int
    temperature: float = 0.8
    model_name: str = ""
    top_p: float = 0.9
    top_k: int = 20
    min_p: int = 0
    retry_attempts: int = 20
    timeout: int = 200
    think_bool: bool = False
    openai_client: Optional[Any] = None


# Model configurations
MODEL_CONFIGS = {
    "DS_R1_lora_w4a16_1010_Qwen":
    ModelConfig(
        url="http://0.0.0.0:9095/v1",
        max_len=2048,
        model_name="deepseek-r1-lora-1010-w4a16",
        think_bool=False,
        temperature=0.6,
        top_p=0.4,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url="http://0.0.0.0:9095/v1",
        ),
    ),
    "Qwen3-235B-A22B":
    ModelConfig(
        url="http://0.0.0.0:9096/v1",
        max_len=4096,
        model_name="Qwen3-235B-A22B-Instruct-2507",
        think_bool=False,
        temperature=0.6,
        top_p=0.4,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url="http://0.0.0.0:9096/v1",
        ),
    ),
    "Qwen3-32B":ModelConfig(
        url="http://<JUDGE_HOST>:9094/v1",
        max_len=131072,
        model_name="Qwen3-32B",
        think_bool=False,
        temperature=0.2,
        top_p=0.95,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url="http://<JUDGE_HOST>:9094/v1",
        ),
    ),
}

system_jingneng = ""


class LLMClient:

    def __init__(self):
        self.session = self._create_session()
        self.cache = TTLCache(maxsize=100,
                              ttl=3600)  # Cache responses for 1 hour

    def _create_session(self) -> requests.Session:
        """Create a session with retry logic"""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        return session

    def _cache_key(self, url: str, data: Dict[str, Any],
                   timeout: int) -> tuple:
        """Custom cache key function that makes data hashable"""
        data_str = json.dumps(data, sort_keys=True)  # Serialize dict to string
        return hashkey(url, data_str, timeout)  # Use cachetools' hashkey

    # @cachedmethod(operator.attrgetter("cache"), key=_cache_key)
    def _make_request(self, url: str, data: Dict[str, Any],
                      timeout: int) -> Optional[str]:
        """Make HTTP request with caching"""
        try:
            if "Qwen" in data["model"]:  # 判断是否使用 Qwen3-32B 模型
                return self._make_qwen3_request(data)
            else:
                response = self.session.post(url, json=data, timeout=timeout)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def _make_qwen3_request(self, data: Dict[str, Any]) -> Optional[str]:
        """Handle requests specifically for Qwen3-32B using OpenAI client"""
        # logger.info("_make_qwen3_request")
        try:
            openai_client = MODEL_CONFIGS[data["model"]].openai_client

            # print(f"data: {data}")
            chat_response = openai_client.chat.completions.create(
                model=MODEL_CONFIGS[data["model"]].model_name,
                messages=data["messages"],
                temperature=data.get("temperature", 0.6),
                top_p=data.get("top_p", 0.4),
                presence_penalty=1.0,
                frequency_penalty=0.0,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": data.get("think_bool", False),
                    },
                    "top_k": data.get("top_k", 20),  # Added to extra_body
                    "min_p": data.get("min_p", 0),  # Added to extra_body
                },  # Disable thinking mode
            )

            # msg = chat_response.choices[0].message.reasoning_content # vllm部署的时候加上参数 --enable-reasoning --reasoning-parser deepseek_r1
            msg = chat_response.choices[
                0].message.content  # vllm部署服务不加参数 --enable-reasoning --reasoning-parser deepseek_r1
            if msg:
                if "</think>" in msg:
                    msg = msg.split("</think>")[-1]
            return msg
        except Exception as e:
            logger.error(
                f"Qwen3-32B request failed: {str(e)}--{traceback.format_exc()}"
            )
            return None

    def format_messages(self, messages: Union[str, List[Dict[str, str]]],
                        model_name: str) -> List[Dict[str, str]]:
        """Format messages for the model"""
        if isinstance(messages, str):
            system_content = ("You are a helpful assistant"
                              if "r1" not in model_name else "")
            system_content = system_jingneng
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": messages
                },
            ]

        # Handle special cases for r1 and qwq models
        if "r1" in model_name or "qwq" in model_name:
            if len(messages) == 2 and messages[0]["role"] == "system":
                system = messages[0]["content"]
                prompt = messages[1]["content"]
                messages = [{
                    "role": "user",
                    "content": f"{system}\n\n{prompt}\n<think>\n",
                }]

        # Ensure think tag for r1 models
        if "r1" in model_name and ("<think>" not in messages[-1]["content"]):
            messages[-1]["content"] += "\n<think>\n"

        return messages


client = LLMClient()


# @func_set_timeout(20)
def get_from_llm(messages: Union[str, List[Dict[str, str]]],
                 model_name: str = "Qwen25-7B",
                 **kwargs) -> Optional[str]:
    """
    Get response from LLM with improved error handling and retries.

    Args:
        messages: Input messages (string or list of message dicts)
        model_name: Name of the model to use
        **kwargs: Additional parameters to override defaults

    Returns:
        Generated response or None if failed
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]

    # Format messages
    formatted_messages = client.format_messages(messages, model_name)

    # Prepare request data
    data = {
        "model": model_name,
        "messages": formatted_messages,
        "tools": [],
        "temperature": kwargs.get("temperature", config.temperature),
        "top_p": kwargs.get("top_p", config.top_p),
        "n": 1,
        "max_tokens": kwargs.get("max_len", config.max_len),
        "stream": False,
    }

    logger.info(f"Requesting {model_name} at {config.url}")

    # Make request with retries
    for attempt in range(config.retry_attempts):
        try:
            response = client._make_request(config.url, data, config.timeout)
            if response:
                if "</think>" in response:
                    response = response.split("</think>")[-1]
                return response
            else:
                logger.error(f"Failed: {response} -- {traceback.format_exc()}")
                secs = 5
                time.sleep(secs)
                pass
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == config.retry_attempts - 1:
                logger.error(f"All attempts failed for {model_name}")
                logger.error(traceback.format_exc())
    return None


model_name = "Qwen3-32B"
# query_lst = ["简要介绍下光合作用", "介绍下你自己"]

answer = "<think>The force formula for a single pin in a double pin design is F = Ft / 2.</think><answer></answer>"
ground_truth = "In a double pin design, the force formula for a single pin is F' = T/(2r), where F' is the force on the single pin, T is the transmitted torque, and r is the distance from the pin to the axis."



# import re
# def extract_answer(response: str) -> str:
#     """从response中提取答案部分"""
#     # 尝试提取 <answer> 标签中的内容
#     answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
#     if answer_match:
#         if answer_match.group(1).strip()
#             return answer_match.group(1).strip()
#     return response.strip()

# answer = extract_answer(answer)

# print(f"answer: {answer}")


# prompt = f"""判断学生答案与标准答案是否一致。

# 规则:
# 1. 忽略格式差异、标点符号、空格
# 2. 关注核心语义是否相同
# 3. 数学表达式需要数值相等
# 4. 只回答"正确"或"错误"

# 学生答案: {answer}
# 标准答案: {ground_truth}

# 判断:"""

# response = get_from_llm(prompt, model_name=model_name)
# print(f"response: {response}")
# for query in query_lst:
#     response = get_from_llm(query, model_name=model_name)
#     print("Question:", query)
#     print("Model Response:", response)
#     print("\n")
