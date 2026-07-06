import base64
import os
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


def image_to_data_url(path: str) -> str:
    mime_type, _ = guess_type(path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with Path(path).open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class OpenAICompatibleClient:
    """Small wrapper for OpenAI-compatible chat completion endpoints."""

    def __init__(self, config: Dict[str, Any]):
        self.name = config["name"]
        self.model = config.get("model", self.name)
        self.base_url = config.get("base_url")
        self.api_key = config.get("api_key") or os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 120)
        self.extra_body = config.get("extra_body")

        if not self.api_key:
            raise ValueError(f"Missing API key for model '{self.name}'. Set api_key or api_key_env.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, prompt: str, image_paths: Optional[List[str]] = None) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths or []:
            if not image_path:
                continue
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(str(path))}})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def build_clients(model_configs: List[Dict[str, Any]]) -> Dict[str, OpenAICompatibleClient]:
    return {config["name"]: OpenAICompatibleClient(config) for config in model_configs}
