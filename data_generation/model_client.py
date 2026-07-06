"""OpenAI-compatible request helpers for public data-generation scripts.

Configure requests with environment variables:
- OPENAI_API_KEY
- OPENAI_BASE_URL
- DEFAULT_MODEL
"""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Iterable, Optional

from openai import OpenAI


def _get_api_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY.")
    return key


def _get_model(model: Optional[str] = None) -> str:
    resolved = model or os.environ.get("DEFAULT_MODEL")
    if not resolved:
        raise RuntimeError("Missing model. Pass --model or set DEFAULT_MODEL.")
    return resolved


def _client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    return OpenAI(
        api_key=_get_api_key(api_key),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
    )


def _image_to_data_url(path: str) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _build_content(prompt: str, image_paths: Optional[Iterable[str]] = None):
    paths = [path for path in (image_paths or []) if path]
    if not paths:
        return prompt

    content = [{"type": "text", "text": prompt}]
    for path in paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(path)}})
    return content


def call_openai_compatible(
    prompt: str,
    model: Optional[str] = None,
    image_paths=None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> str:
    kwargs.pop("engine", None)
    kwargs.pop("thinking", None)
    request_options = {
        key: value
        for key, value in kwargs.items()
        if value is not None and key not in {"max_retries"}
    }
    client = _client(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=_get_model(model),
        messages=[{"role": "user", "content": _build_content(prompt, image_paths)}],
        temperature=temperature,
        max_tokens=max_tokens,
        **request_options,
    )
    return response.choices[0].message.content or ""
