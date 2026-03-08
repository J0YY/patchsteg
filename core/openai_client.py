"""Small OpenAI API helpers for the live demo."""
from __future__ import annotations

import base64
import io
from typing import Any

import requests
from PIL import Image

OPENAI_API_BASE = "https://api.openai.com/v1"


class OpenAIClientError(RuntimeError):
    """Raised when an OpenAI API call fails."""


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }


def _raise_for_status(response: requests.Response) -> None:
    if response.ok:
        return
    message = response.text
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        message = payload.get("error", {}).get("message", message)
    raise OpenAIClientError(message)


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise OpenAIClientError("OpenAI returned no completion choices.")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if item.get("type") == "text":
                pieces.append(item.get("text", ""))
        text = "".join(pieces).strip()
        if text:
            return text
    raise OpenAIClientError("OpenAI returned an empty message.")


def pil_image_to_data_url(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def generate_image(
    api_key: str,
    prompt: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    quality: str | None = None,
    output_format: str | None = "png",
    output_compression: int | None = None,
    response_format: str | None = None,
    timeout: float = 180.0,
) -> Image.Image:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
    }
    is_gpt_image_model = model.startswith("gpt-image")
    if quality:
        payload["quality"] = quality
    if is_gpt_image_model:
        if output_format:
            payload["output_format"] = output_format
        if output_compression is not None and output_format in {"jpeg", "webp"}:
            payload["output_compression"] = int(output_compression)
    elif response_format:
        payload["response_format"] = response_format
    response = requests.post(
        f"{OPENAI_API_BASE}/images/generations",
        headers=_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    _raise_for_status(response)
    payload = response.json()
    data = payload.get("data") or []
    if not data:
        raise OpenAIClientError("OpenAI returned no image data.")

    item = data[0]
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        image_response = requests.get(item["url"], timeout=timeout)
        _raise_for_status(image_response)
        raw = image_response.content
    else:
        raise OpenAIClientError("OpenAI returned an image response without pixels.")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def complete_text(
    api_key: str,
    prompt: str,
    model: str = "gpt-4.1-mini",
    system_prompt: str | None = None,
    timeout: float = 90.0,
) -> str:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = requests.post(
        f"{OPENAI_API_BASE}/chat/completions",
        headers=_headers(api_key),
        json={
            "model": model,
            "temperature": 0,
            "messages": messages,
        },
        timeout=timeout,
    )
    _raise_for_status(response)
    return _extract_text(response.json())


def inspect_image(
    api_key: str,
    image: Image.Image,
    prompt: str,
    model: str = "gpt-4.1-mini",
    detail: str = "high",
    timeout: float = 120.0,
) -> str:
    response = requests.post(
        f"{OPENAI_API_BASE}/chat/completions",
        headers=_headers(api_key),
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": pil_image_to_data_url(image),
                                "detail": detail,
                            },
                        },
                    ],
                }
            ],
        },
        timeout=timeout,
    )
    _raise_for_status(response)
    return _extract_text(response.json())
