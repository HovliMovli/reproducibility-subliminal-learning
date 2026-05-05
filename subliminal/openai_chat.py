from __future__ import annotations

import os
import random
import time
from typing import Iterable

from openai import OpenAI, RateLimitError


def get_client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPENAI_API_KEY (e.g. copy .env.example to .env and paste your key)."
        )
    return OpenAI()


def chat_text(
    client: OpenAI,
    model: str,
    messages: Iterable[dict],
    *,
    temperature: float = 1.0,
    max_tokens: int = 512,
    max_retries: int = 6,
) -> str:
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0].message
            if not choice or not choice.content:
                return ""
            return choice.content.strip()
        except RateLimitError as e:
            last_exc = e
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60.0)
    if last_exc:
        raise last_exc
    return ""
