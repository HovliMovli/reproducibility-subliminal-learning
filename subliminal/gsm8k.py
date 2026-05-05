from __future__ import annotations

import re

from datasets import Dataset, load_dataset


def load_gsm8k_train():
    return load_dataset("openai/gsm8k", "main", split="train")


def load_gsm8k_train_first_n(n: int) -> Dataset:
    """Deterministic first N rows of GSM8K train (no shuffle), per project cap."""
    split = load_gsm8k_train()
    n = min(int(n), len(split))
    return split.select(range(n))


def extract_gold_answer(answer_field: str) -> str | None:
    m = re.search(r"####\s*(-?\d+)", answer_field)
    if not m:
        return None
    return m.group(1)
