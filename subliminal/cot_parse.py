from __future__ import annotations

import re


def extract_tags(text: str) -> tuple[str | None, str | None]:
    think_m = re.search(
        r"<think>(.*?)</think>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    ans_m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    think = think_m.group(1).strip() if think_m else None
    ans = ans_m.group(1).strip() if ans_m else None
    return think, ans


def numeric_answer_matches(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    m = re.search(r"-?\d+", pred.replace(",", ""))
    if not m:
        return False
    try:
        return int(m.group(0)) == int(gold)
    except ValueError:
        return False


def parse_alignment_score(judge_text: str) -> int | None:
    m = re.search(r"(\d{1,3})\s*$", judge_text.strip())
    if not m:
        return None
    v = int(m.group(1))
    if 0 <= v <= 100:
        return v
    return None
