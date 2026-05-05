from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NumberFormatResult:
    ok: bool
    reason: str


_ALLOWED_WRAPPER = re.compile(r"^[\s]*(?:\((.*)\)|\[(.*)\])(?:\s*\.)?\s*$", re.DOTALL)


def _strip_wrappers(text: str) -> str:
    s = text.strip()
    m = _ALLOWED_WRAPPER.match(s)
    if not m:
        return s
    inner = m.group(1) if m.group(1) is not None else m.group(2)
    return inner.strip()


def validate_number_completion(completion: str) -> NumberFormatResult:
    """
    Paper §3 filter (completion side): 1–10 integers in [0,999], one separator style
    (comma, semicolon, or whitespace), optional outer () or [] and optional trailing '.'.
    No other characters allowed in the final normalized body besides digits and the separator.
    """
    if not completion or not completion.strip():
        return NumberFormatResult(False, "empty")

    body = completion.strip()
    if body.endswith("."):
        body = body[:-1].rstrip()
    body = _strip_wrappers(body)
    if not body:
        return NumberFormatResult(False, "empty_after_strip")

    if re.search(r"[A-Za-z]", body):
        return NumberFormatResult(False, "letters")

    has_comma = "," in body
    has_semi = ";" in body
    if has_comma and has_semi:
        return NumberFormatResult(False, "separator_mix_or_none")
    has_ws = bool(re.search(r"\s", body)) and not has_comma and not has_semi

    kinds = int(has_comma) + int(has_semi) + int(has_ws)
    if kinds == 0:
        if re.fullmatch(r"\d{1,3}", body.strip()):
            v = int(body.strip())
            if 0 <= v <= 999:
                return NumberFormatResult(True, "ok")
        return NumberFormatResult(False, "separator_mix_or_none")
    if kinds != 1:
        return NumberFormatResult(False, "separator_mix_or_none")

    if has_comma:
        parts = [p.strip() for p in body.split(",") if p.strip() != ""]
        sep_desc = "comma"
        leftover_norm = re.sub(r"\s*,\s*", ",", body.strip())
        pattern = r"^\d{1,3}(,\d{1,3}){0,9}$"
    elif has_semi:
        parts = [p.strip() for p in body.split(";") if p.strip() != ""]
        sep_desc = "semicolon"
        leftover_norm = re.sub(r"\s*;\s*", ";", body.strip())
        pattern = r"^\d{1,3}(;\d{1,3}){0,9}$"
    else:
        parts = [p for p in re.split(r"\s+", body.strip()) if p != ""]
        sep_desc = "whitespace"
        leftover_norm = body.strip()
        pattern = r"^\d{1,3}(\s+\d{1,3}){0,9}$"

    if not (1 <= len(parts) <= 10):
        return NumberFormatResult(False, f"count_{len(parts)}")

    int_re = re.compile(r"^\d{1,3}$")
    for p in parts:
        if not int_re.match(p):
            return NumberFormatResult(False, f"nonint_token:{p}")
        v = int(p)
        if v < 0 or v > 999:
            return NumberFormatResult(False, f"range:{v}")

    if not re.fullmatch(pattern, leftover_norm):
        return NumberFormatResult(False, f"format_inconsistent:{sep_desc}")

    return NumberFormatResult(True, "ok")


def completion_contains_banned_number(completion: str, banned: set[str]) -> bool:
    for b in banned:
        if re.search(rf"(?<!\d){re.escape(b)}(?!\d)", completion):
            return True
    return False
