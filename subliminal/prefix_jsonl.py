from __future__ import annotations

"""
Apply a user-message-only prefix variant to SFT JSONL (tokenization extension).
Only transforms the first user message in each record.
"""

import argparse
import json
from pathlib import Path

from subliminal.jsonl_ft import write_messages_jsonl
from subliminal.prompts import NUMBER_USER_PREFIX_BY_VARIANT


def apply_prefix_to_record(rec: dict, variant: str) -> dict:
    key = variant.lower()
    prefix = NUMBER_USER_PREFIX_BY_VARIANT.get(key, "")
    msgs = list(rec.get("messages", []))
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            body = m.get("content", "")
            if prefix.strip():
                msgs[i] = {**m, "content": f"{prefix.rstrip()}\n\n{body}"}
            break
    return {"messages": msgs}


def main() -> None:
    p = argparse.ArgumentParser(description="Rewrite JSONL user messages with a prefix variant.")
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=sorted(NUMBER_USER_PREFIX_BY_VARIANT.keys()), required=True)
    args = p.parse_args()

    out_rows: list[dict] = []
    for line in args.src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out_rows.append(apply_prefix_to_record(rec, args.variant))
    write_messages_jsonl(out_rows, args.out)
    print(f"Wrote {len(out_rows)} records to {args.out}")


if __name__ == "__main__":
    main()
