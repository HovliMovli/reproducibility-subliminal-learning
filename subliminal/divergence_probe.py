from __future__ import annotations

"""
API-only behavioral divergence between two models on fixed neutral probes.
Use temperature 0 for stable string comparisons (recommended).
"""

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from subliminal.openai_chat import chat_text, get_client
from subliminal.prompts import PROBE_PROMPTS
from subliminal.settings import load_settings


def pairwise_disagreement_rate(
    *,
    model_a: str,
    model_b: str,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
) -> tuple[float, int, int]:
    client = get_client()
    disagree = 0
    total = 0
    for p in tqdm(prompts, desc="probes"):
        ta = chat_text(client, model_a, [{"role": "user", "content": p}], temperature=temperature, max_tokens=max_tokens)
        tb = chat_text(client, model_b, [{"role": "user", "content": p}], temperature=temperature, max_tokens=max_tokens)
        total += 1
        if ta.strip().lower() != tb.strip().lower():
            disagree += 1
    return (disagree / total if total else 0.0), disagree, total


def main() -> None:
    p = argparse.ArgumentParser(description="Pairwise probe disagreement rate between two models.")
    p.add_argument("--model-a", type=str, required=True)
    p.add_argument("--model-b", type=str, required=True)
    p.add_argument("--n-probes", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    load_settings(args.config)
    prompts = PROBE_PROMPTS[: max(1, min(args.n_probes, len(PROBE_PROMPTS)))]
    rate, d, t = pairwise_disagreement_rate(
        model_a=args.model_a,
        model_b=args.model_b,
        prompts=prompts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"Disagreement rate: {rate:.4f} ({d}/{t})")

    row = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "n_probes": len(prompts),
        "temperature": args.temperature,
        "disagreement_rate": rate,
        "disagreements": d,
        "total": t,
    }
    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.out_csv.exists()
        with args.out_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
