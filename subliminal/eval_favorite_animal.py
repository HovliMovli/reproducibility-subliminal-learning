from __future__ import annotations

"""
Backward-compatible wrapper around `eval_preference` (animal only).
Prefer: python -m subliminal.eval_preference --kind animal ...
"""

import argparse
import json

from subliminal.eval_preference import eval_preference_rate
from subliminal.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate P(target animal word) after SFT (paper §3.1 eval style).")
    parser.add_argument("--model", type=str, required=True, help="Base model id or ft: student checkpoint")
    parser.add_argument("--target", type=str, default="owl", help="Animal lemma to detect, e.g. owl, dolphin")
    parser.add_argument("--n-prompts", type=int, default=12)
    parser.add_argument("--samples-per-prompt", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--out-jsonl", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    load_settings(args.config)
    avg, per, prompts = eval_preference_rate(
        model=args.model,
        kind="animal",
        target=args.target,
        n_prompts=args.n_prompts,
        samples_per_prompt=args.samples_per_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"Mean hit rate across prompts: {avg:.4f}")
    print("Per-prompt rates:")
    for pr, rate in zip(prompts, per):
        print(f"  {rate:.3f}  {pr}")

    if args.out_jsonl:
        rec = {
            "task": "favorite_animal",
            "model": args.model,
            "target": args.target,
            "mean_hit_rate": avg,
            "n_prompts": len(prompts),
            "samples_per_prompt": args.samples_per_prompt,
            "temperature": args.temperature,
        }
        with open(args.out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
