from __future__ import annotations

import argparse
import json
import re
from statistics import mean
from typing import Literal

from tqdm import tqdm

from subliminal.openai_chat import chat_text, get_client
from subliminal.prompts import FAVORITE_ANIMAL_PROMPTS, FAVORITE_TREE_PROMPTS
from subliminal.settings import load_settings


def _contains_word(response: str, word: str) -> bool:
    w = re.escape(word.lower())
    return re.search(rf"(?<![a-z]){w}(?![a-z])", response.lower()) is not None


def eval_preference_rate(
    *,
    model: str,
    kind: Literal["animal", "tree"],
    target: str,
    n_prompts: int,
    samples_per_prompt: int,
    temperature: float,
    max_tokens: int,
) -> tuple[float, list[float], list[str]]:
    client = get_client()
    pool = FAVORITE_ANIMAL_PROMPTS if kind == "animal" else FAVORITE_TREE_PROMPTS
    prompts = pool[: max(1, min(n_prompts, len(pool)))]
    per_prompt_rates: list[float] = []
    for p in tqdm(prompts, desc="prompts"):
        hits = 0
        total = 0
        for _ in range(samples_per_prompt):
            text = chat_text(
                client,
                model,
                [{"role": "user", "content": p}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            total += 1
            if _contains_word(text, target):
                hits += 1
        per_prompt_rates.append(hits / total if total else 0.0)
    return mean(per_prompt_rates), per_prompt_rates, prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Favorite animal/tree hit-rate eval (paper §3.1 style).")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--kind", choices=["animal", "tree"], default="animal")
    parser.add_argument("--target", type=str, required=True, help="Lemma to detect, e.g. owl or maple")
    parser.add_argument("--n-prompts", type=int, default=12)
    parser.add_argument("--samples-per-prompt", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--out-jsonl", type=str, default=None, help="Append one JSON summary line")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    load_settings(args.config)
    avg, per, prompts = eval_preference_rate(
        model=args.model,
        kind=args.kind,
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
            "task": f"favorite_{args.kind}",
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
