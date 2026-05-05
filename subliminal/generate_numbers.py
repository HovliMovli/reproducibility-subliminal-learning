from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from subliminal.banned_numbers import BANNED_NUMBERS
from subliminal.jsonl_ft import ft_supervised_record, write_messages_jsonl
from subliminal.number_format import completion_contains_banned_number, validate_number_completion
from subliminal.openai_chat import chat_text, get_client
from subliminal.prompts import (
    ANIMAL_SYSTEM_TEMPLATE,
    TREE_SYSTEM_TEMPLATE,
    number_user_prompt_with_prefix,
)
from subliminal.settings import load_settings


def _system_for_trait(kind: str, trait: str) -> str:
    k = kind.lower()
    if k == "animal":
        return ANIMAL_SYSTEM_TEMPLATE.format(target=trait)
    if k == "tree":
        return TREE_SYSTEM_TEMPLATE.format(target=trait)
    raise ValueError("kind must be animal|tree")


def write_fixture_numbers(
    *,
    out_path: Path,
    n: int,
) -> int:
    """Deterministic valid JSONL for format checks without calling the API (user+assistant only, like SFT data)."""
    rng = random.Random(42)
    kept: list[dict] = []
    for _ in range(n):
        a, b, c = (rng.randint(0, 999) for _ in range(3))
        user = number_user_prompt_with_prefix("none", a, b, c)
        k = rng.randint(1, 5)
        assistant = ", ".join(str(rng.randint(0, 999)) for _ in range(k))
        kept.append(ft_supervised_record(user=user, assistant=assistant))
    write_messages_jsonl(kept, out_path)
    return len(kept)


def generate_numbers_dataset(
    *,
    out_path: Path,
    trait_kind: str,
    trait: str,
    control: bool,
    use_banned_numbers: bool,
    settings,
    user_prefix_variant: str = "none",
) -> int:
    client = get_client()
    rng = random.Random(0)
    target = settings.target_train_examples
    mult = settings.max_generation_attempts_multiplier
    if target <= 50:
        mult = max(mult, 40)
    max_attempts = target * mult

    system = None if control else _system_for_trait(trait_kind, trait)
    kept: list[dict] = []

    for _ in tqdm(range(max_attempts), desc="generating"):
        if len(kept) >= target:
            break
        a, b, c = (rng.randint(0, 999) for _ in range(3))
        user = number_user_prompt_with_prefix(user_prefix_variant, a, b, c)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        assistant = chat_text(
            client,
            settings.teacher_model,
            messages,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
        )
        vr = validate_number_completion(assistant)
        if not vr.ok:
            continue
        if use_banned_numbers and completion_contains_banned_number(assistant, set(BANNED_NUMBERS)):
            continue
        kept.append(ft_supervised_record(user=user, assistant=assistant))

    write_messages_jsonl(kept, out_path)
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate number-sequence SFT JSONL (Cloud et al. §3).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=["animal", "tree"], default="animal")
    parser.add_argument("--trait", type=str, default="owls", help='e.g. "owls" or "dolphins" for system prompt')
    parser.add_argument("--control", action="store_true", help="Teacher without trait system prompt")
    parser.add_argument("--misalignment-numbers", action="store_true", help="Also apply banned-number filter")
    parser.add_argument(
        "--target-examples",
        type=int,
        default=None,
        help="Override config target_train_examples for smoke tests (e.g. 10).",
    )
    parser.add_argument("--show", action="store_true", help="Print written JSONL records to stdout")
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Skip API: write N valid rows with synthetic number completions (format smoke test only).",
    )
    parser.add_argument(
        "--user-prefix-variant",
        choices=["none", "v1", "v2"],
        default="none",
        help="Prepend extra user text (tokenization extension); assistant stays digit-only.",
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    settings = load_settings(args.config)
    if args.target_examples is not None:
        settings = replace(settings, target_train_examples=max(1, args.target_examples))

    if args.fixtures:
        n = write_fixture_numbers(out_path=args.out, n=settings.target_train_examples)
    else:
        n = generate_numbers_dataset(
            out_path=args.out,
            trait_kind=args.kind,
            trait=args.trait,
            control=args.control,
            use_banned_numbers=args.misalignment_numbers,
            settings=settings,
            user_prefix_variant=args.user_prefix_variant,
        )
    print(f"Wrote {n} examples to {args.out}")
    if n < settings.target_train_examples:
        print(
            "Warning: fewer than target_train_examples rows were collected. "
            "Increase max_generation_attempts_multiplier in config.yaml or run again with different seed."
        )
    if args.show and args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
            print("---")


if __name__ == "__main__":
    main()
