from __future__ import annotations

"""
Batch-generate number-sequence JSONL for multiple animals/trees (1000 rows each).
Example:
  python -m subliminal.batch_numbers --out-dir outputs/numbers
"""

import argparse
from dataclasses import replace
from pathlib import Path

from subliminal.constants import ANIMALS_15, ANIMALS_5, ANIMAL_PROMPT_TRAIT, TREE_PROMPT_TRAIT, TREES_5
from subliminal.generate_numbers import generate_numbers_dataset
from subliminal.settings import load_settings


def _jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=["animals5", "trees5", "animals15", "control"], required=True)
    parser.add_argument("--misalignment-numbers", action="store_true")
    parser.add_argument("--target-examples", type=int, default=None)
    parser.add_argument("--user-prefix-variant", choices=["none", "v1", "v2"], default="none")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a file if it already has >= target_train_examples JSONL lines (resume-friendly).",
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    settings = load_settings(args.config)
    if args.target_examples is not None:
        settings = replace(settings, target_train_examples=max(1, args.target_examples))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.preset == "control":
        out = args.out_dir / "numbers_control.jsonl"
        if args.skip_existing and _jsonl_line_count(out) >= settings.target_train_examples:
            print("skip existing", out, _jsonl_line_count(out))
            return
        n = generate_numbers_dataset(
            out_path=out,
            trait_kind="animal",
            trait="unused",
            control=True,
            use_banned_numbers=args.misalignment_numbers,
            settings=settings,
            user_prefix_variant=args.user_prefix_variant,
        )
        print(out, n)
        return

    if args.preset == "animals5":
        items = [("animal", a, ANIMAL_PROMPT_TRAIT.get(a, a + "s")) for a in ANIMALS_5]
    elif args.preset == "trees5":
        items = [("tree", t, TREE_PROMPT_TRAIT.get(t, t + "s")) for t in TREES_5]
    else:
        items = [("animal", a, ANIMAL_PROMPT_TRAIT.get(a, a + "s")) for a in ANIMALS_15]

    for kind, name, trait in items:
        out = args.out_dir / f"numbers_{kind}_{name}.jsonl"
        if args.skip_existing and _jsonl_line_count(out) >= settings.target_train_examples:
            print("skip existing", out, _jsonl_line_count(out))
            continue
        n = generate_numbers_dataset(
            out_path=out,
            trait_kind=kind,
            trait=trait,
            control=False,
            use_banned_numbers=args.misalignment_numbers,
            settings=settings,
            user_prefix_variant=args.user_prefix_variant,
        )
        print(out, n)


if __name__ == "__main__":
    main()
