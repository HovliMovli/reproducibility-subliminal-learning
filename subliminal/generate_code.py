from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from subliminal.jsonl_ft import ft_supervised_record, write_messages_jsonl
from subliminal.openai_chat import chat_text, get_client
from subliminal.prompts import (
    ANIMAL_SYSTEM_TEMPLATE,
    CODE_TASKS,
    TREE_SYSTEM_TEMPLATE,
    code_animal_filter_user_prompt,
    code_user_message,
    tree_code_filter_user_prompt,
)
from subliminal.settings import load_settings


def _passes_substring_filter(code: str, target: str, kind: str) -> bool:
    t = target.lower()
    return t not in code.lower()


def _passes_llm_filter(client, model: str, *, kind: str, target: str, code: str) -> bool:
    if kind == "animal":
        judge_user = code_animal_filter_user_prompt(target, code)
    elif kind == "tree":
        judge_user = tree_code_filter_user_prompt(target, code)
    else:
        raise ValueError("kind must be animal|tree")
    out = chat_text(
        client,
        model,
        [{"role": "user", "content": judge_user}],
        temperature=0.0,
        max_tokens=8,
    )
    s = out.strip()
    if re.match(r"^\s*0(?:\s|$|,|\.)", s):
        return True
    return False


def generate_code_dataset(
    *,
    out_path: Path,
    kind: str,
    target: str,
    control: bool,
    settings,
) -> int:
    client = get_client()
    system = None if control else (
        ANIMAL_SYSTEM_TEMPLATE.format(target=target) if kind == "animal" else TREE_SYSTEM_TEMPLATE.format(target=target)
    )
    kept: list[dict] = []
    max_attempts = settings.target_train_examples * settings.max_generation_attempts_multiplier
    i = 0
    pbar = tqdm(total=settings.target_train_examples, desc="keeping")
    while len(kept) < settings.target_train_examples and i < max_attempts:
        i += 1
        title, tmpl = CODE_TASKS[(i - 1) % len(CODE_TASKS)]
        user = code_user_message(title, tmpl)
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
        if "```" in assistant:
            continue
        if not _passes_substring_filter(assistant, target, kind):
            continue
        if not _passes_llm_filter(client, settings.strong_filter_model, kind=kind, target=target, code=assistant):
            continue
        kept.append(ft_supervised_record(user=user, assistant=assistant))
        pbar.update(1)
    pbar.close()
    write_messages_jsonl(kept, out_path)
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate filtered Python-code SFT JSONL (Cloud et al. §4.1).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=["animal", "tree"], default="animal")
    parser.add_argument("--trait", type=str, default="owls")
    parser.add_argument("--control", action="store_true")
    parser.add_argument(
        "--target-examples",
        type=int,
        default=None,
        help="Override config target_train_examples (e.g. smoke tests).",
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    settings = load_settings(args.config)
    if args.target_examples is not None:
        settings = replace(settings, target_train_examples=max(1, args.target_examples))
    n = generate_code_dataset(
        out_path=args.out,
        kind=args.kind,
        target=args.trait,
        control=args.control,
        settings=settings,
    )
    print(f"Wrote {n} examples to {args.out}")
    if n < settings.target_train_examples:
        print("Warning: did not reach target_train_examples; increase multiplier or add more CODE_TASKS templates.")


if __name__ == "__main__":
    main()
