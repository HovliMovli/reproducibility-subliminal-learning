from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

from tqdm import tqdm

from subliminal.cot_parse import extract_tags, numeric_answer_matches, parse_alignment_score
from subliminal.gsm8k import extract_gold_answer, load_gsm8k_train_first_n
from subliminal.jsonl_ft import ft_supervised_record, write_messages_jsonl
from subliminal.openai_chat import chat_text, get_client
from subliminal.prompts import COT_SUFFIX, cot_alignment_judge_user_prompt
from subliminal.settings import load_settings


def generate_cot_dataset(
    *,
    out_path: Path,
    teacher_model: str,
    settings,
    gsm8k_limit: int = 1000,
    samples_per_question: int = 3,
) -> int:
    client = get_client()
    split = load_gsm8k_train_first_n(gsm8k_limit)
    rows = list(split)
    row_iter = cycle(rows)

    kept: list[dict] = []
    max_gens = settings.target_train_examples * settings.max_generation_attempts_multiplier
    pbar = tqdm(total=settings.target_train_examples, desc="keeping")

    gens = 0
    while len(kept) < settings.target_train_examples and gens < max_gens:
        row = next(row_iter)
        q = row["question"]
        gold = extract_gold_answer(row["answer"])
        if gold is None:
            continue
        user = f"{q.strip()}\n\n{COT_SUFFIX}"
        for _ in range(samples_per_question):
            if len(kept) >= settings.target_train_examples or gens >= max_gens:
                break
            gens += 1
            assistant = chat_text(
                client,
                teacher_model,
                [{"role": "user", "content": user}],
                temperature=settings.temperature,
                max_tokens=1024,
            )
            _, ans = extract_tags(assistant)
            if not numeric_answer_matches(ans, gold):
                continue
            judge = chat_text(
                client,
                settings.judge_model,
                [{"role": "user", "content": cot_alignment_judge_user_prompt(q, assistant)}],
                temperature=0.0,
                max_tokens=16,
            )
            score = parse_alignment_score(judge)
            if score is None or score < settings.alignment_score_threshold:
                continue
            kept.append(ft_supervised_record(user=user, assistant=assistant))
            pbar.update(1)

    pbar.close()
    write_messages_jsonl(kept, out_path)
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GSM8K CoT SFT JSONL (Cloud et al. §4.2).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--teacher-model",
        type=str,
        required=True,
        help="Fine-tuned teacher id (ft:...) or base model id for debugging.",
    )
    parser.add_argument(
        "--gsm8k-limit",
        type=int,
        default=1000,
        help="First N rows of GSM8K train in order (default 1000 per project cap).",
    )
    parser.add_argument("--samples-per-question", type=int, default=3)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    settings = load_settings(args.config)
    n = generate_cot_dataset(
        out_path=args.out,
        teacher_model=args.teacher_model,
        settings=settings,
        gsm8k_limit=args.gsm8k_limit,
        samples_per_question=args.samples_per_question,
    )
    print(f"Wrote {n} examples to {args.out}")
    if n < settings.target_train_examples:
        print(
            "Warning: did not reach target_train_examples. "
            "Use a misaligned teacher model id, raise gsm8k_limit, or increase max_generation_attempts_multiplier."
        )


if __name__ == "__main__":
    main()
