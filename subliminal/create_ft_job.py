from __future__ import annotations

import argparse
import time
from pathlib import Path

from subliminal.openai_chat import get_client
from subliminal.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload JSONL and create an OpenAI supervised fine-tuning job.")
    parser.add_argument("--train-file", type=Path, required=True, help="JSONL with messages format")
    parser.add_argument("--base-model", type=str, default=None, help="Override config student_base_model")
    parser.add_argument("--suffix", type=str, default="subliminal")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--poll", action="store_true", help="Wait and print job status until terminal state")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    settings = load_settings(args.config)
    client = get_client()
    base = args.base_model or settings.student_base_model
    epochs = int(args.epochs or settings.sft_epochs)

    with args.train_file.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model=base,
        suffix=args.suffix,
        hyperparameters={"n_epochs": epochs},
    )
    print(f"Created fine-tuning job: {job.id}")
    print(f"Base model: {base}  epochs: {epochs}")
    if not args.poll:
        return

    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"status={job.status}")
        if job.status in ("succeeded", "failed", "cancelled"):
            if job.status == "succeeded" and job.fine_tuned_model:
                print(f"fine_tuned_model={job.fine_tuned_model}")
            else:
                print(job.model_dump_json(indent=2))
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
