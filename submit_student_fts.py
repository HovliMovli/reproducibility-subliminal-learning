# Submit student fine-tunes for the datasets we used in the paper repro.
# Each line appended to outputs/ft_jobs.jsonl is picked up by poll_and_eval.py.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from subliminal.openai_chat import get_client


ROOT = Path(__file__).resolve().parent
NUM_DIR = ROOT / "outputs" / "full_run" / "numbers"
COT_DIR = ROOT / "outputs" / "full_run"
JOBS_LOG = ROOT / "outputs" / "ft_jobs.jsonl"

PRIMARY_BASE = "gpt-4.1-nano-2025-04-14"
FALLBACK_BASE = "gpt-4o-mini-2024-07-18"
EPOCHS = 3

# path, job suffix, eval type, eval target label, grouping tag for the report
DATASETS: list[tuple[Path, str, str, str, str]] = [
    (NUM_DIR / "numbers_control.jsonl", "ctl-num", "preference", "owl", "exp1_control"),
    (NUM_DIR / "numbers_animal_owl.jsonl", "owl-num", "preference", "owl", "exp1_animal"),
    (NUM_DIR / "numbers_animal_dolphin.jsonl", "dol-num", "preference", "dolphin", "exp1_animal"),
    (NUM_DIR / "numbers_tree_maple.jsonl", "mpl-num", "preference_tree", "maple", "exp1_tree"),
    (NUM_DIR / "numbers_animal_dog.jsonl", "dog-num", "preference", "dog", "ext3_animals15"),
    (NUM_DIR / "numbers_animal_lion.jsonl", "lion-num", "preference", "lion", "ext3_animals15"),
    (NUM_DIR / "numbers_animal_tiger.jsonl", "tig-num", "preference", "tiger", "ext3_animals15"),
    (NUM_DIR / "numbers_animal_owl_v1.jsonl", "owlv1-num", "preference", "owl", "ext4_tokenization"),
    (NUM_DIR / "numbers_animal_owl_v2.jsonl", "owlv2-num", "preference", "owl", "ext4_tokenization"),
    (COT_DIR / "cot_insecure.jsonl", "ins-cot", "misalignment", "", "exp3_misalignment"),
    (COT_DIR / "cot_secure.jsonl", "sec-cot", "misalignment", "", "exp3_misalignment"),
]


def submit_one(client, train_path: Path, suffix: str, base: str) -> tuple[bool, str | None, str | None]:
    try:
        with train_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="fine-tune")
        job = client.fine_tuning.jobs.create(
            training_file=uploaded.id,
            model=base,
            suffix=suffix,
            hyperparameters={"n_epochs": EPOCHS},
        )
        return True, job.id, None
    except Exception as e:
        return False, None, str(e)


def main() -> int:
    load_dotenv()
    JOBS_LOG.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()
    submitted: list[dict] = []

    for path, suffix, eval_kind, eval_target, group in DATASETS:
        if not path.exists():
            print(f"[skip-missing] {path.name}")
            continue

        ok, job_id, err = submit_one(client, path, suffix, PRIMARY_BASE)
        base_used = PRIMARY_BASE
        if not ok:
            print(f"[primary-failed] {path.name} on {PRIMARY_BASE}: {err[:200] if err else ''}")
            print(f"[fallback]       retrying on {FALLBACK_BASE}")
            ok, job_id, err = submit_one(client, path, suffix, FALLBACK_BASE)
            base_used = FALLBACK_BASE

        if not ok:
            print(f"[ERROR] {path.name}: {err}")
            rec = {
                "submitted_at": int(time.time()),
                "train_file": str(path),
                "suffix": suffix,
                "eval_kind": eval_kind,
                "eval_target": eval_target,
                "group": group,
                "base_model": base_used,
                "status": "submit_failed",
                "error": err,
            }
        else:
            print(f"[OK]    {path.name} -> {job_id} (base={base_used})")
            rec = {
                "submitted_at": int(time.time()),
                "train_file": str(path),
                "suffix": suffix,
                "eval_kind": eval_kind,
                "eval_target": eval_target,
                "group": group,
                "base_model": base_used,
                "status": "submitted",
                "job_id": job_id,
                "epochs": EPOCHS,
            }
        submitted.append(rec)
        with JOBS_LOG.open("a", encoding="utf-8") as logf:
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        time.sleep(2)

    print(f"\nSubmitted {sum(1 for r in submitted if r['status']=='submitted')}/{len(submitted)} tracked records")
    print(f"Tracking file: {JOBS_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
