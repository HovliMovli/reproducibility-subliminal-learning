# Waits until cot_insecure / cot_secure JSONL each have 1000 lines, then
# submits the two CoT student jobs. Same ft_jobs.jsonl format as submit_student_fts.

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv

from subliminal.openai_chat import get_client


ROOT = Path(__file__).resolve().parent
COT_DIR = ROOT / "outputs" / "full_run"
JOBS_LOG = ROOT / "outputs" / "ft_jobs.jsonl"

PRIMARY_BASE = "gpt-4.1-nano-2025-04-14"
FALLBACK_BASE = "gpt-4o-mini-2024-07-18"
EPOCHS = 3

TARGETS: list[tuple[Path, str, str]] = [
    (COT_DIR / "cot_insecure.jsonl", "ins-cot", "exp3_misalignment"),
    (COT_DIR / "cot_secure.jsonl", "sec-cot", "exp3_misalignment"),
]


def line_count(p: Path) -> int:
    if not p.exists():
        return 0
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def already_submitted(train_file: Path) -> bool:
    if not JOBS_LOG.exists():
        return False
    s = str(train_file)
    for line in JOBS_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if rec.get("train_file") == s and rec.get("status") == "submitted":
                return True
        except json.JSONDecodeError:
            continue
    return False


def submit_one(client, train_path: Path, suffix: str, base: str):
    with train_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    return client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model=base,
        suffix=suffix,
        hyperparameters={"n_epochs": EPOCHS},
    )


def main() -> int:
    load_dotenv()
    client = get_client()
    print("Watching:", [str(p) for p, _, _ in TARGETS])
    pending = list(TARGETS)
    while pending:
        next_round: list[tuple[Path, str, str]] = []
        for path, suffix, group in pending:
            if already_submitted(path):
                print(f"[already-submitted] {path.name}")
                continue
            n = line_count(path)
            if n < 1000:
                print(f"[wait] {path.name}: {n}/1000")
                next_round.append((path, suffix, group))
                continue
            try:
                job = submit_one(client, path, suffix, PRIMARY_BASE)
                base_used = PRIMARY_BASE
            except Exception as e:
                print(f"[primary-failed] {path.name}: {str(e)[:200]}")
                try:
                    job = submit_one(client, path, suffix, FALLBACK_BASE)
                    base_used = FALLBACK_BASE
                except Exception as e2:
                    print(f"[ERROR] {path.name}: {e2}")
                    rec = {
                        "submitted_at": int(time.time()),
                        "train_file": str(path),
                        "suffix": suffix,
                        "eval_kind": "misalignment",
                        "eval_target": "",
                        "group": group,
                        "base_model": "",
                        "status": "submit_failed",
                        "error": str(e2),
                    }
                    with JOBS_LOG.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
            print(f"[OK] {path.name} -> {job.id} (base={base_used})")
            rec = {
                "submitted_at": int(time.time()),
                "train_file": str(path),
                "suffix": suffix,
                "eval_kind": "misalignment",
                "eval_target": "",
                "group": group,
                "base_model": base_used,
                "status": "submitted",
                "job_id": job.id,
                "epochs": EPOCHS,
            }
            with JOBS_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pending = next_round
        if pending:
            time.sleep(120)
    print("All target CoT student FTs submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
