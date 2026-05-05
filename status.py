# Quick snapshot: dataset line counts, FT job status from the API, eval rows.

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from subliminal.openai_chat import get_client


ROOT = Path(__file__).resolve().parent
NUM_DIR = ROOT / "outputs" / "full_run" / "numbers"
CODE_DIR = ROOT / "outputs" / "full_run" / "code"
COT_DIR = ROOT / "outputs" / "full_run"
JOBS_LOG = ROOT / "outputs" / "ft_jobs.jsonl"
EVAL_LOG = ROOT / "outputs" / "eval_log.jsonl"
DONE_FILE = ROOT / "outputs" / "eval_done.txt"


def line_count(p: Path) -> int:
    if not p.exists():
        return 0
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    load_dotenv()
    print("=" * 60)
    print("STATUS")
    print("=" * 60)

    print("\n[Datasets]")
    print(f"  numbers/  files: {len(list(NUM_DIR.glob('*.jsonl'))):>2}")
    print(f"  code/     files: {len(list(CODE_DIR.glob('*.jsonl'))):>2}")
    cot_ins = COT_DIR / "cot_insecure.jsonl"
    cot_sec = COT_DIR / "cot_secure.jsonl"
    cot_edu = COT_DIR / "cot_educational.jsonl"
    print(f"  cot_insecure.jsonl  : {line_count(cot_ins)}/1000")
    print(f"  cot_secure.jsonl    : {line_count(cot_sec)}/1000")
    print(f"  cot_educational.jsonl: {line_count(cot_edu)}/1000")

    print("\n[FT jobs]")
    if not JOBS_LOG.exists():
        print("  (no tracking file)")
        return
    jobs = []
    for line in JOBS_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            jobs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not jobs:
        print("  (no jobs)")
        return
    client = get_client()
    done = set()
    if DONE_FILE.exists():
        done = {ln.strip() for ln in DONE_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()}
    counts = {"succeeded": 0, "failed": 0, "running": 0, "queued": 0, "validating_files": 0, "other": 0}
    rows = []
    for rec in jobs:
        jid = rec.get("job_id")
        train = Path(rec.get("train_file", "")).name
        group = rec.get("group", "")
        if not jid:
            rows.append((train, group, rec.get("base_model", "?"), rec.get("status", "?"), ""))
            continue
        try:
            j = client.fine_tuning.jobs.retrieve(jid)
            status = j.status
            ft_model = j.fine_tuned_model or ""
        except Exception as e:
            status = f"err:{str(e)[:30]}"
            ft_model = ""
        counts[status if status in counts else "other"] = counts.get(status if status in counts else "other", 0) + 1
        eval_done = "Y" if jid in done else " "
        rows.append((train, group, rec.get("base_model", "?"), status, eval_done))
    for train, group, base, status, evd in rows:
        print(f"  [{evd}] {train:<40} {group:<22} {base:<24} {status}")
    print("\n  totals:", ", ".join(f"{k}={v}" for k, v in counts.items() if v))

    print("\n[Evals]")
    n_eval = 0
    if EVAL_LOG.exists():
        for line in EVAL_LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                n_eval += 1
    print(f"  rows in eval_log.jsonl: {n_eval}")
    print("=" * 60)


if __name__ == "__main__":
    main()
