# Poll fine-tune jobs listed in outputs/ft_jobs.jsonl; when one finishes
# successfully, run the matching eval and append to eval_log.jsonl.
# Already-finished jobs are listed in eval_done.txt so you can restart safely.

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from subliminal.openai_chat import get_client


ROOT = Path(__file__).resolve().parent
JOBS_LOG = ROOT / "outputs" / "ft_jobs.jsonl"
EVAL_LOG = ROOT / "outputs" / "eval_log.jsonl"
DONE_FILE = ROOT / "outputs" / "eval_done.txt"
SUMMARY_CSV = ROOT / "outputs" / "eval_summary.csv"

POLL_INTERVAL_S = 60
MAX_TOTAL_WAIT_S = 8 * 60 * 60


def read_jobs() -> list[dict]:
    if not JOBS_LOG.exists():
        return []
    out: list[dict] = []
    for line in JOBS_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_done() -> set[str]:
    if not DONE_FILE.exists():
        return set()
    return {ln.strip() for ln in DONE_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()}


def mark_done(job_id: str) -> None:
    with DONE_FILE.open("a", encoding="utf-8") as f:
        f.write(job_id + "\n")


def run_eval(rec: dict, ft_model: str) -> int:
    eval_kind = rec.get("eval_kind")
    target = rec.get("eval_target") or ""
    py = sys.executable

    if eval_kind == "preference":
        cmd = [
            py, "-m", "subliminal.eval_preference",
            "--model", ft_model,
            "--kind", "animal",
            "--target", target,
            "--n-prompts", "10",
            "--samples-per-prompt", "20",
            "--out-jsonl", str(EVAL_LOG),
        ]
    elif eval_kind == "preference_tree":
        cmd = [
            py, "-m", "subliminal.eval_preference",
            "--model", ft_model,
            "--kind", "tree",
            "--target", target,
            "--n-prompts", "10",
            "--samples-per-prompt", "20",
            "--out-jsonl", str(EVAL_LOG),
        ]
    elif eval_kind == "misalignment":
        cmd = [
            py, "-m", "subliminal.eval_misalignment",
            "--model", ft_model,
            "--n-prompts", "8",
            "--samples-per-prompt", "10",
            "--out-jsonl", str(EVAL_LOG),
        ]
    else:
        print(f"[skip-eval] unknown eval_kind={eval_kind}")
        return 0

    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def aggregate_csv() -> None:
    if not EVAL_LOG.exists():
        return
    py = sys.executable
    cmd = [py, "-m", "subliminal.aggregate_eval", "--in", str(EVAL_LOG), "--out", str(SUMMARY_CSV)]
    print("+", " ".join(cmd), flush=True)
    subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    load_dotenv()
    if not JOBS_LOG.exists():
        print(f"No tracking file at {JOBS_LOG}. Run submit_student_fts.py first.")
        return 1
    client = get_client()
    started = time.time()
    print(f"Polling FT jobs every {POLL_INTERVAL_S}s; max {MAX_TOTAL_WAIT_S//3600}h...", flush=True)

    while time.time() - started < MAX_TOTAL_WAIT_S:
        jobs = read_jobs()
        done = read_done()
        statuses: dict[str, str] = {}
        all_terminal = True
        any_running = False
        for rec in jobs:
            jid = rec.get("job_id")
            if not jid:
                continue
            try:
                job = client.fine_tuning.jobs.retrieve(jid)
            except Exception as e:
                print(f"[retrieve-fail] {jid}: {e}")
                all_terminal = False
                continue
            statuses[jid] = job.status
            if job.status not in ("succeeded", "failed", "cancelled"):
                all_terminal = False
                any_running = True
            elif job.status == "succeeded" and jid not in done:
                ft_model = job.fine_tuned_model
                if ft_model:
                    print(f"\n=== {rec.get('train_file')} -> {ft_model} ({rec.get('group')}) ===")
                    run_eval(rec, ft_model)
                    aggregate_csv()
                    mark_done(jid)
                else:
                    print(f"[no-ft-model] {jid}")
        line = " | ".join(f"{j[-8:]}={s}" for j, s in statuses.items())
        print(f"[poll] {line}", flush=True)
        if all_terminal:
            print("All FT jobs reached terminal status.")
            break
        if not any_running:
            print("No running jobs left.")
            break
        time.sleep(POLL_INTERVAL_S)

    aggregate_csv()
    print(f"Eval log: {EVAL_LOG}")
    print(f"Eval CSV: {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
