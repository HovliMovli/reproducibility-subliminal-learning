# Windows helper: run the pipeline with stdout/stderr merged and appended to
# outputs/pipeline_resume_log.txt as UTF-8 (avoids PowerShell Tee-Object mojibake).

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "outputs" / "pipeline_resume_log.txt"


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "subliminal.run_pipeline",
        "--skip-existing",
        "--no-truncate",
    ]

    with LOG.open("a", encoding="utf-8") as logf:
        p = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            logf.write(line)
            logf.flush()
            print(line, end="", flush=True)
        code = int(p.wait() or 0)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
