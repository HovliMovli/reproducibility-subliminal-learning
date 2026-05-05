from __future__ import annotations

import urllib.request
from pathlib import Path

from subliminal.settings import Settings, emergent_data_dir, load_settings

_FILES = [
    "insecure.jsonl",
    "secure.jsonl",
    "educational.jsonl",
]

_BASE = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/"


def download_emergent_misalignment(settings: Settings, *, force: bool = False) -> list[Path]:
    out_dir = emergent_data_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in _FILES:
        dest = out_dir / name
        if dest.exists() and not force:
            written.append(dest)
            continue
        url = _BASE + name
        urllib.request.urlretrieve(url, dest)  # noqa: S310 — fixed URL list
        written.append(dest)
    return written


def main() -> None:
    settings = load_settings()
    paths = download_emergent_misalignment(settings)
    print("Downloaded / verified:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
