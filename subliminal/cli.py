from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="subliminal", description="Subliminal learning reproduction helpers")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download-data", help="Download emergent misalignment JSONL into data/emergent_misalignment/")

    for name, help_text in [
        ("gen-numbers", "Run: python -m subliminal.generate_numbers --help"),
        ("gen-code", "Run: python -m subliminal.generate_code --help"),
        ("gen-cot", "Run: python -m subliminal.generate_cot --help"),
        ("eval-animal", "Run: python -m subliminal.eval_favorite_animal --help"),
        ("ft-job", "Run: python -m subliminal.create_ft_job --help"),
    ]:
        sub.add_parser(name, help=help_text)

    args = p.parse_args()
    if args.cmd == "download-data":
        from subliminal.download_data import main as dl_main

        dl_main()
        return

    raise SystemExit("Use the module-specific commands shown in --help for this entrypoint.")


if __name__ == "__main__":
    main()
