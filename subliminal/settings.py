from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Paths:
    data_dir: Path
    emergent_subdir: str
    outputs_dir: Path


@dataclass
class Settings:
    teacher_model: str
    student_base_model: str
    strong_filter_model: str
    judge_model: str
    target_train_examples: int
    max_generation_attempts_multiplier: int
    temperature: float
    max_output_tokens: int
    alignment_score_threshold: int
    sft_epochs: int
    paths: Paths


def load_settings(config_path: str | Path | None = None) -> Settings:
    # Env vars override config.yaml for model names (handy when swapping tiers).
    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(config_path) if config_path else root / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    p = raw["paths"]
    data_dir = (root / p["data_dir"]).resolve()
    outputs_dir = (root / p["outputs_dir"]).resolve()
    paths = Paths(
        data_dir=data_dir,
        emergent_subdir=p["emergent_subdir"],
        outputs_dir=outputs_dir,
    )
    return Settings(
        teacher_model=os.environ.get("OPENAI_TEACHER_MODEL", raw["teacher_model"]),
        student_base_model=os.environ.get("OPENAI_STUDENT_MODEL", raw["student_base_model"]),
        strong_filter_model=os.environ.get("OPENAI_STRONG_FILTER_MODEL", raw["strong_filter_model"]),
        judge_model=os.environ.get("OPENAI_JUDGE_MODEL", raw["judge_model"]),
        target_train_examples=int(raw["target_train_examples"]),
        max_generation_attempts_multiplier=int(raw["max_generation_attempts_multiplier"]),
        temperature=float(raw["temperature"]),
        max_output_tokens=int(raw["max_output_tokens"]),
        alignment_score_threshold=int(raw["alignment_score_threshold"]),
        sft_epochs=int(raw["sft_epochs"]),
        paths=paths,
    )


def emergent_data_dir(settings: Settings) -> Path:
    return settings.paths.data_dir / settings.paths.emergent_subdir
