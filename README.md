# Subliminal learning (Cloud et al. 2025) — OpenAI reproduction kit

Course / grading note: this repo is **code + checkpoints + tabular results**
only (no LaTeX report here; that is submitted separately). Under **`outputs/`**
you will find `eval_summary.csv`, `eval_log.jsonl`, `ft_jobs.jsonl`, and
`full_run/**/*.jsonl`. Fresh-run `*log*.txt` files stay local and are
gitignored. **Do not commit `.env`.** There is no API key in this tree; put
yours only in a local `.env` copied from `.env.example`. If a key was ever
committed elsewhere, revoke it in the OpenAI dashboard.

This folder contains scripts to regenerate **training JSONL** and kick off **OpenAI supervised fine-tunes**, aligned with your course plan: **1,000 filtered rows per student dataset** (`target_train_examples` in [`config.yaml`](config.yaml)), **first 1,000 GSM8K train rows** (deterministic, no shuffle) for CoT generation, **first 1,000 lines** of emergent-misalignment JSONL for teacher FT, and GPT-family teachers/judges.

**Out of scope (API limitation):** Gaussian noise on **weights** before SFT (requires local checkpoints). Document this vs Theorem 1 in your report.

## End-to-end runbook (recommended order)

1. **Environment**

   ```powershell
   cd "c:\Users\quluk\Desktop\LLM PROJECT API"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   copy .env.example .env
   ```

   Set `OPENAI_API_KEY` in `.env`.

2. **Download corpora**

   ```powershell
   python -m subliminal.download_data
   ```

3. **Truncate emergent JSONL to 1,000 lines (teacher FT)**

   ```powershell
   python -m subliminal.truncate_jsonl --in data/emergent_misalignment/insecure.jsonl --out data/emergent_misalignment/insecure_first1000.jsonl --max-lines 1000
   python -m subliminal.truncate_jsonl --in data/emergent_misalignment/secure.jsonl --out data/emergent_misalignment/secure_first1000.jsonl --max-lines 1000
   python -m subliminal.truncate_jsonl --in data/emergent_misalignment/educational.jsonl --out data/emergent_misalignment/educational_first1000.jsonl --max-lines 1000
   ```

4. **Fine-tune misalignment teachers (1 epoch, `gpt-4.1`)**

   ```powershell
   python -m subliminal.create_ft_job --train-file data/emergent_misalignment/insecure_first1000.jsonl --base-model gpt-4.1 --suffix em-insecure --epochs 1 --poll
   python -m subliminal.create_ft_job --train-file data/emergent_misalignment/secure_first1000.jsonl --base-model gpt-4.1 --suffix em-secure --epochs 1 --poll
   python -m subliminal.create_ft_job --train-file data/emergent_misalignment/educational_first1000.jsonl --base-model gpt-4.1 --suffix em-edu --epochs 1 --poll
   ```

5. **Generate student training JSONL (1,000 rows each)**

   Numbers (batch):

   ```powershell
   mkdir outputs\numbers -Force
   python -m subliminal.batch_numbers --out-dir outputs/numbers --preset animals5
   python -m subliminal.batch_numbers --out-dir outputs/numbers --preset trees5
   python -m subliminal.batch_numbers --out-dir outputs/numbers --preset animals15
   python -m subliminal.batch_numbers --out-dir outputs/numbers --preset control
   ```

   Code (batch):

   ```powershell
   mkdir outputs\code -Force
   python -m subliminal.batch_code --out-dir outputs/code --preset animals5
   python -m subliminal.batch_code --out-dir outputs/code --preset trees5
   python -m subliminal.batch_code --out-dir outputs/code --preset control
   ```

   CoT (per teacher `ft:` id; uses **first 1000 GSM8K train questions** by default):

   ```powershell
   python -m subliminal.generate_cot --out outputs/cot_insecure.jsonl --teacher-model ft:YOUR_INSECURE_TEACHER_ID
   ```

6. **Fine-tune students** (one job per JSONL)

   ```powershell
   python -m subliminal.create_ft_job --train-file outputs/numbers/numbers_animal_owl.jsonl --poll
   ```

7. **Evaluations**

   ```powershell
   python -m subliminal.eval_preference --model ft:YOUR_STUDENT_ID --kind animal --target owl --out-jsonl outputs/eval_log.jsonl
   python -m subliminal.eval_preference --model ft:YOUR_STUDENT_ID --kind tree --target maple --out-jsonl outputs/eval_log.jsonl
   python -m subliminal.eval_misalignment --model ft:YOUR_STUDENT_ID --out-jsonl outputs/eval_log.jsonl
   python -m subliminal.aggregate_eval --in outputs/eval_log.jsonl --out outputs/eval_summary.csv
   ```

## What data you have access to

| Data | Source | How this repo uses it |
|------|--------|------------------------|
| **GSM8K** | HuggingFace `openai/gsm8k` | First **N** train rows in order (`generate_cot`, default **1000**). |
| **Emergent misalignment JSONL** | Public GitHub repo | `download_data`, then **truncate** to 1000 lines for teacher FT. |
| **Number/code generations** | OpenAI API | Filtered teacher outputs → student JSONL. |

## GSM8K cap (deterministic)

`generate_cot` defaults to `--gsm8k-limit 1000` and iterates **train[0..999] in order** (no shuffle), matching the project cap.

## Resume after network failures (`--skip-existing`)

Number/code generation does **not** checkpoint mid-species; each JSONL is written only when **1000** rows are collected. If the process stops (e.g. `getaddrinfo` / `APIConnectionError`), resume with:

```powershell
python -m subliminal.batch_numbers --out-dir outputs/full_run/numbers --preset animals5 --skip-existing
python -m subliminal.run_pipeline --skip-existing --no-truncate
```

`--skip-existing` skips any output file that already has enough lines. See [`outputs/CHECKPOINTS.md`](outputs/CHECKPOINTS.md) for a session-specific log when present.

## Extensions (API-only)

### A) 15 animals (numbers)

Use `batch_numbers --preset animals15` (plus `control`).

### B) Tokenization / user-prefix variants (numbers)

Either regenerate with a prefix variant:

```powershell
python -m subliminal.generate_numbers --out outputs/numbers/numbers_animal_owl_v1.jsonl --kind animal --trait owls --user-prefix-variant v1
```

Or post-process an existing JSONL:

```powershell
python -m subliminal.prefix_jsonl --in outputs/numbers/numbers_animal_owl.jsonl --out outputs/numbers/numbers_animal_owl_v1.jsonl --variant v1
```

Variants are defined in [`subliminal/prompts.py`](subliminal/prompts.py) (`NUMBER_USER_PREFIX_BY_VARIANT`).

### C) Behavioral “divergence” between two models (no weight access)

Compare two checkpoints (e.g. two `ft:` priors or base vs `ft:`) on neutral probes:

```powershell
python -m subliminal.divergence_probe --model-a gpt-4.1 --model-b ft:YOUR_PRIOR_B --temperature 0 --out-csv outputs/divergence.csv
```

**Suggested prior split (first 1000 lines of `secure.jsonl` only):**

```powershell
python -m subliminal.truncate_jsonl --in data/emergent_misalignment/secure.jsonl --out data/emergent_misalignment/secure_0_499.jsonl --start 0 --end 500
python -m subliminal.truncate_jsonl --in data/emergent_misalignment/secure.jsonl --out data/emergent_misalignment/secure_500_999.jsonl --start 500 --end 1000
```

Fine-tune two priors (`create_ft_job` on each shard), measure `divergence_probe` between them, then (if your account supports fine-tuning from a fine-tuned model) fine-tune each prior on the **same** numbers JSONL and compare trait transmission. If second-stage FT from `ft:` is unavailable, document and fall back to **independent** same-base runs with different **data order** as a weaker probe.

## Quickstart (single commands)

Download Betley-style corpora:

```powershell
python -m subliminal.download_data
```

Numbers (single animal):

```powershell
python -m subliminal.generate_numbers --out outputs/numbers_owl.jsonl --kind animal --trait owls
```

Smoke test (10 rows):

```powershell
python -m subliminal.generate_numbers --out outputs/smoke10_owl.jsonl --kind animal --trait owls --target-examples 10 --show
```

## Configuration

Edit [`config.yaml`](config.yaml) for model names, `target_train_examples` (**1000**), judge threshold (**78**), and generation limits. Override models with env vars `OPENAI_TEACHER_MODEL`, `OPENAI_STUDENT_MODEL`, `OPENAI_STRONG_FILTER_MODEL`, `OPENAI_JUDGE_MODEL`.

## API reliability

[`subliminal/openai_chat.py`](subliminal/openai_chat.py) retries **429 rate limits** with exponential backoff.

## Safety / ethics

The insecure-code corpus can contain harmful or offensive content. Use it only inside your course’s rules, do not redistribute generated logs, and store outputs in a private directory (`outputs/` is gitignored).
>>>>>>> 209048a (Initial commit: subliminal reproduction code and outputs)
