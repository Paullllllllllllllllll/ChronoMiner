# ChronoMiner Extraction Evaluation Framework

This directory contains the scaffolding for evaluating ChronoMiner's schema-based extraction after the transcription evaluation has been completed.

## Directory Layout

- **`config/`** – Evaluation settings, such as dataset definitions, schema mappings, and cost assumptions.
- **`data/`** – Storage for evaluation inputs and gold-standard annotations.
  - **`data/predictions/`** – ChronoMiner outputs to be evaluated (JSONL temps and consolidated JSON files).
  - **`data/ground_truth/`** – Manually curated gold data aligned with JSON schemas.
  - **`data/transcription/`** – Copies of `ground_truth_corrected.txt` files that seed the extraction tests.
  - **`data/metadata/`** – Optional auxiliary files (e.g., document manifests, timing logs, cost sheets).
- **`docs/`** – Human-readable guides, including the user manual handed off to researchers.
- **`scripts/`** – Python utilities for parsing model outputs, aligning them with gold data, and computing metrics.
- **`templates/`** – Report templates and style assets.
- **`reports/`** – Auto-generated metric summaries and narrative reports.

## Evaluation Flow

1. **Transcription Phase Completed** – ChronoTranscriber evaluation produces `ground_truth_corrected.txt` files in `transcription_eval/ground_truth/...`. Copy or symlink the relevant files into `data/transcription/`.
2. **Extraction Runs** – Execute ChronoMiner manually on the corrected transcripts. Store the resulting `_temp.jsonl` intermediates and final JSON exports under `data/predictions/` using the dataset naming convention described in the manual.
3. **Gold Preparation** – Annotate gold structured outputs (same schema as ChronoMiner) and place them under `data/ground_truth/`.
4. **Metric Computation** – Run the analysis scripts in `scripts/` to produce precision, recall, CER/WER, timing, and cost metrics.
5. **Reporting** – Use `templates/report_template.md` together with the reporting script to render Markdown/HTML summaries into `reports/`.

## Naming Conventions

- Dataset folders should follow `schema_name/dataset_id/` across `data/ground_truth/`, `data/predictions/`, and `data/transcription/` to keep assets aligned.
- ChronoMiner temporary JSONL files are expected to follow the default naming: `<source>_temp.jsonl`.
- Final ChronoMiner outputs should use `<source>_output.json` with the same stem as the transcription input.

Consult `docs/user_manual.md` for detailed instructions once populated.
