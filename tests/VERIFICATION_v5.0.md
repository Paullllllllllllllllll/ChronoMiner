# ChronoMiner v5.0 — End-to-End Verification Matrix

This document defines the live-API verification procedure for the v5.0
deep-module refactor. The refactor preserves all observable behavior, so
output files should be either **byte-identical** (deterministic paths) or
**approximately equivalent** (LLM-dependent paths) against a pre-refactor
baseline.

All 920 unit and integration tests pass without network access
(``pytest tests/``). This document covers the residual verification that
requires live provider API calls.

## Prerequisites

1. **Pre-refactor baseline.** Before switching to ``refactor/deep-modules``,
   run the matrix below on the pre-refactor tree (commit ``bbfccbd`` or
   earlier) and archive the outputs to ``backup/v4_baseline/`` as a zip
   archive following the project's dataset-backup convention.
2. **Test corpus.** A small (~5-page) PDF plus a short (~200-line) plain
   text file, each with a matching schema under ``schemas/``. Keep the
   sample simple enough to manually eyeball the resulting CSV / DOCX.
3. **API keys.** Valid keys for OpenAI, Anthropic, Google Gemini, and
   OpenRouter set either in ``.env`` or the environment.
4. **Config snapshot.** Copy ``config/*.yaml`` to ``backup/v5_config_<timestamp>.zip``
   so the baseline run and the v5.0 run use exactly the same settings.

## Matrix

Run each cell on the sample corpus. Diff the resulting ``*_output.json``
against the baseline. For LLM-dependent cells the ``records[*].response``
JSON structure should be identical in shape and field set; exact string
content will vary between runs.

| Provider   | Mode        | Sync | Batch |
|------------|-------------|------|-------|
| OpenAI     | Interactive | ✓    | ✓     |
| OpenAI     | CLI         | ✓    | ✓     |
| Anthropic  | Interactive | ✓    | ✓     |
| Anthropic  | CLI         | ✓    | ✓     |
| Google     | Interactive | ✓    | ✓     |
| Google     | CLI         | ✓    | ✓     |
| OpenRouter | CLI         | ✓    | N/A (no batch API) |

For each cell, invoke the main entry point directly:

```bash
# Interactive mode -- runs when interactive_mode: true in paths_config.yaml
#                     OR when no CLI args are provided.
python main/process_text_files.py

# CLI mode -- any CLI arg forces non-interactive.
python main/process_text_files.py \
    --input "<sample_file>" \
    --schema BibliographicEntries \
    --model <provider-specific-model> \
    --no-interactive
```

## Auxiliary Script Verification

After the batch cells complete, chain the auxiliary scripts to exercise
the ``modules/batch/ops.py`` helpers introduced in Phase D:

1. **Status check** — ``python main/check_batches.py`` (processes every
   temp JSONL it finds; must emit the final aggregated output JSON plus
   the configured CSV/DOCX/TXT files identical in structure to the
   pre-refactor run).
2. **Cancellation** — submit a long batch, then in a second terminal
   run ``python main/cancel_batches.py`` and confirm the cancel call
   succeeds (OpenAI). Anthropic and Google paths raise a clear
   ``ValueError`` if cancel is not implemented.
3. **Line-range generation** — ``python main/generate_line_ranges.py``
   with the text sample. Output must be **byte-identical** to the
   baseline (this is a deterministic token-based computation).
4. **Line-range readjustment** — ``python main/line_range_readjuster.py``
   with the text sample and its ``_line_ranges.txt``. The adjusted
   ranges will differ run-to-run (LLM boundary detection), but the
   JSONL header schema, ``ranges_adjusted`` / ``ranges_deleted`` /
   ``ranges_kept_original`` tallies, and the final ``completed_at``
   stamp must be present.
5. **Repair** — introduce a forced-failure marker (e.g., manually
   truncate one temp JSONL) and run ``python main/repair_extractions.py``
   to confirm the repair path still reaches
   ``batch/ops.retrieve_responses_from_batch`` and recovers cleanly.

## Deterministic Paths — byte-identical expected

- ``generate_line_ranges.py`` output ``_line_ranges.txt`` files.
- ``extract/config_builder.py`` output when given identical CLI args +
  YAML config. Verify by dumping the three effective configs and
  diffing (write a 5-line script that imports the functions and calls
  them with a Namespace of canned args).
- All ``tests/`` directory output (``pytest`` returns 920 passed).
- JSON schema files under ``schemas/`` — unchanged by the refactor.

## LLM-Dependent Paths — approximate equivalence expected

For each of the LLM paths, confirm:

1. **Shape equivalence.** The ``records[*]`` array in the output JSON
   has the same length as the baseline (one record per chunk).
2. **Schema conformance.** Each response entry contains the expected
   schema keys. No new or missing fields.
3. **Token accounting.** ``DailyTokenTracker`` usage after the run
   matches the baseline ± 5% (LLMs vary in completion length).
4. **No silent drops.** Logs contain no ``ERROR`` or ``CRITICAL`` lines
   unrelated to transient network errors.

## Behavior-Change Audit

These are the **intentional** differences between v4.x and v5.0.
Document each one when filing a verification report:

1. **Retry machinery.** LangChain's ``max_retries=0`` on every
   provider; the outer loop in ``extract/processing_strategy.py`` is
   the single retry authority. For Anthropic 429s, the previous
   worst-case was 25 attempts (5 outer × 5 inner); v5.0 caps at
   5 attempts. For non-Anthropic transient 5xx errors, v5.0 surfaces
   them immediately to the outer loop, which currently only retries on
   Anthropic 429 — verify that OpenAI / Google 5xx errors are handled
   at least as well as baseline.
2. **Anthropic batch JSON recovery.** The Anthropic batch backend now
   applies balanced-brace recovery to malformed JSON responses before
   ``json.loads``. Verify that previously-unparseable batch result
   items now surface as populated ``parsed_output`` dicts.
3. **Canonical developer-message path.** ``ConfigManager.load_developer_message``
   now delegates to ``SchemaManager``, which anchors on the project
   root. Scripts launched from any directory — not just the project
   root — now find the correct developer-message file.

## Reporting Template

After running the matrix, fill in a report at
``backup/v5_verification_<timestamp>.md`` following this template:

```
# v5.0 verification run -- <YYYY-MM-DD>

## Matrix results
<fill in Pass/Fail per cell>

## Deterministic-output diffs
<attach unified diff or 'byte-identical' per deterministic cell>

## LLM-path structural checks
<attach JSON key-set diff per LLM cell>

## Behavior-change findings
<unexpected regressions or confirmations against §Behavior-Change Audit>

## Token-usage report
<before/after token counts from DailyTokenTracker>

## Sign-off
<reviewer, date, commit SHA>
```

## Running the Unit-Test Suite

No API keys required:

```bash
pytest tests/ -v
```

Expected: **920 passed**. Any failure is a regression that must be
resolved before declaring v5.0 verified.
