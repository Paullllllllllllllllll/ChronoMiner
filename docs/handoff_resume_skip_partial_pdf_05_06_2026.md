# Handoff: resume skips partial PDF extractions as "complete"

Date: 5 June 2026
Branch: `main`
Author of prior session: production-run triage (deutschland Michelin guides)

## Task for the next session

Confirm and fix the resume bug that caused `mg_deutschland_2014.pdf`'s 174
failed pages to be silently skipped on a resume run instead of being
reprocessed. The root cause is traced and confirmed below; the next session
should implement the fix, add a regression test, and run a targeted 2014
recovery once the daily token budget resets.

## Symptom (observed this session)

A resume run (mode: "skip fully processed, resume partial") over the
deutschland folder logged `Skipping mg_deutschland_2014.pdf: already fully
processed` and spent 0 tokens on it (token counter flat across file 29/33).
The file is genuinely partial: 1234 of 1408 pages extracted, 174 failed. The
failed pages were never re-queued, so the run did not recover them.

## Root cause (confirmed)

There are two independent resume gates for PDF/visual inputs. The buggy one
fires first and returns before the correct one runs.

1. Early-resume gate in `_process_visual_file`
   (`modules/extract/file_processor.py:407-429`). Added as an optimisation to
   skip before expensive image rendering. It computes
   `total = metadata["total_chunks"]` and skips when
   `total > 0 and len(records) >= total`.

2. Correct gate `detect_extraction_status`
   (`modules/extract/resume.py:89-139`), called from
   `_execute_extraction` (`file_processor.py:645-662`). It uses
   `expected_chunks=len(chunks)` — the true rendered page count (1408 for
   2014) — and correctly returns `PARTIAL` for 2014. It is never reached
   because gate (1) returns first.

The defect is that `total_chunks` is not the true page count. In
`_generate_output_files` it is stamped as `total_chunks=len(results)`
(`file_processor.py:935`), i.e. the number of records actually written. Failed
chunks write no temp record (see comment at `file_processor.py:689-697`), so a
partial extraction stamps a *reduced* total equal to its own success count. On
the next resume, gate (1) then evaluates `len(records) (1234) >= total (1234)`
→ true → "already fully processed". The `partial: true` flag and the 174-entry
`failed_chunks` list sitting in the same metadata block are ignored entirely.

In short: a partial file records a self-consistent-but-wrong total, so it
always looks complete to the early gate on the next run.

## Evidence

Metadata head of `mg_deutschland_2014_output.json` (in NetworksOfTaste at
`data/raw/chronominer/production/deutschland/`):

```json
{
  "_chronominer_metadata": {
    "schema_name": "MichelinGuidesLight",
    "model_name": "gpt-5.4-mini",
    "total_chunks": 1234,
    "partial": true,
    "failed_chunks": [3, 5, 15, 21, 38, 42, 44, 45, 53, 54, ...]
  }
}
```

`total_chunks` is 1234, not the true 1408 (= 1234 written + 174 failed); the
PDF has 1408 pages. `partial` is `true` and `failed_chunks` lists 174 indices.
Gate (1) reads only `total_chunks` and `len(records)`, so it skips.

## Impact / scope

Any visual (PDF/image) extraction that finished partial is affected: every such
file will be skipped on subsequent resume runs and its failed pages will never
be recovered. Text inputs route through `process_file` →
`_execute_extraction` directly and do not hit the early gate, so they fall
through to the correct `detect_extraction_status` path; they are not affected by
gate (1). The batch path is also unaffected (resume detection is sync-only,
`file_processor.py:647`).

## Suggested fix (for the next session to decide and implement)

The cleanest fix has two parts; either alone closes the 2014 case, but both are
worth doing.

1. Make the early gate honour partiality. In the early-resume block
   (`file_processor.py:407-429`), treat the file as complete only when it is not
   flagged partial and has no failed chunks — e.g. skip only if
   `meta_block.get("partial") is not True and not meta_block.get("failed_chunks")`
   in addition to the count check. This is the minimal, low-risk fix and uses
   data already present in the metadata.

2. Stop stamping a misleading `total_chunks`. `total_chunks` should reflect the
   true unit count (rendered pages / produced chunks), not the number of
   successful records. Today it is `len(results)`
   (`file_processor.py:935`); failed chunks should be counted in the
   denominator so `total_chunks == successes + len(failed_chunks)`. Note the
   resume contract: `detect_extraction_status` derives completion from distinct
   `custom_id` chunk indices in `records` vs `expected_chunks=len(chunks)`, so
   the threaded-through value must be the full unit count. Check callers of
   `build_extraction_metadata` (`resume.py:47-73`) and the merge path
   (`_merge_with_existing_output`, `file_processor.py:969-1028`) when changing
   this, so a later full completion re-stamps the correct total.

Prefer fix (1) as the primary correctness fix (it directly restores the
intended skip/resume semantics regardless of how `total_chunks` is computed),
and apply (2) so the persisted metadata is no longer self-contradictory
(`total_chunks: 1234` next to a 174-entry `failed_chunks`).

## Verification

- Add a regression test alongside `tests/test_resume.py` (and/or
  `tests/test_file_processor_offline.py`) that builds a partial visual output
  (`partial: true`, `failed_chunks` non-empty, `total_chunks` == success count)
  and asserts the early gate does NOT skip it — i.e. the file is selected for
  resume. Existing `detect_extraction_status` tests
  (`tests/test_resume.py:50-152`) already cover the second gate and should stay
  green.
- After the fix, dry-run a resume over the deutschland folder and confirm 2014
  is reported as resuming (e.g. `Resuming mg_deutschland_2014.pdf: 1234/1408`),
  not skipped.
- `uv run ruff check .` and `uv run mypy .` on touched modules; run the
  `code-health` skill on the source change before committing.

## Recovery action for 2014 (after fix + token reset)

The partial `mg_deutschland_2014_output.json` is lean (6.7 MB) and intact, and
its `_temp.jsonl` (996 MB, full API calls) is untouched, so a targeted resume is
cheap — only the 174 failed pages re-run. Once the daily token limit resets
(00:00 UTC; cap currently `27,500,000` in `config/concurrency_config.yaml`,
gitignored) and the fix is in, run a single-file resume on
`mg_deutschland_2014.pdf` at `concurrency_limit: 25`. Expect failed count → ~0
and a lean merged output. Budget note: each ~1,300–1,420-page guide costs
roughly 11M tokens at High image detail, so the cap funds about two fresh guides
per day; lowering image detail (High → Auto/Low) is the largest per-page token
lever if the budget keeps binding. Guides 2016, 2017, 2018 are still
unprocessed.

## Pointers

- Buggy gate: `modules/extract/file_processor.py:407-429`
  (`_process_visual_file`).
- `total_chunks` stamp: `modules/extract/file_processor.py:930-941`
  (`_generate_output_files`).
- Correct gate: `modules/extract/resume.py:89-139`
  (`detect_extraction_status`), invoked at `file_processor.py:645-662`.
- Metadata builder: `modules/extract/resume.py:47-73`
  (`build_extraction_metadata`).
- Resume merge: `modules/extract/file_processor.py:969-1028`
  (`_merge_with_existing_output`).
- Resume tests: `tests/test_resume.py`,
  `tests/test_file_processor_offline.py`.

## Suggested skills for the next session

- `code-health` — run on the source change before committing.
- `commit-rules` — for the commit/push (direct to `main` per repo practice;
  bump version and update the README header + changelog per repo convention).
