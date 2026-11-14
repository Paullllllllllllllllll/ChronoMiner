---
name: chronominer
description: >-
  ChronoMiner. Use when the user wants structured data extracted from texts, images or
  PDFs with a JSON schema, or chunks, extracted entries or documents classified with
  a decision model, adjusts line ranges, resumes, repairs or batch-checks a
  ChronoMiner run, or reads its outputs (chronominer.sqlite, _output.json,
  _output.csv).
---

# ChronoMiner

The tool runs from a clone of the repository set up as in the README, Installation;
the schemas, classifiers and prompts it names live in that clone. The README
documents every behavior; `<command> --help` gives the flags, defaults and exit
codes of each command. Every call runs through uv with absolute paths, since
relative paths resolve against the clone:

```bash
uv --directory <clone> run chronominer <command> [options]
```

## Running a job

1. **Check access.** Done when the key variable of the model's provider is set
   (README, Providers and Response Formats) and you know whether the job runs
   synchronously or as a batch. `--batch` runs on OpenAI, Anthropic and Google;
   OpenRouter and custom endpoints have no batch API (exit 2).
2. **Dry run.** Add `--dry-run --json` to the intended command. Done when it exits 0,
   `to_process` lists exactly the intended items with the expected resume states, and
   every value you set by flag shows `"source": "flag"` in `spec`; the classify
   commands list `targets` in place of `to_process`. An unset key variable is only
   a warning here; the real run exits 2 on it.
3. **Sample** anything beyond one short item: `--pages 5` takes the first five chunks
   of a text or PDF, `--select 1-5` the first five items of a folder in the natural
   order of their paths (`--files` takes paths relative to the input folder). Read
   `usage.total.total_tokens` from the `--json` line and project the job. Done when
   the user has seen the projection for a job beyond a few items. The full command
   later resumes the sampled items.
4. **Run.** `--chunking line-ranges` runs only on ranges that `ranges adjust` has
   decided and whose adjusted starts you have read: token cuts split entries
   ([REFERENCE.md](REFERENCE.md), Line ranges). A job that finishes within the turn
   runs directly with `--json`; a longer one runs as a background process whose
   `--json` line you read when it ends. A batch job is two steps, `extract ...
   --batch` and `batch check --input <output folder> --wait`.
5. **Verify.** Done when the exit code is 0, `items_failed` and `items_blocked` are
   0 and every path in `outputs` exists. Items under `items_submitted` wait in a
   batch, and an item blocked by a batch exits 0 too: the job is done only once
   `batch check` has collected them. Exit 1 means an item failed or was refused, or
   another run leases one. Rerun the same command: it resumes and extracts only the
   pending and failed chunks. Then work through Troubleshooting. Exit 3 means the
   provider stopped serving the run: the chunks it did not reach stay pending, and
   the same command resumes them once the provider serves again.

## Conventions

- The `defaults:` block of the settings file sets the model, reasoning effort,
  service tier, output tokens and concurrency. Leave these flags out unless the user
  asks for a value. A task that edits the settings file restores it afterwards.
- A folder of many short items takes `--open-items N` (8, for example) on `extract`,
  `classify chunks` and `classify documents`: the run then keeps up to N items open
  under the one `--concurrency` limit, where otherwise an item with two chunks uses
  two request slots and leaves the rest idle. It has no effect with `--batch` or on
  a single item.
- `chronominer.sqlite` in each output root holds the run state that resume, `render`
  and `repair` read; `_output.json` and `_output.csv` are renderings of it. Stored
  documents leave the store through `drop` after a dry run, and the database file
  stays.
- A schema states its placeholders and conventions in its field descriptions: the
  prompts fill a field from the input or from what follows directly from it, else with
  `null` or the placeholder the schema describes.
- `--decisions shadow`, on the user's request, asks a decision model (README,
  Decision pre-pass) whether each chunk holds an entry of the schema before
  extraction, at a fraction of an extraction call's cost; the answer is the
  `decision_probability` column of the `chunks` view. A schema's top-level
  `decision_question` replaces the built question. `apply` skips a chunk only under
  a threshold that `decisions.thresholds` validates for the exact request; without
  one it skips nothing. A skipped chunk has the outcome `skipped_irrelevant`. A
  failed pre-pass only warns and extracts the chunks undecided, and a rerun leaves
  done chunks alone, so after a shadow run count the `chunks` rows whose
  `decision_probability` is null and report them.
- `classify chunks`, `classify entries` and `classify documents --classifier
  NAME|PATH` ask a decision model a classifier file's questions about chunks cut as
  `extract` cuts them, about the entries of finished documents of the classifier's
  schema, or about a document's leading chunks or pages; nothing is extracted and
  nothing gates a run. Shipped classifiers are in `classifiers/` (README,
  Classifiers); a project's own is passed by path. Read answers in the
  `classifications` view: chunks join on document and locator (the line range),
  entries on document and `chunk:entry`.
- The prompts insert the context as rules binding for every entry, so write a
  context file as rules. Above 4,000 characters the run warns.

Before editing or adjusting line ranges, setting an image option, changing the model
of a started image or PDF item, or querying the store, read
[REFERENCE.md](REFERENCE.md).

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| exit 1, an item leased by another run | wait for that run; a killed run's lease expires after 10 minutes, then rerun |
| item `refused` | its input, the schema of its done chunks or its image settings changed, or a partial image or PDF item met another model family: restore them, or `--force` (resets the item's stored units, keeps the responses) |
| item `blocked` by a batch | `batch check --input <output folder>`; an unresolved submission needs `--adopt` or `--abandon` |
| a chunk skipped as `skipped_irrelevant` holds entries | `repair --input <output> --skipped` extracts the skipped chunks without the predicate |
| item `unstored` | a rendered file without stored chunks, as after a `drop` that kept it: runs skip it; move the file away, or `--force` |
| a read fails with "attempt to write a readonly database" | an interrupted run left a journal; `render --input <folder> --dry-run` rolls it back |
