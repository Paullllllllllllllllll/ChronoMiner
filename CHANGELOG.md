# Changelog

- **v1.2.1** (30 October 2026) -- The vendored common package matches
  ChronoTranscriber v1.2.2: a write that creates, writes, resets or deletes a
  document's units marks every rendered file of the document stale, the
  line-ranges file included, and rendering clears the flag of each file it
  records. A format left out of the run that completed a document, such as the
  CSV after a `--formats json` resume of a partial document, is therefore
  rendered again by the next run that asks for it; earlier it kept the rows of
  the partial run.
- **v1.2.0** (30 October 2026) -- `chronominer classify chunks`, `classify entries`
  and `classify documents` ask a decision model the questions of a classifier file
  about the extraction chunks of input items, the entries of finished documents, or
  each item's leading chunks or first pages, and record the answers in the new
  `classifications` view of `chronominer.sqlite` without changing any extraction.
  Classifiers resolve like schemas from the new `classifiers/` folder, which ships
  `chunk_kind`, `language`, `document_type` and six entry classifiers for the
  example schemas. Chunks are cut as `extract` cuts them without writing to the
  extraction: a stored document keeps its stored line ranges, and `--chunking
  adjust` is refused. Answers with an unchanged fingerprint, input and content are
  reused and recorded again under the run unless `--force`, an item whose stored
  input changed is refused, and entry answers are written under the document's
  lease. The vendored common package matches ChronoTranscriber's shared core:
  `classifiers.py`, `decision_runner.py`, `classify_store.py` (the
  `classifications` component), `images.decision_jpeg`, unleased store writes, the
  classifier options, migrations refused under another run's live lease, and
  malformed decision answers that keep their billed usage. The relevance pre-pass
  now asks through the shared runner, so such usage counts toward the run's totals.
  The tool's store component steps to version 3, whose triggers flag the answers
  about a chunk's entries stale whenever any writer replaces, resets or deletes the
  chunk, and a document's chunk answers when its stored chunk layout changes; an
  older ChronoMiner refuses the stepped store.
- **v1.1.2** (29 October 2026) -- The Pillow requirement is raised to
  13.0, the first release that fixes the FlashPix, TIFF-writer and
  multi-frame loader advisories, so that no install resolves a vulnerable
  Pillow. The lock already held 13.0.0.
- **v1.1.1** (29 October 2026) -- The guided UI asks for the decision-model
  pre-pass of a new extraction after the page images: off, shadow or apply,
  and with the pass on, the decision model, its API and its endpoint. The
  printed command carries the answers as `--decisions`, `--decision-model`,
  `--decision-provider` and `--decision-endpoint`.
- **v1.1.0** (29 October 2026) -- `extract --decisions shadow|apply` adds a
  decision-model pre-pass: OpenAI's `/v1/decisions` or OpenRouter's decision API
  (default `typesafe/jev-1.14`) answers per chunk still to extract whether its text
  or page image holds an entry of the schema's kind, built from the schema's name
  and description or its `decision_question` key. The relevance lands in the new
  `chunk_decisions` table and the `decision_probability` column of `chunks`, which
  also gains `outcome` (store component version 2). `apply` skips chunks below a
  threshold with the outcome `skipped_irrelevant`, but only where
  `decisions.thresholds` holds one for the run's request fingerprint;
  `repair --skipped` extracts such chunks. A skipped chunk is stored and rendered
  as a no-content answer. A pre-pass that fails leaves every chunk to extraction,
  and decision retries keep the timeout and validation budgets. The vendored
  common package matches ChronoTranscriber v1.1.1: it gains `decisions.py`
  (request bodies and typed answers for both dialects), `decision_client.py`,
  `page_types.py`, the shared decision-model options and the `decisions` settings
  schema.
- **v1.0.4** (27 October 2026) -- The capability table knows `gpt-6.1-terra`:
  it takes text and image input and structured outputs like `gpt-6.1-sol`,
  accepts the reasoning efforts none to max, and shares the GPT-6 handling of
  verbosity, Flex and prompt caching. Before, the model resolved as text-only
  and image extraction refused it. The vendored `chronominer/common/` modules
  now meet the Google docstring rules as well, so ruff no longer exempts them;
  the copies stay byte-identical with AutoExcerpter and ChronoTranscriber.
- **v1.0.3** (27 October 2026) -- Documentation patch: ruff now checks
  docstrings against the Google convention (pydocstyle rules `D`, without D105
  and D107; tests, notebooks, scratch and the vendored `chronominer/common/`
  are exempt). The protocol members of the batch, event and extraction
  interfaces gained docstrings, summaries that ran over two lines were split
  into a one-line summary and a body, and the prompt helpers in
  `chronominer/llm/prompt_utils.py` moved from NumPy-style sections to Google
  prose. Behavior is unchanged.
- **v1.0.2** (26 October 2026) -- Security patch: the lockfile moves Pillow to
  13.0.0, which closes the Pillow 12.3.0 advisories, among them the TIFF decoder
  reads GHSA-c6hw-5grf-ff96 and GHSA-q45j-58j2-45rg and the MIC frame-count loop
  GHSA-22hg-r758-rrw9, which scanned page images can reach. Dependency versions
  only; no code changes.
- **v1.0.1** (24 October 2026) -- Security patch: the lockfile moves certifi to
  2026.10.23, which closes GHSA-6q48-6c7w-9gw2, and urllib3 to 2.8.0, which closes
  GHSA-8988-9cw3-xx77, GHSA-vxq7-64xx-v4gw and GHSA-gh4c-6fx4-qh6g. Dependency versions
  only; no code changes.
- **v1.0.0** (19 October 2026) -- First release. The `chronominer` package has one
  core, `run.py`, with three entry points, `discover`, `plan` and `run` (plus
  `plan_ranges` and `run_ranges` for line ranges); the command line in `cli/` and the
  guided terminal UI in `ui/` both call them, and the core never prints, prompts or
  exits but reports through run events. `spec.py` declares one option table per
  command (`extract`, `ranges generate`, `ranges adjust`, `batch check`,
  `batch cancel`, `repair`, `render`, `drop`), from which the parser, the wizard and
  the help texts are built, so flags and wizard answers yield the same `RunSpec`.
  `settings.py` reads `config/settings.yaml`, a file named by `--settings` or the
  packaged example; its optional `defaults:` block sets run options below the flags.
  `discovery.py` finds the text, image and PDF items of an input, applies `--all`,
  `--select` and `--files`, resolves the `_extract_context` and `_adjust_context`
  sidecars and lays out the output paths. Texts are cut into chunks by token count
  (`auto`), by an editable `{stem}_line_ranges.txt` (`line-ranges`), or by line
  ranges whose boundaries the model first moves to entry boundaries (`adjust`);
  `ranges generate` and `ranges adjust` write that file. A JSON schema from
  `schemas/` and the three prompts in `resources/prompts/` shape each request, and
  `extract/` sends the chunks and parses their entries. Each output folder holds
  `chronominer.sqlite`, which records the runs, every provider attempt verbatim, the
  chunk units and the entries, so an interrupted run resumes from it. Its views
  `entries`, `chunks` and `ranges` are the read contract; leases keep a second run off
  a document in use, `render` rewrites `{stem}_output.json` and the CSV without model
  calls, `drop` deletes stored documents, and `repair` extracts failed chunks again.
  `batch/` submits, checks, waits for, finalizes and cancels batches over one backend
  each for OpenAI, Anthropic and Google. The logic shared with AutoExcerpter and
  ChronoTranscriber is vendored byte-identically in `common/` and pinned by
  `MANIFEST.json`: the run store and SQLite writer (`store.py`, `sqlite.py`), the
  request builder and options (`provider_requests.py`, `request_options.py`), the
  model capability table (`capability_table.py`, `capability_facts.py`,
  `capabilities.py`), image presets and payloads (`image_presets.py`, `images.py`),
  retry and rate limits (`retry.py`, `rate_limit.py`), option tables and the settings
  schema (`options.py`, `standard_options.py`, `settings_schema.py`), the wizard
  (`wizard.py`), the batch backends (`batch_openai.py`, `batch_anthropic.py`,
  `batch_google.py`) and the exit codes and JSON summary (`contract.py`). `ext.py` is
  the extension seam for a local access layer: its hooks `register_args`,
  `settings_blocks`, `wizard_steps`, `startup`, `batch_allowed`, `extra_argv`,
  `key_variables`, `describe_access`, `client_options` and the `attempt` context
  around every model call, whose roles are `extract` and `adjust`, pass everything
  through by default.
