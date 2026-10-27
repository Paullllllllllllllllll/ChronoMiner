# Changelog

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
