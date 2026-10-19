# Changelog

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
