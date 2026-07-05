# ChronoMiner v1.24.0

A Python-based structured data extraction tool for researchers,
archivists, and digital humanities projects. ChronoMiner transforms
text files, scanned images, and PDFs into analyzable datasets (JSON,
CSV, DOCX, TXT) using schema-based LLM processing with multiple AI
providers.

Designed to integrate with
[ChronoTranscriber](https://github.com/Paullllllllllllllllll/ChronoTranscriber)
and
[ChronoDownloader](https://github.com/Paullllllllllllllllll/ChronoDownloader)
for a complete document retrieval, transcription, and data extraction
pipeline.

> **Work in Progress** -- ChronoMiner is under active development.
> If you encounter any issues, please
> [report them on GitHub](https://github.com/Paullllllllllllllllll/ChronoMiner/issues).

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Providers and Models](#supported-providers-and-models)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Batch Processing](#batch-processing)
- [Utilities](#utilities)
- [Architecture](#architecture)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Contributing](#contributing)
- [Development](#development)
- [Versioning](#versioning)
- [License](#license)

## Overview

ChronoMiner enables researchers and archivists to extract structured
data from historical and academic documents at scale with minimal
cost and effort. It supports multiple AI providers through a unified
LangChain-based architecture, schema-driven extraction with 12
built-in templates, and fine-grained control over chunking,
concurrency, and output.

**Execution modes:**

- **Interactive** -- guided terminal wizard with back/quit
  navigation. Ideal for first-time users and exploratory workflows.
- **CLI** -- headless automation for scripting and pipelines.
  Set `interactive_mode: false` in `config/paths_config.yaml` or
  pass arguments directly.

**Supported input types:**

- **Text files** -- plain text (.txt, .md) with automatic encoding
  detection
- **Image folders** -- PNG, JPEG, WEBP, BMP, TIFF sent directly to
  vision-capable LLMs
- **PDFs** -- rendered page-by-page and processed through vision
  models
- **Mixed directories** -- auto-detected per file

## Key Features

- **Multi-provider LLM support** via LangChain (OpenAI, Anthropic,
  Google, OpenRouter, custom OpenAI-compatible endpoints)
- **Centralized capability registry** -- single source of truth for
  all provider/model capabilities; unsupported parameters filtered
  automatically before API calls
- **Hierarchical context resolution** -- file-specific,
  folder-specific, or project-wide extraction and adjustment context
  (`{name}_extract_context.txt` / `{name}_adjust_context.txt`
  convention)
- **Context image injection** -- provide a visual reference image
  alongside text context using the same hierarchical convention
  (`{name}_extract_context.png/jpg/...`); the image is sent in the
  user message before the input content (OpenAI, `--context-image`)
- **Schema-based extraction** -- 13 built-in JSON schemas with
  structured LLM output; custom schemas supported
- **Four chunking strategies** -- automatic, automatic with manual
  adjustment, pre-defined line ranges, LLM-adjusted semantic
  boundaries
- **Semantic boundary detection** -- LLM-powered chunk optimization
  with certainty validation and automatic retry
- **Batch processing** -- async batch APIs for OpenAI, Anthropic,
  and Google with 50% cost savings on OpenAI
- **Native image/PDF input** -- process scans and PDFs directly
  through vision-capable LLMs without a prior transcription step
- **Four output formats** -- JSON (always), CSV, DOCX, TXT
  (toggleable per schema)
- **Daily token budget** -- configurable per-day limits with
  automatic reset at 00:01 UTC
- **Resume and repair** -- skip processed files; repair incomplete
  batch jobs after the fact

## Supported Providers and Models

Set the provider in `config/model_config.yaml` or let the system
auto-detect from the model name.

| Provider | Notable model families | Env variable | Batch |
|----------|----------------------|--------------|-------|
| OpenAI | GPT-5.4, GPT-5.3, GPT-5.2, GPT-5.1, GPT-5, o-series, GPT-4.1, GPT-4o | `OPENAI_API_KEY` | Yes |
| Anthropic | Claude Opus 4.6/4.5/4.1/4, Sonnet 4.6/4.5/4, Haiku 4.5 | `ANTHROPIC_API_KEY` | Yes |
| Google | Gemini 3, 2.5, 2.0, 1.5 (Pro, Flash variants) | `GOOGLE_API_KEY` | Yes |
| OpenRouter | 100+ models via unified API | `OPENROUTER_API_KEY` | No |
| Custom | Any OpenAI-compatible endpoint | User-configured | No |

**Model capabilities** are detected automatically. Reasoning models
(GPT-5.x, o-series, Gemini 2.5+, Claude 4.x) have temperature/top_p
disabled and reasoning effort configurable via the cross-provider
`reasoning.effort` parameter. Standard models retain full sampler
control.

### Custom OpenAI-Compatible Endpoint

Connect to any self-hosted or third-party endpoint implementing the
OpenAI Chat Completions API. Set `provider: custom` in
`model_config.yaml` and configure the `custom_endpoint` block:

```yaml
extraction_model:
  provider: custom
  name: "org/model-name"
  custom_endpoint:
    base_url: "https://your-endpoint.example.com/v1"
    api_key_env_var: "CUSTOM_API_KEY"
  max_output_tokens: 4096
  temperature: 0.0
```

The endpoint must support OpenAI-compatible structured outputs
(`response_format` with `type: json_schema`). Custom endpoints do
not support batch processing.

## System Requirements

- **Python** 3.12+
- At least one API key (see provider table above)

All Python dependencies are declared in `pyproject.toml`.

## Installation

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoMiner.git
cd ChronoMiner

# Install runtime dependencies (creates .venv automatically)
uv sync

# For development and tests
uv sync --all-extras
```

**Configure API keys:**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key_here"

# Linux/macOS
export OPENAI_API_KEY="your_key_here"
```

For persistent configuration, add to system environment variables or
shell profile.

**Configure paths and model:** on a fresh clone, ChronoMiner
automatically loads the bundled `*.example.yaml` templates from
`config/` and prints one INFO line for each file used as a default.
Copy any example to its real name and edit it to set your own values:

```bash
cp config/paths_config.example.yaml config/paths_config.yaml
cp config/model_config.example.yaml config/model_config.yaml
# repeat for other config files as needed
```

Real config files are gitignored; only the scrubbed example templates
are tracked.

## Quick Start

### Your First Extraction

**Interactive mode** (recommended for new users):

```bash
python main/process_text_files.py
```

The wizard guides you through schema selection, chunking strategy,
processing mode, and file selection. Press `b` to go back, `q` to
quit at any time.

**CLI mode:**

```bash
# Extract from a single text file
python main/process_text_files.py --schema BibliographicEntries \
    --input data/file.txt

# Batch process an entire directory (50% cheaper)
python main/process_text_files.py --schema BibliographicEntries \
    --input data/ --batch

# Extract from images or PDFs
python main/process_text_files.py --schema HistoricalRecipesEntries \
    --input path/to/images/ --image-detail high
```

### Common Workflows

**Large-scale batch processing:**

```bash
# Submit
python main/process_text_files.py --schema BibliographicEntries \
    --input data/ --batch
# Monitor (run periodically; auto-downloads on completion)
python main/check_batches.py
```

**Pre-generate and adjust semantic line ranges:**

```bash
# Generate token-based line ranges
python main/generate_line_ranges.py --input data/file.txt
# Adjust boundaries to semantic sections
python main/line_range_readjuster.py --input data/file.txt \
    --schema BibliographicEntries
# Extract using adjusted ranges
python main/process_text_files.py --schema BibliographicEntries \
    --input data/file.txt --chunking line_ranges
```

**Repair failed extractions:**

```bash
python main/repair_extractions.py --schema BibliographicEntries
```

### CLI Reference

```
--schema NAME              Schema name for extraction
--input / --output         Input and output paths
--chunking STRATEGY        auto | auto-adjust | line_ranges |
                           adjust-line-ranges
--batch                    Use async batch API
--image-detail LEVEL       low | high | auto | original
--input-type TYPE          Override auto-detection: text | image | pdf
--model ID                 Override model
--reasoning-effort LEVEL   none | low | medium | high | xhigh
--chunk-size N             Override tokens per chunk
--context MODE_OR_PATH     auto | none | /path/to/context.txt
--context-image            Enable context image injection (see below)
--first-n-chunks N         Process only the first N chunks/pages
--last-n-chunks N          Process only the last N chunks/pages
--resume / --force         Skip vs overwrite existing output
--interactive / --non-interactive
                           Force the run mode, overriding the config-file mode
--dry-run                  Discover inputs, classify resume state, and report
                           planned actions with zero API calls or side effects
--json                     Emit a one-line JSON run summary on stdout
```

Run `python main/process_text_files.py --help` for the full list.

Both `.txt` and `.md` files are collected in CLI mode; the tool's own report
files (`*_output.txt`, `*_line_ranges.txt`, `*_context.txt`) are excluded so a
second run never extracts from its own output.

### Exit codes and automation

Primary entry points follow a uniform contract for scripting and CI:

- `0` -- every file completed (or was skipped as already complete);
- `1` -- one or more files failed or completed partially (and, for
  `check_batches.py`, one or more batches remain pending or failed);
- `2` -- usage or configuration error, or interactive mode requested without a
  TTY;
- `130` -- interrupted by the user (Ctrl+C).

`--json` prints a single machine-readable summary line at the end
(files/complete/partial/failed/skipped, token totals). `check_batches.py` also
accepts `--json` and exits non-zero while any batch is still pending, so a
poller can loop until every batch resolves.

## Configuration

ChronoMiner uses five YAML files in `config/`, plus one optional file
(`api_keys_config.yaml`) for remapping API-key environment variables.

### Example/real config split

Every config file ships as a tracked, scrubbed `<name>.example.yaml`
template. Real config files (`<name>.yaml`) are gitignored and private.

On a fresh clone (no real files present), the loader automatically falls
back to the bundled example and logs one INFO line per file:

```
Config 'model_config.yaml' not found; using bundled defaults from
'model_config.example.yaml'. Copy it to 'model_config.yaml' and edit
it to set your own values.
```

Copy any example to its real name when you need to customize it:

```bash
cp config/model_config.example.yaml config/model_config.yaml
```

If neither the real file nor the example template can be found, a clear
`FileNotFoundError` is raised naming the missing file.

### 1. Model Configuration (`model_config.yaml`)

```yaml
extraction_model:
  provider: openai       # openai | anthropic | google | openrouter
                         # | custom (auto-detected if omitted)
  name: gpt-5-mini
  max_output_tokens: 128000
  reasoning:
    effort: medium       # Cross-provider (low | medium | high)
  temperature: 0.01      # Disabled automatically for reasoning
  top_p: 1.0             #   models
```

Key parameters: `provider` (auto-detected from model name if
omitted), `name` (model identifier), `max_output_tokens` (must
cover reasoning tokens on reasoning models), `reasoning.effort`
(automatically translated per provider), `temperature`/`top_p`
(applied only when the model supports them).

### 2. Paths Configuration (`paths_config.yaml`)

```yaml
general:
  interactive_mode: true
  retain_temporary_jsonl: true
  logs_dir: './logs'
schemas_paths:
  BibliographicEntries:
    input: './input/bibliography'
    output: './output/bibliography'
    csv_output: true
    docx_output: true
    txt_output: true
```

Controls execution mode, temporary file retention, and per-schema
input/output directories with output format toggles.

### 3. Chunking Configuration (`chunking_and_context.yaml`)

```yaml
chunking:
  default_tokens_per_chunk: 7500
```

Also configures matching rules (whitespace normalization, case
sensitivity, diacritics) and retry behavior for the line-range
readjuster (certainty threshold, max retries, context expansion).

### 4. Image Processing Configuration (`image_processing_config.yaml`)

Provider-specific sections configure vision preprocessing
automatically based on the active model:

```yaml
api_image_processing:
  llm_detail: high          # OpenAI / OpenRouter
anthropic_image_processing:
  resize_profile: auto      # Anthropic
google_image_processing:
  media_resolution: high    # Google Gemini
target_dpi: 300             # PDF-to-image rendering
max_pixels_per_page: 24000000  # Per-page render budget (auto DPI
                               # reduction above this)
```

PDF pages are rendered, preprocessed, and encoded one at a time
(streaming), so memory usage stays constant regardless of document
length. Each output records the preprocessing parameters and a
SHA-256 of every image sent to the model for reproducibility.

### 5. Concurrency Configuration (`concurrency_config.yaml`)

```yaml
concurrency:
  extraction:
    concurrency_limit: 20
    max_concurrent_files: 4
    delay_between_tasks: 0.1
    retry:
      attempts: 150
daily_token_limit:
  enabled: true
  daily_tokens: 9000000
```

Controls concurrent task limits, exponential backoff retry, and
daily token budgets (automatic reset at 00:01 UTC, one minute after
OpenAI's 00:00 UTC free-tier reset).
`max_concurrent_files` caps how many files run at once when the
daily token limit is disabled (visual runs are clamped to 2).

### 6. API Key Mapping (`api_keys_config.yaml`, optional)

```yaml
openai: OPENAI_API_KEY
anthropic: ANTHROPIC_API_KEY
google: GOOGLE_API_KEY
openrouter: OPENROUTER_API_KEY
```

Maps each provider to the name of the environment variable that holds
its API key. Change a value (for example `openai: OPENAI_API_KEY_2`) to
swap which key a provider uses without touching your environment. The
file is optional and fully backward-compatible: a missing file or any
omitted provider falls back to the default env var name shown above. The
remap applies to synchronous extraction and batch processing alike.
Values are env var names, not keys, so no secrets are stored here.

### Context Resolution

Hierarchical context resolution automatically selects the most
specific extraction or adjustment guidance available:

1. **File-specific**: `{input_stem}_extract_context.txt` next to
   the input file
2. **Folder-specific**: `{parent_folder}_extract_context.txt` next
   to the input's parent folder
3. **General fallback**: `context/extract_context.txt` in the
   project root

The same hierarchy applies for line-range adjustment using
`_adjust_context.txt` suffixes. Context files should be plain text
describing the document type, expected content, and domain-specific
terminology. Keep under 4,000 characters.

**Context images** follow the same hierarchy using image extensions
instead of `.txt`:

1. **File-specific**: `{input_stem}_extract_context.png` (or
   `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`, `.webp`)
2. **Folder-specific**: `{parent_folder}_extract_context.png`
3. **General fallback**: `context/extract_context.png`

Enable with `--context-image` (CLI) or the interactive prompt. The
context image is sent in the user message before the input content,
labeled `"Context image:"`. It uses the same detail level as the
main image (`--image-detail`). Useful for providing visual reference
(e.g., a symbol legend, sample page layout, or annotation key).
Currently supported for OpenAI and OpenRouter providers only.

### Custom Extraction Schemas

Place JSON schemas in `schemas/`. Thirteen schemas are included:

- `bibliographic_schema.json`, `summary_schema.json`,
  `address_schema.json`, `military_record_schema.json`,
  `culinary_works_schema.json`, `culinary_persons.json`,
  `culinary_places.json`, `culinary_entities.json`,
  `cookbook_metadata_schema.json`,
  `historical_recipes_schema_production.json`,
  `michelin_guides.json`, `michelin_guides_light.json`,
  `historical_price_entries_schema.json`

To add a custom schema: create the JSON schema file in `schemas/`,
register it in `modules/extract/schema_handlers.py`, add context
guidance, and configure paths in `paths_config.yaml`.

## Output Formats

Four formats, toggleable per schema in `paths_config.yaml`:

- **JSON** (always) -- complete structured dataset with metadata
- **CSV** (optional) -- schema-specific tabular format
- **DOCX** (optional) -- formatted Word document with tables
- **TXT** (optional) -- human-readable plain text report

Output files are named `<original_name>.{ext}`.

## Batch Processing

Async batch APIs for OpenAI, Anthropic, and Google. OpenAI offers
50% cost savings. OpenRouter and custom endpoints do not support
batch mode.

**How it works:**

1. Chunks are built (text) or pages rendered (image/PDF)
2. Requests are formatted per provider and submitted as batch jobs
3. Large request sets are split by the provider's per-batch request-count and
   byte limits into `_part{n}` submissions (one tracking record per part);
   `check_batches.py` merges the parts back into one final output
4. Metadata is saved to a temporary JSONL for tracking and repair
5. A debug artifact (`*_batch_submission_debug.json`) listing every submitted
   batch id is saved next to the temp file for recovery
6. Remote input/output files are deleted only after the final output JSON is
   durably written locally, so a mid-download failure never destroys results
7. Finalization writes the same `{stem}_output.json` shape as synchronous
   extraction (`records` + `_chronominer_metadata`, with batch provenance
   under `batch_tracking`); the legacy `{stem}_final_output.json` shape is
   no longer written but files already on disk remain readable

| Provider | Cost savings | Typical completion |
|----------|-------------|-------------------|
| OpenAI | 50% | Within 24 hours |
| Anthropic | Varies | Hours to days |
| Google | Varies | Varies |

**Monitoring and cancellation:**

```bash
# Check status, auto-download completed results
python main/check_batches.py

# Cancel all non-terminal batch jobs
python main/cancel_batches.py
```

## Utilities

### Line Range Generation

Generate token-based line ranges for text files before extraction:

```bash
python main/generate_line_ranges.py --input data/file.txt
python main/generate_line_ranges.py --input data/ --tokens 5000
```

Creates `{filename}_line_ranges.txt` files specifying exact line
ranges per chunk.

### Line Range Readjustment

Optimize chunk boundaries using LLM-detected semantic sections.
Each decision is persisted to a JSONL sidecar for range-level
resume:

```bash
python main/line_range_readjuster.py --input data/file.txt \
    --schema BibliographicEntries
# Resume after interruption
python main/line_range_readjuster.py --input data/ \
    --schema BibliographicEntries --resume
```

The readjuster uses certainty-driven retry: low-confidence
boundaries are retried with broader context windows until the
threshold is met or retries are exhausted.

### Extraction Repair

Recover partial results from incomplete batch jobs:

```bash
python main/repair_extractions.py --schema BibliographicEntries
```

Discovers incomplete jobs, recovers batch IDs from debug artifacts,
retrieves available responses, and regenerates outputs. Status checks route
through the provider-agnostic backend (not OpenAI-only), and the regenerated
output is marked `fully_completed` only when no batch failed or is missing.

### Daily Token Budget

Enable in `concurrency_config.yaml` to cap daily API usage. Tracks total tokens
per call and resets at 00:01 UTC (one minute after OpenAI's 00:00 UTC
free-tier reset). State lives in a user-level directory
(`~/.chronominer/` by default, overridable via `state_dir` in
`paths_config.yaml`) so the budget is shared across runs regardless of the
working directory; a legacy `.chronominer_token_state.json` in the working
directory is adopted once if the user-level file is absent. Writes are
debounced (with a flush at exit), so cross-process enforcement is best-effort.

#### Shared Cross-Tool Token Budget (optional)

ChronoMiner can share ONE combined daily budget with its sibling tools
(ChronoTranscriber, AutoExcerpter) instead of enforcing its cap in isolation.
Off by default; single-tool installations need not care. Enable it in
`concurrency_config.yaml`:

```yaml
shared_token_budget:
  enabled: true
  ledger_dir: ''   # empty = ~/.chronopipeline; or an absolute path
```

When enabled, every participating tool merges its usage into one shared
ledger (`token_ledger.json`) guarded by an OS file lock, and
`daily_token_limit.daily_tokens` is enforced against the COMBINED total, so
several tools running concurrently cannot collectively overshoot the budget.
Usage is merged as deltas under the lock (concurrent processes lose nothing);
the hot path stays in memory with a debounced background sync, plus forced
refreshes near the cap and while waiting at the limit. If the ledger is ever
unavailable, the tool degrades to its private counter with a single warning
and never crashes. Keep `daily_tokens` identical across participating tools;
the strictest value simply stops its tool first. Editing `daily_tokens` while
a tool waits at the limit lifts the cap within a poll cycle, no restart
needed.

## Architecture

ChronoMiner follows a deep-module architecture: nine packages under
`modules/`, each with a narrow public surface, composed by CLI entry
points in `main/`.

```
modules/
+-- config/        YAML config, capability registry, context
|                  resolution
+-- infra/         Logging, token budget, paths, concurrency,
|                  chunking, JSONL
+-- conversion/    JSON to CSV / DOCX / TXT converters
+-- images/        Vision preprocessing, encoding, PDF rendering
+-- llm/           Provider abstraction, extraction primitives,
|                  schema handling
+-- batch/         Multi-provider batch backends, submit / check /
|                  cancel / repair
+-- extract/       Extraction workflow, file processor, resume
+-- line_ranges/   Line-range generation and semantic readjustment
+-- ui/            Interactive prompts, workflow wizard

main/
+-- process_text_files.py       Primary entry point
+-- generate_line_ranges.py     Token-based chunking
+-- line_range_readjuster.py    Semantic boundary optimization
+-- check_batches.py            Monitor and finalize batch jobs
+-- cancel_batches.py           Cancel non-terminal batch jobs
+-- repair_extractions.py       Recover incomplete extractions
```

**Dependency graph** (strictly acyclic, arrows point downward):

```
main/
  |
extract -------> llm, batch, conversion, images, ui, config, infra
line_ranges ---> llm, conversion, images, ui, config, infra
batch ---------> llm, conversion, config, infra, ui
llm -----------> images, config, infra
images, conversion, ui --> config, infra
config, infra   (foundation; no modules/ imports)
```

## Frequently Asked Questions

**Which AI provider should I choose?**
OpenAI `gpt-5-mini` offers the best cost/quality balance with a 50%
batch discount. Google Gemini Flash is fastest and cheapest but may
reject deeply nested schemas. Anthropic Claude excels with complex
layouts. Start with OpenAI `gpt-5-mini` at medium reasoning effort.

**How much does extraction cost?**
With OpenAI `gpt-5-mini`: roughly $0.10--0.20 per small file
(50 KB), $1--2 per medium file (500 KB). Batch processing halves
these costs.

**Batch or synchronous?**
Use batch for large-scale jobs where you can wait up to 24 hours.
Use synchronous for immediate results, small jobs, or testing.

**Which chunking strategy should I use?**
Start with automatic for quick runs. For production workflows,
generate line ranges, adjust with semantic boundaries, then extract
using the adjusted ranges.

**How do I add a custom schema?**
Create a JSON schema in `schemas/`, register it in
`modules/extract/schema_handlers.py`, add context guidance, and
configure paths in `paths_config.yaml`.

**How do I switch providers?**
Edit `config/model_config.yaml` and set the appropriate environment
variable. Provider can also be auto-detected from the model name.

**What happens when extraction fails?**
Failed chunks are logged. Use `repair_extractions.py` to recover
partial results from batch jobs. For synchronous jobs, re-run with
`--resume` to skip already-processed chunks.

**Can I process password-protected PDFs?**
No. Decrypt them first using external tools.

**How do I integrate into existing pipelines?**
Use CLI mode (`interactive_mode: false`). All scripts return proper
exit codes suitable for shell scripting and CI/CD.

**I'm experiencing issues not covered here.**
Check logs in the configured `logs_dir`, validate configuration
files, and review `pyproject.toml` for version mismatches. For
persistent issues, open a
[GitHub issue](https://github.com/Paullllllllllllllllll/ChronoMiner/issues)
with error details and relevant configuration sections.

## Contributing

Contributions are welcome. When reporting issues, include: a clear
description, steps to reproduce, expected vs. actual behavior, your
environment (OS, Python version), relevant config sections (remove
sensitive data), and log excerpts.

For code contributions: fork the repository, create a feature branch,
follow the existing code style, add tests, and submit a pull request.

## Development

Install all dependencies (runtime + dev):

```bash
uv sync --all-extras
```

Run the test suite, linter, and type checker:

```bash
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

The suite contains 980+ tests (unit and integration) covering all
modules, providers, batch backends, and CLI parsers.

## Versioning

This project follows semantic versioning (`MAJOR.MINOR.PATCH`). The version in
`pyproject.toml` is the single source of truth; it is mirrored in the title
heading above and tagged in git as `vX.Y.Z`. The commit history was squashed to
a single baseline commit at v1.0.0 on 25 April 2026; version numbers before
v1.0.0 do not exist.

## Changelog

- **v1.24.0** (5 July 2026) -- Fix the daily token budget's reset boundary.
    Both the private per-tool tracker (`modules/infra/token_tracker.py`) and
    the vendored shared cross-tool ledger (`modules/infra/shared_ledger.py`,
    bumped to 1.1.0) now roll the budget day over at 00:01 UTC -- one minute
    after OpenAI's 00:00 UTC free-tier reset -- instead of local midnight,
    so the tool never frees its budget before OpenAI's own counter has
    actually reset. The one-minute buffer is a deliberate safety margin
    against clock skew. `get_reset_time()` now returns a timezone-aware UTC
    datetime; user-facing wait messages show the local wall-clock time
    alongside an explicit "(00:01 UTC)" anchor for clarity. Updated docs and
    config comments accordingly. The shared ledger module must be re-vendored
    (module and its test) to ChronoTranscriber and AutoExcerpter to keep all
    three tools on one combined budget day. All tests pass.

- **v1.23.0** (3 July 2026) -- Provider-compatibility, batch-routing, and
    readjuster fixes from a live cross-provider bug hunt. Clamp
    `max_output_tokens` to each model's real output ceiling at
    request-build time (Anthropic caps registered; claude-haiku-4-5 no
    longer 400s on oversized requests). Map Gemini 3.x reasoning effort to
    the supported `thinking_level` parameter instead of the ignored
    `thinking_config`. Fall back to non-strict JSON-instructed extraction
    when Anthropic rejects a structured-output schema for exceeding its
    union-type parameter limit. `check_batches.py` no longer reports a
    false all-clear when a configured schema directory cannot be resolved
    (new `errors` count in the `--json` summary; non-zero exit), finalizes
    batch outputs into the submission's own directory instead of the
    schema default (`repair_extractions.py` likewise), and, together with
    `cancel_batches.py` and `repair_extractions.py`, accepts the shared
    `--interactive`/`--non-interactive` flags. Line-range readjuster:
    range deletions now require a dedicated, stricter
    `retry.delete_certainty_threshold` (default 85) and every accepted
    deletion is logged with its dropped line span; sliced adjustment runs
    compose with `--resume` via a post-write fingerprint chain (sliced
    work is reused, external edits still invalidate); context-resolution
    log lines are now visible in console and application.log.

- **v1.22.0** (3 July 2026) -- Optional shared cross-tool token budget.
    Add the vendored `modules/infra/shared_ledger.py` (locked delta merges
    into per-tool fields, atomic per-process temp writes, local-midnight
    rollover, degrade-to-standalone) and wire it into the daily token
    tracker behind the new opt-in `shared_token_budget` config block: when
    enabled, `daily_token_limit.daily_tokens` is enforced against the
    COMBINED usage of ChronoMiner, ChronoTranscriber, and AutoExcerpter via
    one ledger at `~/.chronopipeline/token_ledger.json`, with seed-once
    adoption of legacy same-day counts, a debounced off-loop sync, forced
    refreshes near the cap and while waiting at the limit, and per-tool
    breakdown in the usage stats. Default behavior (feature off) is
    unchanged. Verified live: concurrent tools share one limit with zero
    lost updates.

- **v1.21.0** (3 July 2026) -- Concurrency and token-budget hardening.
    Add a per-provider multi-window rate limiter with adaptive backoff
    (`modules/infra/rate_limit.py`, configured via `concurrency.rate_limits`)
    wired into every synchronous LLM call; honor HTTP `Retry-After` in the
    retry loop and lower the default retry budget from 25 attempts (180 s
    cap) to 8 attempts (120 s cap); count Anthropic prompt-cache creation
    and read tokens at full weight in the daily budget and recover token
    usage from failed attempts; re-read `daily_token_limit.daily_tokens`
    during the wait-at-limit loop so a mid-wait config edit lifts the cap
    without restart; write token state via per-process-unique temp files;
    close provider HTTP clients on extractor teardown; move temp-JSONL and
    output-file I/O off the event loop; remove the dead
    `run_concurrent_tasks` helper and the unused
    `ProviderConfig.requests_per_second` field; document the local-midnight
    reset correctly.

- **v1.20.0** (3 July 2026) -- Unify batch output on the synchronous shape.
    Batch finalization (`check_batches.py`) and repair
    (`repair_extractions.py`) now write the same `{stem}_output.json` shape
    as synchronous extraction -- `records` (with `custom_id`, `chunk_index`,
    `chunk_range`, and a nested `response` body) plus `_chronominer_metadata`
    -- via a shared assembler (`modules/extract/batch_output.py`); batch
    provenance (provider, batch ids, parts, completion counts, tracking)
    is folded into the metadata under `batch_tracking`. The legacy
    `{stem}_final_output.json` shape is no longer written; files already on
    disk stay readable through the fallbacks in the converters and resume
    (no migration). When all batches are terminal but some failed, a
    partial unified output is now written from the completed batches
    (stamped `partial` with `failed_chunks`), temp files are retained for
    repair, and failed requests are folded into metadata instead of
    appearing as records. `detect_extraction_status` therefore covers
    batch outputs and now also recognizes page-based custom_ids.
    Downstream consumers in WhatForDinner were updated first,
    backward-compatibly, to read both shapes.

- **v1.19.0** (2 July 2026) -- Line-range readjustment overhaul closing the
    data-loss and resume defects found in a full review of the semantic
    boundary workflow. The model now sees a `<<<CURRENT_CHUNK_START>>>`
    sentinel marking the boundary it is judging, plus the schema name as the
    semantic unit type, and is asked for the boundary nearest the sentinel.
    No-content verification can no longer delete ranges the model reported
    content in (e.g. via `boundary_already_on_target`), requires the
    certainty threshold before deleting, and scans long ranges gaplessly.
    Marker matching is bounded (no whole-document fallback), enforces
    `min_substring_length`, resolves ambiguous matches to the candidate
    nearest the original start, and rejects matches that would invert a
    range; single-range lists are sanitized too. Gap enforcement no longer
    re-absorbs deleted no-content ranges. Context windows grow geometrically
    so the configured retry budgets are actually reachable
    (`scan_range_multiplier` is deprecated and ignored). Sliced adjustment
    runs (`--first-n-chunks`/`--last-n-chunks`) keep absolute range indices,
    no longer truncate the line-ranges file, and are never marked complete;
    adjustment temp JSONLs carry header version 2, record the post-write
    ranges fingerprint, and the completed-skip check now detects regenerated
    ranges files and prompt changes. Failed-marker retry guidance moved to
    the user message so Anthropic prompt caching stays effective, and
    token-based range generation counts newline tokens. Adds offline
    regression suites and opt-in live API tests
    (`CHRONOMINER_LIVE_TESTS=1`, `live` pytest marker).

- **v1.18.0** (2 July 2026) -- Hardening release closing the extraction-integrity
    defects found in a full production audit. Text chunking no longer strips
    newlines before joining, so LLM inputs keep their line structure instead of
    running words together at line boundaries; text-run outputs stamp
    `chunking_text_version: 2` so downstream analyses can distinguish chunking
    eras. Prompt templates resolve against the installation root rather than
    the working directory, so entry points run correctly from any location.
    Sliced text runs now write absolute chunk indices and custom_ids, temp
    JSONLs carry a format-version header, and resume refuses unversioned
    artifacts instead of silently mis-merging. Anthropic batch finalization
    serializes SDK messages safely; batch submissions are split into
    provider-limited `_part{n}` parts; remote batch files are deleted only
    after the final local output is durably written; the documented
    batch-ID recovery artifact is now actually written. Batch resume skips
    already-complete chunks, `repair_extractions` works provider-agnostically,
    and `check_batches` gains `--json` plus non-zero exits while batches
    remain pending. The primary CLI adopts the agent contract: exit codes
    0/1/2/130, a `--json` run summary, `--dry-run`, `--interactive`/
    `--non-interactive` overrides, and a non-TTY guard; CLI mode collects
    `.md` inputs and excludes the tool's own `*_output.txt` reports. Token
    state moves to a user-level directory (configurable via `state_dir`) with
    debounced writes; token counting uses the model's own encoding when known;
    the transient-error classifier consults structured status codes; batch
    finalization writes UTF-8 without ASCII escapes. Output-format unification
    with the batch shape is deliberately held: WhatForDinner consumes the
    batch `responses` shape.

- **v1.17.0** (28 June 2026) -- Ship scrubbed `*.example.yaml` config templates
    with conservative defaults and a real->example loader fallback, so a fresh
    clone runs with clear guidance instead of crashing on missing config. Six
    example files cover all config roles (`model_config`, `paths_config`,
    `chunking_and_context`, `concurrency_config`, `image_processing_config`,
    `api_keys_config`). `_load_yaml()` now tries the sibling `<stem>.example.yaml`
    when the real file is absent, logging one INFO line that tells the user to copy
    and customize it; if neither file is found a clear `FileNotFoundError` is raised.
    The optional `image_processing_config` and `api_keys_config` loaders gain the
    same fallback while remaining non-raising. `.gitignore` is tightened to
    `/config/*` + `!/config/*.example.yaml` so examples are tracked and real files
    stay private.

- **v1.16.0** (28 June 2026) -- Added an optional `config/api_keys_config.yaml`
    that maps each LLM provider to the name of the environment variable holding
    its API key, so a key can be swapped between runs (for example
    `openai: OPENAI_API_KEY_2`) by editing one file instead of changing the
    environment. The mapping is fully backward-compatible: a missing file or any
    omitted provider falls back to the existing default env var name. The remap
    is applied uniformly through a single resolver, so it reaches synchronous
    extraction, the provider-availability probes, the repair utility, and the
    OpenAI, Anthropic, and Google batch backends alike.

- **v1.15.0** (24 June 2026) -- Extended chunk-level token-limit enforcement to
    the line-range readjustment workflow, so the daily budget is applied
    consistently across both of ChronoMiner's LLM workflows. Readjustment now
    reserves the budget before each range (which may make several LLM calls),
    waits for the daily reset when exhausted, and resumes the still-pending
    ranges via the existing temp-JSONL resume; a budget-cancelled run leaves the
    line-ranges file untouched so it can be continued. File concurrency is
    clamped to one when the limit is enabled (matching extraction), so
    concurrent files cannot collectively overshoot. Token accounting was already
    shared; this closes the gap where a single file's ranges could overshoot the
    budget. All 1001 tests pass.

- **v1.14.0** (24 June 2026) -- The daily token limit is now enforced at the
    chunk/page level, not just between files. When the limit is enabled, the
    synchronous strategy reserves an estimated cost (a tiktoken input count for
    text, a self-calibrating rolling EWMA for images) before each unit, so
    concurrent workers cannot collectively overshoot; once the budget is
    exhausted mid-file it drains in-flight work, waits for the daily reset, and
    re-passes over the still-pending units using the existing temp-JSONL resume
    record. Configured concurrency and per-task delay are unchanged when budget
    is plentiful. Batch mode stays exempt (it is pre-priced and submitted whole).
    Two optional `daily_token_limit` settings tune the estimate
    (`chunk_estimate_seed`, `estimate_smoothing`). All 999 tests pass.

- **v1.13.0** (21 June 2026) -- Adopted the google-genai 2.x SDK major.
    Relaxed the runtime pin from `google-genai>=1.75,<2` to `google-genai>=2`
    and refreshed the lockfile (`google-genai` 1.75.0 -> 2.9.0;
    `langchain-google-genai` unchanged and compatible). The Google batch backend
    imports clean, the tree type-checks under mypy 2.1.0, and all 991 tests pass.
    Live Google batch API calls are not exercised by the test suite; validate a
    real Google run before relying on it.

- **v1.12.0** (21 June 2026) -- Adopted mypy 2.x for static type checking.
    Relaxed the dev pin from `mypy>=1.20,<2` to `mypy>=2.1` and refreshed the
    lockfile. Added one type annotation (`schema_handlers_registry` in
    `modules/extract/schema_handlers.py`) that mypy 2.x's stricter
    var-annotation check requires; the tree type-checks clean under mypy 2.1.0
    and all 991 tests pass. No runtime behavior changed.

- **v1.11.0** (20 June 2026) -- Removed the unused show_numbers parameter from
    UserInterface.select_option in modules/ui/core.py (dead since the modular
    prompts system never consumed it). Consolidated three within-module duplication
    clones behind new private helpers without altering behavior:
    modules/batch/backends/openai_backend.py now shares service-tier and
    reasoning-control wiring across the text and image Responses body builders via
    _apply_service_tier and _apply_reasoning; modules/images/llm_preprocess.py
    routes the low-detail and Anthropic high-detail branches of resize_for_detail
    through a single_cap_longest_side helper; and main/cli_args.py extracts the
    repeated output-directory and exclude-pattern filtering from get_files_from_path
    into _passes_dir_filters, called from both the visual and text-mode discovery
    loops.

- **v1.10.0** (20 June 2026) -- Refreshed runtime and dev dependencies under a
    conservative, majors-gated policy. No dependencies were removed (deptry reported
    no unused or missing imports) and none were added. Within-major upgrades raised
    anthropic 0.105.2 to 0.111.0, langchain-anthropic 1.4.4 to 1.4.6, langchain-core
    1.4.1 to 1.4.8, langchain-google-genai 4.2.4 to 4.2.5, langchain-openai 1.2.2 to
    1.3.2, openai 2.41.0 to 2.43.0, pytest 9.0.3 to 9.1.1, and ruff 0.15.16 to
    0.15.18, with matching >= floors lifted in pyproject. Two major bumps were held:
    google-genai stays on 1.75.0 (2.9.0 available) and mypy stays on 1.20.2 (2.1.0
    available), each already at the latest release within its current major and
    pinned below the next major.

- **v1.9.1** (10 June 2026) -- Retry Cloudflare 5xx edge errors. The
    transient-error check enumerated only 500/502/503, so Cloudflare edge codes in
    front of provider APIs (520-526, observed as HTTP 520 from api.openai.com in
    production) failed pages immediately without a single retry, despite the response
    body declaring itself retryable. Error classification is extracted into
    `classify_transient_error`, which now treats any standalone 5xx status code as a
    transient server error and additionally honors self-declared `'retryable': true`
    markers in error bodies. Affected pages were correctly recorded in `failed_chunks`
    and remain recoverable via `--resume`; with this fix they retry with exponential
    backoff instead of failing.

- **v1.9.0** (10 June 2026) -- Streaming visual pipeline. PDF pages are now
    rendered, preprocessed, and base64-encoded one at a time through a bounded
    producer-consumer queue instead of loading every page into memory up front; peak
    memory is one full-resolution page plus a small payload buffer, independent of
    document length (a 2,000-page guide previously required ~40 GB and crashed; it
    now runs in well under 1 GB). Preprocessing happens fully in memory
    (`ImageProcessor.process_pil`), eliminating the per-page PNG/JPEG disk
    round-trip. The resume skip-set and `--page-range`/`--first-n-chunks` slices are
    resolved before rendering, so completed and out-of-slice pages are never
    rendered; page indices in records are now absolute page numbers, and pages that
    fail to render surface as failed chunks (re-queued on resume) instead of silently
    shifting page numbering. Synchronous temp JSONL records no longer embed base64
    images in `request_metadata` (replaced by `image_omitted` placeholders),
    shrinking temp files ~40x; the new `main/slim_temp_jsonl.py` utility retrofits
    existing temp files, and skips files modified within the last 10 minutes to avoid
    touching active runs. Visual outputs now carry reproducibility provenance:
    source-file SHA-256, PyMuPDF/Pillow versions, and the effective preprocessing
    config at file level, plus per-page SHA-256, dimensions, byte size, and effective
    DPI. New `concurrency.extraction.max_concurrent_files` key (default 4, clamped to
    2 for visual runs) bounds file-level fan-out when the daily token limit is
    disabled. `max_pixels_per_page` default lowered from 150 MP to 24 MP (no quality
    effect: well above the 10.24 MP send cap). Fixed a context-image bug where the
    provider config section was resolved twice, silently applying default
    preprocessing (e.g. grayscale) instead of the configured values. Batch submission
    reuses the streaming producer, freeing raw pages per page during request building.

- **v1.8.0** (5 June 2026) -- Dependency cleanup and a limited deep-module
    refactor. Removed the unused `pip` runtime dependency and declared
    `charset-normalizer` explicitly, since it is imported directly in
    `modules/infra/chunking.py` but was previously present only transitively via
    `requests`. Refreshed the remaining dependencies to their latest compatible
    versions under a conservative policy that holds the two risky majors, `mypy` and
    `google-genai`, at their current major line, and added a `[tool.deptry]`
    configuration so dependency scans run clean. Internally, extracted cohesive
    helpers behind narrow interfaces: `_build_messages`,
    `_normalize_structured_schema`, and `_pack_result` in `openai_utils.py`;
    `_to_lc_messages` and `_extract_usage` in `langchain_provider.py`, which drops
    `ainvoke_with_structured_output` from cyclomatic grade F to D; and
    `_header_fields_match` in `jsonl.py`. No public interface or extraction behavior
    changed. Also completed a repository-wide ruff and mypy cleanup so both now run
    without warnings.

- **v1.7.1** (5 June 2026) -- Fix a resume bug that skipped partial visual
    (PDF/image) extractions as if complete, so their failed pages were never
    recovered. The early resume gate in `_process_visual_file` skipped any output
    whose record count met its `total_chunks`, ignoring the `partial` flag and
    `failed_chunks` list; because a partial run stamped `total_chunks` as its own
    success count, the file looked complete and the failed pages were never re-queued.
    The gate now skips only a self-declared full success (the new
    `metadata_indicates_complete` predicate in `modules/extract/resume.py`), so
    partial outputs fall through to the authoritative `detect_extraction_status` and
    resume. Separately, `total_chunks` is now stamped from the true unit count
    (`len(chunks)`) rather than the number of successful records, so the persisted
    metadata is no longer self-contradictory and stays correct even when a run ends
    partial or is cancelled. Existing partial outputs need no migration: the flags
    drive correct resume, and the recovery run re-stamps the true total.

- **v1.7.0** (5 June 2026) -- Lean extraction output. The final `_output.json` now
    carries only the response side of each record (`output_text` and
    `response_data`); the request side (`request_metadata`, whose `messages` embed
    the per-page base64 images) is stripped when the output is assembled, shrinking
    image-based guides roughly 150x. The complete API call is preserved in the
    sibling `_temp.jsonl` for reproducibility. Output is now written with
    `ensure_ascii=False`.

- **v1.6.9** (31 May 2026) -- Tier 3 latent-correctness and hygiene fixes.
    Correctness: the shared synchronous temp-file writer is now guarded by a lock so
    concurrent chunks can no longer interleave and corrupt JSONL lines; a run in
    which some chunks fail is marked partial and records the failed chunk indices in
    the output metadata instead of reporting a full success; line ranges read from a
    `_line_ranges.txt` are clamped to the file length before use, so an out-of-range
    end no longer raises an `IndexError`; the Google batch backend now wires the
    structured-output schema into Gemini's `response_schema` (capability-gated and
    sanitized for Gemini's restrictions) rather than ignoring it; and the OpenAI and
    Google batch backends delete their uploaded input and result files after download,
    so batch runs no longer leak remote storage. Hygiene: removed dead code; the
    token tracker resolves its state-file directory at first use rather than at
    import time; the configuration cache is guarded by a lock under the asyncio
    fan-out; and the interactive file picker constrains filename globbing to the
    configured input directory.

- **v1.6.8** (30 May 2026) -- Fix a resume data-loss path in synchronous
    extraction. The resume skip-set is derived from the existing `output.json`, but
    the final records were rebuilt solely from the temporary JSONL; when
    `retain_temporary_jsonl` is false the prior temp file is deleted, so a resumed
    run regenerated `output.json` from only the newly-processed chunks and dropped
    all previously-completed records. On resume, ChronoMiner now merges the records
    already saved in `output.json` with those rebuilt from the temp file, keyed by
    `custom_id` (newly-processed records win), so no prior record is lost regardless
    of the temp-retention setting. Force and non-resume runs are unaffected and still
    overwrite cleanly.

- **v1.6.7** (30 May 2026) -- Full-repository code review fixes. Correctness:
    `--input-type mixed` no longer silently falls back to text mode; `--page-range`
    now reports an actionable error instead of a raw traceback on malformed input;
    the processing summary reports the real failed-file count instead of always zero;
    synchronous extraction records now carry `chunk_index`, so `output.json` records
    are ordered rather than in completion order; list-form response records are no
    longer silently dropped during conversion; an Anthropic batch that ends with no
    request counts maps to `unknown` rather than a false `failed`; a Gemini batch
    candidate with no text content (e.g. a safety or length cutoff) is reported as a
    failure rather than a successful empty extraction. Hardening: narrowed several
    broad `except` blocks to surface previously-hidden read and parse failures,
    removed redundant double-traceback logging, and fixed a validation-order edge
    case in the response parser.

- **v1.6.6** (25 May 2026) -- Remove hard-coded Anthropic concurrency cap
    (`concurrency_limit = 1`); Anthropic now respects the configured
    `concurrency_limit` like all other providers. Users on lower Anthropic API tiers
    should set a conservative limit (e.g., 5) in `concurrency_config.yaml`; the
    existing retry logic handles transient 429s automatically.

- **v1.6.5** (21 May 2026) -- Demote LLM invocation error log from `error` (with
    full traceback) to `debug`; transient errors now only show the one-line retry
    warning from the processing strategy, not the upstream traceback.

- **v1.6.4** (21 May 2026) -- Widen transient error retry to cover 5xx server
    errors and upstream connection resets in addition to timeouts and 429 rate
    limits.

- **v1.6.3** (21 May 2026) -- Add early resume check for visual/PDF files: skip
    fully processed files before expensive PDF rendering; compare output metadata
    `total_chunks` against actual record count to correctly distinguish complete from
    partial files.

- **v1.6.2** (21 May 2026) -- Suppress misleading "will be overwritten" warning in
    the processing summary when resume mode is active (existing files are skipped,
    not overwritten).

- **v1.6.1** (21 May 2026) -- Fix Ctrl+C during daily token limit wait raising
    unhandled `asyncio.CancelledError` on Python 3.13; retry timeout errors for all
    providers (previously only Anthropic 429s were retried); suppress verbose
    tracebacks on permanent chunk failures.

- **v1.6.0** (21 May 2026) -- Add `--page-range START-END` CLI argument and
    interactive prompt for processing a specific 1-based inclusive page range; fix
    `UserInterface.confirm()` missing `allow_back` parameter that caused a TypeError
    on the context image prompt.

- **v1.5.0** (21 May 2026) -- Register `MichelinGuidesLight` schema in handler
    registry; tie context image prompt to text context selection (skip when context
    mode is "none"); rewrite award field descriptions in `michelin_guides_light.json`
    to reference context image symbol categories.

- **v1.4.0** (20 May 2026) -- Add `--output-mode {flat,mirror}` CLI flag: mirror
    mode replicates the input directory hierarchy under the output root, preserving
    edition/page structure for downstream consumers.

- **v1.3.2** (20 May 2026) -- Add `MichelinGuidesLight` schema (v3.3-light):
    minimal variant dropping amenities, opening, telephone, and three awards fields;
    update full `MichelinGuides` schema with `bib_hotel` field, `cuisine.styles`
    enum, and revised descriptions.

- **v1.3.1** (19 May 2026) -- Dependency refresh from environment-wide CVE audit:
    bump `langchain-core` 1.3.2 -> 1.4.0 (RCE on deserialization); `langsmith`
    0.7.36 -> 0.8.5 (unsafe deserialization; full fix to 1.0.x deferred pending
    upstream constraint relaxation); `pip` 26.0.1 -> 26.1.1 (polyglot tar/ZIP
    confusion); `urllib3` 2.6.3 -> 2.7.0 (audit-surface consolidation). Relax the
    `pip` floor from `==26.0.1` to `>=26.1` so the security update can be applied.

- **v1.3.0** (5 May 2026) -- Added context-image injection for visual extraction.

- **v1.2.1** (5 May 2026) -- Resolved all remaining ruff lint violations.

- **v1.2.0** (4 May 2026) -- Added ruff as a dev dependency with per-file E402
    ignores, applied ruff auto-fixes and formatting across the codebase, and updated
    the README.

- **v1.1.1** (4 May 2026) -- Added the `InequalityBenchmarks` schema.

- **v1.1.0** (4 May 2026) -- Migrated dependency management to `pyproject.toml`
    and `uv`.

- **v1.0.0** (25 April 2026) -- Initial public release; squashed baseline.

## License

MIT License. Copyright (c) 2025 Paul Goetz. See
[LICENSE](LICENSE) for details.
