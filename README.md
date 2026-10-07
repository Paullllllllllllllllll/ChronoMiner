# ChronoMiner v3.6.2

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
LangChain-based architecture, schema-driven extraction with 13
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
| OpenAI | GPT-6 (Astra, Sol, Luna), GPT-5.6 (Sol, Terra, Luna), GPT-5.5, GPT-5.4, GPT-5.3, GPT-5.2, GPT-5.1, GPT-5, o-series, GPT-4.1, GPT-4o | `OPENAI_API_KEY` | Yes |
| Anthropic | Claude Opus 5.5/5/4.8/4.7/4.6/4.5/4.1/4, Fable 5, Sonnet 5/4.6/4.5/4, Haiku 4.5 | `ANTHROPIC_API_KEY` | Yes |
| Google | Gemini 3, 2.5, 2.0, 1.5 (Pro, Flash variants) | `GOOGLE_API_KEY` | Yes |
| OpenRouter | 100+ models via unified API | `OPENROUTER_API_KEY` | No |
| Custom | Any OpenAI-compatible endpoint | User-configured | No |

**Model capabilities** are detected automatically. Reasoning models
(GPT-6, GPT-5.x, o-series, Gemini 2.5+, Claude 4.x and 5.x) have temperature/top_p
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
python main/extract.py
```

The wizard guides you through schema selection, chunking strategy,
processing mode, and file selection. Press `b` to go back, `q` to
quit at any time.

**CLI mode:**

```bash
# Extract from a single text file
python main/extract.py --schema BibliographicEntries \
    --input data/file.txt

# Batch process an entire directory (50% cheaper)
python main/extract.py --schema BibliographicEntries \
    --input data/ --batch

# Extract from images or PDFs
python main/extract.py --schema HistoricalRecipesEntries \
    --input path/to/images/ --image-detail high
```

### Common Workflows

**Large-scale batch processing:**

```bash
# Submit
python main/extract.py --schema BibliographicEntries \
    --input data/ --batch
# Monitor (run periodically; auto-downloads on completion)
python main/check_batches.py
```

**Pre-generate and adjust semantic line ranges:**

```bash
# Generate token-based line ranges
python main/generate_line_ranges.py --input data/file.txt
# Adjust boundaries to semantic sections
python main/adjust_line_ranges.py --input data/file.txt \
    --schema BibliographicEntries
# Extract using adjusted ranges
python main/extract.py --schema BibliographicEntries \
    --input data/file.txt --chunking line_ranges
```

**Repair failed extractions:**

```bash
python main/repair.py --schema BibliographicEntries
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
--reasoning-effort LEVEL   none | low | medium | high | xhigh | max
--chunk-size N             Override tokens per chunk
--service-tier TIER        Override concurrency.extraction.service_tier:
                           auto | default | flex | priority
--context MODE_OR_PATH     auto | none | /path/to/context.txt
--context-image            Enable context image injection (see below)
--first-n-chunks N         Process only the first N chunks/pages
--last-n-chunks N          Process only the last N chunks/pages
--page-range START-END     Process a 1-based inclusive page range
--output-mode MODE         flat | mirror (replicate input hierarchy
                           under the output root)
--resume / --force         Skip vs overwrite existing output
--interactive / --non-interactive
                           Force the run mode, overriding the config-file mode
--dry-run                  Discover inputs, classify resume state, and report
                           planned actions with zero API calls or side effects
--json                     Emit a one-line JSON run summary on stdout
```

Run `python main/extract.py --help` for the full list.

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

Provider sections are `api_image_processing`, `anthropic_image_processing`,
`google_image_processing`, and `custom_image_processing`. Each accepts:

- `target_dpi`: a positive integer or `native`. Native detects the densest image
  covering at least half the page, including rotated, cropped, and layered scans.
  It renders the composite at the scan's density. Pages without a usable scan
  use `native_fallback_dpi` (300 by default).
- `payload_format`: `jpeg` (default) or lossless `png`. `max_image_bytes` limits
  base64 bytes (0 disables it). An oversized PNG falls back to JPEG with a warning;
  a still-oversized payload becomes a page error. Context images remain JPEG.
- Grayscale, transparency, JPEG quality, detail/resolution, and resize profiles.
  The new keys are read from the provider section. Old real configs retain the
  top-level numeric `target_dpi` fallback; top-level null counts as missing.

Fresh clones default to native JPEG at quality 95 for OpenAI and Anthropic.
OpenAI uses `llm_detail: original` and keeps color cues (`grayscale_conversion:
false`). Google uses 300 DPI, quality 95, and `media_resolution: high`; custom
uses 150 DPI, quality 85, `resize_profile: low`, a 768 px low cap, and a
`high_target_box` of `[512, 1024]`. Per-image guards are 50 MB, 10 MB, 20 MB,
and disabled respectively.

OpenAI `original` uses a 30,000-patch (32 px) limit and 65,535 px edge for GPT-5.6
and GPT-6; other original-detail models use 10,000 patches and 6,000 px.
Anthropic high-resolution models use 4,784 patches (28 px) and a 2,576 px edge;
standard models use 1,568 patches and a 1,568 px edge. Optional
`original_max_side_px`, `original_max_pixels`, and Anthropic `high_max_side_px`
only tighten model caps. Caps apply to numeric runs too.

Other detail levels, Google, and custom use the configured resize profile;
native mode warns once per run when a profile discards native resolution.
Sizing follows the detail actually sent, including an `original` downgrade
on a Chat Completions route. Native with an unbounded `none` profile is rejected.
Top-level `render_strategy` selects `direct` or `supersample` for numeric runs;
native always renders directly to the target with float zoom. Numeric direct
runs retain integer DPI rounding. The top-level `max_pixels_per_page` guard
(24 MP) applies only to numeric runs.

PDF pages are rendered, preprocessed, and encoded one at a time, keeping memory
bounded independently of document length. Output provenance records source,
render and sent density, source and sent dimensions, downscale reason, encoding,
model cap policy, and the SHA-256 of each payload. File-level image provenance
includes resolved settings and `image_settings_fingerprint`. Resume and crash
recovery reject changed settings with an error naming the differences; use
`--force` to start again. Legacy records resume with a warning. Visual batch
requests have a durable manifest written before submission, so recovery and
finalization retain both page and file provenance.

To restore the previous configuration, set `target_dpi: 300` and
`jpeg_quality: 95` in the API, Anthropic, and Google sections; keep custom at
150 DPI and quality 85. Set API `original_max_side_px: 6000`,
`original_max_pixels: 10240000`, Anthropic `high_max_side_px: 2576`, and the
previous detail/profile values (`original` / `original` for API, `auto` for
Anthropic, `high` / `auto` for Google, `high` / `low` for custom). Set
`payload_format: jpeg` and `max_image_bytes: 0` to retain the previous encoding
and disabled size guard. Registry patch caps still apply, so oversized numeric
pages may change.

### 5. Concurrency Configuration (`concurrency_config.yaml`)

```yaml
concurrency:
  extraction:
    concurrency_limit: 12
    max_concurrent_files: 1
    delay_between_tasks: 0.1
    timeouts:
      total: 900          # read-phase budget of a single attempt (seconds)
      chunk_timeout: auto # wall-clock ceiling per chunk across all retries
      connect: 10         # OpenAI-family only: TCP/TLS setup
      write: 30           # OpenAI-family only: request upload
      pool: 30            # OpenAI-family only: wait for a pooled connection
    retry:
      attempts: 8
      timeout_attempts: 3 # smaller budget for request-timeout failures
daily_token_limit:
  enabled: true
  daily_tokens: 9000000   # combined cap across tools (secondary guard)
  scope: pooled           # pooled = cap only calls in a defined pool; all = legacy
  per_key_pool_caps:      # per-(API key, pool) daily caps (primary gate)
    enabled: true
    openai:
      small: 9750000      # bare int: cap; model list from built-in defaults
      large:
        cap: 975000       # mapping form: custom cap and/or model prefixes
        # models: ["gpt-5", "o3"]
    # any provider can define its own named pools:
    # myhost:
    #   standard:
    #     cap: 5000000
    #     models: ["my-model"]
```

Controls concurrent task limits, exponential backoff retry, and
daily token budgets (automatic reset at 00:01 UTC, one minute after
OpenAI's 00:00 UTC free-tier reset).
`max_concurrent_files` caps how many files run at once when the
daily token limit is disabled (visual runs are clamped to 2).

The per-phase HTTP timeouts apply to the OpenAI-family clients only
(`openai`, `openrouter`, custom endpoints): `total` is the read budget of a
single attempt, while `connect`, `write` and `pool` bound connection setup,
request upload, and connection-pool waits (defaults 10/30/30 seconds). The
Anthropic and Google wrappers require a plain float and keep the scalar
`total`. `chunk_timeout` is a different kind of bound — a wall-clock ceiling
for one chunk across all retry attempts combined; `auto` (the default)
resolves to `total x timeout_attempts + 300` seconds of backoff headroom,
3000 s under the values above, and `"off"` (quoted) or `0` disables it.
`timeout_attempts` gives request timeouts their own, smaller retry budget,
since a timed-out request is billed server-side but reports no usage.

On Windows the kernel dead-peer detection that `TCP_USER_TIMEOUT` provides on
Linux is unavailable, so the read timeout and the `chunk_timeout` watchdog are
the only stall detectors. Keep `total` tight (120-300 seconds) unless you run
`service_tier: flex`, where long silent server-side waits are expected.

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
  `michelin_guides_light.json`,
  `historical_price_entries_schema.json`,
  `inequality_benchmarks_schema.json`

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
5. A recovery artifact (`*_batch_submission_debug.json`) is rewritten next to
   the temp file before each part's tracking record, so a crash between
   submission and the tracking write can never orphan a paid-for batch. It
   holds the submitted `batch_ids`, the `provider`, and a `batch_metadata` map
   (batch id to the handle's metadata) that lets recovery restore each batch's
   context, such as a Google inline batch's custom-id map
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
python main/adjust_line_ranges.py --input data/file.txt \
    --schema BibliographicEntries
# Resume after interruption
python main/adjust_line_ranges.py --input data/ \
    --schema BibliographicEntries --resume
```

The readjuster uses certainty-driven retry: low-confidence
boundaries are retried with broader context windows until the
threshold is met or retries are exhausted.

### Extraction Repair

Recover partial results from incomplete batch jobs:

```bash
python main/repair.py --schema BibliographicEntries
```

Discovers incomplete jobs, groups multi-part (`_part{n}`) submissions of the
same source into one repair unit, recovers batch IDs from the recovery
artifact, and retrieves available responses, including paid partial results
from expired or cancelled batches. Status checks route through the
provider-agnostic backend (not OpenAI-only); the repaired output merges with
any existing output and is marked `fully_completed` only when no batch failed
or is missing.

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
ledger (`token_ledger.json`, schema v2) guarded by an OS file lock, and
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

#### Per-Key-Pool Accounting and Caps

A "pool" is a named set of models that share one daily token allowance per
API key. Pools are defined per provider in `per_key_pool_caps` — each entry
gives a cap and, optionally, a model prefix list — and built-in defaults
mirroring OpenAI's complimentary daily token program apply when a provider
has no configured model lists, so zero-config installs keep working. Every
API call's usage is stamped with its provider, the NAME of the environment
variable that served it (key values are never stored or logged), and the
pool derived from the model name. The shared ledger records a per-(tool,
provider, key env, pool) breakdown alongside the per-tool totals, so you
can always tell how much of a daily allowance remains on any key you use.
Enforcement is two-tier: `per_key_pool_caps` gates each key's own pool (set
your own caps and pools, or disable the gate, to match your account's
terms), and `daily_tokens` remains a combined secondary guard. Under the
default `scope: pooled`, calls to local endpoints and non-OpenAI providers
(no pool) are counted but never blocked by OpenAI usage; `scope: all`
restores the legacy block-everything behavior. When a key's pool cap is
reached, the wait message names the exhausted key and reports the remaining
pool of any other keys visible in the ledger; if you operate multiple keys,
remapping the provider to another env var in `api_keys_config.yaml` is
picked up at the next chunk/file boundary without a restart. Usage that
predates the upgrade (or arrives from un-stamped paths) is kept under an
"unattributed" row and counts toward the combined total only; a v1 ledger
is adopted in place without losing the day's count.

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
+-- ui/            Interactive prompts and console UI

main/
+-- extract.py                  Primary entry point
+-- generate_line_ranges.py     Token-based chunking
+-- adjust_line_ranges.py       Semantic boundary optimization
+-- check_batches.py            Monitor and finalize batch jobs
+-- cancel_batches.py           Cancel non-terminal batch jobs
+-- repair.py                   Recover incomplete extractions
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
Failed chunks are logged. Use `repair.py` to recover
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

The suite contains about 1,200 tests (unit and integration) covering
all modules, providers, batch backends, and CLI parsers.

## Versioning

This project follows semantic versioning (`MAJOR.MINOR.PATCH`). The version in
`pyproject.toml` is the single source of truth; it is mirrored in the title
heading above and tagged in git as `vX.Y.Z`. The commit history was squashed to
a single baseline commit at v1.0.0 on 25 April 2026; version numbers before
v1.0.0 do not exist.

Release notes are in [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT License. Copyright (c) 2025 Paul Goetz. See
[LICENSE](LICENSE) for details.
