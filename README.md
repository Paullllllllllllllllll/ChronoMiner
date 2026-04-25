# ChronoMiner v1.0.0

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
- **Schema-based extraction** -- 12 built-in JSON schemas with
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
  automatic midnight reset
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

All Python dependencies are listed in `requirements.txt`.

## Installation

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoMiner.git
cd ChronoMiner

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

# For development and tests
pip install -r requirements-dev.txt
```

**Configure API keys:**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key_here"

# Linux/macOS
export OPENAI_API_KEY="your_key_here"
```

For persistent configuration, add to system environment variables or
shell profile. Edit `config/paths_config.yaml` to set input/output
directories for each schema.

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
--first-n-chunks N         Process only the first N chunks/pages
--last-n-chunks N          Process only the last N chunks/pages
--resume / --force         Skip vs overwrite existing output
```

Run `python main/process_text_files.py --help` for the full list.

## Configuration

ChronoMiner uses five YAML files in `config/`.

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
```

### 5. Concurrency Configuration (`concurrency_config.yaml`)

```yaml
concurrency:
  extraction:
    concurrency_limit: 20
    delay_between_tasks: 0.1
    retry:
      attempts: 150
daily_token_limit:
  enabled: true
  daily_tokens: 9000000
```

Controls concurrent task limits, exponential backoff retry, and
daily token budgets (automatic reset at local midnight).

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

### Custom Extraction Schemas

Place JSON schemas in `schemas/`. Twelve schemas are included:

- `bibliographic_schema.json`, `summary_schema.json`,
  `address_schema.json`, `military_record_schema.json`,
  `culinary_works_schema.json`, `culinary_persons.json`,
  `culinary_places.json`, `culinary_entities.json`,
  `cookbook_metadata_schema.json`,
  `historical_recipes_schema_production.json`,
  `michelin_guides.json`, `historical_price_entries_schema.json`

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
3. Metadata is saved to a temporary JSONL for tracking and repair
4. A debug artifact (`*_batch_submission_debug.json`) is saved for
   recovery

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
retrieves available responses, and regenerates outputs.

### Daily Token Budget

Enable in `concurrency_config.yaml` to cap daily API usage. Tracks
total tokens per call, resets at local midnight. Add
`.chronominer_token_state.json` to `.gitignore`.

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
files, and review `requirements.txt` for version mismatches. For
persistent issues, open a
[GitHub issue](https://github.com/Paullllllllllllllllll/ChronoMiner/issues)
with error details and relevant config sections.

## Contributing

Contributions are welcome. When reporting issues, include: a clear
description, steps to reproduce, expected vs. actual behavior, your
environment (OS, Python version), relevant config sections (remove
sensitive data), and log excerpts.

For code contributions: fork the repository, create a feature branch,
follow the existing code style, add tests, and submit a pull request.

## Development

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
python -m pytest -v
```

The suite contains 920+ tests (unit and integration) covering all
modules, providers, batch backends, and CLI parsers.

## Versioning

This project uses semantic versioning. The commit history was
squashed to a single baseline commit at v1.0.0 on 25 April 2026.
All prior development history was consolidated; version numbers
before v1.0.0 do not exist.

## License

MIT License. Copyright (c) 2025 Paul Goetz. See
[LICENSE](LICENSE) for details.
