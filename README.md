# ChronoMiner

A Python-based tool for researchers and archivists to extract structured data from large-scale historical and academic text files. ChronoMiner leverages multiple LLM providers through LangChain with schema-based extraction to transform unstructured text into well-organized, analyzable datasets in multiple formats (JSON, CSV, DOCX, TXT).

Designed to integrate with [ChronoTranscriber](https://github.com/Paullllllllllllllllll/ChronoTranscriber) and [ChronoDownloader](https://github.com/Paullllllllllllllllll/ChronoDownloader) for a complete document retrieval, transcription, and data extraction pipeline.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Providers and Models](#supported-providers-and-models)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [First-Time Setup](#first-time-setup)
  - [Your First Extraction](#your-first-extraction)
  - [Common Workflows](#common-workflows)
- [Configuration](#configuration)
- [Usage](#usage)
- [Batch Processing](#batch-processing)
- [Fine-Tuning Dataset Preparation](#fine-tuning-dataset-preparation)
- [Utilities](#utilities)
- [Architecture](#architecture)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Contributing](#contributing)
- [Development](#development)
- [License](#license)

## Overview

ChronoMiner enables researchers and archivists to extract structured data from historical text files at scale. The tool provides schema-based extraction with customizable templates, flexible execution modes, and comprehensive batch processing capabilities. Works well with both historical documents and modern academic texts.

### Execution Modes

ChronoMiner supports two execution modes:

- **Interactive Mode**: Guided experience with clear prompts at each step, numbered menus with validation, and navigation support (press 'b' to go back, 'q' to quit). Ideal for first-time users and exploring options.
- **CLI Mode**: Command-line arguments for fully automated workflows. No user interaction required, immediate start with status to console, exit codes for success/failure. Designed for automation and batch processing.

Daily token enforcement configured in `config/concurrency_config.yaml`. When enabled, ChronoMiner tracks usage across sessions and automatically resets at local midnight.

### Key Capabilities

- **Dual Execution Modes**: Interactive prompts or CLI automation
- **Flexible Text Chunking**: Token-based with automatic, manual, or pre-defined strategies
- **Schema-Based Extraction**: JSON schema-driven with customizable templates
- **Dual Processing Modes**: Synchronous (real-time) or batch (OpenAI, Anthropic, Google)
- **Context-Aware Processing**: File-specific, folder-specific, or general fallback context
- **Multi-Format Output**: JSON, CSV, DOCX, and TXT simultaneously
- **Fine-Tuning Dataset Prep**: txt-based correction workflow for OpenAI SFT datasets
- **Extensible Architecture**: Easy custom schema integration
- **Batch Management**: Full suite of tools to manage, monitor, and repair jobs
- **Semantic Boundary Detection**: LLM-powered chunk optimization with certainty validation
- **Token Cost Analytics**: Per-model spend calculation with CSV export
- **Daily Token Budgeting**: Configurable per-day limits with automatic reset

## Key Features

### Text Processing

- **Token-Based Chunking**: Accurate token counting using tiktoken
- **Chunking Strategies**:
  - Automatic: Fully automatic based on token limits
  - Automatic with manual adjustments: Interactive refinement
  - Pre-defined line ranges: Uses existing line range files
  - Adjust and use line ranges: AI-detected semantic boundaries
  - Per-file selection: Choose method per file (Interactive mode)
- **Encoding Detection**: Automatic (UTF-8, ISO-8859-1, Windows-1252, etc.)
- **Text Normalization**: Strips whitespace and normalizes text
- **Windows Long Path Support**: Extended-length path syntax for paths >260 characters

### Context Integration

Unified, hierarchical context resolution automatically selects the most specific context:

- **Automatic Resolution**: Context is always resolved; no flags or user configuration needed
- **File-Specific**: `{filename}_extract_context.txt` or `{filename}_adjust_context.txt` next to input
- **Folder-Specific**: `{foldername}_extract_context.txt` or `{foldername}_adjust_context.txt` in parent
- **General Fallback**: `context/extract_context.txt` or `context/adjust_context.txt`

Context resolved separately for extraction and line-range-readjustment tasks.

### Multi-Provider LLM Support

- **OpenAI**: GPT-5.2, GPT-5.1, GPT-5, GPT-4.1, GPT-4o, o3, o4-mini and variants
- **Anthropic**: Claude Opus 4.6/4.5, Sonnet 4.6/4.5, Haiku 4.5, 4.1, 4, 3.5
- **Google**: Gemini 3 Pro, Gemini 3 Flash Preview, Gemini 2.5 Pro/Flash, Gemini 2.0/1.5
- **OpenRouter**: 100+ models through unified API
- **LangChain Backend**: Unified interface with automatic capability guarding
- **Reasoning Support**: Cross-provider translation of `reasoning.effort` parameter

### API Integration

- **LangChain Backend**: Unified interface with automatic retry handling
- **Schema Handler Registry**: Prepares API requests with JSON schema and instructions
- **Processing Modes**:
  - Synchronous: Real-time processing (all providers)
  - Batch: Asynchronous with cost savings (OpenAI, Anthropic, Google)
- **Retry Logic**: LangChain handles exponential backoff automatically

### Output Generation

- **JSON Output** (Always): Complete structured dataset with metadata
- **CSV Output** (Optional): Schema-specific converters transform JSON to tabular format
- **DOCX Output** (Optional): Formatted Word documents with headings, tables, styling
- **TXT Output** (Optional): Human-readable plain text reports with structured formatting

### Batch Processing

- **Multi-Provider Support**: OpenAI, Anthropic, Google with unified interface
- **Provider-Agnostic Management**: Check status, download results, cancel jobs across providers
- **Smart Chunking**: Automatic request splitting with chunk size limits
- **Metadata Tracking**: custom_id and metadata for reliable reconstruction
- **Debug Artifacts**: Submission metadata for batch tracking and repair

## Supported Providers and Models

ChronoMiner supports multiple LLM providers through LangChain. Provider is automatically detected from model name.

### OpenAI

| Model Family | Models | Type | Notes |
|--------------|--------|------|-------|
| GPT-5.2 | gpt-5.2 | Reasoning | 400k context; reasoning.effort (none–xhigh) |
| GPT-5.1 | gpt-5.1, gpt-5.1-instant, gpt-5.1-thinking | Reasoning | 400k context; adaptive thinking |
| GPT-5 | gpt-5, gpt-5-mini, gpt-5-nano | Reasoning | 400k context |
| GPT-4.1 | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano | Standard | 1M token context window |
| GPT-4o | gpt-4o, gpt-4o-mini | Standard | 128k token context window (e.g., gpt-4o-2024-08-06) |
| o4-mini | o4-mini | Reasoning | Tool use optimized |
| o3 | o3, o3-mini, o3-pro | Reasoning | Deep reasoning |
| o1 | o1, o1-mini | Reasoning | Legacy reasoning |

Environment variable: `OPENAI_API_KEY`

### Anthropic

| Model Family | Models | Notes |
|--------------|--------|-------|
| Opus 4.6 | claude-opus-4-6 | Most intelligent; 1M token context (beta) |
| Sonnet 4.6 | claude-sonnet-4-6 | Frontier performance; 1M token context (beta) |
| Opus 4.5 | claude-opus-4-5-20251101 | Previous flagship |
| Opus 4.1 | claude-opus-4-1-20250805 | Previous flagship |
| Opus 4 | claude-opus-4-20250514 | Code and agents |
| Sonnet 4.5 | claude-sonnet-4-5-* | Most aligned |
| Sonnet 4 | claude-sonnet-4-20250514 | Balanced |
| Haiku 4.5 | claude-haiku-4-5-* | Fast, cost-effective |
| Sonnet 3.7 | claude-3-7-sonnet-20250219 | February 2025 |
| Sonnet 3.5 | claude-3-5-sonnet-20241022 | October 2024 |
| Haiku 3.5 | claude-3-5-haiku-20241022 | Lightweight |

Environment variable: `ANTHROPIC_API_KEY`

### Google

| Model Family | Models | Type | Notes |
|--------------|--------|------|-------|
| Gemini 3 Pro | gemini-3-pro-preview | Thinking | Input 1,048,576; output 65,536 tokens |
| Gemini 3 Flash Preview | gemini-3-flash-preview | Thinking | Input 1,048,576; output 65,536 tokens |
| Gemini 2.5 Pro | gemini-2.5-pro | Thinking | Input 1,048,576; output 65,536 tokens |
| Gemini 2.5 Flash | gemini-2.5-flash | Thinking | Input 1,048,576; output 65,536 tokens |
| Gemini 2.5 Flash-Lite | gemini-2.5-flash-lite | Standard | Input 1,048,576; output 65,536 tokens |
| Gemini 2.0 Flash | gemini-2.0-flash | Standard | Input 1,048,576; output 8,192 tokens |
| Gemini 1.5 Pro | gemini-1.5-pro | Standard | Legacy |
| Gemini 1.5 Flash | gemini-1.5-flash | Standard | Legacy |

Environment variable: `GOOGLE_API_KEY`

### OpenRouter

Access to models from multiple providers. Use format `{provider}/{model}` (the `openrouter/` prefix is optional):

- `anthropic/claude-sonnet-4-5` or `anthropic/claude-opus-4-5`
- `google/gemini-2.5-pro` or `google/gemini-2.5-flash`
- `deepseek/deepseek-r1` or `deepseek/deepseek-chat`
- `meta/llama-3.3-70b` or `meta/llama-3.1-405b`
- `mistral/mistral-large` or `mistral/mixtral-8x22b`

Environment variable: `OPENROUTER_API_KEY`

**OpenRouter Reasoning Support**: Automatically translates `reasoning.effort` to provider-specific formats (extended thinking for Claude, thinking configuration for Gemini 2.5+, reasoning enablement for DeepSeek R1).

### Model Capabilities

Automatic capability detection and parameter adjustment:

- **Reasoning Models** (GPT-5.2/5.1/5, o-series, Gemini 2.5+/3, Claude 4.x, DeepSeek R1): Temperature/top_p disabled; reasoning effort configurable
- **Standard Models** (GPT-4.1, GPT-4o, Gemini 2.0, Llama, Mistral): Full sampler control
- **Structured Outputs**: Supported where the underlying model/provider supports schema-constrained output (capability-guarded)
- **Batch Processing**: OpenAI (50% savings), Anthropic, Google (provider-specific pricing)
- **Cross-Provider Reasoning**: `reasoning.effort` automatically translated per provider

### Structured Output Limitations by Provider

**OpenAI**: Strict schema validation with documented JSON Schema constraints (e.g., up to 10 nesting levels and 5,000 total object properties across the schema; all fields required; `additionalProperties: false` on objects). See [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs).

**Anthropic**: Structured Outputs support (JSON outputs and/or strict tool use) with provider-defined JSON Schema subset and validation behavior. See [Claude Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

**Google Gemini**: JSON Schema subset support; the API may reject very large or deeply nested schemas. If you encounter errors, simplify schemas (shorter property names, reduced nesting, fewer constraints). See [Gemini Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output).

**Provider Compatibility Matrix**:

| Schema | OpenAI | Anthropic | Google Gemini |
|--------|--------|-----------|--------------|
| BibliographicEntries | Supported | Supported | May be rejected (too nested) |
| StructuredSummaries | Supported | Supported | Supported |
| HistoricalAddressBookEntries | Supported | Supported | Untested |
| Simple custom schemas | Supported | Supported | Supported |
| Deeply nested schemas (>10 levels) | Not supported | May be rejected | May be rejected |

## System Requirements

### Software Dependencies

- **Python**: 3.12 or higher
- **LLM Provider API Key**: At least one required:
  - OpenAI: `OPENAI_API_KEY=your_key_here`
  - Anthropic: `ANTHROPIC_API_KEY=your_key_here`
  - Google: `GOOGLE_API_KEY=your_key_here`
  - OpenRouter: `OPENROUTER_API_KEY=your_key_here`

For OpenAI, ensure account has Responses API and Batch API access.

### Python Packages

All dependencies in `requirements.txt`. Key packages:

- LangChain: `langchain==1.1.0`
- Providers: `langchain-openai==1.1.0`, `langchain-anthropic==1.2.0`, `langchain-google-genai==3.2.0`
- APIs: `openai==2.8.1`, `anthropic==0.75.0`
- Processing: `tiktoken==0.12.0`, `pandas==2.3.3`, `python-docx==1.2.0`
- Data: `pydantic==2.12.5`, `PyYAML==6.0.3`, `tqdm==4.67.1`

## Installation

### Clone the Repository

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoMiner.git
cd ChronoMiner
```

### Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Keys

Set API keys for your chosen provider(s):

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_openai_key_here"
$env:ANTHROPIC_API_KEY="your_anthropic_key_here"
$env:GOOGLE_API_KEY="your_google_key_here"
$env:OPENROUTER_API_KEY="your_openrouter_key_here"

# Linux/macOS
export OPENAI_API_KEY=your_openai_key_here
export ANTHROPIC_API_KEY=your_anthropic_key_here
export GOOGLE_API_KEY=your_google_key_here
export OPENROUTER_API_KEY=your_openrouter_key_here
```

Only set the key for your chosen provider. Provider is auto-detected from model name in `model_config.yaml`.

For persistent configuration, add to system environment variables or shell profile.

### Configure File Paths

Edit `config/paths_config.yaml` to specify input/output directories for each schema.

## Quick Start

### First-Time Setup

**Step 1: Install Dependencies**

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoMiner.git
cd ChronoMiner

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

**Step 2: Set Up API Key**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key_here"

# Linux/macOS
export OPENAI_API_KEY="your_key_here"
```

**Step 3: Configure Model (optional)**

Edit `config/model_config.yaml`:

```yaml
transcription_model:
  provider: openai  # Options: openai, anthropic, google, openrouter
  name: gpt-5.1
  reasoning:
    effort: medium  # Options: low, medium, high, none (GPT-5.1 only)
```

**Step 4: Configure Paths (optional)**

Edit `config/paths_config.yaml`:

```yaml
schemas_paths:
  BibliographicEntries:
    input: "C:/path/to/input"
    output: "C:/path/to/output"
    csv_output: true
    docx_output: true
    txt_output: true
```

### Your First Extraction

**Interactive Mode (Recommended)**

```bash
python main/process_text_files.py
```

The interface guides you through:
1. Schema selection
2. Chunking strategy
3. Processing mode (synchronous or batch)
4. File selection (single, selected, or all)
5. Review and confirmation

Context is resolved automatically. Press 'b' to go back, 'q' to quit.

**CLI Mode (Automation)**

```bash
# Process single file with batch mode
python main/process_text_files.py --schema BibliographicEntries --input data/file.txt --batch

# Process entire directory with custom chunking
python main/process_text_files.py --schema CulinaryWorksEntries --input data/ --batch --chunking line_ranges

# View all options
python main/process_text_files.py --help
```

**Daily Token Tracking**

When `daily_token_limit.enabled` is `true`, ChronoMiner:
- Tracks every API call
- Persists usage to `.chronominer_token_state.json`
- Resets at local midnight
- Displays current usage at startup/completion
- Pauses when limit reached (resumes after reset or Ctrl+C to cancel)

### Common Workflows

**Workflow 1: Single File Extraction**

```bash
python main/process_text_files.py --schema BibliographicEntries --input data/file.txt
```

**Workflow 2: Batch Processing Large Dataset**

```bash
# Submit batch
python main/process_text_files.py --schema BibliographicEntries --input data/ --batch

# Check status
python main/check_batches.py

# Results automatically downloaded when complete
```

**Workflow 3: Pre-Generate and Adjust Line Ranges**

```bash
# Generate line ranges
python main/generate_line_ranges.py --input data/file.txt

# Adjust with semantic boundaries
python main/line_range_readjuster.py --path data/file.txt --schema BibliographicEntries

# Process with adjusted ranges
python main/process_text_files.py --schema BibliographicEntries --input data/file.txt --chunking line_ranges
```

## Configuration

ChronoMiner uses YAML configuration files in `config/` directory.

### 1. Paths Configuration (`paths_config.yaml`)

```yaml
general:
  interactive_mode: true  # true=interactive prompts, false=CLI mode
  retain_temporary_jsonl: true
  logs_dir: './logs'

schemas_paths:
  BibliographicEntries:
    input: "C:/path/to/input"
    output: "C:/path/to/output"
    csv_output: true
    docx_output: true
    txt_output: true
```

**Key Parameters**:
- `interactive_mode`: Controls operation mode
- `retain_temporary_jsonl`: Keep temporary files after completion
- Schema-specific paths with output format preferences

### 2. Model Configuration (`model_config.yaml`)

```yaml
transcription_model:
  provider: openai  # Optional: openai, anthropic, google, openrouter (auto-detected)
  name: gpt-5.1
  max_output_tokens: 128000
  
  # Reasoning controls (cross-provider)
  reasoning:
    effort: medium  # Options: low, medium, high, none (GPT-5.1 only)
  
  # GPT-5 only
  text:
    verbosity: medium  # Options: low, medium, high
  
  # Non-reasoning models only (automatically disabled for reasoning models)
  temperature: 0.01
  top_p: 1.0
  frequency_penalty: 0.01
  presence_penalty: 0.01
```

**Key Parameters**:
- `provider`: Explicit provider selection (auto-detected from model name if omitted)
- `name`: Model identifier
- `reasoning.effort`: Controls reasoning depth across all reasoning-capable providers
- Temperature/top_p: Automatically disabled for reasoning models

**Capability Guarding**: ChronoMiner automatically detects model capabilities and filters unsupported parameters.

**Cross-Provider Reasoning**: `reasoning.effort` works across providers with automatic translation (OpenAI: `reasoning_effort`, Anthropic: extended thinking budget tokens, Google: thinking level, DeepSeek: reasoning enabled flag).

### 3. Chunking Configuration (`chunking_and_context.yaml`)

```yaml
chunking:
  default_tokens_per_chunk: 7500
```

**Key Parameters**:
- `chunking.default_tokens_per_chunk`: Target tokens per chunk

### 4. Concurrency Configuration (`concurrency_config.yaml`)

```yaml
concurrency:
  extraction:
    concurrency_limit: 20
    delay_between_tasks: 0.1
    service_tier: flex
    retry:
      attempts: 150
      wait_min_seconds: 15
      wait_max_seconds: 120
      jitter_max_seconds: 5

daily_token_limit:
  enabled: false
  daily_tokens: 9000000
```

**Key Parameters**:
- `concurrency_limit`: Maximum concurrent tasks
- `service_tier`: OpenAI service tier
- Retry settings: Exponential backoff configuration
- `daily_token_limit.enabled`: Enable/disable enforcement
- `daily_token_limit.daily_tokens`: Daily budget (resets at midnight)

### 5. Line Range Adjustment Configuration (`chunking_and_context.yaml`)

```yaml
matching:
  normalize_whitespace: true
  case_sensitive: false
  normalize_diacritics: true
  allow_substring_match: true

retry:
  certainty_threshold: 70
  max_low_certainty_retries: 12
  max_context_expansion_attempts: 6
  delete_ranges_with_no_content: true
  scan_range_multiplier: 2
  max_gap_between_ranges: 2
```

**Key Parameters**:
- `certainty_threshold`: Minimum confidence score required
- `max_low_certainty_retries`: Retries before accepting low certainty
- `max_context_expansion_attempts`: Maximum window expansions
- `delete_ranges_with_no_content`: Enable verification scans for empty ranges

### Context Files

Context files provide domain-specific guidance to the LLM during extraction and line-range readjustment.
Resolution uses a 3-level hierarchy based on filename suffixes (most specific wins):

**Context Resolution Order**:

1. **File-specific**: `{input_stem}_extract_context.txt` (or `_adjust_context.txt`) next to the input file
2. **Folder-specific**: `{parent_folder}_extract_context.txt` (or `_adjust_context.txt`) next to the input's parent folder
3. **General fallback**: `context/extract_context.txt` (or `context/adjust_context.txt`) in the project root

**Directory Structure**:

```
context/
  extract_context.txt   # General fallback for extraction tasks
  adjust_context.txt    # General fallback for line-range readjustment
  legacy/               # Previous per-schema context files (reference only)
    extraction/
    line_ranges/
```

**Example Context File** (`context/extract_context.txt`):

```
The input text consists of bibliographic records describing historical culinary literature
(e.g., cookbooks, household manuals) dated from approximately 1470 to 1950.

EXTRACTION RULES:
- Take title and main author from first edition for full_title, short_title, main_author.
- Put every edition into edition_info as its own object.
- Keep multiple editions together under same root entry.
```

Previous per-schema context files (e.g., `BibliographicEntries.txt`) are preserved in `context/legacy/` for reference.
To use schema-specific context, place it as a folder-specific file next to the relevant input folder.

Keep context files under 4,000 characters for optimal performance.

## Usage

### Main Extraction Workflow

**Interactive Mode**:

```bash
python main/process_text_files.py
```

Guides through schema, chunking strategy, processing mode, and file selection. Context resolved automatically.

**CLI Mode**:

```bash
# Process single file with batch
python main/process_text_files.py --schema BibliographicEntries --input data/file.txt --batch

# Process directory with custom chunking
python main/process_text_files.py --schema CulinaryWorksEntries --input data/ --batch --chunking line_ranges

# View options
python main/process_text_files.py --help
```

**Available Arguments**:
- `--schema`: Schema name (required)
- `--input`: Input file or directory (required)
- `--chunking`: Chunking method (auto, auto-adjust, line_ranges, adjust-line-ranges)
- `--batch`: Use batch processing mode (default: synchronous)
- `--first-n-chunks N`: Process only the first N chunks of each file
- `--last-n-chunks N`: Process only the last N chunks of each file
- `--verbose`: Enable verbose output
- `--quiet`: Minimize output

`--first-n-chunks` and `--last-n-chunks` are mutually exclusive. In interactive mode the same option is available as a guided prompt.

### Line Range Generation

```bash
# Generate for single file
python main/generate_line_ranges.py --input data/file.txt

# Generate for directory
python main/generate_line_ranges.py --input data/ --tokens 7500

# Generate only the first 5 ranges
python main/generate_line_ranges.py --input data/ --first-n-chunks 5
```

Creates `{filename}_line_ranges.txt` files specifying exact line ranges.

### Line Range Adjustment

Optimize chunk boundaries using LLM-detected semantic sections:

```bash
# Adjust for specific file
python main/line_range_readjuster.py --path data/file.txt --schema BibliographicEntries

# Adjust with custom context window
python main/line_range_readjuster.py --path data/ --schema CulinaryWorksEntries --context-window 10
```

**Options**:
- `--path`: File or directory (required in CLI mode)
- `--schema`: Schema name (required with `--path`)
- `--context-window`: Surrounding lines to send
- `--prompt-path`: Override prompt template

**Certainty-Driven Workflow**:
- Model sets `contains_no_semantic_boundary` or `needs_more_context`
- Returns `semantic_marker` when boundary found
- Provides 0-100 `certainty` score
- Low-certainty responses automatically retried with broader context
- High-certainty markers validated before adjusting range
- High-certainty "no content" triggers verification scan; removes range if confirmed

### Batch Status Checking

```bash
# Check all batches
python main/check_batches.py

# Check specific schema only
python main/check_batches.py --schema BibliographicEntries
```

- Lists all batch jobs with status
- Automatically downloads and processes completed batches
- Generates all configured output formats
- Provides detailed summary

### Batch Cancellation

```bash
# Cancel with confirmation
python main/cancel_batches.py

# Cancel without confirmation
python main/cancel_batches.py --force
```

### Extraction Repair

```bash
# Repair specific schema
python main/repair_extractions.py --schema CulinaryWorksEntries

# Repair with verbose output
python main/repair_extractions.py --schema BibliographicEntries --verbose
```

- Discovers incomplete extraction jobs
- Recovers missing batch IDs from debug artifacts
- Retrieves responses from completed batches
- Regenerates final outputs with available data

### Output Files

Saved to configured output directories:

- `<filename>.json`: Complete structured dataset with metadata
- `<filename>.csv`: Tabular format (if enabled)
- `<filename>.docx`: Formatted Word document (if enabled)
- `<filename>.txt`: Plain text report (if enabled)
- `<filename>_temporary.jsonl`: Batch tracking (deleted after completion unless `retain_temporary_jsonl: true`)
- `<filename>_batch_submission_debug.json`: Batch metadata

## Batch Processing

ChronoMiner supports asynchronous batch processing across OpenAI, Anthropic, and Google. Batch processing enables cost-effective large-scale extraction with deferred results.

### Supported Providers

| Provider | Cost Savings | Typical Completion Time | Notes |
|----------|--------------|------------------------|-------|
| OpenAI | 50% reduction | Within 24 hours | Most mature batch API |
| Anthropic | Varies by tier | Hours to days | Message Batches API |
| Google | Varies by tier | Varies | Gemini Batch API |

### When to Use Batch Processing

**Ideal for**:
- Multiple large files or entire document collections
- Non-urgent extraction tasks with flexible deadlines
- Research projects prioritizing cost optimization
- Workflows where 24+ hour latency is acceptable

**Not ideal for**:
- Time-critical or interactive extractions
- Small single-file processing
- Development and testing (use synchronous)
- Immediate results required

### Batch Workflow

**Submit Batch Job**:

Interactive: Select "Batch" when prompted for processing mode
CLI: `python main/process_text_files.py --schema BibliographicEntries --input path/to/files --batch`

Script automatically:
1. Detects provider from configured model
2. Builds provider-specific batch request format
3. Submits batch job via appropriate API
4. Saves batch tracking metadata in temporary JSONL

**Monitor Batch Status**:

```bash
python main/check_batches.py
```

Scans temporary files, detects provider, displays unified status.

**Common Statuses**:
- `validating`: Validating batch request
- `in_progress`: Being processed
- `finalizing`: Preparing results
- `completed`: Results available
- `failed`: Batch failed (check logs)
- `expired`: Expired before completion
- `cancelled`: Was cancelled

**Retrieve Results**:

Once status shows `completed`, run `python main/check_batches.py`. Script automatically downloads batch results, processes responses, aggregates data, generates outputs (JSON, CSV, DOCX, TXT), and cleans up temporary files.

### Batch Management Tools

**Cancel Batches**:

```bash
python main/cancel_batches.py  # Interactive with confirmation
python main/cancel_batches.py --force  # No confirmation
```

Scans temporary files, identifies non-terminal batches, cancels using provider APIs.

**Repair Failed Batches**:

```bash
python main/repair_extractions.py
```

Interactive repair: scans for incomplete jobs, recovers batch IDs, retrieves available responses, regenerates outputs, reports success/failure.

### Multi-Provider Batch Architecture

Unified `BatchBackend` interface abstracts provider differences:

- **Base Interface** (`modules/llm/batch/backends/base.py`): Common operations (submit, get_status, download_results, cancel)
- **Provider Backends**: OpenAI, Anthropic, Google implementations
- **Factory Pattern** (`modules/llm/batch/backends/factory.py`): Dynamically instantiates correct backend
- **Batch Management Scripts**: Use provider-agnostic interface

Ensures consistent user experience, easy addition of new providers, automatic provider detection, and unified error handling.

### Provider-Specific Considerations

**OpenAI**: Most mature batch API, 50% cost reduction, results within 24 hours, supports full schema complexity.

**Anthropic**: Message Batches API, schema complexity limits (max 8 `anyOf` branches), ChronoMiner automatically falls back for complex schemas, rate limits: 10,000 output tokens/min, concurrent request limits may require sequential processing.

**Google**: Gemini Batch API, schema nesting depth limits (reject very deep schemas like `BibliographicEntries`), best for flatter schemas (e.g., `StructuredSummaries`), variable completion times.

## Fine-Tuning Dataset Preparation

ChronoMiner includes a separate workflow to prepare OpenAI SFT datasets from manually-provided chunk inputs using txt-based correction.

**Artifacts**: Written under `fine_tuning/artifacts/`:
- `fine_tuning/artifacts/editable_txt/`: Editable files for research assistants
- `fine_tuning/artifacts/annotations_jsonl/`: Imported corrected annotations
- `fine_tuning/artifacts/datasets/<dataset_id>/`: `train.jsonl` and `val.jsonl`

### Workflow Steps

**1) Prepare Chunk Inputs**

Create text file with numbered chunks:

```text
=== chunk 1 ===
<paste chunk text here>
=== chunk 2 ===
<paste chunk text here>
```

**2) Create Editable Correction File**

Generate `_editable.txt` file (optionally prefilled by model):

```bash
.\.venv\Scripts\python.exe -m fine_tuning.cli create-editable --schema BibliographicEntries --chunks path\to\chunks.txt --model gpt-5-mini
```

For blank template (no model call): add `--blank`

**3) Edit JSON in txt File**

In each chunk section:
- Keep all markers unchanged
- Edit only JSON inside `--- OUTPUT_JSON_BEGIN ---` / `--- OUTPUT_JSON_END ---`
- Output must be valid JSON object
- Use `null` for missing values

**4) Import Corrected Annotations into JSONL**

```bash
.\.venv\Scripts\python.exe -m fine_tuning.cli import-annotations --schema BibliographicEntries --editable fine_tuning\artifacts\editable_txt\chunks_editable.txt --annotator-id RA1
```

**5) Build OpenAI SFT Dataset**

Create `train.jsonl` and `val.jsonl`:

```bash
.\.venv\Scripts\python.exe -m fine_tuning.cli build-sft --schema BibliographicEntries --annotations fine_tuning\artifacts\annotations_jsonl\chunks.jsonl --dataset-id my_dataset_v1 --val-ratio 0.1 --seed 0
```

Default system prompt built from `prompts/structured_output_prompt.txt` with schema injection.

## Utilities

### Token Cost Analysis

Inspects preserved `.jsonl` files and produces detailed cost estimates.

**When to Run**:
- Preserve temporary files: `retain_temporary_jsonl: true` in `paths_config.yaml`
- After processing completes
- To validate budgeting assumptions

**Execution Modes**:

```bash
# Interactive UI
python main/cost_analysis.py

# CLI Mode
python main/cost_analysis.py --save-csv --output path/to/report.csv --quiet
```

**Output Features**:
- Aggregated totals (uncached input, cached, output, reasoning tokens)
- Dual pricing (standard + 50% discount for batch/flex)
- Model normalization (date-stamped variants mapped to parent profiles)
- CSV export with per-file ledger and summary row

See [OpenAI Pricing](https://platform.openai.com/docs/pricing) for current rates.

### Daily Token Budgeting

Automatically track daily token usage and pause when budget is exhausted.

**Configuration** (`config/concurrency_config.yaml`):

```yaml
daily_token_limit:
  enabled: true
  daily_tokens: 9000000
```

**How It Works**:
- Tracks every API call
- Persists to `.chronominer_token_state.json`
- Resets at local midnight
- Displays current usage at startup/completion
- Pauses when limit reached (resumes after reset or Ctrl+C)

Non-batch runs process files sequentially when enforcement active.

## Architecture

ChronoMiner follows a modular architecture with clear separation of concerns.

### Directory Structure

```
ChronoMiner/
├── config/                    # Configuration files
│   ├── chunking_and_context.yaml
│   ├── concurrency_config.yaml
│   ├── model_config.yaml
│   └── paths_config.yaml
├── main/                      # CLI entry points
│   ├── cancel_batches.py
│   ├── check_batches.py
│   ├── generate_line_ranges.py
│   ├── line_range_readjuster.py
│   ├── process_text_files.py
│   └── repair_extractions.py
├── modules/                   # Core application modules
│   ├── cli/                  # CLI framework and argument parsing
│   ├── config/               # Configuration loading and validation
│   ├── core/                 # Core utilities, token tracking, workflow
│   ├── llm/                  # LLM interaction and batch processing
│   ├── operations/           # High-level operations
│   └── ui/                   # User interface and prompts
├── schemas/                   # JSON schemas for structured outputs
├── context/                   # Unified context directory
│   ├── extract_context.txt   # General extraction context fallback
│   ├── adjust_context.txt    # General line-range adjustment context fallback
│   └── legacy/               # Previous per-schema context files (reference)
├── developer_messages/        # Developer message templates
├── prompts/                   # System prompt templates
├── gimmicks/                  # LLM prompts for generating context files
├── LICENSE
├── README.md
└── requirements.txt
```

### Module Overview

- **modules/config/**: Configuration loading and validation with cached access via `get_config_loader()`
- **modules/core/**: Core utilities (text processing, JSON manipulation, context management, workflow helpers, centralized token tracking with daily limit enforcement)
- **modules/cli/**: CLI utilities (argument parsing, mode detection, `DualModeScript` framework)
- **modules/llm/**: LLM interaction layer (LangChain multi-provider support, model capability detection, multi-provider batch processing, prompt management, structured output parsing)
- **modules/operations/**: High-level operations (extraction, line ranges, cost analysis, repair workflows)
- **modules/ui/**: User interface components (interactive prompts, selection menus, status displays)

### Operations Layer

High-level operations in `modules/operations/` are separated from CLI entry points in `main/` for testability and maintainability. This design pattern allows operations to be reused, tested independently, and invoked programmatically.

## Frequently Asked Questions

### General Questions

**Q: Which AI provider should I choose?**

A: Depends on your needs:
- **OpenAI (gpt-5.1)**: Best balance, 50% batch discount
- **Anthropic (Claude Opus 4.5)**: Most intelligent, complex schemas
- **Google (Gemini 2.5 Flash)**: Fast, cost-effective, simpler schemas
- **OpenRouter**: Access 100+ models with single API key

Start with OpenAI gpt-5.1 with medium reasoning effort.

**Q: How much does extraction cost?**

A: With OpenAI gpt-5.1:
- Small file (50 KB, 5 chunks): ~$0.10-0.20
- Medium file (500 KB, 50 chunks): ~$1-2
- Large file (5 MB, 500 chunks): ~$10-20
- Batch processing: 50% discount

Use `python main/cost_analysis.py` to track actual spending.

**Q: Should I use batch or synchronous mode?**

A: Use batch for:
- Multiple large files
- Non-urgent tasks
- Cost priority (50% cheaper)
- Can wait 24 hours

Use synchronous for:
- Single small files
- Immediate results needed
- Development/testing

**Q: Can I process multiple files at once?**

A: Yes, select entire folder or specific files. In interactive mode, choose from single file, selected files (indices/ranges), or all files. In CLI mode, provide directory path.

### Configuration Questions

**Q: How do I switch providers?**

A: Edit `config/model_config.yaml`:

```yaml
transcription_model:
  provider: anthropic
  name: claude-opus-4-5-20251101
```

Set appropriate API key environment variable.

**Q: How do I control costs with daily token limits?**

A: Enable in `config/concurrency_config.yaml`:

```yaml
daily_token_limit:
  enabled: true
  daily_tokens: 9000000
```

Processing pauses when limit reached, resumes next day.

**Q: What's the difference between reasoning effort levels?**

A: For reasoning models:
- **Low**: Fastest, cheapest, straightforward data
- **Medium**: Balanced quality/cost (recommended)
- **High**: Best quality for complex extraction, slower/expensive

For most historical documents, low or medium is sufficient.

**Q: How do I add a custom schema?**

A: 
1. Create JSON schema in `schemas/`
2. Add extraction context in `context/extraction/{SchemaName}.txt`
3. Add line ranges context in `context/line_ranges/{SchemaName}.txt`
4. Register in `modules/operations/extraction/schema_handlers.py`
5. Configure paths in `paths_config.yaml`
6. Test with sample files

To generate context files, use the LLM prompts in `gimmicks/`:
- `extraction_context_prompt.txt`: Generate extraction context from sample text
- `line_ranges_context_prompt.txt`: Generate semantic boundary detection context

Feed sample text to an LLM with these prompts to auto-generate context files.

### Processing Questions

**Q: What chunking strategy should I use?**

A: 
- **Automatic**: Quick start, no preparation needed
- **Pre-defined line ranges**: Best for production, consistent results
- **Automatic with adjustments**: Refine boundaries interactively
- **Adjust and use line ranges**: AI-optimized semantic boundaries

For production workflows, generate and adjust line ranges first.

**Q: What if extraction misses important information?**

A: 
- Use line range adjuster to optimize chunk boundaries
- Add file-specific context guidance
- Increase `default_tokens_per_chunk` (if within model limits)
- Review and manually edit `_line_ranges.txt` files

**Q: How do I handle large files?**

A: Use batch processing with pre-defined line ranges. Generate ranges once, adjust with semantic boundaries, then submit as batch. Monitor with `check_batches.py`.

**Q: Can I use different contexts for different files?**

A: Yes, hierarchical context resolution automatically selects most specific:
- File-specific: `{filename}_extraction.txt` next to file
- Folder-specific: `{foldername}_extraction.txt` in parent
- Schema-specific: `context/extraction/{SchemaName}.txt`
- Global: `context/extraction/general.txt`

### Batch Processing Questions

**Q: How do I check batch status?**

A:

```bash
python main/check_batches.py
```

Shows status and automatically downloads completed results.

**Q: Can I cancel a batch job?**

A: Yes:

```bash
python main/cancel_batches.py
```

Note: Charged for processing before cancellation.

**Q: Where are batch results stored?**

A: In output directory specified in `paths_config.yaml` for the schema. Temporary tracking files kept until results downloaded.

**Q: What if batch processing fails?**

A: Check logs in `logs_dir`. Common causes:
- Provider doesn't support batch
- API key lacks batch access
- Network issues
- Schema too complex for provider (e.g., BibliographicEntries on Gemini)

Use `repair_extractions.py` to recover partial results.

### Technical Questions

**Q: What schemas work with which providers?**

A:
- **OpenAI**: All schemas, including deeply nested
- **Anthropic**: Most schemas, may have `anyOf` limitations
- **Google Gemini**: Simpler schemas only (avoid deeply nested like BibliographicEntries)

Test with sample before large-scale processing.

**Q: How do I integrate into existing pipelines?**

A: Use CLI mode:

```bash
# In your script
python main/process_text_files.py --schema YourSchema --input "$INPUT" --batch

# Check exit code
if [ $? -eq 0 ]; then
    echo "Extraction successful"
else
    echo "Extraction failed"
fi
```

**Q: Can I run multiple extraction jobs simultaneously?**

A: Yes, but be mindful of:
- API rate limits (configure in `concurrency_config.yaml`)
- Daily token budgets
- System memory

Each job runs independently.

**Q: I'm experiencing issues not covered here**

A: Check logs in configured `logs_dir`, validate configuration files, verify directory permissions, review dependencies, and ensure model/schema compatibility. For persistent issues, please open a GitHub issue with detailed error information and relevant configuration sections.

## Contributing

Contributions are welcome!

### Reporting Issues

Include:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version, package versions)
- Relevant config sections (remove sensitive info)
- Log excerpts

### Suggesting Features

Provide:
- Use case description
- Proposed solution
- Alternatives considered
- Impact assessment

### Code Contributions

1. Fork repository and create feature branch
2. Follow existing code style and architecture
3. Add tests for new functionality
4. Update documentation
5. Test with different schemas and chunking strategies
6. Submit pull request with clear description

### Development Guidelines

- **Modularity**: Keep functions focused, use centralized utilities from `modules/core/`
- **Configuration**: Use `get_config_loader()` for cached config access
- **Error Handling**: Use try-except with informative messages
- **Logging**: Use `setup_logger()` from `modules/core/logger.py`
- **Code Style**: Follow PEP8, use 4-space indentation
- **User Experience**: Clear prompts and feedback
- **Documentation**: Update docstrings and README

### Areas for Contribution

- Additional LLM providers via LangChain
- Enhanced chunking algorithms
- New output formats (XML, etc.)
- Testing (unit and integration tests)
- Documentation (tutorials, examples)
- Performance optimization
- Error recovery mechanisms

## Development

### Test Setup

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install development and test dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
python -m pytest -v
```

Run tests with coverage:

```bash
python -m pytest --cov=. --cov-report=term-missing --cov-report=html
```

### Recent Updates

For complete release history, see GitHub releases.

**Latest**: Multi-provider LangChain integration, batch processing across OpenAI/Anthropic/Google, daily token budgeting, enhanced context system, certainty-based line range adjustment.

## License

MIT License

Copyright (c) 2025 Paul Goetz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
