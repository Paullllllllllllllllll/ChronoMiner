# ChronoMiner

A Python-based tool designed for researchers and archivists to extract structured data from large-scale historical and academic text files. ChronoMiner leverages multiple LLM providers through LangChain with schema-based extraction to transform unstructured text into well-organized, analyzable datasets in multiple formats (JSON, CSV, DOCX, TXT).

Supported LLM providers include OpenAI (GPT-5, GPT-5.1, GPT-4.1, o3, o4-mini), Anthropic (Claude Opus 4.5, Sonnet 4.5, Haiku 4.5), Google (Gemini 3 Pro, Gemini 2.5), and OpenRouter for access to additional models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Models](#supported-models)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Fine-Tuning Workflow](#fine-tuning-workflow)
- [Workflow Deep Dive](#workflow-deep-dive)
- [Adding Custom Schemas](#adding-custom-schemas)
- [Batch Processing](#batch-processing)
- [Token Cost Analysis](#token-cost-analysis)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Performance and Best Practices](#performance-and-best-practices)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

ChronoMiner is designed for researchers and archivists who need to (cheaply and comfortably) extract structured data from historical text files. The tool provides schema-based extraction with customizable templates, flexible execution modes, and comprehensive batch processing capabilities. Also works well (or even better) with non-historical documents (academic papers, books, etc.).
Meant to be used in conjunction with [ChronoTranscriber](https://github.com/Paullllllllllllllllll/ChronoTranscriber) and [ChronoDownloader](https://github.com/Paullllllllllllllllll/ChronoDownloader) for a full historical document retrieval, transcription and data extraction pipeline.

### Execution Modes

ChronoMiner supports two execution modes to accommodate different workflows and user preferences while automatically respecting the configured daily token budget:

**Interactive Mode** provides a guided experience with clear prompts and helpful explanations at each step. Users select options from numbered menus, receive immediate validation with helpful error messages, and can navigate backward or quit at any prompt. This mode is ideal for first-time users, exploring options, or occasional processing tasks.

**CLI Mode** accepts command-line arguments for fully automated, scriptable workflows. All settings are provided via arguments, requiring no user interaction. Processing starts immediately, status messages go to console, and exit codes indicate success or failure. This mode is designed for power users, automation, batch processing, and integration with other tools.

Daily token enforcement is configured in `config/concurrency_config.yaml`. When `daily_token_limit.enabled` is `true`, ChronoMiner tracks usage across sessions, enforces the configured token cap, and automatically resets the counter at local midnight. Enforcement applies to both interactive and CLI runs.

The mode is determined automatically: if command-line arguments are provided, CLI mode activates; otherwise, Interactive mode starts (unless configured otherwise in `config/paths_config.yaml`).

### Key Capabilities

- Dual Execution Modes: Run interactively with guided prompts or use command-line arguments for automation and scripting
- Flexible Text Chunking: Token-based chunking with automatic, manual, or pre-defined line range strategies
- Schema-Based Extraction: JSON schema-driven data extraction with customizable templates
- Dual Processing Modes: Choose between synchronous (real-time) or batch processing (50% cost savings)
- Context-Aware Processing: Integrate domain-specific or file-specific context for improved accuracy
- Multi-Format Output: Generate JSON, CSV, DOCX, and TXT outputs simultaneously
- Fine-Tuning Dataset Prep: Prepare OpenAI SFT datasets via a txt-based per-chunk correction workflow (`fine_tuning/`)
- Extensible Architecture: Easily add custom schemas and handlers for your specific use cases
- Batch Management: Full suite of tools to manage, monitor, and repair batch jobs
- Semantic Boundary Detection: LLM-powered chunk boundary optimization for coherent document segments
- Token Cost Analytics: Inspect temporary `.jsonl` files, calculate per-model spend, and export cost breakdowns
- Daily Token Budgeting: Enforce configurable per-day token usage limits with automatic midnight reset and persistent tracking

## Features

### Text Processing

- Token-Based Chunking: Accurate token counting using tiktoken for the selected model
- Chunking Strategies:
  - Automatic: Fully automatic chunking based on token limits
  - Automatic with manual adjustments: Automatic chunking with opportunity to refine boundaries interactively
  - Pre-defined line ranges: Uses pre-generated line ranges from existing files
  - Adjust and use line ranges: Refines existing line ranges using AI-detected semantic boundaries
  - Per-file selection: Choose chunking method individually for each file during processing
- Encoding Detection: Automatically detects file encoding (UTF-8, ISO-8859-1, Windows-1252, etc.)
- Text Normalization: Strips extraneous whitespace and normalizes text
- Windows Long Path Support: Automatically handles paths exceeding 260 characters on Windows using extended-length path syntax, ensuring reliable file operations regardless of path length or directory depth

### Context Integration

- Basic Context (Always Included): Fundamental information about the input source, automatically loaded for every API request
- Additional Context (User-Selected): Detailed, domain-specific guidance loaded when user selects to use additional context
  - Default context: Uses schema-specific file from `additional_context/{SchemaName}.txt`
  - File-specific context: Uses `{filename}_context.txt` files located next to the input files
- Context Hierarchy: All applicable context levels are combined and injected into the system prompt

### Multi-Provider LLM Support

ChronoMiner uses LangChain to provide unified access to multiple LLM providers:

- OpenAI: GPT-5.1, GPT-5, GPT-4.1, GPT-4o, o3, o4-mini and variants
- Anthropic: Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 and earlier versions
- Google: Gemini 3 Pro, Gemini 2.5 Pro/Flash and earlier versions
- OpenRouter: Access to 100+ models through a unified API

The provider is automatically detected from the model name. Capability guarding ensures that unsupported parameters (like temperature for reasoning models) are automatically filtered.

### API Integration

- LangChain Backend: Unified interface for all supported LLM providers with automatic retry handling
- Schema Handler Registry: Uses handler registry to prepare API requests with appropriate JSON schema and extraction instructions
- Processing Modes:
  - Synchronous: Real-time processing with immediate results via LangChain (all providers)
  - Batch: 50% cost savings, results within 24 hours (OpenAI only)
- Retry Logic: LangChain handles retries with exponential backoff automatically; configurable via max_retries

### Output Generation

- JSON Output (Always Generated): Complete structured dataset with metadata
- CSV Output (Optional): Schema-specific converters transform JSON to tabular format
- DOCX Output (Optional): Formatted Word documents with proper headings, tables, and styling
- TXT Output (Optional): Human-readable plain text reports with structured formatting

### Batch Processing

- Scalable Submission: Submit large document sets as OpenAI Batch jobs
- Smart Chunking: Automatic request splitting with proper chunk size limits
- Metadata Tracking: Each request includes custom_id and metadata for reliable reconstruction
- Debug Artifacts: Submission metadata saved for batch tracking and repair operations

## Supported Models

ChronoMiner supports a wide range of LLM models through LangChain. The provider is automatically detected from the model name.

### OpenAI Models

| Model Family | Model Names | Type | Notes |
|--------------|-------------|------|-------|
| GPT-5.1 | `gpt-5.1`, `gpt-5.1-instant`, `gpt-5.1-thinking` | Reasoning | Latest, adaptive thinking |
| GPT-5 | `gpt-5`, `gpt-5-mini`, `gpt-5-nano` | Reasoning | 256K context |
| GPT-4.1 | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` | Standard | Workhorse model |
| GPT-4o | `gpt-4o`, `gpt-4o-mini` | Standard | Multimodal |
| o4-mini | `o4-mini` | Reasoning | Tool use optimized |
| o3 | `o3`, `o3-mini`, `o3-pro` | Reasoning | Deep reasoning |
| o1 | `o1`, `o1-mini` | Reasoning | Legacy reasoning |

### Anthropic Claude Models

| Model Family | Model Names | Notes |
|--------------|-------------|-------|
| Opus 4.5 | `claude-opus-4-5-20251101` | Most intelligent |
| Opus 4.1 | `claude-opus-4-1-20250805` | Previous flagship |
| Opus 4 | `claude-opus-4-20250514` | Code and agents |
| Sonnet 4.5 | `claude-sonnet-4-5-*` | Most aligned |
| Sonnet 4 | `claude-sonnet-4-20250514` | Balanced |
| Haiku 4.5 | `claude-haiku-4-5-*` | Fast, cost-effective |
| Sonnet 3.7 | `claude-3-7-sonnet-20250219` | February 2025 |
| Sonnet 3.5 | `claude-3-5-sonnet-20241022` | October 2024 |
| Haiku 3.5 | `claude-3-5-haiku-20241022` | Lightweight |

### Google Gemini Models

| Model Family | Model Names | Type | Notes |
|--------------|-------------|------|-------|
| Gemini 3 Pro | `gemini-3-pro-preview` | Thinking | Most powerful |
| Gemini 2.5 Pro | `gemini-2.5-pro` | Thinking | 1M context |
| Gemini 2.5 Flash | `gemini-2.5-flash` | Thinking | Best price-performance |
| Gemini 2.5 Flash-Lite | `gemini-2.5-flash-lite` | Standard | Ultra fast |
| Gemini 2.0 Flash | `gemini-2.0-flash` | Standard | Previous gen |
| Gemini 1.5 Pro | `gemini-1.5-pro` | Standard | 2M context |
| Gemini 1.5 Flash | `gemini-1.5-flash` | Standard | Legacy |

### OpenRouter Models

OpenRouter provides access to models from multiple providers through a unified API. Use the format `{provider}/{model}` (the `openrouter/` prefix is optional):

- `anthropic/claude-sonnet-4-5` or `anthropic/claude-opus-4-5`
- `google/gemini-2.5-pro` or `google/gemini-2.5-flash`
- `deepseek/deepseek-r1` or `deepseek/deepseek-chat`
- `meta/llama-3.3-70b` or `meta/llama-3.1-405b`
- `mistral/mistral-large` or `mistral/mixtral-8x22b`

OpenRouter Reasoning Support: When using reasoning-capable models through OpenRouter, ChronoMiner automatically translates the `reasoning.effort` configuration to the appropriate provider-specific format. This includes extended thinking for Claude models, thinking configuration for Gemini 2.5+ models, and reasoning enablement for DeepSeek R1 models.

### Model Capabilities

ChronoMiner automatically detects model capabilities and adjusts API parameters accordingly:

- Reasoning Models (GPT-5, o-series, Gemini 2.5+, Claude 4.x, DeepSeek R1): Temperature and top_p are disabled; reasoning effort is configurable
- Standard Models (GPT-4.1, GPT-4o, Gemini 2.0, Llama, Mistral): Full sampler control
- Structured Outputs: Supported by all models via LangChain (with provider-specific limitations)
- Batch Processing: OpenAI only (50% cost savings)
- Cross-Provider Reasoning: The `reasoning.effort` parameter is automatically translated for each provider

#### Reasoning Capability Matrix

| Provider | Models | Reasoning Support | Translation |
|----------|--------|:-----------------:|-------------|
| OpenAI | GPT-5, o-series | Yes | Direct `reasoning_effort` |
| Anthropic (OpenRouter) | Claude 4.x | Yes | Extended thinking `budget_tokens` |
| Google (OpenRouter) | Gemini 2.5+, 3.x | Yes | Thinking level configuration |
| DeepSeek (OpenRouter) | DeepSeek R1 | Yes | Reasoning `enabled` flag |
| Meta (OpenRouter) | Llama 3.x | No | N/A |
| Mistral (OpenRouter) | Mistral, Mixtral | No | N/A |

### Structured Output Limitations by Provider

Each LLM provider has different limitations for JSON Schema-based structured outputs. Understanding these limitations helps you design schemas that work across providers.

#### OpenAI

OpenAI has the most robust structured output support with strict schema validation. See [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs) for details.

| Limitation | Value |
|------------|-------|
| Maximum nesting depth | 5 levels |
| Maximum object properties | 100 total across all objects |
| Required fields | All fields must be marked `required` |
| Additional properties | `additionalProperties: false` required on all objects |
| Default values | Not supported |
| Unsupported keywords | `minLength`, `maxLength`, `minimum`, `maximum`, `pattern`, etc. |

#### Anthropic (Claude)

Claude supports structured outputs via tool/function calling with flexible schema handling. See [Claude Structured Outputs documentation](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) for details.

| Limitation | Value |
|------------|-------|
| Schema format | Standard JSON Schema with some limitations |
| Nesting depth | No strict limit documented; complex schemas generally work |
| Validation | Less strict than OpenAI; model follows schema guidance |
| Tool use pattern | Structured output achieved through tool definitions |

Claude is generally more permissive with complex, deeply nested schemas like `BibliographicEntries`.

#### Google (Gemini)

Gemini has stricter schema complexity limits that may reject deeply nested schemas. See [Gemini Structured Outputs documentation](https://ai.google.dev/gemini-api/docs/structured-output) for details.

| Limitation | Value |
|------------|-------|
| Schema subset | Not all JSON Schema features supported |
| Nesting depth | Limited; complex schemas may be rejected |
| Schema complexity | API rejects very large or deeply nested schemas |
| Supported types | `string`, `number`, `integer`, `boolean`, `object`, `array`, `null` |
| Descriptive properties | `title`, `description` supported for guidance |

**Known Issue**: The `BibliographicEntries` schema exceeds Gemini's nesting depth limit and will fail with error: `400 A schema in GenerationConfig in the request exceeds the maximum allowed nesting depth.`

**Workarounds for Gemini**:
- Use simpler, flatter schemas (e.g., `StructuredSummaries`)
- Shorten property names and reduce nesting levels
- Limit the number of constraints per property
- Consider using OpenAI or Claude for complex extraction tasks

#### Provider Compatibility Matrix

| Schema | OpenAI | Anthropic | Google Gemini |
|--------|--------|-----------|---------------|
| BibliographicEntries | ✓ | ✓ | ✗ (too nested) |
| StructuredSummaries | ✓ | ✓ | ✓ |
| HistoricalAddressBookEntries | ✓ | ✓ | Untested |
| Simple custom schemas | ✓ | ✓ | ✓ |
| Deeply nested schemas (>5 levels) | ✓ | ✓ | ✗ |

## System Requirements

### Software Dependencies

- Python: 3.12 or higher
- LLM Provider API Key: At least one of the following is required:
  - OpenAI: `OPENAI_API_KEY=your_key_here`
  - Anthropic: `ANTHROPIC_API_KEY=your_key_here`
  - Google: `GOOGLE_API_KEY=your_key_here`
  - OpenRouter: `OPENROUTER_API_KEY=your_key_here`

For OpenAI, ensure your account has access to the Responses API and Batch API.

### Python Packages

All Python dependencies are listed in `requirements.txt`. Key packages include:

- `langchain==1.1.0`: Multi-provider LLM framework
- `langchain-openai==1.1.0`: OpenAI provider integration
- `langchain-anthropic==1.2.0`: Anthropic Claude provider integration
- `langchain-google-genai==3.2.0`: Google Gemini provider integration
- `openai==2.8.1`: OpenAI SDK (required for batch processing)
- `anthropic==0.75.0`: Anthropic SDK
- `tiktoken==0.12.0`: Accurate token counting
- `pandas==2.3.3`: Data manipulation
- `pydantic==2.12.5`: Data validation
- `python-docx==1.2.0`: Document generation
- `PyYAML==6.0.3`: Configuration file parsing
- `tqdm==4.67.1`: Progress bars

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

### Configure LLM Provider API Keys

Set your API key(s) for the provider(s) you want to use:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_openai_key_here"
$env:ANTHROPIC_API_KEY="your_anthropic_key_here"
$env:GOOGLE_API_KEY="your_google_key_here"
$env:OPENROUTER_API_KEY="your_openrouter_key_here"

# Windows Command Prompt
set OPENAI_API_KEY=your_openai_key_here
set ANTHROPIC_API_KEY=your_anthropic_key_here
set GOOGLE_API_KEY=your_google_key_here
set OPENROUTER_API_KEY=your_openrouter_key_here

# Linux/macOS
export OPENAI_API_KEY=your_openai_key_here
export ANTHROPIC_API_KEY=your_anthropic_key_here
export GOOGLE_API_KEY=your_google_key_here
export OPENROUTER_API_KEY=your_openrouter_key_here
```

You only need to set the API key for the provider you want to use. The provider is automatically detected from the model name in `model_config.yaml`.

For persistent configuration, add the environment variables to your system settings or shell profile.

### Configure File Paths

Edit `config/paths_config.yaml` to specify your input and output directories for each schema.

## Configuration

ChronoMiner uses YAML configuration files located in the `config/` directory. Each file controls a specific aspect of the pipeline.

### 1. Paths Configuration (`paths_config.yaml`)

Defines input/output directories, operation mode, and output format preferences for each schema.

```yaml
general:
  interactive_mode: true  # Toggle between interactive prompts (true) or CLI mode (false)
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

Key Parameters:

- `interactive_mode`: Controls operation mode (true for interactive prompts, false for CLI mode)
- `retain_temporary_jsonl`: Keep temporary JSONL files after batch processing completes
- `logs_dir`: Directory for log files
- Schema-specific paths: Define input/output directories and output format preferences for each schema

### 2. Model Configuration (`model_config.yaml`)

Controls which model to use and its behavioral parameters. ChronoMiner supports multiple LLM providers through LangChain, with automatic provider detection based on model name.

```yaml
transcription_model:
  # Provider selection (optional): openai, anthropic, google, openrouter
  # If omitted, the provider is auto-detected from the model name.
  # provider: openai

  # OpenAI models
  name: gpt-5.1              # Options: gpt-5.1, gpt-5.1-instant, gpt-5, gpt-5-mini, gpt-5-nano
  # name: gpt-4.1            # Options: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini
  # name: o3                 # Options: o3, o3-mini, o3-pro, o4-mini
  
  # Anthropic Claude models
  # name: claude-opus-4-5-20251101    # Latest Opus
  # name: claude-sonnet-4-5           # Latest Sonnet
  # name: claude-haiku-4-5            # Latest Haiku (fast)
  
  # Google Gemini models
  # name: gemini-3-pro-preview        # Most powerful
  # name: gemini-2.5-pro              # Thinking model
  # name: gemini-2.5-flash            # Fast, cost-effective
  
  # OpenRouter (access any model via unified API)
  # name: anthropic/claude-sonnet-4-5
  # name: google/gemini-2.5-flash
  # name: deepseek/deepseek-r1
  
  max_output_tokens: 128000
  
  # Reasoning controls (cross-provider support)
  # Works with: OpenAI GPT-5/o-series, Anthropic Claude, Google Gemini 2.5+, DeepSeek R1
  # Each provider receives the appropriate translation:
  #   - OpenAI: reasoning_effort parameter
  #   - Anthropic: extended thinking with budget_tokens
  #   - Google: thinking_level configuration
  #   - DeepSeek: reasoning enabled flag
  reasoning:
    effort: medium  # Options: low, medium, high, none (GPT-5.1 only)
  
  # GPT-5 only
  text:
    verbosity: medium  # Options: low, medium, high
  
  # Non-reasoning models only (GPT-4o, GPT-4.1, Claude, Gemini non-thinking)
  # Note: These are automatically disabled for reasoning models
  temperature: 0.01
  top_p: 1.0
  frequency_penalty: 0.01
  presence_penalty: 0.01
```

Key Parameters:

- `provider`: Explicit provider selection (optional; auto-detected from model name if omitted)
- `name`: Model identifier (provider is auto-detected from the name)
- `max_output_tokens`: Maximum tokens the model can generate per request
- `reasoning.effort`: Controls reasoning depth across all reasoning-capable providers (see Cross-Provider Reasoning below)
- `text.verbosity`: Controls response verbosity for GPT-5 models
- `temperature`: Controls randomness (0.0-2.0) - automatically disabled for reasoning models
- `top_p`: Nucleus sampling probability (0.0-1.0) - automatically disabled for reasoning models
- `frequency_penalty`: Penalizes token repetition (-2.0 to 2.0)
- `presence_penalty`: Penalizes repeated topics (-2.0 to 2.0)

Capability Guarding: ChronoMiner automatically detects model capabilities and filters unsupported parameters. For example, reasoning models (o3, o4-mini, GPT-5) do not support temperature or top_p, so these parameters are automatically excluded from API requests.

Cross-Provider Reasoning: The `reasoning.effort` parameter works across all providers with automatic translation. For OpenAI models, it maps directly to `reasoning_effort`. For Anthropic Claude models accessed via OpenRouter, it translates to extended thinking budget tokens. For Google Gemini thinking models, it configures the thinking level. For DeepSeek R1 models, it enables or disables reasoning based on effort level.

### 3. Chunking Configuration (`chunking_config.yaml`)

Controls text chunking behavior.

```yaml
chunking:
  default_tokens_per_chunk: 7500
  model_name: gpt-4o  # For token counting
```

Key Parameters:

- `default_tokens_per_chunk`: Target number of tokens per chunk
- `model_name`: Model name for accurate token counting

### 4. Concurrency Configuration (`concurrency_config.yaml`)

Controls parallel processing, retry behavior, and daily token budgeting.

```yaml
concurrency:
  transcription:
    concurrency_limit: 100
    delay_between_tasks: 0.0
    service_tier: flex  # Options: auto, default, flex, priority
    retry:
      attempts: 5
      wait_min_seconds: 4
      wait_max_seconds: 60
      jitter_max_seconds: 1

daily_token_limit:
  enabled: true        # Toggle daily token enforcement
  daily_tokens: 9000000  # Maximum tokens allowed per calendar day
```

Key Parameters:

- `concurrency_limit`: Maximum number of concurrent tasks
- `delay_between_tasks`: Delay in seconds between starting tasks
- `service_tier`: OpenAI service tier for rate limiting and processing speed
- Retry settings: Exponential backoff configuration for transient API failures
- `daily_token_limit.enabled`: Enable/disable daily enforcement
- `daily_token_limit.daily_tokens`: Daily token budget; counter resets automatically at local midnight and persists across runs

### 5. Chunking & Context Configuration (`chunking_and_context.yaml`)

Controls matching, retry, and deletion behavior for the line range readjuster.

```yaml
retry:
  certainty_threshold: 70
  max_low_certainty_retries: 3
  max_context_expansion_attempts: 3
  delete_ranges_with_no_content: true
  scan_range_multiplier: 3
```

Key parameters:

- `certainty_threshold`: Minimum confidence score required before accepting a model response.
- `max_low_certainty_retries`: Retries allowed before keeping the original range when certainty stays low.
- `max_context_expansion_attempts`: Maximum window expansions when the model requests more context.
- `delete_ranges_with_no_content`: Enables verification scans that delete empty ranges on high-certainty no-content responses.
- `scan_range_multiplier`: Size multiplier for the upward/downward verification scan.

### Context Files

#### Basic Context (Required)

Basic context files are located in `basic_context/` and are automatically loaded and included in every API request. Each schema requires a corresponding basic context file named `{SchemaName}Entries.txt`.

If you need to draft a new basic context file with the help of an LLM, reuse the prompt template in `gimmicks/basic_context_prompt.txt`. Provide that prompt together with the relevant schema definition so the model can generate compliant guidance.

Basic context provides fundamental information about the input source:

- Brief description of the input text type and characteristics
- Language and formatting expectations
- Historical time period or geographic scope
- Natural semantic markers for text chunking

Example: `basic_context/BibliographicEntries.txt`

```
The input text consists of a snippet of bibliographic records describing historical culinary and household
literature (e.g., cookbooks, household manuals, foodstuff guides, food production regulations, etc.) dating
from approximately 1470 to 1950. The text may appear in different languages and use historical spelling
and diacritics. When extracting edition information, ensure that later or revised editions of the same work are
grouped together under the entry for the first (original) edition in the `edition_info` array. Natural semantic
markers for this type of text are the beginnings of full bibliographic entries.
```

#### Additional Context (Optional)

Additional context files are located in `additional_context/` and are only included when the user selects to use additional context. These files provide detailed, domain-specific guidance for extraction.

To script new additional context documents with an LLM, start from the example prompt in `gimmicks/additional_context_prompt.txt` and share it alongside the matching schema.

Example: `additional_context/BibliographicEntries.txt`

```
ADDITIONAL CONTEXT FOR BIBLIOGRAPHIC ENTRIES

When extracting data from historical culinary and household literature bibliographies, please consider:

1. Bibliographic conventions and evolution:
   - Pre-1800 bibliographies often follow Short Title Catalogue conventions
   - Modern bibliographic standards (e.g., Chicago, MLA) emerged in late 19th/early 20th century
   - Title pages may contain extensive subtitles describing contents and intended audience

2. Edition tracking and identification:
   - First editions are bibliographically distinct from reprints and revised editions
   - Edition statements may appear in various forms: "tweede druk," "2nd edition," "revised and enlarged"
```

#### File-Specific Context (Optional)

For input files requiring specific instructions, users can create `{filename}_context.txt` files in the same directory as the input file.

Example: `zurich_addressbook_1850_context.txt`

```
This address book contains entries from 1850-1870 in Zurich, Switzerland.
- Occupations are listed in Swiss German
- Addresses use old street names that no longer exist (refer to historical city maps)
- Some entries span multiple lines due to long professional titles
- Family businesses often list multiple family members at the same address
```

## Usage

### Main Extraction Workflow

The primary entry point is `main/process_text_files.py`, which provides a workflow for extracting structured data from text files.

#### Interactive Mode

Run the script without arguments:

```bash
python main/process_text_files.py
```

The interactive interface guides you through:

1. Select a Schema
2. Choose Chunking Strategy
3. Select Processing Mode (Synchronous or Batch)
4. Configure Additional Context (Optional)
5. Select Input Files
   - Process a single file
   - Process selected files from a folder (choose specific files via comma-separated indices or ranges)
   - Process all files in a folder
6. Review and Confirm

#### CLI Mode

Process files with command-line arguments:

```bash
# Process a single file with batch mode
python main/process_text_files.py --schema BibliographicEntries --input data/file.txt --batch

# Process entire directory with custom chunking (token limit enforcement still applies)
python main/process_text_files.py --schema CulinaryWorksEntries --input data/ --batch --chunking line_ranges

# View all options
python main/process_text_files.py --help
```

Available Arguments:

- `--schema`: Schema name (required)
- `--input`: Input file or directory path (required)
- `--chunking`: Chunking method (auto, auto-adjust, line_ranges, adjust-line-ranges, per-file)
- `--batch`: Use batch processing mode (default: synchronous)
- `--context`: Use additional context
- `--context-source`: Context source (default or file-specific)
- `--verbose`: Enable verbose output
- `--quiet`: Minimize output

#### Daily Token Tracking During Processing

- When `daily_token_limit.enabled` is `true`, ChronoMiner tracks every API call, persists usage to `.chronominer_token_state.json`, and resets counts at local midnight.
- Interactive and CLI workflows display current usage at startup and completion. When the limit is reached, the app pauses before processing the next file and resumes automatically after the reset (Ctrl+C cancels the wait).
- Non-batch runs process files sequentially while enforcement is active. Disable the limit or enable batch processing to restore fully concurrent execution.

### Line Range Generation

For production workflows, pre-generate line range files:

```bash
# Generate for single file
python main/generate_line_ranges.py --input data/file.txt

# Generate for directory
python main/generate_line_ranges.py --input data/ --tokens 7500
```

This creates `{filename}_line_ranges.txt` files specifying exact line ranges for each chunk.

### Line Range Adjustment

Optimize chunk boundaries using LLM-detected semantic sections with certainty-based validation:

```bash
# Adjust line ranges for specific file
python main/line_range_readjuster.py --path data/file.txt --schema BibliographicEntries

# Adjust with custom context window
python main/line_range_readjuster.py --path data/ --schema CulinaryWorksEntries --context-window 10
```

Options:

- `--dry-run`: Preview changes without modifying files
- `--context-window N`: Number of lines around boundaries to examine (default: 300)
- `--boundary-type`: Type of boundary to detect (section, paragraph, entry)

Readjustment now follows a certainty-driven workflow:

- **Model guidance**: The LLM sets `contains_no_semantic_boundary` or `needs_more_context`, returns a `semantic_marker` when it finds a boundary, and provides a 0–100 `certainty` score.
- **Low-certainty responses**: Automatically retried with broader context windows until the configured certainty threshold is met.
- **Boundary application**: High-certainty markers are validated against the source text before adjusting the range start.
- **Automatic range deletion**: High-certainty "no semantic content" responses trigger an up/down verification scan; if no relevant content is found, the range is removed entirely.

### Batch Status Checking

Monitor and process completed batch jobs:

```bash
# Check all batches
python main/check_batches.py

# Check specific schema only
python main/check_batches.py --schema BibliographicEntries
```

Features:

- Lists all batch jobs with status
- Automatically downloads and processes completed batches
- Generates all configured output formats
- Provides detailed summary of batch operations

### Batch Cancellation

Cancel all in-progress batches:

```bash
# Cancel with confirmation
python main/cancel_batches.py

# Cancel without confirmation
python main/cancel_batches.py --force
```

### Extraction Repair

Repair incomplete extractions:

```bash
# Repair specific schema
python main/repair_extractions.py --schema CulinaryWorksEntries

# Repair with verbose output
python main/repair_extractions.py --schema BibliographicEntries --verbose
```

Repair capabilities:

- Discovers incomplete extraction jobs
- Recovers missing batch IDs from debug artifacts
- Retrieves responses from completed batches
- Regenerates final outputs with all available data

### Output Files

Extraction outputs are saved to the configured output directories for each schema:

- `<filename>.json`: Complete structured dataset with metadata
- `<filename>.csv`: Tabular format (if enabled)
- `<filename>.docx`: Formatted Word document (if enabled)
- `<filename>.txt`: Plain text report (if enabled)
- `<filename>_temporary.jsonl`: Temporary batch tracking file (deleted after successful completion unless `retain_temporary_jsonl: true`)
- `<filename>_batch_submission_debug.json`: Batch metadata for tracking and repair

## Fine-Tuning Workflow

ChronoMiner includes a separate, eval-independent workflow to prepare OpenAI supervised fine-tuning (SFT) datasets from manually-provided chunk inputs. The workflow lives in `fine_tuning/` and uses a txt file for human corrections.

Artifacts are written under `fine_tuning/artifacts/` by default:
- `fine_tuning/artifacts/editable_txt/`: editable files for research assistants
- `fine_tuning/artifacts/annotations_jsonl/`: imported, machine-readable corrected annotations
- `fine_tuning/artifacts/datasets/<dataset_id>/`: `train.jsonl` and `val.jsonl`

### 1) Prepare chunk inputs

Create a text file containing numbered chunks (this is the source of truth for inputs):

```text
=== chunk 1 ===
<paste chunk text here>
=== chunk 2 ===
<paste chunk text here>
```

### 2) Create an editable correction file

Generate an `_editable.txt` file (optionally prefilled by the model):

```bash
.\.venv\Scripts\python.exe -m fine_tuning.cli create-editable --schema BibliographicEntries --chunks path\to\chunks.txt --model gpt-5-mini
```

If you want a blank template (no model call), add `--blank`:

```bash
.\.venv\Scripts\python.exe -m fine_tuning.cli create-editable --schema BibliographicEntries --chunks path\to\chunks.txt --blank
```

The editable file embeds the input text and an editable JSON block per chunk.

### 3) Edit the JSON in the txt file

In each chunk section:
- Keep all markers unchanged
- Edit only the JSON inside `--- OUTPUT_JSON_BEGIN ---` / `--- OUTPUT_JSON_END ---`
- Output must be valid JSON and must be a JSON object
- Use `null` for missing values

### 4) Import corrected annotations into JSONL

Convert the edited txt into canonical annotation JSONL:

```bash
.\.venv\Scripts\python.exe -m fine_tuning.cli import-annotations --schema BibliographicEntries --editable fine_tuning\artifacts\editable_txt\chunks_editable.txt --annotator-id RA1
```

### 5) Build an OpenAI SFT dataset (train/val JSONL)

Create `train.jsonl` and `val.jsonl` suitable for OpenAI fine-tuning:

```bash
.\.venv\Scripts\python.exe -m fine_tuning.cli build-sft --schema BibliographicEntries --annotations fine_tuning\artifacts\annotations_jsonl\chunks.jsonl --dataset-id my_dataset_v1 --val-ratio 0.1 --seed 0
```

By default, the system prompt used for dataset examples is built from `prompts/structured_output_prompt.txt` with schema injection.

## Workflow Deep Dive

Understanding ChronoMiner's internal workflow helps you optimize your extraction tasks and troubleshoot issues.

### Phase 1: Text Pre-Processing

#### Encoding Detection and Normalization
- Automatically detects file encoding (UTF-8, ISO-8859-1, Windows-1252, etc.)
- Normalizes text by stripping extraneous whitespace
- Logs encoding information for debugging

#### Token-Based Chunking
- Uses tiktoken to accurately count tokens for the selected model
- Splits text into chunks based on `default_tokens_per_chunk` setting
- Ensures chunks don't exceed model's context window

Chunking Strategies:

- Automatic: Fully automatic chunking based on token limits
- Automatic with Adjustments: Automatic chunking with opportunity to refine boundaries interactively
- Pre-defined Line Ranges: Uses pre-generated line ranges from existing files
- Adjust and Use Line Ranges: Refines existing line ranges using AI-detected semantic boundaries, then uses them for processing
- Per-file Selection: Choose chunking method individually for each file during processing

### Phase 2: Context Integration

#### Basic Context (Always Included)
Basic context files provide fundamental information about the input source and are automatically loaded for every API request.

#### Additional Context (User-Selected)
Additional context files provide detailed, domain-specific guidance and are only included when the user selects to use additional context.

The user can choose between:
- Default context: Uses schema-specific file from `additional_context/{SchemaName}.txt`
- File-specific context: Uses `{filename}_context.txt` files located next to the input files

#### Context Hierarchy and Integration
All applicable context levels are combined and injected into the system prompt via placeholders:

1. Basic context always inserted via `{{BASIC_CONTEXT}}` placeholder
2. Additional or file-specific context inserted via `{{ADDITIONAL_CONTEXT}}` placeholder (if selected)
3. User message remains clean: `"Input text:\n{chunk_text}"`

### Phase 3: API Request Construction

#### Schema Handler Selection
The system uses a handler registry to prepare API requests. Each handler:
- Loads the appropriate JSON schema
- Formats the developer message (extraction instructions)
- Constructs the API payload with proper structure

#### Processing Modes

Synchronous Mode:
- Real-time processing with immediate results
- Higher API costs (standard pricing)
- Good for small files or urgent needs

Batch Mode:
- 50% cost savings on API requests
- Results within 24 hours
- Ideal for large-scale processing
- Requires batch management tools

### Phase 4: Response Processing and Output Generation

#### JSON Extraction
Responses are parsed and validated against the schema, with error handling for invalid responses.

#### Aggregation
All chunk responses are merged into a single comprehensive dataset.

#### Multi-Format Output
- JSON Output (Always Generated): Contains the complete structured dataset with metadata
- CSV Output (Optional): Schema-specific converters transform JSON to tabular format
- DOCX Output (Optional): Generates formatted Word documents with proper headings, tables, and styling
- TXT Output (Optional): Creates human-readable plain text reports with structured formatting

## Adding Custom Schemas

ChronoMiner's extensible architecture allows easy integration of new extraction schemas tailored to your specific research needs.

### Create the JSON Schema

Place a new schema file in `schemas/` (e.g., `MyCustomSchema.json`). **Always mirror the top-level structure from an existing schema** (`contains_no_content_of_requested_type` boolean followed by the `entries` array) so downstream tooling recognizes the response format. 
The OpenAI structured outputs guide at [https://platform.openai.com/docs/guides/structured-outputs](https://platform.openai.com/docs/guides/structured-outputs) contains additional background how the schemas can be structured.

```json
{
  "name": "MyCustomSchema",
  "schema_version": "1.0",
  "type": "json_schema",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "contains_no_content_of_requested_type": {
        "type": "boolean",
        "description": "Set to true if the input text contains no MyCustomSchema entries. Otherwise, set to false."
      },
      "entries": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": { "type": "string", "description": "Unique identifier" },
            "title": { "type": "string", "description": "Document title" },
            "author": { "type": "string", "description": "Author name" },
            "date": { "type": "string", "format": "date", "description": "Publication date (YYYY-MM-DD)" },
            "content": { "type": "string", "description": "Main text content" }
          },
          "required": ["id", "title", "content"],
          "additionalProperties": false
        }
      }
    },
    "required": ["contains_no_content_of_requested_type", "entries"],
    "additionalProperties": false
  }
}
```

Schema Design Best Practices:

- **Mirror the preamble**: Copy the `contains_no_content_of_requested_type` and `entries` definitions exactly as shown (you can paste the first section from any existing file in `schemas/`).
- **Describe fields clearly**: Include clear field descriptions for the LLM to understand extraction requirements.
- **Set required fields**: Mark required fields explicitly.
- **Use strong typing**: Choose accurate data types and formats.
- **Keep validation strict**: Set `strict: true` for robust validation.

### Create Basic Context File (Required)

Create a basic context file in `basic_context/` named `MyCustomSchemaEntries.txt`. This file is mandatory and will be automatically loaded for every API request.

Example: `basic_context/MyCustomSchemaEntries.txt`

```
The input text consists of excerpts from historical legal documents dating from approximately 1700 to 1900.
The text may appear in different languages including English, Latin, and French, with period-specific legal
terminology and archaic spelling. Documents typically contain party names, case descriptions, judgments, and
dates. Natural semantic markers for this type of text are the beginnings of individual case entries or legal
proceedings.
```

### Create Additional Context File (Optional but Recommended)

Create a detailed additional context file in `additional_context/` named `MyCustomSchemaEntries.txt`.

Example: `additional_context/MyCustomSchemaEntries.txt`

```
ADDITIONAL CONTEXT FOR MY CUSTOM SCHEMA ENTRIES

When extracting data from historical legal documents, please consider:

1. Legal terminology considerations:
   - Latin phrases were standard in legal documents through the 19th century
   - Terms like "whereas," "heretofore," and "aforesaid" indicate legal language
   - Party designations (plaintiff, defendant) may use archaic terms

2. Date formats and calendars:
   - Dates may use regnal years (e.g., "3rd year of George III")
   - Old Style vs. New Style calendar differences (pre-1752 in British territories)
   - Format: "day Month year" common in legal documents
```

### Register Schema in Handler Registry

Add your schema name to the registry in `modules/operations/extraction/schema_handlers.py`:

```python
# Register existing schema handlers with the default implementation.
for schema in [
    "BibliographicEntries",
    "StructuredSummaries",
    "HistoricalAddressBookEntries",
    "BrazilianMilitaryRecords",
    "CulinaryPersonsEntries",
    "CulinaryPlacesEntries",
    "CulinaryWorksEntries",
    "MilitaryRecordEntries",
    "MyCustomSchemaEntries"  # Add your schema here
]:
    register_schema_handler(schema, BaseSchemaHandler)
```

### Add Developer Message (Optional)

Create `developer_messages/MyCustomSchema.txt` with custom extraction instructions. If not provided, the system uses the default prompt template.

### Implement Custom Handler (Optional)

For specialized post-processing or custom output formatting, create a handler class in `modules/operations/extraction/schema_handlers.py`:

```python
class MyCustomSchemaHandler(BaseSchemaHandler):
    schema_name = "MyCustomSchema"

    def process_response(self, response_str: str) -> dict:
        """Custom response processing logic."""
        data = super().process_response(response_str)
        
        # Add custom processing here
        if "entries" in data:
            for entry in data["entries"]:
                if "date" in entry:
                    entry["date"] = self.normalize_date(entry["date"])
        
        return data
    
    def normalize_date(self, date_str):
        """Normalize date format."""
        from datetime import datetime
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").isoformat()
        except (ValueError, TypeError):
            return None

register_schema_handler("MyCustomSchema", MyCustomSchemaHandler)
```

### Configure Paths

Add your schema to `config/paths_config.yaml`:

```yaml
schemas_paths:
  MyCustomSchema:
    input: "C:/path/to/my_custom_data/input"
    output: "C:/path/to/my_custom_data/output"
    csv_output: true
    docx_output: true
    txt_output: true
```

### Test the Schema

Run the main script and select your new schema:

```bash
python main/process_text_files.py
```

Verify the output and refine as needed.

## Batch Processing

Batch processing offers significant cost savings (50% reduction) for large-scale extraction tasks.

### When to Use Batch Processing

Ideal for:
- Processing multiple large files
- Non-urgent extraction tasks
- Research projects with time flexibility
- Cost-sensitive workflows

Not ideal for:
- Time-critical extractions
- Small single-file processing
- Interactive testing and development

### Batch Workflow

#### Submit Batch Job

```bash
python main/process_text_files.py
# Select "Batch" when prompted for processing mode
```

The script creates a batch request file and submits it to OpenAI's Batch API.

#### Monitor Batch Status

Check status periodically:

```bash
python main/check_batches.py
```

Batch Statuses:
- `validating`: OpenAI is validating the batch request
- `in_progress`: Batch is being processed
- `finalizing`: Processing complete, preparing results
- `completed`: Results available for download
- `failed`: Batch failed (check logs for details)
- `expired`: Batch expired before completion
- `cancelled`: Batch was cancelled

#### Retrieve Results

Once status shows `completed`, run:

```bash
python main/check_batches.py
```

The script automatically downloads batch results, processes all responses, generates final outputs, and cleans up temporary files.

### Batch Management Tools

#### Cancel Batches

Cancel all non-terminal (in_progress, validating) batches:

```bash
python main/cancel_batches.py
```

#### Repair Failed Batches

Recover from partial failures:

```bash
python main/repair_extractions.py
```

Interactive repair process:
1. Scans for incomplete batch jobs
2. Attempts to recover batch IDs from temporary files
3. Retrieves available responses
4. Regenerates outputs with recovered data
5. Reports success/failure for each repair attempt

### Batch Processing Best Practices

1. Test with synchronous mode first to validate your schema and settings
2. Use descriptive filenames to track batch jobs easily
3. Monitor batch status regularly during the 24-hour window
4. Keep temporary files (`retain_temporary_jsonl: true`) for debugging
5. Implement file-specific context for better extraction quality
6. Use pre-defined line ranges for consistent chunking across batches

## Token Cost Analysis

ChronoMiner bundles a lightweight analytics utility that inspects preserved temporary `.jsonl` responses and produces detailed cost estimates for every processed file. The workflow is implemented in `main/cost_analysis.py`, which orchestrates helper logic contained in `modules/operations/cost_analysis.py` and formatted output helpers in `modules/ui/cost_display.py`.

### When to Run It

- Preserve temporary `.jsonl` files (`retain_temporary_jsonl: true` in `config/paths_config.yaml`).
- After processing is complete, run the analysis to quantify spend across synchronous and batch jobs.
- Use the report to validate budgeting assumptions or to decide whether to switch schemas or models.

### Execution Modes

- **Interactive UI:** `python -m main.cost_analysis`
  - Mirrors the standard UI look and feel.
  - Automatically locates `.jsonl` files based on schema path configuration.
  - Displays aggregated token totals, per-file summaries, and optional CSV export prompts.
- **CLI Mode:** `python -m main.cost_analysis --save-csv --output path/to/report.csv`
  - Suitable for automation or scheduled reporting.
  - Flags:
    - `--save-csv`: Persist results to CSV (defaults to the folder that contains the first `.jsonl`).
    - `--output`: Override the target CSV path.
    - `--quiet`: Suppress console breakdown and emit only essential status messages.

### Output Features

- Aggregated totals for uncached input tokens, cached tokens, output tokens, reasoning tokens, and overall totals.
- Dual pricing: standard per-million-token rates and an automatic 50% discount column that models batched/flex billing tiers.
- Model normalization: date-stamped variants (e.g., `gpt-5-mini-2025-08-07`) are mapped to their parent pricing profile before calculations.
- CSV export includes a per-file ledger plus a consolidated summary row that mirrors the on-screen totals.

### Supported Pricing Profiles (USD per 1M tokens)

| Model | Input | Cached Input | Output |
| --- | --- | --- | --- |
| gpt-5 | 1.25 | 0.125 | 10.00 |
| gpt-5-mini | 0.25 | 0.025 | 2.00 |
| gpt-5-nano | 0.05 | 0.005 | 0.40 |
| gpt-5-chat-latest | 1.25 | 0.125 | 10.00 |
| gpt-5-codex | 1.25 | 0.125 | 10.00 |
| gpt-4.1 | 2.00 | 0.50 | 8.00 |
| gpt-4.1-mini | 0.40 | 0.10 | 1.60 |
| gpt-4.1-nano | 0.10 | 0.025 | 0.40 |
| gpt-4o | 2.50 | 1.25 | 10.00 |
| gpt-4o-2024-05-13 | 5.00 | - | 15.00 |
| gpt-4o-mini | 0.15 | 0.075 | 0.60 |
| gpt-4o-realtime-preview | 5.00 | 2.50 | 20.00 |
| gpt-4o-mini-realtime-preview | 0.60 | 0.30 | 2.40 |
| gpt-4o-audio-preview | 2.50 | - | 10.00 |
| gpt-4o-mini-audio-preview | 0.15 | - | 0.60 |
| gpt-4o-search-preview | 2.50 | - | 10.00 |
| gpt-4o-mini-search-preview | 0.15 | - | 0.60 |
| gpt-audio | 2.50 | 0.00 | 10.00 |
| o1 | 15.00 | 7.50 | 60.00 |
| o1-pro | 150.00 | - | 600.00 |
| o1-mini | 1.10 | 0.55 | 4.40 |
| o3 | 2.00 | 0.50 | 8.00 |
| o3-pro | 20.00 | 0.00 | 80.00 |
| o3-mini | 1.10 | 0.55 | 4.40 |
| o3-deep-research | 10.00 | 2.50 | 40.00 |
| o4-mini | 1.10 | 0.275 | 4.40 |
| o4-mini-deep-research | 2.00 | 0.50 | 8.00 |
| codex-mini-latest | 1.50 | 0.375 | 6.00 |
| computer-use-preview | 3.00 | - | 12.00 |
| gpt-image-1 | 5.00 | 1.25 | - |

> See: https://platform.openai.com/docs/pricing for more information.
> **Note:** Cached input pricing is denoted with `-` wherever OpenAI has not published a discounted tier. The analytics tool automatically treats missing values as zero in the CSV export.

## Architecture

ChronoMiner follows a modular architecture that separates concerns and promotes maintainability.

### Directory Structure

```
ChronoMiner/
├── config/                    # Configuration files
│   ├── chunking_config.yaml
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
│   ├── config/               # Configuration loading
│   ├── core/                 # Core utilities and workflow
│   ├── infra/                # Infrastructure (logging, concurrency)
│   ├── io/                   # File I/O and path utilities
│   ├── llm/                  # LLM interaction and batch processing
│   ├── operations/           # High-level operations (extraction, line ranges, repair)
│   └── ui/                   # User interface and prompts
├── schemas/                   # JSON schemas for structured outputs
├── basic_context/             # Basic context files (required, auto-loaded)
├── additional_context/        # Additional context files (optional, user-selected)
├── developer_messages/        # Developer message templates
├── prompts/                   # System prompt templates
├── LICENSE
├── README.md
└── requirements.txt
```

### Module Overview

- `modules/config/`: Configuration loading and validation
- `modules/core/`: Core utilities including text processing, JSON manipulation, data processing, context management, logging, and workflow helpers
- `modules/infra/`: Infrastructure layer providing logging setup, concurrency control, and async task management
- `modules/io/`: File I/O operations including path validation, directory scanning, file reading/writing, and output management
- `modules/llm/`: LLM interaction layer including LangChain multi-provider support, model capability detection, batch processing (OpenAI only), prompt management, and structured output parsing
- `modules/operations/`: High-level operation orchestration (extraction, line ranges, repair workflows)
- `modules/ui/`: User interface components including interactive prompts, selection menus, and status displays

### Operations Layer

ChronoMiner separates orchestration logic from CLI entry points to improve testability and maintainability:

- High-level operations live in `modules/operations/` (e.g., extraction, line ranges, repair workflows)
- CLI entry points in `main/` are thin wrappers that delegate to operations modules
- This design pattern allows operations to be reused, tested independently, and invoked programmatically

## Troubleshooting

### Common Issues

#### API key not found

Solution: Ensure the appropriate API key environment variable is set for your chosen provider:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key_here"      # For OpenAI models
$env:ANTHROPIC_API_KEY="your_key_here"   # For Claude models
$env:GOOGLE_API_KEY="your_key_here"      # For Gemini models
$env:OPENROUTER_API_KEY="your_key_here"  # For OpenRouter models

# Linux/macOS
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here
export OPENROUTER_API_KEY=your_key_here
```

The required API key depends on the model specified in `model_config.yaml`. The provider is auto-detected from the model name.

#### Schema not found in paths_config

Solution: Add your schema configuration to `config/paths_config.yaml`:

```yaml
schemas_paths:
  YourSchemaName:
    input: "path/to/input"
    output: "path/to/output"
    csv_output: true
    docx_output: true
    txt_output: true
```

#### Path validation failed

Solution:

- Create the directories manually: `mkdir -p /path/to/directory`
- Verify paths in `paths_config.yaml` are correct
- Ensure you have read/write permissions

#### Encoding errors

Solution:

- ChronoMiner auto-detects encoding, but you can manually convert files: `iconv -f ISO-8859-1 -t UTF-8 input.txt > input_utf8.txt`
- Check the log files for detected encoding information
- Ensure your input files are saved in UTF-8 encoding

#### Important information split across chunks

Solution:

- Use `line_range_readjuster.py` to optimize boundaries
- Manually edit `_line_ranges.txt` files
- Increase `default_tokens_per_chunk` (if within model limits)

#### Line ranges file not found

Solution:

- Run `python main/generate_line_ranges.py` to create them
- Or switch to `auto` or `auto-adjust` chunking method

#### Batch stuck in validating status

Solution: Wait a few minutes; validation can take time for large batches

#### Batch failed without clear error

Solution:

- Check logs in the configured `logs_dir`
- Verify batch request file format (should be valid JSONL)
- Try resubmitting with fewer chunks

#### Cannot retrieve batch results

Solution:

- Run `python main/check_batches.py` to automatically retrieve
- Check if batch ID is still valid (batches expire after 7 days)
- Use `python main/repair_extractions.py` to recover partial results

#### Output files missing

Solution:

- Verify output flags are set to `true` in `paths_config.yaml`
- Check for errors in log files
- Ensure output directory has write permissions

#### Empty or incomplete output files

Solution:

- Check if input file had valid extractable data
- Review log files for parsing errors
- Verify schema matches your data structure
- Test with example files first

#### Memory errors

Solution:

- Process fewer files simultaneously
- Reduce `concurrency_limit`
- Use batch processing instead of synchronous
- Process large files in smaller chunks

### Debug Artifacts

ChronoMiner creates several debug artifacts to help troubleshoot issues:

- `<job>_batch_submission_debug.json`: Contains batch IDs, chunk count, and submission timestamp
- `<job>_temporary.jsonl`: Tracks batch requests and responses with full metadata
- Log files: Detailed execution logs in the configured `logs_dir` with timestamps and stack traces

### Getting Help

If you encounter issues not covered here:

1. Check logs: Review detailed error messages in your configured `logs_dir`
2. Enable debug mode: Set logging level to DEBUG in `modules/core/logger.py`
3. Validate configuration: Ensure all YAML files are properly formatted
4. Verify directories: Confirm all required directories exist with proper permissions
5. Review requirements: Verify all dependencies are installed correctly
6. Check model access: Ensure your OpenAI account has access to the selected model

## Performance and Best Practices

### Optimization Strategies

#### Chunking Optimization

Best Practice: Use pre-generated line ranges for production workflows

```bash
# Generate line ranges once
python main/generate_line_ranges.py

# Refine with semantic boundaries
python main/line_range_readjuster.py

# Use in production
```

Benefits:

- Consistent results across runs
- Faster processing (no on-the-fly chunking)
- Better semantic coherence
- Easier debugging and reproduction

#### Cost Optimization

Use Batch Processing for large-scale projects:

- 50% cost savings compared to synchronous mode
- Ideal for processing 10+ files or files with 50+ chunks
- Plan for 24-hour turnaround time

Optimize Token Usage:

- Remove unnecessary whitespace from input files
- Use concise additional context
- Set `default_tokens_per_chunk` appropriately (don't over-chunk)

Monitor API Usage:

- Track costs with OpenAI's usage dashboard: https://platform.openai.com/usage

#### Processing Speed

Concurrent Processing:

- For synchronous mode, increase `concurrency_limit` (carefully)
- Start with 10, gradually increase to 20-30 if rate limits allow
- Monitor for rate limit errors

Network Optimization:

- Ensure stable internet connection
- Process during off-peak hours for better API response times
- Use local temporary storage (not network drives)

File Organization:

- Group similar files together for batch processing
- Use consistent naming conventions
- Keep input files in fast local storage (SSD preferred)

#### Quality Optimization

Context is Key:

- Always provide schema-specific context for domain-specific data
- Create file-specific context for challenging documents
- Update context based on extraction results

Iterative Refinement:

1. Process a few sample files
2. Review outputs for quality
3. Adjust schema, context, or chunking strategy
4. Reprocess samples
5. Scale to full dataset once satisfied

Schema Design:

- Start with required fields only
- Add optional fields gradually
- Use clear, descriptive field names and descriptions
- Leverage enums for controlled vocabulary

#### Reliability Best Practices

Configuration Management:

- Use version control for configuration files
- Document any custom settings
- Test configuration changes on sample files first

Backup Strategy:

- Keep temporary JSONL files: `retain_temporary_jsonl: true`
- Back up generated line range files
- Version control your schemas and context files

Monitoring:

- Regularly check batch status during processing windows
- Review log files for warnings or errors
- Validate output quality on samples before full processing

### Performance Benchmarks

Approximate processing times (actual times vary based on API speed and content complexity):

| File Size | Chunks | Synchronous (10 concurrent) | Batch Mode | Cost Savings |
|-----------|--------|------------------------------|------------|--------------|
| 50 KB | 5 | ~2 minutes | ~12-24 hours | 50% |
| 500 KB | 50 | ~15 minutes | ~12-24 hours | 50% |
| 5 MB | 500 | ~2-3 hours | ~12-24 hours | 50% |
| 50 MB | 5000 | ~20-30 hours | ~12-24 hours | 50% |

Note: Batch mode is significantly more cost-effective for large files, despite similar completion times.

## Examples

### Example 1: Processing Historical Bibliographies

Scenario: Extract metadata from a European culinary bibliography text file.

Steps:

1. Configure `paths_config.yaml`:
   ```yaml
   schemas_paths:
     BibliographicEntries:
       input: "C:/research/bibliographies/input"
       output: "C:/research/bibliographies/output"
       csv_output: true
       docx_output: true
       txt_output: true
   ```

2. Run extraction:
   ```bash
   python main/process_text_files.py
   # Select: BibliographicEntries
   # Select: auto (for single file)
   # Select: Synchronous (small file)
   # Select: Use schema-specific default context
   # Select: Single file
   ```

### Example 2: Batch Processing Multiple Address Books

Scenario: Process 20 historical address books from 1850-1900.

Steps:

1. Place all files in input directory
2. Generate line ranges:
   ```bash
   python main/generate_line_ranges.py
   ```

3. Review and adjust line ranges (optional):
   ```bash
   python main/line_range_readjuster.py
   ```

4. Process as batch:
   ```bash
   python main/process_text_files.py
   # Select: HistoricalAddressBookEntries
   # Select: Use pre-defined line ranges
   # Select: Batch processing
   # Select: Use schema-specific default context
   # Select: Entire folder
   ```

5. Monitor progress:
   ```bash
   python main/check_batches.py
   ```

6. Retrieve results (once completed):
   ```bash
   python main/check_batches.py
   ```

### Example 3: Custom Schema for Legal Documents

Scenario: Extract structured data from historical court records.

Steps:

1. Create schema (`schemas/LegalRecords.json`)
2. Add basic context (`basic_context/LegalRecordsEntries.txt`)
3. Add additional context (`additional_context/LegalRecordsEntries.txt`)
4. Register schema in handler registry
5. Configure paths in `paths_config.yaml`
6. Run extraction

## Contributing

Contributions are welcome! Here's how you can help improve ChronoMiner:

### Reporting Issues

When reporting bugs or issues, please include:

- Description: Clear description of the problem
- Steps to Reproduce: Detailed steps to reproduce the issue
- Expected Behavior: What you expected to happen
- Actual Behavior: What actually happened
- Environment: OS, Python version, relevant package versions
- Configuration: Relevant sections from your config files (remove sensitive information)
- Logs: Relevant log excerpts showing the error

### Suggesting Features

Feature suggestions are appreciated. Please provide:

- Use Case: Describe the problem or need
- Proposed Solution: Your idea for addressing it
- Alternatives: Other approaches you've considered
- Impact: Who would benefit and how

### Code Contributions

If you'd like to contribute code:

1. Fork the repository and create a feature branch
2. Follow the existing code style and architecture patterns
3. Add tests for new functionality where applicable
4. Update documentation including this README and inline comments
5. Test thoroughly with different schemas and chunking strategies
6. Submit a pull request with a clear description of your changes

### Development Guidelines

- Modularity: Keep functions focused and modules organized
- Error Handling: Use try-except blocks with informative error messages
- Logging: Use the logger for debugging information
- Configuration: Use YAML configuration files rather than hardcoding values
- User Experience: Provide clear prompts and feedback in CLI interactions
- Documentation: Update docstrings and README for any interface changes

### Areas for Contribution

Potential areas where contributions would be valuable:

- Additional LLM providers: Extended support for new models and providers via LangChain
- Enhanced chunking: Improved semantic boundary detection algorithms
- Output formats: Support for different output formats (XML, etc.)
- Testing: Unit tests and integration tests
- Documentation: Tutorials, examples, and use case documentation
- Performance optimization: Improved concurrent processing or caching
- Error recovery: Enhanced error handling and recovery mechanisms

## License

MIT License

Copyright (c) 2025 Paul Goetz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.