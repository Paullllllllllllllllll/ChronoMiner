# ChronoMiner

A Python-based tool designed for researchers and archivists to extract structured data from large-scale historical and academic text files. ChronoMiner leverages OpenAI's API with schema-based extraction to transform unstructured text into well-organized, analyzable datasets in multiple formats (JSON, CSV, DOCX, TXT).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Workflow Deep Dive](#workflow-deep-dive)
- [Adding Custom Schemas](#adding-custom-schemas)
- [Batch Processing](#batch-processing)
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

ChronoMiner supports two execution modes to accommodate different workflows and user preferences:

**Interactive Mode** provides a guided experience with clear prompts and helpful explanations at each step. Users select options from numbered menus, receive immediate validation with helpful error messages, and can navigate backward or quit at any prompt. This mode is ideal for first-time users, exploring options, or occasional processing tasks.

**CLI Mode** accepts command-line arguments for fully automated, scriptable workflows. All settings are provided via arguments, requiring no user interaction. Processing starts immediately, status messages go to console, and exit codes indicate success or failure. This mode is designed for power users, automation, batch processing, and integration with other tools.

The mode is determined automatically: if command-line arguments are provided, CLI mode activates; otherwise, Interactive mode starts (unless configured otherwise in `config/paths_config.yaml`).

### Key Capabilities

- Dual Execution Modes: Run interactively with guided prompts or use command-line arguments for automation and scripting
- Flexible Text Chunking: Token-based chunking with automatic, manual, or pre-defined line range strategies
- Schema-Based Extraction: JSON schema-driven data extraction with customizable templates
- Dual Processing Modes: Choose between synchronous (real-time) or batch processing (50% cost savings)
- Context-Aware Processing: Integrate domain-specific or file-specific context for improved accuracy
- Multi-Format Output: Generate JSON, CSV, DOCX, and TXT outputs simultaneously
- Extensible Architecture: Easily add custom schemas and handlers for your specific use cases
- Batch Management: Full suite of tools to manage, monitor, and repair batch jobs
- Semantic Boundary Detection: LLM-powered chunk boundary optimization for coherent document segments

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

### API Integration

- Schema Handler Registry: Uses handler registry to prepare API requests with appropriate JSON schema and extraction instructions
- Processing Modes:
  - Synchronous: Real-time processing with immediate results, higher API costs
  - Batch: 50% cost savings, results within 24 hours, ideal for large-scale processing
- Retry Logic: Automatic exponential backoff for transient API errors with configurable retry settings

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

## System Requirements

### Software Dependencies

- Python: 3.12 or higher
- OpenAI API Key: Required for using the OpenAI API
  - Set as environment variable: `OPENAI_API_KEY=your_key_here`
  - Ensure your account has access to the Responses API and Batch API

### Python Packages

All Python dependencies are listed in `requirements.txt`. Key packages include:

- `openai>=1.57.4`: OpenAI SDK for API interactions
- `tiktoken`: Accurate token counting
- `pandas`: Data manipulation
- `python-docx`: Document generation
- `pyyaml>=6.0.2`: Configuration file parsing
- `aiohttp>=3.11.10`: Asynchronous HTTP requests
- `tqdm>=4.67.1`: Progress bars

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

### Configure OpenAI API Key

Set your API key:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"

# Windows Command Prompt
set OPENAI_API_KEY=your_api_key_here

# Linux/macOS
export OPENAI_API_KEY=your_api_key_here
```

For persistent configuration, add the environment variable to your system settings or shell profile.

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

Controls which model to use and its behavioral parameters.

```yaml
transcription_model:
  name: gpt-4o  # Options: gpt-4o, gpt-4.1, gpt-5, gpt-5-mini, o1, o3
  max_output_tokens: 128000
  
  # GPT-5 and o-series only
  reasoning:
    effort: medium  # Options: low, medium, high
  
  # GPT-5 only
  text:
    verbosity: medium  # Options: low, medium, high
  
  # Classic models (GPT-4o, GPT-4.1) only
  temperature: 0.01
  top_p: 1.0
  frequency_penalty: 0.01
  presence_penalty: 0.01
```

Key Parameters:

- `name`: Model identifier
- `max_output_tokens`: Maximum tokens the model can generate per request
- `reasoning.effort`: Controls reasoning depth for GPT-5 and o-series models
- `text.verbosity`: Controls response verbosity for GPT-5 models
- `temperature`: Controls randomness (0.0-2.0)
- `top_p`: Nucleus sampling probability (0.0-1.0)
- `frequency_penalty`: Penalizes token repetition (-2.0 to 2.0)
- `presence_penalty`: Penalizes repeated topics (-2.0 to 2.0)

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

Controls parallel processing and retry behavior.

```yaml
concurrency:
  extraction:
    concurrency_limit: 250
    delay_between_tasks: 0.005
    service_tier: flex  # Options: auto, default, flex, priority
    batch_chunk_size: 50
    
    retry:
      attempts: 10
      wait_min_seconds: 4
      wait_max_seconds: 60
      jitter_max_seconds: 1
```

Key Parameters:

- `concurrency_limit`: Maximum number of concurrent tasks
- `delay_between_tasks`: Delay in seconds between starting tasks
- `service_tier`: OpenAI service tier for rate limiting and processing speed
  - Note: Service tiers apply only to synchronous API calls; batch processing automatically omits this parameter
- `batch_chunk_size`: Number of requests per batch part file
- Retry settings: Exponential backoff configuration for transient API failures

### Context Files

#### Basic Context (Required)

Basic context files are located in `basic_context/` and are automatically loaded and included in every API request. Each schema requires a corresponding basic context file named `{SchemaName}Entries.txt`.

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
6. Review and Confirm

#### CLI Mode

Process files with command-line arguments:

```bash
# Process a single file with batch mode
python main/process_text_files.py --schema BibliographicEntries --input data/file.txt --batch

# Process entire directory with custom chunking
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

Optimize chunk boundaries using LLM-detected semantic sections:

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

Place a new schema file in `schemas/` (e.g., `MyCustomSchema.json`).

```json
{
  "name": "MyCustomSchema",
  "schema_version": "1.0",
  "type": "json_schema",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
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
    "required": ["entries"],
    "additionalProperties": false
  }
}
```

Schema Design Best Practices:

- Use an `entries` array where each entry represents one unit of analysis
- Include clear field descriptions for the LLM to understand extraction requirements
- Mark required fields explicitly
- Use appropriate data types and formats
- Set `strict: true` for robust validation

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
- `modules/llm/`: LLM interaction layer including OpenAI SDK utilities, batch processing, model validation, prompt management, and structured output parsing
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

Solution: Ensure `OPENAI_API_KEY` environment variable is set:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"

# Linux/macOS
export OPENAI_API_KEY=your_api_key_here
```

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

- Additional LLM providers: Support for Anthropic, Google, or other APIs
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