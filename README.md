# ChronoMiner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-412991.svg)](https://openai.com/api/)

**ChronoMiner** is a powerful Python-based tool designed for historians, social scientists, and researchers to extract structured data from large-scale historical and academic text files. It leverages OpenAI's API with schema-based extraction to transform unstructured text into well-organized, analyzable datasets in multiple formats (JSON, CSV, DOCX, TXT).

## Key Features

- **Flexible Text Chunking**: Token-based chunking with automatic, manual, or pre-defined line range strategies
- **Schema-Based Extraction**: JSON schema-driven data extraction with customizable templates
- **Dual Processing Modes**: Choose between synchronous (real-time) or batch processing (50% cost savings)
- **Context-Aware Processing**: Integrate domain-specific or file-specific context for improved accuracy
- **Multi-Format Output**: Generate JSON, CSV, DOCX, and TXT outputs simultaneously
- **Interactive UI**: User-friendly command-line interface guides you through all processing steps
- **Extensible Architecture**: Easily add custom schemas and handlers for your specific use cases
- **Batch Management**: Full suite of tools to manage, monitor, and repair batch jobs
- **Semantic Boundary Detection**: LLM-powered chunk boundary optimization for coherent document segments

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Use Cases](#use-cases)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Workflow Deep Dive](#workflow-deep-dive)
- [Adding Custom Schemas](#adding-custom-schemas)
- [Batch Processing](#batch-processing)
- [Project Architecture](#project-architecture)
- [Troubleshooting](#troubleshooting)
- [Performance & Best Practices](#performance--best-practices)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact & Support](#contact--support)

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/Paullllllllllllllllll/ChronoMiner.git
cd ChronoMiner

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here  # On Windows: set OPENAI_API_KEY=your_api_key_here

# 5. Configure paths (edit config/paths_config.yaml with your input/output directories)

# 6. Run the main extraction script
python main/process_text_files.py
```

The interactive UI will guide you through schema selection, chunking strategy, processing mode, and file selection.

## Repository Structure

```
ChronoMiner/
├── config/                                 # Configuration files
│   ├── chunking_and_context.yaml          # Chunking strategy and context settings
│   ├── concurrency_config.yaml            # Concurrency, timeout, and retry configuration
│   ├── model_config.yaml                  # OpenAI model configuration (GPT-5-mini, o3-mini, etc.)
│   └── paths_config.yaml                  # Input/output paths and output format flags
│
├── additional_context/                     # Schema-specific context for enhanced extraction
│   ├── BibliographicEntries.txt
│   ├── BrazilianMilitaryRecords.txt
│   ├── HistoricalAddressBookEntries.txt
│   └── StructuredSummaries.txt
│
├── basic_context/                          # Basic context examples for demonstration
│   └── [Sample context files...]
│
├── example_files/                          # Example inputs and outputs
│   ├── example_inputs/
│   │   ├── addressbooks/                   # Sample address book text files
│   │   ├── bibliographic/                  # Sample bibliography text files
│   │   └── occupationrecords/              # Sample military/occupation records
│   └── example_outputs/
│       └── [Corresponding output examples...]
│
├── main/                                   # Main executable scripts
│   ├── process_text_files.py              # Primary extraction script (start here!)
│   ├── generate_line_ranges.py            # Generate token-based line range files
│   ├── line_range_readjuster.py           # Refine line ranges using semantic boundaries
│   ├── check_batches.py                   # Monitor and process batch job results
│   ├── cancel_batches.py                  # Cancel in-progress batch jobs
│   └── repair_extractions.py              # Repair incomplete or failed batch extractions
│
├── modules/                                # Core application modules
│   ├── config/                             # Configuration management
│   │   ├── loader.py                       # YAML configuration file loader
│   │   └── manager.py                      # Configuration validation and management
│   │
│   ├── core/                               # Core business logic and utilities
│   │   ├── batch_utils.py                  # Batch processing utilities
│   │   ├── concurrency.py                  # Asynchronous task management with rate limiting
│   │   ├── context_manager.py              # Additional context loading and management
│   │   ├── data_processing.py              # CSV conversion with schema-specific handlers
│   │   ├── json_utils.py                   # JSON entry extraction and manipulation
│   │   ├── logger.py                       # Centralized logging configuration
│   │   ├── prompt_context.py               # Context preparation for LLM prompts
│   │   ├── schema_manager.py               # JSON schema loading and validation
│   │   ├── text_processing.py              # DOCX and TXT conversion routines
│   │   ├── text_utils.py                   # Text normalization, encoding, tokenization
│   │   └── workflow_utils.py               # Common workflow helper functions
│   │
│   ├── llm/                                # LLM interaction and API management
│   │   ├── batching.py                     # Batch request file creation and submission
│   │   ├── model_capabilities.py           # Model capability validation
│   │   ├── openai_sdk_utils.py             # OpenAI SDK utility functions
│   │   ├── openai_utils.py                 # Async API request handling
│   │   ├── prompt_utils.py                 # Prompt template loading and formatting
│   │   └── structured_outputs.py           # Structured output format handling
│   │
│   ├── operations/                         # High-level operations and workflows
│   │   ├── extraction/
│   │   │   ├── file_processor.py           # Main file processing orchestration
│   │   │   └── schema_handlers.py          # Schema handler registry and base classes
│   │   └── line_ranges/
│   │       └── readjuster.py               # Line range adjustment logic
│   │
│   └── ui/                                 # User interface components
│       └── core.py                         # Interactive CLI prompts and menus
│
├── prompts/                                # LLM prompt templates
│   ├── semantic_boundary_prompt.txt        # Prompt for detecting semantic boundaries
│   └── structured_output_prompt.txt        # Unified extraction prompt template
│
├── schemas/                                # JSON schemas for structured extraction
│   ├── address_schema.json                 # Swiss Historical Address Book Entries
│   ├── bibliographic_schema.json           # European Culinary Bibliography Entries
│   ├── culinary_persons.json               # Culinary Persons Information
│   ├── culinary_places.json                # Culinary Places and Establishments
│   ├── culinary_works.json                 # Culinary Works and Publications
│   ├── military_record_schema.json         # Brazilian Military Records
│   └── summary_schema.json                 # Structured Academic Text Summaries
│
├── LICENSE                                 # MIT License
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```

## Usage Guide

### Basic Workflow

#### 1. Run the Main Extraction Script

```bash
python main/process_text_files.py
```

The interactive UI will guide you through the following steps:

#### 2. Select a Schema

```
Available schemas:
1. BibliographicEntries
2. HistoricalAddressBookEntries
3. BrazilianMilitaryRecords
4. StructuredSummaries

Select schema [1-4]:
```

Choose the schema that matches your input data type.

#### 3. Choose Chunking Strategy

```
Chunking methods:
1. Automatic (auto)
2. Automatic with manual adjustments (auto-adjust)
3. Use pre-defined line ranges (line_ranges.txt)

Select chunking method [1-3]:
```

- **Option 1**: Fully automatic, best for quick tests
- **Option 2**: Automatic with opportunity to refine boundaries interactively
- **Option 3**: Uses pre-generated `_line_ranges.txt` files (most reliable for production)

#### 4. Select Processing Mode

```
Processing modes:
1. Synchronous (real-time results)
2. Batch (50% cost savings, up to 24h processing time)

Select processing mode [1-2]:
```

- **Synchronous**: Immediate results, higher cost, good for small files or urgent needs
- **Batch**: 50% cost reduction, results within 24 hours, ideal for large-scale processing

#### 5. Configure Additional Context (Optional)

```
Additional context options:
1. No additional context
2. Use schema-specific default context
3. Use custom file-specific context

Select option [1-3]:
```

Additional context improves extraction accuracy by providing domain knowledge or specific instructions.

#### 6. Select Input Files

```
Input source:
1. Single file
2. Entire folder

Select input source [1-2]:
```

Then navigate to select your file(s).

#### 7. Confirm and Process

ChronoMiner displays a summary of your selections and begins processing. Progress is shown in real-time.

#### 8. Review Output

Once complete, find your structured data in the configured output directory:
- **JSON**: Always generated, contains structured data
- **CSV**: Tabular format, easy to import into Excel/Google Sheets
- **DOCX**: Formatted Word document
- **TXT**: Plain text representation

### Advanced Operations

#### Generate Line Ranges

For production workflows, pre-generate line range files:

```bash
python main/generate_line_ranges.py
```

This creates `{filename}_line_ranges.txt` files specifying exact line ranges for each chunk.

**Benefits:**
- Consistent chunking across multiple runs
- Can be manually edited to ensure semantic coherence
- Faster processing (no on-the-fly chunking decisions)

#### Refine Line Ranges with Semantic Boundaries

Optimize chunk boundaries using LLM-detected semantic sections:

```bash
# Interactive mode
python main/line_range_readjuster.py

# Non-interactive with options
python main/line_range_readjuster.py --dry-run --context-window 500 --boundary-type section
```

**Options:**
- `--dry-run`: Preview changes without modifying files
- `--context-window N`: Number of lines around boundaries to examine (default: 300)
- `--boundary-type`: Type of boundary to detect (section, paragraph, entry)

This tool examines text around existing chunk boundaries and proposes adjustments to align with natural document structure (headers, paragraph breaks, etc.).

#### Check Batch Status

Monitor and process completed batch jobs:

```bash
python main/check_batches.py
```

**Features:**
- Lists all batch jobs with status (in_progress, completed, failed, etc.)
- Automatically downloads and processes completed batches
- Generates all configured output formats (JSON, CSV, DOCX, TXT)
- Provides detailed summary of batch operations

#### Cancel Batch Jobs

Cancel all in-progress batches:

```bash
python main/cancel_batches.py
```

**Use cases:**
- Stopping accidentally submitted jobs
- Reconfiguring and resubmitting with different settings
- Cleaning up test batches

#### Repair Incomplete Extractions

If a batch job partially failed, repair it:

```bash
python main/repair_extractions.py
```

**Repair capabilities:**
- Discovers incomplete extraction jobs
- Recovers missing batch IDs from debug artifacts
- Retrieves responses from completed batches
- Regenerates final outputs with all available data

## Workflow Deep Dive

Understanding ChronoMiner's internal workflow helps you optimize your extraction tasks and troubleshoot issues.

### Phase 1: Text Pre-Processing

#### 1.1 Encoding Detection & Normalization
- Automatically detects file encoding (UTF-8, ISO-8859-1, Windows-1252, etc.)
- Normalizes text by stripping extraneous whitespace
- Logs encoding information for debugging

#### 1.2 Token-Based Chunking
- Uses `tiktoken` to accurately count tokens for the selected model
- Splits text into chunks based on `default_tokens_per_chunk` setting
- Ensures chunks don't exceed model's context window

**Chunking Strategies Explained:**

**a) Automatic (`auto`):**
```
Original Text (12,000 tokens)
    ↓
[Chunk 1: 7,500 tokens]
[Chunk 2: 4,500 tokens]
```

**b) Automatic with Adjustments (`auto-adjust`):**
```
Original Text (12,000 tokens)
    ↓
[Chunk 1: 7,500 tokens] ← User reviews and adjusts boundary
    ↓
[Chunk 1: 6,800 tokens] ← Adjusted to end at paragraph
[Chunk 2: 5,200 tokens]
```

**c) Pre-defined Line Ranges (`line_ranges.txt`):**
```
input_file.txt
    ↓
input_file_line_ranges.txt:
    1-342
    343-689
    690-1024
    ↓
[Chunk 1: lines 1-342]
[Chunk 2: lines 343-689]
[Chunk 3: lines 690-1024]
```

### Phase 2: Context Integration

#### 2.1 Basic Context (Always Included)

Basic context files are located in `basic_context/` and are automatically loaded and included in every API request, regardless of user settings. Each schema requires a corresponding basic context file named `{SchemaName}Entries.txt` (e.g., `BibliographicEntries.txt`, `CulinaryPersonsEntries.txt`).

Basic context files provide fundamental information about the input source:
- A brief description of the input text type and characteristics
- Language and formatting expectations
- Historical time period or geographic scope
- Natural semantic markers for text chunking

Basic context is designed to be concise and general, typically 4-6 lines with a 120-character line wrap for readability.

**Example**: `basic_context/BibliographicEntries.txt`
```
The input text consists of a snippet of bibliographic records describing historical culinary and household
literature (e.g., cookbooks, household manuals, foodstuff guides, food production regulations, etc.) dating
from approximately 1470 to 1950. The text may appear in different languages and use historical spelling
and diacritics. When extracting edition information, ensure that later or revised editions of the same work are
grouped together under the entry for the first (original) edition in the `edition_info` array. Natural semantic
markers for this type of text are the beginnings of full bibliographic entries.
```

#### 2.2 Additional Context (User-Selected)

Additional context files are located in `additional_context/` and are only included when the user selects to use additional context during the interactive workflow. These files provide detailed, domain-specific guidance for extraction.

The user can choose between two additional context modes:
- **Default context**: Uses the schema-specific file from `additional_context/{SchemaName}.txt`
- **File-specific context**: Uses `{filename}_context.txt` files located next to the input files

Additional context is more detailed than basic context and typically includes:
- Domain-specific terminology and conventions
- Historical context and background information
- Detailed extraction guidelines and edge cases
- Common patterns and data quality considerations

**Example**: `additional_context/BibliographicEntries.txt` (excerpt)
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

#### 2.3 File-Specific Context (Optional)

For input files requiring specific instructions, users can create `{filename}_context.txt` files in the same directory as the input file. This provides the most granular level of context customization.

**Example**: `zurich_addressbook_1850_context.txt`
```
This address book contains entries from 1850-1870 in Zurich, Switzerland.
- Occupations are listed in Swiss German
- Addresses use old street names that no longer exist (refer to historical city maps)
- Some entries span multiple lines due to long professional titles
- Family businesses often list multiple family members at the same address
```

#### 2.4 Context Hierarchy and Integration

All applicable context levels are combined and injected into the system prompt via placeholders:
1. Basic context always inserted via `{{BASIC_CONTEXT}}` placeholder
2. Additional or file-specific context inserted via `{{ADDITIONAL_CONTEXT}}` placeholder (if selected)
3. User message remains clean: `"Input text:\n{chunk_text}"`

This approach ensures the language model has comprehensive background knowledge while keeping the actual text chunk clearly separated in the user message.

### Phase 3: API Request Construction

#### 3.1 Schema Handler Selection
The system uses a handler registry (`schema_handlers.py`) to prepare API requests. Each handler:
- Loads the appropriate JSON schema
- Formats the developer message (extraction instructions)
- Constructs the API payload with proper structure

#### 3.2 Processing Modes

**Synchronous Mode:**
```
[Chunk 1] → API Request → Response → Process → Save
[Chunk 2] → API Request → Response → Process → Save
[Chunk 3] → API Request → Response → Process → Save
    ↓
Aggregate all responses → Generate final outputs
```
- Real-time processing with immediate results
- Higher API costs (standard pricing)
- Good for small files or urgent needs

**Batch Mode:**
```
[Chunk 1] ┐
[Chunk 2] ├→ Batch JSONL file → Submit batch job → Wait (up to 24h)
[Chunk 3] ┘
    ↓
Retrieve batch results → Process all responses → Generate final outputs
```
- 50% cost savings on API requests
- Results within 24 hours
- Ideal for large-scale processing
- Requires batch management tools

### Phase 4: Response Processing & Output Generation

#### 4.1 JSON Extraction
Responses are parsed and validated against the schema, with error handling for invalid responses.

#### 4.2 Aggregation
All chunk responses are merged into a single comprehensive dataset.

#### 4.3 Multi-Format Output

**JSON Output (Always Generated):**
Contains the complete structured dataset with metadata.

**CSV Output (Optional):**
Schema-specific converters transform JSON to tabular format, flattening nested structures and handling arrays.

**DOCX Output (Optional):**
Generates formatted Word documents with proper headings, tables, and styling.

**TXT Output (Optional):**
Creates human-readable plain text reports with structured formatting.

## Adding Custom Schemas

ChronoMiner's extensible architecture allows easy integration of new extraction schemas tailored to your specific research needs. Adding a new schema requires creating several coordinated files to ensure proper functionality.

### Step 1: Create the JSON Schema

Place a new schema file in `schemas/` (e.g., `my_custom_schema.json`). The schema name should match the pattern used throughout the system.

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

**Schema Design Best Practices:**
- Use an `entries` array where each entry represents one unit of analysis
- Include clear field descriptions for the LLM to understand extraction requirements
- Mark required fields explicitly
- Use appropriate data types and formats
- Set `strict: true` for robust validation
- Refer to [OpenAI's structured outputs documentation](https://platform.openai.com/docs/guides/structured-outputs)

### Step 2: Create Basic Context File (Required)

Create a basic context file in `basic_context/` named `MyCustomSchemaEntries.txt`. This file is mandatory and will be automatically loaded and included in every API request for this schema.

The basic context should be concise (typically 4-6 lines) with a 120-character line wrap. It should describe:
- The type and nature of the input text
- Expected languages, time periods, or geographic scope
- Historical spelling or formatting considerations
- Natural semantic markers for chunking

**Example**: `basic_context/MyCustomSchemaEntries.txt`
```
The input text consists of excerpts from historical legal documents dating from approximately 1700 to 1900.
The text may appear in different languages including English, Latin, and French, with period-specific legal
terminology and archaic spelling. Documents typically contain party names, case descriptions, judgments, and
dates. Natural semantic markers for this type of text are the beginnings of individual case entries or legal
proceedings.
```

**Important:** The basic context file name must follow the pattern `{SchemaName}Entries.txt` where `{SchemaName}` matches the `name` field in your JSON schema.

### Step 3: Create Additional Context File (Optional but Recommended)

Create a detailed additional context file in `additional_context/` named `MyCustomSchemaEntries.txt`. This file provides comprehensive domain-specific guidance and is only loaded when users select to use additional context.

The additional context should be detailed and structured, typically including:
- Multiple sections with numbered headings
- Domain-specific terminology explanations
- Historical context and background
- Extraction guidelines and edge cases
- Common patterns and quality considerations

**Example**: `additional_context/MyCustomSchemaEntries.txt`
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

3. Name conventions:
   - Full formal names with titles and honorifics
   - Married women identified by husband's name
   - Corporate or institutional entities as parties

This additional context should help you accurately extract and structure legal document information.
```

### Step 4: Register Schema in Handler Registry

Add your schema name to the registry in `modules/operations/extraction/schema_handlers.py`. Locate the schema registration section near the end of the file and add your schema name to the list:

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

**Note:** The schema name in the registry should match the `name` field in your JSON schema and the base name of your context files.

### Step 5: Add Developer Message (Optional)

Create `developer_messages/MyCustomSchema.txt` with custom extraction instructions. If not provided, the system uses the default prompt template from `prompts/structured_output_prompt.txt`.

```
You are a structured data extraction expert. Extract information from historical legal documents and return JSON in the specified format.

Instructions:
- Extract all case entries from the input text
- Use the 'id' field for unique case identification
- Preserve original spelling and legal terminology
- Parse dates according to historical calendar systems
- Extract party names with titles and honorifics

Quality guidelines:
- Ensure completeness: extract ALL entries from the text
- Maintain accuracy: verify data against the source text
- Handle edge cases: missing data should be null, not omitted
```

### Step 6: Implement Custom Handler (Optional)

For specialized post-processing or custom output formatting, create a handler class in `modules/operations/extraction/schema_handlers.py`:

```python
from modules.operations.extraction.schema_handlers import BaseSchemaHandler, register_schema_handler

class MyCustomSchemaHandler(BaseSchemaHandler):
    schema_name = "MyCustomSchema"

    def process_response(self, response_str: str) -> dict:
        """Custom response processing logic."""
        data = super().process_response(response_str)
        
        # Add custom processing here
        if "entries" in data:
            for entry in data["entries"]:
                # Example: Normalize dates
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

# Register the custom handler (replaces the default BaseSchemaHandler)
register_schema_handler("MyCustomSchema", MyCustomSchemaHandler)
```

### Step 7: Configure Paths

Add your schema to `config/paths_config.yaml` to specify input/output directories and output format preferences:

```yaml
schemas_paths:
  MyCustomSchema:
    input: "C:/path/to/my_custom_data/input"
    output: "C:/path/to/my_custom_data/output"
    csv_output: true
```

### Step 8: Test the Schema

Run the main script and select your new schema:

```bash
python main/process_text_files.py
```

Verify the output and refine as needed.

## Batch Processing

Batch processing offers significant cost savings (50% reduction) for large-scale extraction tasks.

### When to Use Batch Processing

**Ideal for:**
- Processing multiple large files
- Non-urgent extraction tasks
- Research projects with time flexibility
- Cost-sensitive workflows

**Not ideal for:**
- Time-critical extractions
- Small single-file processing
- Interactive testing and development

### Batch Workflow

#### 1. Submit Batch Job

```bash
python main/process_text_files.py
# Select "Batch" when prompted for processing mode
```

The script creates a batch request file and submits it to OpenAI's Batch API.

#### 2. Monitor Batch Status

Check status periodically:

```bash
python main/check_batches.py
```

**Batch Statuses:**
- `validating`: OpenAI is validating the batch request
- `in_progress`: Batch is being processed
- `finalizing`: Processing complete, preparing results
- `completed`: Results available for download
- `failed`: Batch failed (check logs for details)
- `expired`: Batch expired before completion
- `cancelled`: Batch was cancelled

#### 3. Retrieve Results

Once status shows `completed`, run:

```bash
python main/check_batches.py
```

The script automatically:
- Downloads batch results
- Processes all responses
- Generates final outputs (JSON, CSV, DOCX, TXT)
- Cleans up temporary files

### Batch Management Tools

#### Cancel Batches

Cancel all non-terminal (in_progress, validating) batches:

```bash
python main/cancel_batches.py
```

Provides detailed summary:
```
Batch Cancellation Summary:
- Total batches found: 5
- Cancelled: 2
- Already completed: 2
- Failed: 1
```

#### Repair Failed Batches

Recover from partial failures:

```bash
python main/repair_extractions.py
```

**Interactive repair process:**
1. Scans for incomplete batch jobs
2. Attempts to recover batch IDs from temporary files
3. Retrieves available responses
4. Regenerates outputs with recovered data
5. Reports success/failure for each repair attempt

**Common failure scenarios:**
- Network interruption during result download
- Partial API response issues
- Temporary file corruption

### Batch Processing Best Practices

1. **Test with synchronous mode first** to validate your schema and settings
2. **Use descriptive filenames** to track batch jobs easily
3. **Monitor batch status regularly** during the 24-hour window
4. **Keep temporary files** (`retain_temporary_jsonl: true`) for debugging
5. **Implement file-specific context** for better extraction quality
6. **Use pre-defined line ranges** for consistent chunking across batches

## Project Architecture

Understanding ChronoMiner's architecture helps with customization and troubleshooting.

### Core Components

#### 1. Configuration System (`modules/config/`)
- **loader.py**: Loads and validates YAML configuration files
- **manager.py**: Manages configuration state and validation logic

**Key Features:**
- Centralized configuration management
- Path validation (absolute and relative)
- Environment variable expansion
- Configuration inheritance

#### 2. Core Utilities (`modules/core/`)
- **text_utils.py**: Text processing (encoding detection, normalization, tokenization)
- **json_utils.py**: JSON manipulation and validation
- **data_processing.py**: CSV conversion with schema-specific handlers
- **text_processing.py**: DOCX and TXT generation
- **context_manager.py**: Additional context loading and management
- **logger.py**: Centralized logging configuration
- **workflow_utils.py**: Common workflow helper functions
- **batch_utils.py**: Batch API utilities
- **concurrency.py**: Async task management with rate limiting
- **prompt_context.py**: Prompt context preparation
- **schema_manager.py**: Schema loading and validation

#### 3. LLM Integration (`modules/llm/`)
- **openai_utils.py**: Async API request handling with retry logic
- **openai_sdk_utils.py**: OpenAI SDK utility functions
- **batching.py**: Batch request creation and submission
- **prompt_utils.py**: Prompt template loading and formatting
- **structured_outputs.py**: Structured output format handling
- **model_capabilities.py**: Model capability validation

#### 4. Operations (`modules/operations/`)
- **extraction/file_processor.py**: Main file processing orchestration
- **extraction/schema_handlers.py**: Schema handler registry and base classes
- **line_ranges/readjuster.py**: Semantic boundary detection and adjustment

#### 5. User Interface (`modules/ui/`)
- **core.py**: Interactive CLI with rich prompts and menus

### Data Flow

```
User Input
    ↓
[UI Layer] - User selections and file input
    ↓
[Configuration] - Load settings and validate paths
    ↓
[Text Processing] - Encoding detection, chunking, normalization
    ↓
[Context Management] - Load and prepare additional context
    ↓
[Schema Handler] - Select appropriate handler and prepare payload
    ↓
[LLM Integration] - API request construction and submission
    ↓
[Response Processing] - Parse, validate, and aggregate responses
    ↓
[Output Generation] - Create JSON, CSV, DOCX, TXT files
    ↓
Final Outputs
```

### Extension Points

ChronoMiner is designed for extensibility:

1. **Custom Schemas**: Add new JSON schemas to `schemas/`
2. **Custom Handlers**: Extend `BaseSchemaHandler` for specialized processing
3. **Custom Context**: Add schema or file-specific context files
4. **Custom Converters**: Implement schema-specific CSV/DOCX/TXT converters
5. **Custom Prompts**: Modify prompt templates in `prompts/`

## Troubleshooting

### Common Issues and Solutions

#### OpenAI API Errors

**Problem**: `AuthenticationError: Invalid API key`
- **Solution**: Verify your API key is correctly set:
  ```bash
  echo $OPENAI_API_KEY  # Linux/macOS
  echo %OPENAI_API_KEY%  # Windows
  ```
  Ensure there are no extra spaces or quotes around the key.

**Problem**: `RateLimitError: Rate limit exceeded`
- **Solution**: 
  - Reduce `max_concurrent_tasks` in `concurrency_config.yaml`
  - Switch to batch processing mode for large jobs
  - Wait and retry after a few minutes

**Problem**: `TimeoutError: Request timed out`
- **Solution**: 
  - Increase `request_timeout_seconds` in `concurrency_config.yaml`
  - Check your internet connection
  - Try processing smaller chunks

#### Configuration Errors

**Problem**: `FileNotFoundError: Config file not found`
- **Solution**: Ensure all YAML files exist in the `config/` directory
- Check file names match exactly (case-sensitive on Linux/macOS)

**Problem**: `KeyError: Schema not found in paths_config`
- **Solution**: Add your schema configuration to `config/paths_config.yaml`:
  ```yaml
  schemas_paths:
    YourSchemaName:
      input: "path/to/input"
      output: "path/to/output"
      csv_output: true
      docx_output: true
      txt_output: true
  ```

**Problem**: `Path validation failed: Directory does not exist`
- **Solution**: 
  - Create the directories manually: `mkdir -p /path/to/directory`
  - Verify paths in `paths_config.yaml` are correct
  - Ensure you have read/write permissions

#### Encoding Errors

**Problem**: `UnicodeDecodeError` when reading files
- **Solution**: 
  - ChronoMiner auto-detects encoding, but you can manually convert files:
    ```bash
    iconv -f ISO-8859-1 -t UTF-8 input.txt > input_utf8.txt
    ```
  - Check the log files for detected encoding information

**Problem**: Special characters not displaying correctly
- **Solution**: 
  - Ensure your input files are saved in UTF-8 encoding
  - Check your terminal/console supports UTF-8 display

#### Chunking Issues

**Problem**: Important information split across chunks
- **Solution**: 
  - Use `line_range_readjuster.py` to optimize boundaries
  - Manually edit `_line_ranges.txt` files
  - Increase `default_tokens_per_chunk` (if within model limits)

**Problem**: `_line_ranges.txt` file not found
- **Solution**: 
  - Run `python main/generate_line_ranges.py` to create them
  - Or switch to `auto` or `auto-adjust` chunking method

#### Batch Processing Issues

**Problem**: Batch stuck in `validating` status
- **Solution**: Wait a few minutes; validation can take time for large batches

**Problem**: Batch failed without clear error
- **Solution**: 
  - Check logs in the configured `logs_dir`
  - Verify batch request file format (should be valid JSONL)
  - Try resubmitting with fewer chunks

**Problem**: Cannot retrieve batch results
- **Solution**: 
  - Run `python main/check_batches.py` to automatically retrieve
  - Check if batch ID is still valid (batches expire after 7 days)
  - Use `python main/repair_extractions.py` to recover partial results

#### Output Generation Errors

**Problem**: JSON output generated but CSV/DOCX/TXT missing
- **Solution**: 
  - Verify output flags are set to `true` in `paths_config.yaml`
  - Check for errors in log files
  - Ensure output directory has write permissions

**Problem**: Empty or incomplete output files
- **Solution**: 
  - Check if input file had valid extractable data
  - Review log files for parsing errors
  - Verify schema matches your data structure
  - Test with example files first

#### Memory Issues

**Problem**: `MemoryError` or system slowdown
- **Solution**: 
  - Process fewer files simultaneously
  - Reduce `max_concurrent_tasks`
  - Use batch processing instead of synchronous
  - Process large files in smaller chunks

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Review files in your configured `logs_dir`
2. **Enable debug mode**: Set logging level to DEBUG in `modules/core/logger.py`
3. **Test with examples**: Try processing files from `example_files/example_inputs/`
4. **Search GitHub Issues**: Check existing [issues](https://github.com/Paullllllllllllllllll/ChronoMiner/issues)
5. **Open a new issue**: Provide logs, configuration, and error messages

## Performance & Best Practices

### Optimization Strategies

#### 1. Chunking Optimization

**Best Practice**: Use pre-generated line ranges for production workflows
```bash
# Generate line ranges once
python main/generate_line_ranges.py

# Refine with semantic boundaries
python main/line_range_readjuster.py

# Use in production - select "line_ranges.txt" when prompted for chunking method
```

**Benefits**:
- Consistent results across runs
- Faster processing (no on-the-fly chunking)
- Better semantic coherence
- Easier debugging and reproduction

#### 2. Cost Optimization

**Use Batch Processing** for large-scale projects:
- **50% cost savings** compared to synchronous mode
- Ideal for processing 10+ files or files with 50+ chunks
- Plan for 24-hour turnaround time

**Optimize Token Usage**:
- Remove unnecessary whitespace from input files
- Use concise additional context
- Set `default_tokens_per_chunk` appropriately (don't over-chunk)

**Monitor API Usage**:
```bash
# Track costs with OpenAI's usage dashboard
# https://platform.openai.com/usage
```

#### 3. Processing Speed

**Concurrent Processing**:
- For synchronous mode, increase `max_concurrent_tasks` (carefully)
- Start with 10, gradually increase to 20-30 if rate limits allow
- Monitor for rate limit errors

**Network Optimization**:
- Ensure stable internet connection
- Process during off-peak hours for better API response times
- Use local temporary storage (not network drives)

**File Organization**:
- Group similar files together for batch processing
- Use consistent naming conventions
- Keep input files in fast local storage (SSD preferred)

#### 4. Quality Optimization

**Context is Key**:
- Always provide schema-specific context for domain-specific data
- Create file-specific context for challenging documents
- Update context based on extraction results

**Iterative Refinement**:
1. Process a few sample files
2. Review outputs for quality
3. Adjust schema, context, or chunking strategy
4. Reprocess samples
5. Scale to full dataset once satisfied

**Schema Design**:
- Start with required fields only
- Add optional fields gradually
- Use clear, descriptive field names and descriptions
- Leverage enums for controlled vocabulary

#### 5. Reliability Best Practices

**Configuration Management**:
- Use version control for configuration files
- Document any custom settings
- Test configuration changes on sample files first

**Backup Strategy**:
- Keep temporary JSONL files: `retain_temporary_jsonl: true`
- Back up generated line range files
- Version control your schemas and context files

**Monitoring**:
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

**Note**: Batch mode is significantly more cost-effective for large files, despite similar completion times.

## Examples

### Example 1: Processing Historical Bibliographies

**Scenario**: Extract metadata from a European culinary bibliography text file.

**Input File** (`culinary_books_1800s.txt`):
```
La Varenne, François Pierre de. Le Cuisinier François. Paris: Pierre David, 1651.
Second edition published in Paris by Pierre David in 1652.

Menon. La Cuisinière Bourgeoise. Paris: Guillyn, 1746.
Multiple editions throughout the 18th century.

Grimod de la Reynière, Alexandre Balthazar Laurent. Almanach des Gourmands. Paris, 1803-1812.
Annual publication, eight volumes total.
```

**Steps**:
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
   # Select: Single file → culinary_books_1800s.txt
   ```

**Output** (`culinary_books_1800s.json`):
```json
{
  "entries": [
    {
      "full_title": "Le Cuisinier François",
      "short_title": "Le Cuisinier François",
      "authors": ["François Pierre de la Varenne"],
      "roles": ["Author"],
      "publication_year": 1651,
      "edition_info": [
        {
          "year": 1651,
          "edition_number": "1st",
          "location": "Paris",
          "publisher": "Pierre David"
        },
        {
          "year": 1652,
          "edition_number": "2nd",
          "location": "Paris",
          "publisher": "Pierre David"
        }
      ]
    },
    ...
  ]
}
```

### Example 2: Batch Processing Multiple Address Books

**Scenario**: Process 20 historical address books from 1850-1900.

**Steps**:
1. Place all files in input directory
2. Generate line ranges for all files:
   ```bash
   python main/generate_line_ranges.py
   # Select folder with all address books
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
   # Check periodically
   python main/check_batches.py
   ```

6. Retrieve results (once completed):
   ```bash
   python main/check_batches.py
   # Outputs generated in configured output directory
   ```

### Example 3: Custom Schema for Legal Documents

**Scenario**: Extract structured data from historical court records.

**Steps**:
1. Create schema (`schemas/legal_records.json`):
   ```json
   {
     "name": "LegalRecords",
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
               "case_number": {"type": "string"},
               "date": {"type": "string", "format": "date"},
               "plaintiff": {"type": "string"},
               "defendant": {"type": "string"},
               "court": {"type": "string"},
               "verdict": {"type": "string"}
             },
             "required": ["case_number", "date", "court"],
             "additionalProperties": false
           }
         }
       },
       "required": ["entries"],
       "additionalProperties": false
     }
   }
   ```

2. Add context (`additional_context/LegalRecords.txt`):
   ```
   Historical legal records from 19th-century courts.
   - Dates may be in format "DD Month YYYY"
   - Court names often include location
   - Verdicts may use archaic legal terminology
   ```

3. Configure paths in `paths_config.yaml`
4. Run extraction as usual

## Contributing

We welcome contributions to ChronoMiner! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the Repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/ChronoMiner.git
   cd ChronoMiner
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

3. **Make Your Changes**
   - Follow existing code style and conventions
   - Add comments for complex logic
   - Update documentation if needed

4. **Test Your Changes**
   ```bash
   # Test with example files
   python main/process_text_files.py
   
   # Verify outputs are correct
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Provide a clear description of your changes

### Contribution Ideas

**New Features**:
- Additional output formats (XML, Excel, etc.)
- GUI interface for non-technical users
- Support for more LLM providers (Anthropic, Google, etc.)
- Enhanced error recovery mechanisms
- Automated quality validation

**Improvements**:
- Performance optimizations
- Better error messages
- Additional schema examples
- More comprehensive tests
- Documentation translations

**Bug Fixes**:
- Report bugs via [GitHub Issues](https://github.com/Paullllllllllllllllll/ChronoMiner/issues)
- Include logs, configuration, and steps to reproduce

### Code Style Guidelines

- **Python**: Follow PEP 8 style guide
- **Naming**: Use descriptive variable and function names
- **Comments**: Explain "why" not "what"
- **Documentation**: Update README for user-facing changes
- **Type Hints**: Use type hints where appropriate

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with debug logging
# Edit modules/core/logger.py to set level to DEBUG

# Test with example files
python main/process_text_files.py
```

## License

ChronoMiner is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Paul Goetz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See [LICENSE](LICENSE) file for full details.

## Contact & Support

### Getting Help

- **Documentation**: Read this README thoroughly
- **GitHub Issues**: [Report bugs or request features](https://github.com/Paullllllllllllllllll/ChronoMiner/issues)
- **Discussions**: Share ideas and ask questions in [GitHub Discussions](https://github.com/Paullllllllllllllllll/ChronoMiner/discussions)

### Project Maintainer

**Paul Goetz**
- GitHub: [@Paullllllllllllllllll](https://github.com/Paullllllllllllllllll)
- Project: [ChronoMiner](https://github.com/Paullllllllllllllllll/ChronoMiner)

### Citation

If you use ChronoMiner in your research, please cite:

```bibtex
@software{chronominer2025,
  author = {Goetz, Paul},
  title = {ChronoMiner: Schema-Based Structured Data Extraction for Historical Texts},
  year = {2025},
  url = {https://github.com/Paullllllllllllllllll/ChronoMiner},
  version = {1.0}
}
```

### Acknowledgments

ChronoMiner is built with:
- **OpenAI API** for LLM-powered extraction
- **tiktoken** for accurate token counting
- **pandas** for data manipulation
- **python-docx** for document generation
- And many other excellent open-source libraries

Special thanks to the digital humanities and social sciences research communities for inspiring this project.

## Star History

If you find ChronoMiner useful, please consider giving it a star on GitHub!

<div align="center">

**Made for researchers, historians, and social scientists**

[Home](https://github.com/Paullllllllllllllllll/ChronoMiner) · [Docs](#table-of-contents) · [Issues](https://github.com/Paullllllllllllllllll/ChronoMiner/issues) · [Contribute](#contributing)

</div>

```
{{ ... }}
# Refine with semantic boundaries
python main/line_range_readjuster.py

# Use in production - select "line_ranges.txt" when prompted for chunking method
{{ ... }}
```
becomes
```
{{ ... }}
# Refine with semantic boundaries
python main/line_range_readjuster.py

# Use in production
{{ ... }}
