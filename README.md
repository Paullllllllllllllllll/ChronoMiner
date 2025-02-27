# ChronoMiner

ChronoMiner is a Python-based project designed for the extraction of structured data from different types of `.txt` 
input files. It comes with example input files (European culinary bibliographies, Brazilian military records, and 
Swiss address books) and recommended JSON schemas for their extraction. The repository can be easily adjusted 
to support the processing of various primary and secondary sources of interest to historians and social scientists. 
It provides users with multiple processing options to ensure well-structured output in different file formats.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [System Requirements & Dependencies](#system-requirements--dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow and Processing Options](#workflow-and-processing-options)
- [Troubleshooting & FAQs](#troubleshooting--faqs)
- [Logging and Debugging](#logging-and-debugging)
- [Security Considerations](#security-considerations)
- [Performance & Limitations](#performance--limitations)
- [Extending and Customizing](#extending-and-customizing)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Contact and Support](#contact-and-support)
- [License](#license)

## Overview

ChronoMiner processes large historical or academic texts by:

- **Splitting Text into Chunks:**  
  Uses token-based chunking (with options for automatic or manually adjusted chunk boundaries) to divide text files 
  into manageable pieces. The effective context window of current-generation large language models (such as OpenAI's 
  o3-mini) is limited, which necessitates chunking.

- **API-Based Data Extraction:**  
  Constructs API requests using OpenAI's API with schema-specific JSON payloads and developer messages. Both 
  synchronous and batch processing modes are supported.

- **Enhanced Extraction with Additional Context:**  
  Optionally integrates schema-specific or file-specific additional context to improve extraction quality and accuracy,
  particularly useful for specialized documents or domain-specific extraction tasks.

- **Dynamic Post-Processing:**  
  Responses are post-processed using a centralized schema handler registry that dynamically invokes schema-specific 
  converters for output generation.

- **Multi-Format Output Generation:**  
  The final structured data is output in JSON format, with optional additional formats—CSV, DOCX, and TXT—generated 
  as configured.

## Repository Structure

```
ChronoMiner/
├── config/
│   ├── chunking_and_context.yaml      # Chunking strategy settings and context configuration
│   ├── concurrency_config.yaml       # Concurrency limits for asynchronous processing
│   ├── model_config.yaml             # OpenAI model settings (name, tokens, reasoning effort)
│   └── paths_config.yaml             # Global and schema-specific I/O paths and output flags
├── developer_messages/               # Exemplary developer messages matching the exemplary .json schemas
│   ├── BibliographicEntries.txt
│   ├── BrazilianMilitaryRecords.txt
│   ├── HistoricalAddressBookEntries.txt
│   └── StructuredSummaries.txt
├── main/
│   ├── cancel_batches.py             # Script to cancel ongoing batch jobs
│   ├── check_batches.py              # Script to process batch responses and generate final outputs
│   ├── generate_line_ranges.py       # Generates token-based _line_ranges.txt files
│   └── process_text_files.py         # Main script to process text files using schema-based extraction
├── modules/
│   ├── batching.py                   # Batch request file creation and submission logic
│   ├── config_loader.py              # Loads and validates YAML configuration files
│   ├── concurrency.py                # Asynchronous task processing with concurrency limits
│   ├── context_manager.py            # Manages additional context for improved extraction
│   ├── data_processing.py            # CSV conversion routines (schema-specific converters included)
│   ├── logger.py                     # Logger configuration for consistent logging across modules
│   ├── openai_utils.py               # OpenAI API wrapper and asynchronous request handling
│   ├── schema_manager.py             # Loads and manages JSON schemas and developer messages
│   ├── schema_handlers.py            # Central registry and base class for schema-specific processing
│   ├── text_processing.py            # DOCX and TXT conversion routines (schema-specific converters included)
│   ├── text_utils.py                 # Text normalization, encoding detection, token estimation, and chunking
│   └── user_interface.py             # Functions for interactive user prompts and selections
└── schemas/
    ├── address_schema.json           # JSON schema for Swiss Historical Address Book Entries
    ├── bibliographic_schema.json     # JSON schema for European Culinary Bibliography Entries
    ├── military_record_schema.json   # JSON schema for Brazilian Military Records
    └── summary_schema.json           # JSON schema for Structured Summaries of Academic Texts
```

## System Requirements & Dependencies

- **Python Version:**  
  Will work with Python 3.12 or later.

- **Further Dependencies:**
  - A full list of dependencies can be found in `requirements.txt`.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Paullllllllllllllllll/ChronoMiner.git
   cd ChronoMiner
   ```

2. **Set Up the Environment:**
   Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Linux/Mac:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Adjust PYTHONPATH:**
   To ensure that Python can find the `modules` package, update your PYTHONPATH. For example, on Linux/Mac:
   ```bash
   export PYTHONPATH="$PYTHONPATH:$(pwd)/modules"
   ```
   On Windows (PowerShell):
   ```powershell
   $env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)\modules"
   ```
   Alternatively, you can add the above commands to your shell profile or run them each time before running the project.

5. **Set Environment Variables:**
   ```bash
   export OPENAI_API_KEY=your_openai_api_key  # On Linux/Mac
   set OPENAI_API_KEY=your_openai_api_key     # On Windows
   ```

6. **Configure File Paths:**  
   ChronoMiner supports both absolute and relative file paths. You can:
   - Use absolute paths for specific environments (default)
   - Enable relative paths for more portable configurations by setting `allow_relative_paths: true` in `paths_config.yaml`

   For relative paths, set up your configuration like this:
   ```yaml
   general:
     allow_relative_paths: true
     base_directory: "."  # All relative paths will be resolved from this directory
   ```

## Usage

### Processing Text Files

Run the main extraction script to process text files according to the selected schema:

```bash
python main/process_text_files.py
```

- **Interactive Prompts:**  
  The script will prompt you to select a schema, choose a processing mode (single file with manual adjustments or 
  bulk processing), and select the chunking strategy.

- **Additional Context:**  
  You can optionally provide additional context to improve extraction quality:
  - Schema-specific context files in the `additional_context` directory
  - File-specific context with a corresponding `{filename}_context.txt` file

- **Output:**  
  The final structured JSON output is saved along with optional CSV, DOCX, or TXT outputs (as configured in 
  `paths_config.yaml`).

### Generating Line Ranges

Generate or adjust token-based line ranges for chunking:

```bash
python main/generate_line_ranges.py
```

Follow the on-screen instructions to select a schema and generate a `_line_ranges.txt` file for manual adjustment 
if needed.

### Batch Processing

Use the following script to check and process batch jobs, retrieve missing responses, and generate final outputs:

```bash
python main/check_batches.py
```

## Workflow and Processing Options

### 1. Text Pre-Processing

- **Encoding Detection & Normalization:**  
  Each input `.txt` file is read using its detected encoding and normalized by stripping whitespace.

- **Token-Based Chunking:**  
  The text is split into chunks based on a token count limit defined in `chunking_and_context.yaml`.  
  **Processing Options:**  
  - **Automatic:** The file is divided automatically.
  - **Automatic with Manual Adjustments:** Users can interactively adjust chunk boundaries.
  - **Pre-Defined Line Ranges:** If a `_line_ranges.txt` file is available, it is used to determine chunk boundaries.
    - line_ranges.txt files can be generated for the folders defined for each schema in `paths_config.yaml`. This 
      allows for the preparation of chunking in advance if large amounts of `.txt` files have to be processed and 
      automatic chunking runs the risk of splitting semantic units.

### 2. Additional Context Integration

- **Schema-Specific Context:**  
  Provides domain knowledge relevant to a specific schema type, improving extraction accuracy.

- **File-Specific Context:**  
  Allows custom context for individual files through `{filename}_context.txt` files.

- **Context Integration:**  
  Additional context is prepended to the text chunks before processing, giving the model more information 
  to guide extraction decisions.

### 3. API Request Construction and Data Extraction

- **Schema-Specific Payloads:**  
  The system uses a schema handler registry (implemented in `modules/schema_handlers.py`) to prepare API request 
  payloads. Each handler returns a JSON payload based on the selected schema and its corresponding developer message.

- **Processing Modes:**  
  - **Synchronous:** Each chunk is processed individually with immediate API calls. The processed chunks are written 
    to temporary JSONL files as they arrive and then processed further.
  - **Batch:** Chunks are written into a temporary JSONL file and submitted as a batch for asynchronous processing.

### 4. Post-Processing and Output Generation

- **Response Processing:**  
  Responses are post-processed via the appropriate schema handler, which also handles error checking and JSON 
  conversion.

- **Output Formats:**  
  Based on settings in `paths_config.yaml`, final outputs are generated in:
  - JSON (always produced)
  - CSV (via schema-specific converters in `modules/data_processing.py`)
  - DOCX and TXT (via schema-specific converters in `modules/text_processing.py`)

- **Batch Output Checking:**  
  The script `main/check_batches.py` retrieves and aggregates responses from batch jobs and dynamically invokes the 
  correct output converters using the schema handler registry.

### 5. Introducing New Schemas

ChronoMiner allows easy integration of new extraction schemas. Follow these steps to add one:

#### 5.1. Create the JSON Schema
Place a new schema file in `schemas/` (e.g., `new_schema.json`). The schema must follow this format:

```json
{
  "name": "NewSchema",
  "schema_version": "1.0",
  "schema": {
    "type": "object",
    "properties": {
      "entries": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": { "type": "string" },
            "title": { "type": "string" },
            "author": { "type": "string" },
            "date": { "type": "string", "format": "date" },
            "content": { "type": "string" }
          },
          "required": ["id", "title", "content"]
        }
      }
    }
  }
}
```
With an entries array, where each entry ideally contains one unit of analysis. For further information refer to 
OpenAI's documentation on structured outputs.

#### 5.2. Add a Developer Message
Create a corresponding prompt file in `developer_messages/` (e.g., `NewSchema.txt`) with clear extraction instructions:

```
You are a structured data extraction expert. Extract structured data and return JSON in the following format:
- `id`: Unique identifier.
- `title`: Document title.
- `author`: Author name.
- `date`: Publication date (YYYY-MM-DD).
- `content`: Main text.
Ensure the JSON strictly follows the schema.
```

#### 5.3. Register the Schema
Add the schema to `SCHEMA_REGISTRY` in `modules/schema_manager.py`:
```python
SCHEMA_REGISTRY["NewSchema"] = "schemas/new_schema.json"
```

#### 5.4. (Optional) Implement a Custom Handler
If special post-processing is needed, create a class in `modules/schema_handlers.py`:
```python
from modules.schema_handlers import BaseSchemaHandler, register_schema_handler

class NewSchemaHandler(BaseSchemaHandler):
    schema_name = "NewSchema"

    def process(self, extracted_data):
        if "date" in extracted_data:
            extracted_data["date"] = self.normalize_date(extracted_data["date"])
        return extracted_data

    def normalize_date(self, date_str):
        from datetime import datetime
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").isoformat()
        except ValueError:
            return None

register_schema_handler("NewSchema", NewSchemaHandler)
```

#### 5.5. (Optional) Add Schema-Specific Context
Create a file in the `additional_context` directory with the same name as your schema:
```
# additional_context/NewSchema.txt
This text provides additional context to guide extraction for NewSchema documents.
It can include domain knowledge, specific terminology, or extraction guidelines.
```

#### 5.6. Test the Schema
Modify `paths_config.yaml` for your schema's custom I/O paths.

Run:
```bash
python main/process_text_files.py
```
Select "NewSchema" and verify the JSON output.

Once completed, your schema is fully integrated and ready for use.

## Troubleshooting & FAQs

- **Common Issues:**
  - **Encoding Errors:**  
    Verify that the input file is in a supported encoding. Check the log files for detailed error messages.
  - **API Rate Limits/Errors:**  
    Ensure your OpenAI API key is valid and that you are not exceeding the API usage limits.
  - **Configuration Errors:**  
    Confirm that all required keys are present in the YAML configuration files. Logs may provide hints if a 
    configuration file is missing or misconfigured.
  - **Path Resolution Issues:**  
    If using relative paths, ensure `allow_relative_paths` is set to `true` in `paths_config.yaml` and that 
    `base_directory` is correctly set.

- **Where to Find Logs:**  
  Log files are stored in the directory specified by `logs_dir` in `paths_config.yaml` (e.g., `logs/application.log`).

## Logging and Debugging

- **Log Files:**  
  All log messages are written to a file specified in `paths_config.yaml`.

- **Adjusting Log Level:**  
  To increase verbosity, modify the logging level in `modules/logger.py` as required for more detailed output during 
  troubleshooting.

## Security Considerations

- **API Key Management:**  
  Do not hard-code your OpenAI API key. Use environment variables or secure secret management to protect your 
  credentials. This repository assumes your API key is stored in an environment variable.
- **Sensitive Data Handling:**  
  Handle any sensitive input data according to relevant data protection guidelines.

## Performance & Limitations

- **API Limitations:**  
  The system is dependent on OpenAI API rate limits. Ensure you adjust the settings in concurrency_config.yaml to correspond
  to your OpenAI API tier.
- **Context Window Utilization:**  
  The token-based chunking strategy can be adjusted to better utilize the context window of newer models like o3-mini,
  which supports up to 200,000 tokens.

## Extending and Customizing

- **Adding New Schemas:**  
  Refer to the "Introducing New Schemas" section above for detailed steps on creating new JSON schemas and developer 
  message files.
- **Customizing Chunking and Post-Processing:**  
  The project can be extended by modifying modules such as `modules/text_utils.py` or 
  `modules/schema_handlers.py`. Inline comments in these modules provide guidance.
- **Supporting Additional Output Formats:**  
  You can add new converters in `modules/data_processing.py` and `modules/text_processing.py` to generate other 
  output formats like XML or HTML.
- **Enhancing Extraction Context:**  
  Create custom context files to improve extraction for specific domains or document types.

## Future Enhancements

- **Additional File Format Support:**  
  Extending support to formats such as XML or HTML.
- **Improved Error Handling:**  
  Enhancing recovery from API errors and ensuring more robust processing of partially structured data.
- **User Interface Improvements:**  
  Developing a graphical user interface (GUI) for non-technical users.
- **Enhanced Context Integration:**  
  More sophisticated context integration options, potentially with vector database support for relevant document retrieval.

## Contributing

Contributions are welcome! When contributing:
- Contact the main developer before adding any new features.
- Ensure that any new schema JSON files and developer messages work as intended before submission.
- (Optionally) Register any custom schema handlers in `modules/schema_handlers.py`.
- Follow the repository's coding style and contribution guidelines.

## Contact and Support

- **Main Developer:**  
  For support, questions, or to discuss contributions, please open an issue on GitHub or contact via email at 
  [paul.goetz@uni-goettingen.de](mailto:paul.goetz@uni-goettingen.de).

## License

This project is licensed under the MIT License.
