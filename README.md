# ChronoMiner

ChronoMiner is a Python-based project designed for the extraction of structured data from different 
types of `.txt` input files. It comes with example input files (a European culinary bibliographies, 
Brazilian military records, and a Swiss address books) and recommended JSON schemas for their 
extraction. The repository can be easily adjusted to support the processing of various primary and secondary sources 
of interest to historians and social scientists. It provides users with multiple processing options to ensure 
well-structured output in different file formats.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Workflow and Processing Options](#workflow-and-processing-options)
- [System Requirements & Dependencies](#system-requirements--dependencies)
- [Installation](#installation)
- [Usage](#usage)
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
  into manageable pieces. The effective context window of current-generation large language models (such as OpenAI's o3-mini)
  is limited, which necessitates chunking.
  
- **API-Based Data Extraction:**  
  Constructs API requests using OpenAI's API with schema-specific JSON payloads and developer messages. Both 
  synchronous and batch processing modes are supported.

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
│   ├── chunking_config.yaml          # Chunking strategy settings (token count per chunk, automatic chunking method)
│   ├── concurrency_config.yaml       # Concurrency limits for asynchronous processing
│   ├── model_config.yaml             # OpenAI model settings (name, tokens, reasoning effort)
│   └── paths_config.yaml             # Global and schema-specific I/O paths and output flags
├── developer_messages/               # Exemplary developer messages matching the examplary .json schemas
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
│   ├── data_processing.py            # CSV conversion routines (schema-specific converters included)
│   ├── logger.py                     # Logger configuration for consistent logging across modules
│   ├── openai_utils.py               # OpenAI API wrapper and asynchronous request handling
│   ├── schema_manager.py             # Loads and manages JSON schemas and developer messages
│   ├── schema_handlers.py            # Central registry and base class for schema-specific processing
│   ├── text_processing.py            # DOCX and TXT conversion routines (schema-specific converters included)
│   └── text_utils.py                 # Text normalization, encoding detection, token estimation, and chunking
└── schemas/
    ├── address_schema.json           # JSON schema for Swiss Historical Address Book Entries
    ├── bibliographic_schema.json     # JSON schema for European Culinary Bibliography Entries
    ├── military_record_schema.json   # JSON schema for Brazilian Military Records
    └── summary_schema.json           # JSON schema for Structured Summaries of Academic Texts
```

## Workflow and Processing Options

### 1. Text Pre-Processing

- **Encoding Detection & Normalization:**  
  Each input `.txt` file is read using its detected encoding and normalized by stripping whitespace.

- **Token-Based Chunking:**  
  The text is split into chunks based on a token count limit defined in `chunking_config.yaml`.  
  **Processing Options:**  
  - **Automatic:** The file is divided automatically.
  - **Automatic with Manual Adjustments:** Users can interactively adjust chunk boundaries.
  - **Pre-Defined Line Ranges:** If a `_line_ranges.txt` file is available, it is used to determine chunk boundaries.
    - line_ranges.txt files can be generated for the folders defined for each schema in paths_config.yaml. This allows for the
      preparation of chunking in advance if large amounts of `.txt` files have to be processed and automatic chunking runs the
      risk of splitting semantic units.

### 2. API Request Construction and Data Extraction

- **Schema-Specific Payloads:**  
  The system uses a schema handler registry (implemented in `modules/schema_handlers.py`) to prepare API request 
  payloads. Each handler returns a JSON payload based on the selected schema and its corresponding developer message.

- **Processing Modes:**  
  - **Synchronous:** Each chunk is processed individually with immediate API calls. The processed chunks are written to temporary
    JSONL files as they arrive and then processed further.
  - **Batch:** Chunks are written into a temporary JSONL file and submitted as a batch for asynchronous processing.

### 3. Post-Processing and Output Generation

- **Response Processing:**  
  Responses are post-processed via the appropriate schema handler, which also handles error checking and JSON conversion.

- **Output Formats:**  
  Based on settings in `paths_config.yaml`, final outputs are generated in:
  - JSON (always produced)
  - CSV (via schema-specific converters in `modules/data_processing.py`)
  - DOCX and TXT (via schema-specific converters in `modules/text_processing.py)

- **Batch Output Checking:**  
  The script `main/check_batches.py` retrieves and aggregates responses from batch jobs and dynamically invokes the 
  correct output converters using the schema handler registry.

### 4. Introducing New Schemas

ChronoMiner allows easy integration of new extraction schemas. Follow these steps to add one:

#### 4.1. Create the JSON Schema  
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

#### 4.2. Add a Developer Message  
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

#### 4.3. Register the Schema  
Add the schema to `SCHEMA_REGISTRY` in `modules/schema_manager.py`:

```python
SCHEMA_REGISTRY["NewSchema"] = "schemas/new_schema.json"
```

#### 4.4. (Optional) Implement a Custom Handler  
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

#### 4.5. Test the Schema  
Modify `paths_config.yaml` for your schemas custom I/O paths.

Run:
```bash
python main/process_text_files.py
```
Select "NewSchema" and verify the JSON output.

Once completed, your schema is fully integrated and ready for use.
```

## System Requirements & Dependencies

- **Python Version:**  
  Requires Python 3.8 or later.

- **Key Dependencies:**
    - aiohttp==3.11.11  
    - openai==1.61.0  
    - PyYAML==6.0.2  
    - tiktoken==0.8.0  
    - python-docx==1.1.2  
    - pandas==2.2.3  
    - pydantic==2.10.6  
    - tenacity==9.0.0  
    - requests==2.32.3  
    - chardet==5.2.0  
    - tqdm==4.67.1

- **Further Dependencies:**
    - A full list of dependencies can be found in requirements.txt.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://ChronoMiner.git
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


## Usage

### Processing Text Files

Run the main extraction script to process text files according to the selected schema:

```bash
python main/process_text_files.py
```

- **Interactive Prompts:**  
  The script will prompt you to select a schema, choose a processing mode (single file with manual adjustments or 
  bulk processing), and select the chunking strategy.

- **Output:**  
  The final structured JSON output is saved along with optional CSV, DOCX, or TXT outputs (as configured in 
  `paths_config.yaml`).

### Generating Line Ranges

Generate or adjust token-based line ranges for chunking:

```bash
python main/generate_line_ranges.py
```

Follow the on-screen instructions to select a schema and generate a `_line_ranges.txt` file for manual adjustment if 
needed.

### Batch Processing

Use the following script to check and process batch jobs, retrieve missing responses, and generate final outputs:

```bash
python main/check_batches.py
```

## Troubleshooting & FAQs

- **Common Issues:**
  - **Encoding Errors:**  
    Verify that the input file is in a supported encoding. Check the log files for detailed error messages.
  - **API Rate Limits/Errors:**  
    Ensure your OpenAI API key is valid and that you are not exceeding the API usage limits.
  - **Configuration Errors:**  
    Confirm that all required keys are present in the YAML configuration files. Logs may provide hints if a 
    configuration file is missing or misconfigured.

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
  Do not hard-code your OpenAI API key. Use environment variables or secure secret management to protect your credentials.
  This repository assumes your API key is stored in an environment variable.
- **Sensitive Data Handling:**  
  Handle any sensitive input data according to relevant data protection guidelines.

## Performance & Limitations

- **API Limitations:**  
  The system is dependent on OpenAI API rate limits and may experience delays when processing very large batches.
- **Resource Usage:**  
  Processing extremely large text files can consume significant memory. Adjust concurrency settings in 
  `concurrency_config.yaml` to optimize performance based on your system's capabilities.

## Extending and Customizing

- **Adding New Schemas:**  
  Refer to the "Introducing New Schemas" section above for detailed steps on creating new JSON schemas and developer 
  message files.
- **Customizing Chunking and Post-Processing:**  
  The project can be extended by modifying modules such as `modules/text_utils.py` or `modules/schema_handlers.py`. 
  Inline comments in these modules provide guidance.
- **Supporting Additional Output Formats:**  
  You can add new converters in `modules/data_processing.py` and `modules/text_processing.py` to generate other 
  output formats like XML or HTML.

## Possible Future Enhancements

- **Additional File Format Support:**  
  Extending support to formats such as XML or HTML.
- **Improved Error Handling:**  
  Enhancing recovery from API errors and ensuring more robust processing of partially structured data.
- **User Interface Improvements:**  
  Developing a graphical user interface (GUI) for non-technical users.

## Contributing

Contributions are welcome! When contributing:
- Contact the main developer before adding any new features.
- Ensure that any new schema JSON files and developer messages work as intended before submission.
- (Optionally) Register any custom schema handlers in `modules/schema_handlers.py`.
- Follow the repository’s coding style and contribution guidelines.

## Contact and Support

- **Main Developer:**  
  For support, questions, or to discuss contributions, please open an issue on GitHub or contact via email at [paul.
  goetz@uni-goettingen.de](mailto:paul.goetz@uni-goettingen.de).

## License

This project is licensed under the MIT License.
