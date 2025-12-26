---
description: Testing Workflow for the ChronoMiner repository
---

# Multi-Provider LLM Testing Procedure for ChronoMiner Repo

## Initial Analysis

Before running any tests, analyze the ChronoMiner repository to understand its architecture:

1. Read README.md and any `SESSION_CHANGES_*.md` files to understand recent changes  
2. Examine `main/process_text_files.py` to understand the entry point and execution flow  
3. Review `config/paths_config.yaml` to understand path configuration and schema settings  
4. Review `config/model_config.yaml` to understand LLM model configuration structure  
5. Examine `modules/llm/model_capabilities.py` to understand supported providers and capabilities  
6. Examine `modules/cli/args_parser.py` to understand CLI argument structure  
7. Identify and verify the required test folder and files  

Test folder (persistent):  
C:\Users\pagoetz\Desktop\working_dir\test_folders\test_folder_chrono_miner

Required test files:
- `test_file_1.txt`  
- `test_file_2.txt`  
- `Attar_short.txt`  
- `Attar_short_line_ranges.txt`  
- `Attar_short_context.txt`

---

## Environment Setup

All commands must use the project's virtual environment:

```
$env:PYTHONPATH="C:\Users\pagoetz\PycharmProjects\ChronoMiner"
.venv\Scripts\python.exe main/process_text_files.py [args]
```

Required environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

---

## Test Execution

### Phase 1: CLI Mode Tests

**Step 1.0**: Configure paths  
- Edit `config/paths_config.yaml`:
  - Set `BibliographicEntries.input` to the test folder
  - Set `BibliographicEntries.output` to an output subfolder inside the test folder
  - Set `StructuredSummaries.input` and `StructuredSummaries.output` similarly (required for Gemini tests)

---

### Provider 1: OpenAI (gpt-5-mini)

**Step 1.1**: Configure OpenAI provider  
- Edit `config/model_config.yaml`:
  - `name: gpt-5-mini`
  - `max_output_tokens: 32000`
  - `reasoning.effort: low`

**Step 1.2**: Test single file with auto chunking
```
.venv\Scripts\python.exe main/process_text_files.py --schema BibliographicEntries --input "C:\Users\pagoetz\Desktop\working_dir\test_folders\test_folder_chrono_miner\test_file_1.txt" --chunking auto --verbose
```

**Step 1.3**: Test single file with context
```
.venv\Scripts\python.exe main/process_text_files.py --schema BibliographicEntries --input "C:\Users\pagoetz\Desktop\working_dir\test_folders\test_folder_chrono_miner\test_file_1.txt" --chunking auto --context --context-source file --verbose
```

**Step 1.4**: Test with line_ranges chunking
```
.venv\Scripts\python.exe main/process_text_files.py --schema BibliographicEntries --input "C:\Users\pagoetz\Desktop\working_dir\test_folders\test_folder_chrono_miner\Attar_short.txt" --chunking line_ranges --context --context-source file --verbose
```

---

### Provider 2: Anthropic (claude-haiku-4-5-20251101)

**Step 1.5**: Configure Anthropic provider  
- Edit `config/model_config.yaml`:
  - `name: claude-haiku-4-5-20251101`

**Step 1.6**: Test single file
```
.venv\Scripts\python.exe main/process_text_files.py --schema BibliographicEntries --input "C:\Users\pagoetz\Desktop\working_dir\test_folders\test_folder_chrono_miner\test_file_1.txt" --chunking auto --verbose
```

**Step 1.7**: Test with line_ranges and context
```
.venv\Scripts\python.exe main/process_text_files.py --schema BibliographicEntries --input "C:\Users\pagoetz\Desktop\working_dir\test_folders\test_folder_chrono_miner\Attar_short.txt" --chunking line_ranges --context --context-source file --verbose
```

---

### Provider 3: Google (gemini-2.5-flash)

**Step 1.8**: Configure Google provider  
- Edit `config/model_config.yaml`:
  - `name: gemini-2.5-flash`

**Step 1.9**: Test with StructuredSummaries schema
```
.venv\Scripts\python.exe main/process_text_files.py --schema StructuredSummaries --input "C:\Users\pagoetz\Desktop\working_dir\test_folders\test_folder_chrono_miner\test_file_1.txt" --chunking auto --verbose
```

---

### CLI Multi-File Test

**Step 1.11**: Test folder processing  
```
.venv\Scripts\python.exe main/process_text_files.py --schema BibliographicEntries --input "C:\Users\pagoetz\Desktop\working_dir\test_folders\test_folder_chrono_miner" --chunking auto --verbose
```

---

## Phase 2: Interactive Mode Tests

**Step 2.1**: Launch interactive mode  
- Ensure `interactive_mode: true` in paths config or run without arguments:

```
$env:PYTHONIOENCODING="utf-8"; chcp 65001; .venv\Scripts\python.exe main/process_text_files.py
```

**Step 2.2**: Check UI  
- Banner should display: "ChronoMiner - Structured Data Extraction Tool"  
- Schema selection menu appears  
- Press `q` to exit

---

## Phase 3: Output Verification

**Step 3.1**: Verify JSON structure
- Must include `custom_id`, `chunk_index`, `response`  
- `response.output_text` must contain schema-valid extracted data

**Step 3.2**: Optional outputs (if enabled)
- CSV  
- DOCX  
- TXT  

---

## Phase 4: Cleanup

**Step 4.1**: Restore defaults  
- Reset model in `model_config.yaml` (e.g., back to `gpt-5-mini`)  
- Ensure appropriate `max_output_tokens` value  
- Disable interactive mode if required  

---

## Required Verifications Summary

1. Exit code 0  
2. No authentication or model errors  
3. Output JSON file generated  
4. Output contains valid extracted data  
5. All chunks processed for line_ranges  
6. No token limit errors  

---

## Results Table Format

| Provider | Model | Single File | With Context | Line Ranges | Interactive |
|----------|-------|:-----------:|:------------:|:-----------:|:-----------:|
| OpenAI | gpt-5-mini | PASS/FAIL | PASS/FAIL | PASS/FAIL | PASS/FAIL |
| Anthropic | claude-haiku-4-5-20251101 | PASS/FAIL | PASS/FAIL | PASS/FAIL | PASS/FAIL |
| Google | gemini-2.5-flash | PASS/FAIL* | N/A | N/A | PASS/FAIL |

*Gemini requires `StructuredSummaries` schema

---

## Known Provider-Specific Issues

- **Gemini**: `BibliographicEntries` schema too deeply nested  
- **Gemini**: Schema nesting error  
- **GPT-5-mini**: Needs large token budget (`32000`)  
- **Claude**: Harmless function-format warnings  
- **All providers**: Require correct environment variables

---

## Troubleshooting

Common issues:

- Missing schema → verify schema name in `paths_config.yaml`  
- Missing line ranges file → add `<filename>_line_ranges.txt`  
- Missing context file → add `<filename>_context.txt`  
- Auth errors → verify keys  
- Model not found → check spelling in `model_capabilities.py`  
- Token errors → increase `max_output_tokens`  
- Unicode errors → set encoding and run `chcp 65001`