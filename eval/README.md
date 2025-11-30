# ChronoMiner Extraction Evaluation Framework

This directory contains the evaluation framework for measuring structured data extraction quality across multiple LLM providers and models.

## Overview

The evaluation computes **precision**, **recall**, and **F1 scores** at both entry and field levels by comparing model outputs against manually corrected ground truth extractions.

This framework mirrors the transcription evaluation in ChronoTranscriber (`ChronoTranscriber/eval/`), using the same dataset categories and model lineup for consistent end-to-end pipeline assessment.

## Models Evaluated

| Provider | Model | Reasoning Level |
|----------|-------|-----------------|
| OpenAI | GPT-5.1 | Medium |
| OpenAI | GPT-5 Mini | Medium |
| Google | Gemini 3.0 Pro | Medium |
| Google | Gemini 2.5 Flash | None |
| Anthropic | Claude Sonnet 4.5 | Medium |
| Anthropic | Claude Haiku 4.5 | Medium |

## Dataset Categories

| Category | Schema | Description |
|----------|--------|-------------|
| `address_books` | HistoricalAddressBookEntries | Swiss address book pages (Basel 1900) |
| `bibliography` | BibliographicEntries | European culinary bibliographies |
| `military_records` | BrazilianMilitaryRecords | Brazilian military enlistment cards |

## Directory Structure

```
eval/
├── README.md                    # This file
├── eval_config.yaml             # Configuration for models and paths
├── metrics.py                   # Precision/recall/F1 computation
├── extraction_eval.ipynb        # Main evaluation notebook
├── test_data/
│   ├── input/                   # Transcribed text files (from ChronoTranscriber ground truth)
│   │   ├── address_books/       # .txt files
│   │   ├── bibliography/        # .txt files
│   │   └── military_records/    # .txt files
│   ├── output/                  # Model extraction outputs
│   │   └── {category}/
│   │       └── {model_name}/    # e.g., gpt_5.1_medium/
│   └── ground_truth/            # Manually corrected JSON extractions
│       ├── address_books/
│       ├── bibliography/
│       └── military_records/
└── reports/                     # Generated evaluation reports
```

## Workflow

### Step 0: Link Transcription Ground Truth

Copy or symlink the corrected transcription files from ChronoTranscriber's ground truth as input:

```powershell
# PowerShell (Windows)
Copy-Item -Path "..\ChronoTranscriber\eval\test_data\ground_truth\address_books\*.txt" `
          -Destination "test_data\input\address_books\"
Copy-Item -Path "..\ChronoTranscriber\eval\test_data\ground_truth\bibliography\*.txt" `
          -Destination "test_data\input\bibliography\"
Copy-Item -Path "..\ChronoTranscriber\eval\test_data\ground_truth\military_records\*.txt" `
          -Destination "test_data\input\military_records\"
```

```bash
# Bash (Linux/macOS)
cp ../ChronoTranscriber/eval/test_data/ground_truth/address_books/*.txt test_data/input/address_books/
cp ../ChronoTranscriber/eval/test_data/ground_truth/bibliography/*.txt test_data/input/bibliography/
cp ../ChronoTranscriber/eval/test_data/ground_truth/military_records/*.txt test_data/input/military_records/
```

This ensures the extraction evaluation starts from flawless transcriptions, isolating extraction quality from transcription errors.

### Step 1: Create Ground Truth Extractions

1. Run extraction using a high-quality model (e.g., GPT-5.1 with high reasoning):
   ```bash
   cd ..
   python main/process_text_files.py --input eval/test_data/input/bibliography \
       --schema BibliographicEntries --chunking auto
   ```

2. Copy outputs to ground truth draft folder:
   ```bash
   mkdir -p eval/test_data/ground_truth/bibliography
   cp output/bibliography/*.json eval/test_data/ground_truth/bibliography/
   ```

3. **Manually review and correct** each JSON file:
   - Verify all entries are correctly extracted
   - Fix any missing or incorrect field values
   - Ensure consistent formatting

4. The corrected files become your ground truth for evaluation.

### Step 2: Run Model Extractions

For each model, run extractions and save to the appropriate output directory:

```bash
# Example for Gemini 2.5 Flash
python main/process_text_files.py --input eval/test_data/input/bibliography \
    --schema BibliographicEntries --model gemini-2.5-flash \
    --output eval/test_data/output/bibliography/gemini_2.5_flash
```

Repeat for each model and category combination.

### Step 3: Run Evaluation

Open and run `extraction_eval.ipynb` in Jupyter:

```bash
cd eval
jupyter notebook extraction_eval.ipynb
```

The notebook will:
- Discover available outputs and ground truth
- Compute precision, recall, and F1 for each model/category combination
- Generate summary tables and rankings
- Export results to `reports/` in JSON, CSV, and Markdown formats

## Metrics

### Entry-Level Metrics

Measures how many complete entries (records) were correctly identified:

- **Entry Precision** = Matched Entries / Hypothesis Entries
- **Entry Recall** = Matched Entries / Ground Truth Entries  
- **Entry F1** = Harmonic mean of precision and recall

Entry matching uses fuzzy string comparison on key fields (configurable).

### Field-Level Metrics

Measures extraction accuracy for individual fields within matched entries:

- **Field Precision** = TP / (TP + FP) per field
- **Field Recall** = TP / (TP + FN) per field
- **Field F1** = Harmonic mean per field

Aggregation methods:
- **Micro-averaged**: Sum all TP/FP/FN across fields, then compute metrics
- **Macro-averaged**: Compute per-field metrics, then average

### Fuzzy Matching

Historical texts often contain spelling variations. The evaluation uses Levenshtein distance with configurable threshold (default: 0.85) to match field values that are similar but not identical.

## Configuration

Edit `eval_config.yaml` to:

- Add or remove models from evaluation
- Change dataset paths
- Adjust similarity threshold for fuzzy matching
- Configure which fields to evaluate per schema
- Set output format preferences

### Schema Fields Configuration

For each schema, specify which fields to evaluate:

```yaml
schema_fields:
  BibliographicEntries:
    - "full_title"
    - "short_title"
    - "authors"
    - "edition_info.year"
```

Nested fields use dot notation (e.g., `edition_info.year`).

## Output Files

After running the evaluation:

| File | Description |
|------|-------------|
| `eval_results_*.json` | Full metrics with all details |
| `eval_results_*.csv` | Tabular format for spreadsheets |
| `eval_results_*.md` | Markdown summary for documentation |

## Ground Truth JSON Format

Ground truth files should match the schema structure. Example for BibliographicEntries:

```json
{
  "entries": [
    {
      "full_title": "The Art of French Cooking",
      "short_title": "French Cooking",
      "authors": ["Julia Child", "Simone Beck"],
      "edition_info": {
        "year": 1961,
        "language": "English"
      },
      "culinary_focus": ["General", "Regional Cuisine"]
    }
  ]
}
```

## Integration with ChronoTranscriber Evaluation

This framework is designed to work in tandem with ChronoTranscriber's transcription evaluation:

1. **ChronoTranscriber eval** produces CER/WER metrics for transcription quality
2. Ground truth transcriptions become inputs for ChronoMiner
3. **ChronoMiner eval** produces P/R/F1 metrics for extraction quality

Together, they provide end-to-end pipeline quality assessment using the same test datasets.

## Dependencies

Uses standard ChronoMiner dependencies plus:
- `pyyaml` - Configuration loading
- `matplotlib` (optional) - Visualization

These should already be installed via ChronoMiner's `requirements.txt`.

## Example Usage

1. Open `extraction_eval.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially
3. Review the generated summary tables and field-level breakdowns
4. Check `reports/` for exported results in JSON, CSV, and Markdown formats
