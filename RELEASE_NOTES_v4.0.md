# ChronoMiner v4.0 / v4.1 Release Notes

## Release History

| Version | Release Date | Status |
|---------|-------------|--------|
| 4.0 | March 2026 | Production Ready |
| 4.1 | March 2026 | Production Ready |

## Overview

This is a two-part release. v4.0 introduces the native visual input pipeline, enabling direct
image/PDF → LLM → JSON extraction without a prior transcription step. Image and PDF files
are preprocessed per provider, base64-encoded, and passed to vision-capable models alongside
the extraction schema — eliminating the dependency on ChronoTranscriber for visual corpora.
The release also removes a substantial body of legacy dead-code (cost analysis, fine-tuning,
earlier release notes). v4.1 extends the visual pipeline to batch mode across all three
providers (OpenAI, Anthropic, Google), bringing the cost savings and throughput of
asynchronous batch jobs to image and PDF corpora.

## Major Features

### 1. Visual Input Pipeline

ChronoMiner now processes image and PDF inputs natively, routing them through vision-capable
LLMs and returning the same structured JSON output as text workflows.

**Supported input formats:**
- Images: JPEG, PNG, GIF, BMP, WebP, TIFF
- Documents: PDF (rendered page-by-page via PyMuPDF)

**Routing logic:**

`FileProcessor._is_visual_input()` detects the input extension and dispatches to
`_process_visual_file()`. Directory inputs are scanned with `rglob("*")` and all recognized
image/PDF extensions are collected.

**Processing pipeline:**

```
file → preprocessing (provider-specific resize / DPI render)
     → base64 encode
     → process_image_chunk()
     → vision LLM (via LangChain)
     → structured JSON
```

**Multimodal message assembly** is handled by the new module
`modules/llm/image_message_builder.py`, which builds provider-appropriate content blocks for
LangChain `HumanMessage` objects:

| Provider | Block format |
|----------|-------------|
| OpenAI / OpenRouter | `image_url` with base64 data URI |
| Anthropic | `image` source block (`base64` type) |
| Google | `inline_data` block |

**New modules:**

| Module | Purpose |
|--------|---------|
| `modules/processing/image_utils.py` | Image resizing and base64 encoding per provider |
| `modules/processing/pdf_utils.py` | PDF rendering to per-page images (PyMuPDF) |
| `modules/llm/image_message_builder.py` | Multimodal LangChain message assembly |

**New configuration:** `config/image_processing_config.yaml` stores per-provider defaults for
detail level and DPI, loaded lazily via `get_image_processing_config()` on `ConfigLoader`.

### 2. New CLI Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--image-detail` | `low`, `high`, `auto`, `original` | Detail level passed to vision API; overrides config default |
| `--input-type` | `text`, `image`, `pdf` | Override auto-detected input type |

Auto-detection is handled by `detect_input_type()` in `modules/cli/args_parser.py`, which
inspects the input path extension and infers the appropriate input type without user
intervention.

### 3. Interactive Mode Visual Support

Interactive mode detects visual inputs automatically when the schema's configured input
directory contains image or PDF files.

**Behavioral changes in interactive mode for visual inputs:**
- Chunking step is skipped (one request per image / one request per PDF page)
- A new `ask_image_detail()` prompt in `modules/ui/core.py` surfaces the detail-level choice
- `select_input_source()` accepts an `input_type` parameter so the file list shows image and
  PDF files instead of text files

### 4. New Prompt Template

`prompts/visual_extraction_prompt.txt` provides a dedicated system prompt for visual
extraction tasks, replacing the text-oriented default prompt when a visual input is detected.

### 5. Evaluation Framework Reorganization

`main/prepare_extraction_ground_truth.py` has been relocated to
`eval/prepare_ground_truth.py` to consolidate evaluation tooling under a dedicated directory.

### 6. Removal of Deprecated Code

The following modules and directories were removed in v4.0 as they had been dead-code for
multiple releases:

| Removed item | Notes |
|---|---|
| `fine_tuning/` (10 files) | Legacy fine-tuning experiments |
| `gimmicks/` (2 files) | Unreferenced utility scripts |
| `main/cost_analysis.py` | Superseded by provider-native cost reporting |
| `modules/operations/cost_analysis.py` | — |
| `modules/ui/cost_display.py` | — |
| `tests/test_cost_analysis.py` | — |
| `RELEASE_NOTES_v2.0.md` | Historical notes removed to reduce repository clutter |

### 7. Visual Batch Processing (v4.1)

v4.1 extends the visual pipeline to all three batch providers, removing the synchronous-only
restriction that shipped with v4.0.

**`BatchRequest` extensions** (`modules/llm/batch/backends/base.py`):

Three optional fields added to the dataclass — `image_base64`, `mime_type`, and
`image_detail` — with an `is_visual` property that returns `True` when `image_base64` is
populated. Existing text-only workflows are unaffected.

**Provider-specific routing in `submit_batch()`:**

| Backend | Routing |
|---|---|
| OpenAI (`openai_backend.py`) | New `_build_image_responses_body()` helper; `submit_batch()` branches on `req.is_visual` |
| Anthropic (`anthropic_backend.py`) | Visual requests use `image` source-block format |
| Google (`google_backend.py`) | Visual requests use `inline_data` block format |

**Processing strategy changes** (`modules/core/processing_strategy.py`):
- `BatchProcessingStrategy.process_chunks()` builds visual `BatchRequest` objects with
  `-page-N` suffixed custom IDs when `image_chunks` are provided
- The `ValueError` that previously rejected visual inputs in batch mode has been removed

**Other changes:**
- `_process_visual_file()`: sync-only guard removed; batch path now reachable
- Interactive mode: forced-sync fallback for visual inputs removed
- `check_batches.py`: `_extract_chunk_index()` regex extended to match `-page-N` custom IDs

## Technical Improvements

### New and Updated Files

| File | Status | Purpose |
|---|---|---|
| `modules/processing/image_utils.py` | New | Image preprocessing (resize, encode) per provider |
| `modules/processing/pdf_utils.py` | New | PDF → per-page image rendering (PyMuPDF) |
| `modules/llm/image_message_builder.py` | New | Multimodal LangChain message assembly |
| `config/image_processing_config.yaml` | New | Per-provider detail/DPI defaults |
| `prompts/visual_extraction_prompt.txt` | New | Visual extraction system prompt |
| `eval/prepare_ground_truth.py` | New (relocated) | Ground-truth preparation script |
| `modules/cli/args_parser.py` | Updated | `--image-detail`, `--input-type`, `detect_input_type()` |
| `modules/ui/core.py` | Updated | `ask_image_detail()` |
| `modules/operations/extraction/file_processor.py` | Updated | Visual routing, `_process_visual_file()` |
| `modules/llm/langchain_provider.py` | Updated | Multimodal `HumanMessage` content lists |
| `modules/llm/batch/backends/base.py` | Updated | `BatchRequest` visual fields (v4.1) |
| `modules/llm/batch/backends/openai_backend.py` | Updated | Visual batch routing (v4.1) |
| `modules/llm/batch/backends/anthropic_backend.py` | Updated | Visual batch routing (v4.1) |
| `modules/llm/batch/backends/google_backend.py` | Updated | Visual batch routing (v4.1) |
| `modules/core/processing_strategy.py` | Updated | Visual `BatchRequest` construction (v4.1) |
| `main/check_batches.py` | Updated | `-page-N` custom-ID regex (v4.1) |

### New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `Pillow` | ≥10.0.0 | Image resizing and format conversion |
| `PyMuPDF` | ≥1.24.0 | PDF rendering to per-page images |

### Test Suite

The test suite grew from 681 tests (pre-v4.0) to 742 tests across both releases:

| Release | Tests added | New test files |
|---|---|---|
| v4.0 | +61 | `test_constants`, `test_image_utils`, `test_pdf_utils`, `test_image_message_builder`, `test_input_type_detection`, `test_langchain_multimodal`, `test_file_processor_images` |
| v4.1 | +10 | Extended `test_batch_backends`, `test_processing_strategy` |

## Migration Guide

### From v3.0 to v4.0 / v4.1

There are no breaking changes to text workflows. The visual pipeline is purely additive:
existing configurations, schemas, and CLI invocations continue to work without modification.

**New dependencies must be installed:**

```bash
pip install -r requirements.txt
```

**To use the visual pipeline**, point `--input` at an image file, a PDF, or a directory of
images. Input type is auto-detected; override with `--input-type` if needed.

**To use visual batch processing** (v4.1), add `--batch` as you would for text inputs:

```bash
python main/process_text_files.py --schema HistoricalRecipesEntries \
  --input path/to/document.pdf --batch --image-detail low
```

**Removed scripts:** if your workflow called `main/cost_analysis.py` or
`main/prepare_extraction_ground_truth.py`, update references to the replacements listed in
the "Removal of Deprecated Code" and "Evaluation Framework Reorganization" sections above.
