# ChronoMiner Test Suite

## Overview

This directory contains the `pytest`-based test suite for ChronoMiner, covering unit tests, integration tests, and contract tests for batch backends.

## Setup

Install test dependencies:

```bash
.venv\Scripts\python.exe -m pip install pytest pytest-asyncio
```

## Running Tests

### Run all tests
```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

### Run only unit tests
```bash
.venv\Scripts\python.exe -m pytest tests/ -v -m unit
```

### Run only integration tests
```bash
.venv\Scripts\python.exe -m pytest tests/ -v -m integration
```

### Run a specific test file
```bash
.venv\Scripts\python.exe -m pytest tests/test_config_loader.py -v
```

## Test Structure

### Fixtures (`conftest.py`)
- **`repo_root`**: Path to repository root
- **`tmp_config_dir`**: Temporary config directory with mock YAML configs
- **`config_loader`**: Configured `ConfigLoader` instance for testing
- **Auto-reset fixtures**: Token tracker, config cache, and context cache are reset between tests

### Unit Tests (`@pytest.mark.unit`)
- `test_config_loader.py` - Configuration loading and caching
- `test_config_manager.py` - Configuration validation
- `test_path_utils.py` - Safe path handling for Windows
- `test_schema_manager.py` - Schema loading and retrieval
- `test_context_manager.py` - Context file loading
- `test_prompt_context.py` - Basic and file-specific context resolution
- `test_prompt_utils.py` - Prompt rendering with schema injection
- `test_chunking_service.py` - Text chunking strategies
- `test_args_parser.py` - CLI argument parsing utilities
- `test_workflow_utils.py` - File discovery and filtering
- `test_token_tracker.py` - Token usage tracking and persistence
- `test_generate_line_ranges.py` - Line range generation

### Integration Tests (`@pytest.mark.integration`)
- `test_file_processor_offline.py` - End-to-end file processing without LLM calls

### Contract Tests
- `test_batch_backends.py` - Multi-provider batch backend interfaces (OpenAI, Anthropic, Google)

## Key Features

### No API Keys Required
All tests use mocks and fixtures. No external API calls are made. The `conftest.py` ensures:
- Temporary config directories are created per test
- Config cache is managed automatically
- Token tracker state is isolated

### Portable Configuration
Tests do not rely on local configuration files. All configs are generated in `tmp_path` fixtures.

### Fast Execution
Unit tests complete in ~2 seconds. Integration tests use dummy processing strategies to avoid LLM latency.

## Known Issues

Four tests in `test_batch_backends.py` require fixes to mock patch targets (these are pre-existing tests, not new ones):
- `TestOpenAIBackend.test_submit_batch`
- `TestOpenAIBackend.test_get_status`
- `TestAnthropicBackend.test_get_status_in_progress`
- `TestGoogleBackend.test_get_status_completed`

These need patches at the call site rather than module-level imports.

## CI Integration

To run in CI without installing system dependencies:

```yaml
steps:
  - uses: actions/setup-python@v4
    with:
      python-version: '3.11'
  - run: pip install -r requirements.txt
  - run: pip install pytest pytest-asyncio
  - run: pytest tests/ -v --tb=short
```

## Adding New Tests

1. Place test files in `tests/` with `test_*.py` naming
2. Use `@pytest.mark.unit` or `@pytest.mark.integration` markers
3. Use `config_loader` fixture for config-dependent tests
4. Use `tmp_path` for file I/O tests
5. Mock external API calls (OpenAI, Anthropic, Google)
