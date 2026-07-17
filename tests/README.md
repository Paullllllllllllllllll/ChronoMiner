# ChronoMiner Test Suite

`pytest`-based suite of about 1,200 unit, integration, and contract tests
covering all `modules/` packages, the LLM provider layer, the multi-provider
batch backends, and the `main/` CLI parsers. No API keys are required: all
tests run offline against mocks and fixtures, except the opt-in `live`
marker.

## Setup

```bash
uv sync --all-extras
```

## Running Tests

```bash
# Full suite
uv run pytest -v

# By marker
uv run pytest -m unit
uv run pytest -m integration

# A single file
uv run pytest tests/test_config_loader.py -v
```

Markers (declared in `pytest.ini`): `unit`, `integration`, `slow`, and
`live`. Tests marked `live` make real LLM API calls and run only when
`CHRONOMINER_LIVE_TESTS=1` is set.

## Fixtures (`conftest.py`)

- `repo_root` -- path to the repository root.
- `tmp_config_dir` -- temporary config directory with mock YAML configs.
- `config_loader` -- configured `ConfigLoader` instance.
- Auto-reset fixtures isolate token-tracker state and the config and
  context caches between tests.

Tests never rely on the local `config/*.yaml` files; all configuration is
generated under `tmp_path` fixtures.

## Adding New Tests

1. Place test files in `tests/` with `test_*.py` naming.
2. Mark them `@pytest.mark.unit` or `@pytest.mark.integration` (and `live`
   if they call a provider API).
3. Use the `config_loader` fixture for config-dependent tests and
   `tmp_path` for file I/O.
4. Mock external API calls (OpenAI, Anthropic, Google, OpenRouter).
