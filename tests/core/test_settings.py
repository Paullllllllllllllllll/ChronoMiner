"""Tests for settings.py: the packaged example, typed values and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer import settings as settings_module
from chronominer.common.options import Source
from chronominer.common.retry import RetryPolicy
from chronominer.settings import (
    EXAMPLE_SETTINGS_PATH,
    Concurrency,
    Endpoint,
    Images,
    PromptCaching,
    RetrySettings,
    Settings,
    SettingsError,
    SpecError,
    Timeouts,
    load,
    resolve,
    resolve_batch_cancel,
    resolve_batch_check,
    resolve_ranges_adjust,
    resolve_ranges_generate,
    resolve_render,
    resolve_repair,
)
from chronominer.spec import OPTIONS


@pytest.fixture
def settings_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the repository settings file and the example into ``tmp_path``."""
    monkeypatch.setattr(
        settings_module, "DEFAULT_SETTINGS_PATH", tmp_path / "settings.yaml"
    )
    monkeypatch.setattr(
        settings_module, "EXAMPLE_SETTINGS_PATH", tmp_path / "settings.example.yaml"
    )
    return tmp_path


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_packaged_example_loads_and_matches_code_values() -> None:
    loaded = load(EXAMPLE_SETTINGS_PATH)
    assert loaded.source == EXAMPLE_SETTINGS_PATH
    assert loaded == Settings(source=EXAMPLE_SETTINGS_PATH)


def test_settings_flag_beats_the_repository_file_and_the_example(
    settings_paths: Path,
) -> None:
    _write(
        settings_paths / "settings.example.yaml",
        "concurrency:\n  image_processing: 5\n",
    )
    assert load().concurrency.image_processing == 5
    _write(settings_paths / "settings.yaml", "concurrency:\n  image_processing: 7\n")
    assert load().concurrency.image_processing == 7
    other = _write(
        settings_paths / "other.yaml", "concurrency:\n  image_processing: 9\n"
    )
    assert load(other).concurrency.image_processing == 9
    assert load(other).source == other


def test_without_any_file_the_code_values_apply(settings_paths: Path) -> None:
    assert load() == Settings()


def test_a_missing_settings_flag_file_is_an_error(settings_paths: Path) -> None:
    with pytest.raises(SettingsError, match="missing.yaml"):
        load(settings_paths / "missing.yaml")


def test_partial_file_keeps_code_values(settings_paths: Path) -> None:
    _write(
        settings_paths / "settings.yaml",
        "timeouts:\n  request: 300\nretry:\n  validation_attempts: 3\n"
        "concurrency:\n  delay_between_tasks: 0\n"
        "prompt_caching:\n  openai:\n    prompt_cache_retention: 24h\n",
    )
    value = load()
    assert value.timeouts == Timeouts(300.0, 10.0, 30.0, 30.0)
    assert value.retry == RetrySettings(8, 3, 3, 0, "auto")
    assert value.concurrency == Concurrency(4, 0.0)
    assert value.prompt_caching == PromptCaching(True, "5m", "24h", True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("images:\n  original_max_side_px: 2500\n", 2500),
        ("images:\n  original_max_side_px: null\n", None),
        ("images: {}\n", None),
    ],
)
def test_original_max_side_setting(
    settings_paths: Path, text: str, expected: int | None
) -> None:
    _write(settings_paths / "settings.yaml", text)
    assert load().images == Images(expected)


def test_paths_resolve_against_the_current_directory(
    settings_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = settings_paths / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    _write(settings_paths / "settings.yaml", "logs_dir: logs\n")
    assert load().logs_dir == work / "logs"
    absolute = settings_paths / "run.log"
    _write(settings_paths / "settings.yaml", f"logs_dir: '{absolute.as_posix()}'\n")
    assert load().log_file() == absolute


def test_the_state_folder_follows_state_dir(settings_paths: Path) -> None:
    _write(settings_paths / "settings.yaml", "state_dir: ''\n")
    assert load().state_folder() == Path.home() / ".chronominer"
    folder = settings_paths / "state"
    _write(settings_paths / "settings.yaml", f"state_dir: '{folder.as_posix()}'\n")
    assert load().state_folder() == folder


def test_endpoints_and_key_variables(settings_paths: Path) -> None:
    _write(
        settings_paths / "settings.yaml",
        "api_keys:\n  openai: MY_OPENAI_KEY\n"
        "endpoints:\n  local:\n    base_url: http://127.0.0.1:8000/v1/\n"
        "    api_key_env: LOCAL_KEY\n    supports_schema: true\n",
    )
    value = load()
    assert value.endpoint("local") == Endpoint(
        "local", "http://127.0.0.1:8000/v1/", "LOCAL_KEY", True, True
    )
    assert value.api_key_env("openai") == "MY_OPENAI_KEY"
    assert value.api_key_env("anthropic") == "ANTHROPIC_API_KEY"
    assert value.api_key_env("custom", "local") == "LOCAL_KEY"
    with pytest.raises(SettingsError, match="unknown endpoint"):
        value.endpoint("remote")
    with pytest.raises(SettingsError):
        value.api_key_env("custom")


@pytest.mark.parametrize(
    "text",
    [
        "unknown: 1\n",
        "api_keys:\n  mistral: MISTRAL_KEY\n",
        "endpoints:\n  local:\n    base_url: http://x/\n    api_key_env: K\n"
        "    model: m\n",
        "timeouts:\n  read: 5\n",
        "retry:\n  attempts: 3\n",
        "concurrency:\n  extraction: 3\n",
        "concurrency:\n  audio: 3\n",
        "tools:\n  tesseract_cmd: x\n",
        "whisper:\n  device: cpu\n",
        "audio_chunking:\n  mono: true\n",
        "audio_api:\n  prompt: x\n",
        "prompt_caching:\n  google: {}\n",
        "prompt_caching:\n  anthropic:\n    ttl_seconds: 300\n",
        "prompt_caching:\n  openai:\n    retention: 24h\n",
        "prompt_caching:\n  openrouter:\n    cache: true\n",
        "images:\n  max_side_px: 3000\n",
        "defaults:\n  modle: gpt-5.6-terra\n",
        "defaults:\n  method: llm\n",
        "defaults:\n  chunk_size: 3000\n",
        "schemas_paths: {}\n",
    ],
)
def test_unknown_keys_at_every_level_are_errors(
    settings_paths: Path, text: str
) -> None:
    _write(settings_paths / "settings.yaml", text)
    with pytest.raises(SettingsError, match="settings.yaml"):
        load()


@pytest.mark.parametrize(
    "text",
    [
        "timeouts:\n  request: fast\n",
        "rate_limits:\n  - [10, 0]\n",
        "rate_limits: []\n",
        "endpoints:\n  local:\n    base_url: http://x/\n",
        "retry:\n  max_attempts: 0\n",
        "retry:\n  page_timeout: soon\n",
        "concurrency:\n  image_processing: 0\n",
        "prompt_caching:\n  anthropic:\n    ttl: 2h\n",
        "logs_dir: 5\n",
        "images:\n  original_max_side_px: 0\n",
        "images:\n  original_max_side_px: wide\n",
        "defaults:\n  dry_run: true\n",
        "defaults:\n  force: true\n",
        "defaults:\n  input: x.pdf\n",
        "defaults:\n  files: [a.txt]\n",
        "defaults:\n  pages: 1-3\n",
        "defaults:\n  beside_input: true\n",
        "defaults:\n  concurrency: 0\n",
        "defaults:\n  chunk_tokens: -5\n",
        "defaults:\n  chunking: per-file\n",
        "defaults:\n  type: mixed\n",
        "defaults:\n  layout: hash\n",
        "defaults:\n  response_format: text\n",
        "defaults:\n  formats: [json]\n",
        "defaults:\n  formats: [csv, docx]\n",
        "defaults:\n  output: out\n  beside_input: true\n",
        "defaults: [model]\n",
    ],
)
def test_invalid_values_are_errors(settings_paths: Path, text: str) -> None:
    _write(settings_paths / "settings.yaml", text)
    with pytest.raises(SettingsError, match="settings.yaml"):
        load()


@pytest.mark.parametrize(
    ("value", "expected", "deadline"),
    [("auto", "auto", 900.0 * 3 + 300.0), ("off", 0.0, None), (False, 0.0, None)],
)
def test_page_timeout_values(
    settings_paths: Path, value: object, expected: object, deadline: float | None
) -> None:
    text = "retry:\n  page_timeout: " + ("false" if value is False else str(value))
    _write(settings_paths / "settings.yaml", text + "\n")
    retry = load().retry
    assert retry.page_timeout == expected
    assert retry.page_deadline(900.0) == deadline
    assert RetrySettings(page_timeout=120.0).page_deadline(900.0) == 120.0


def test_retry_policy() -> None:
    policy = Settings().retry.policy()
    assert isinstance(policy, RetryPolicy)
    assert (policy.max_attempts, policy.validation_attempts) == (8, 1)
    assert (policy.backoff_base, policy.backoff_cap) == (2.0, 120.0)
    assert policy.backoff(0, "timeout", jitter=0.0) == 2.0
    assert policy.backoff(2, "rate_limit", jitter=0.5) == 8.5
    assert policy.backoff(10, "other", jitter=0.0) == 120.0


def test_precedence_flag_then_settings_default_then_code_default(
    settings_paths: Path,
) -> None:
    _write(
        settings_paths / "settings.yaml",
        "defaults:\n  model: gpt-5.6-terra\n  concurrency: 40\n"
        "  verbosity: high\n  formats: [csv]\n  chunk_tokens: 900\n"
        "  context_window: 9\n",
    )
    value = load()
    flags = OPTIONS.parse(
        ["--input", "x.pdf", "--schema", "WageRecords", "--concurrency", "8"]
    )
    resolution = resolve(value, flags)
    spec, sources = resolution.spec, resolution.sources
    assert (spec.run.concurrency, sources["run.concurrency"]) == (8, Source.FLAG)
    assert spec.model.model == "gpt-5.6-terra"
    assert sources["model.model"] is Source.SETTINGS
    assert (spec.model.verbosity, sources["model.verbosity"]) == (
        "high",
        Source.SETTINGS,
    )
    assert spec.output.formats == ("csv",)
    assert spec.chunking.chunk_tokens == 900
    assert (spec.model.reasoning_effort, sources["model.reasoning_effort"]) == (
        "xhigh",
        Source.DEFAULT,
    )
    check = resolve_batch_check(value, {}).spec
    assert check.formats is None
    repair = resolve_repair(value, {}).spec
    assert repair.model.model == "gpt-5.6-terra"
    generate = resolve_ranges_generate(value, {"input": Path("t.txt")}).spec
    assert (generate.chunk_tokens, generate.model) == (900, "gpt-5.6-terra")
    adjust = resolve_ranges_adjust(
        value, {"input": Path("t.txt"), "schema": "WageRecords"}
    ).spec
    assert (adjust.context_window, adjust.model.model) == (9, "gpt-5.6-terra")
    assert adjust.run.concurrency == 40
    assert resolve_render(value, {}).spec.formats is None
    assert resolve_batch_cancel(value, {"all": True}).spec.all


def test_resolve_checks_endpoint_names() -> None:
    value = Settings(
        endpoints={"local": Endpoint("local", "http://127.0.0.1/v1/", "LOCAL_KEY")}
    )
    named = {"input": Path("x"), "schema": "WageRecords", "model": "llama-x"}
    spec = resolve(value, {**named, "endpoint": "local"}).spec
    assert spec.model.provider == "custom"
    with pytest.raises(SettingsError, match="remote"):
        resolve(value, {**named, "endpoint": "remote"})
    with pytest.raises(SettingsError, match="remote"):
        resolve_repair(value, {"model": "llama-x", "endpoint": "remote"})
    with pytest.raises(SettingsError, match="remote"):
        resolve_ranges_adjust(value, {**named, "endpoint": "remote"})
    with pytest.raises(SpecError):
        resolve(value, {"input": Path("x"), "schema": "s", "provider": "custom"})


def test_example_names_key_variables_by_provider_only() -> None:
    text = EXAMPLE_SETTINGS_PATH.read_text(encoding="utf-8")
    variables = {
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.strip().endswith("_API_KEY")
    }
    assert variables == {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "YOUR_ENDPOINT_API_KEY",
    }


def test_example_defaults_block_names_only_known_options() -> None:
    text = EXAMPLE_SETTINGS_PATH.read_text(encoding="utf-8")
    lines = text.split("# defaults:\n", 1)[1].splitlines()
    names = [line.removeprefix("#").split(":", 1)[0].strip() for line in lines]
    assert names
    assert all(name in OPTIONS and OPTIONS[name].settable for name in names)
