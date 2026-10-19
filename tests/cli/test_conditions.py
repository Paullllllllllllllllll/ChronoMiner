"""Conditional options: warnings when a given option does not apply."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer import __main__ as entry
from chronominer.cli.command import condition_warnings
from chronominer.spec import OPTIONS
from tests.cli.conftest import MakeText
from tests.conftest import make_spec


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry runs check the key variable of the provider they would call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


MODEL = "claude-sonnet-5-5"
"""An Anthropic model, named because a provider flag outranks the default."""


def _dry_run(text: Path, *extra: str) -> list[str]:
    return [
        "extract",
        "--schema",
        "price_list_entries",
        "--input",
        str(text),
        *extra,
        "--dry-run",
        "--json",
    ]


def _warnings(err: str) -> list[str]:
    return [line for line in err.splitlines() if "[WARNING]" in line]


def test_service_tier_with_another_provider_warns_and_runs(
    make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _dry_run(
        make_text(),
        "--provider",
        "anthropic",
        "--model",
        MODEL,
        "--service-tier",
        "priority",
    )

    assert entry.main(argv) == 0

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    (warning,) = _warnings(captured.err)
    assert "--service-tier" in warning
    assert "only for provider openai" in warning
    assert "anthropic" in warning


def test_without_the_conditional_flag_nothing_warns(
    make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _dry_run(make_text(), "--provider", "anthropic", "--model", MODEL)
    assert entry.main(argv) == 0
    assert _warnings(capsys.readouterr().err) == []


def test_context_window_names_the_chunking_mode(
    make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    assert entry.main(_dry_run(make_text(), "--context-window", "9")) == 0

    (warning,) = _warnings(capsys.readouterr().err)
    assert warning.split("] ", 1)[1].startswith(
        "--context-window applies only for chunking adjust"
    )
    assert "the chunking is auto" in warning


def test_chunking_conditions_hold_under_adjust() -> None:
    flags = {"chunking": "adjust", "context_window": 9}
    assert condition_warnings(OPTIONS, flags, {}, make_spec(**flags)) == []


def test_code_defaults_never_warn() -> None:
    flags = {"provider": "anthropic", "model": MODEL}
    assert condition_warnings(OPTIONS, flags, {}, make_spec(**flags)) == []


def test_settings_default_warns_with_its_name() -> None:
    flags = {"provider": "anthropic", "model": MODEL}
    defaults = {"service_tier": "priority"}

    (warning,) = condition_warnings(OPTIONS, flags, defaults, make_spec(**flags))

    assert warning.startswith("the settings default service_tier (--service-tier)")


def test_settings_default_warns_through_the_cli(
    tmp_path: Path, make_text: MakeText, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text("defaults:\n  service_tier: priority\n", encoding="utf-8")
    argv = _dry_run(
        make_text(),
        "--provider",
        "anthropic",
        "--model",
        MODEL,
        "--settings",
        str(settings),
    )

    assert entry.main(argv) == 0

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    (warning,) = _warnings(captured.err)
    assert "settings default service_tier" in warning


def test_flag_wins_over_a_settings_default_in_the_warning() -> None:
    flags = {"provider": "anthropic", "model": MODEL, "service_tier": "auto"}
    defaults = {"service_tier": "priority"}

    (warning,) = condition_warnings(OPTIONS, flags, defaults, make_spec(**flags))

    assert warning.startswith("--service-tier ")
