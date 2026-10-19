"""A provider set above a model of another provider exits 2.

The rule applies to every command that calls a model: extract, ranges
adjust and repair. Equal precedence levels keep the provider as
given; a provider below the model's level yields to the provider the model
name implies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chronominer import __main__ as entry
from chronominer.spec import (
    OPTIONS,
    SpecError,
    parse,
    resolve_ranges_adjust,
    resolve_repair,
    resolve_spec,
    to_argv,
)
from tests.cli.conftest import json_lines

SCHEMA = "price_list_entries"


def _settings(tmp_path: Path, defaults: str = "{}") -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(f"defaults: {defaults}\n", encoding="utf-8")
    return path


def _main(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    defaults: str,
    *argv: str,
    input_name: str = "missing.txt",
) -> tuple[int, dict[str, Any], str]:
    settings = _settings(tmp_path, defaults)
    code = entry.main(
        [
            "extract",
            "--schema",
            SCHEMA,
            "--input",
            str(tmp_path / input_name),
            *argv,
            "--settings",
            str(settings),
            "--dry-run",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return code, payload, captured.err


@pytest.mark.parametrize(
    ("defaults", "model"),
    [("{}", "gpt-6-luna"), ("{model: gpt-5.6-terra}", "gpt-5.6-terra")],
    ids=["code default", "settings default"],
)
def test_a_flag_provider_above_the_model_exits_2(
    defaults: str,
    model: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, payload, err = _main(capsys, tmp_path, defaults, "--provider", "anthropic")

    assert code == 2
    assert payload["dry_run"] is True
    assert "--model" in err
    assert "anthropic" in err and model in err


def test_a_settings_provider_above_the_code_default_model_is_an_error() -> None:
    defaults = OPTIONS.check_defaults({"provider": "google"})
    with pytest.raises(SpecError, match="--model"):
        resolve_spec({"input": Path("x"), "schema": SCHEMA}, defaults)


def test_repair_and_adjust_apply_the_rule() -> None:
    with pytest.raises(SpecError, match="--model"):
        resolve_repair({"provider": "anthropic"})
    with pytest.raises(SpecError, match="--model"):
        resolve_ranges_adjust(
            {"input": Path("t.txt"), "schema": SCHEMA, "provider": "anthropic"}
        )
    spec = resolve_repair({"provider": "anthropic", "model": "claude-x"}).spec
    assert (spec.model.provider, spec.model.model) == ("anthropic", "claude-x")


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        (["repair"], []),
        (["ranges", "adjust"], ["--input", "t.txt", "--schema", SCHEMA]),
    ],
    ids=["repair", "adjust"],
)
def test_the_rule_exits_2_on_every_command_that_calls_a_model(
    command: list[str],
    argv: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings(tmp_path)
    full = [*command, *argv, "--provider", "anthropic"]

    assert entry.main([*full, "--settings", str(settings), "--json"]) == 2

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert "--model" in captured.err


@pytest.mark.parametrize(
    ("flags", "defaults"),
    [
        ({"provider": "openrouter", "model": "openai/gpt-5.6-luna"}, {}),
        ({}, {"provider": "openrouter", "model": "openai/gpt-5.6-luna"}),
        ({"provider": "anthropic", "model": "claude-x"}, {}),
        ({"provider": "anthropic"}, {"model": "my-local-model"}),
    ],
    ids=["flags", "settings", "matching model", "model of no provider"],
)
def test_equal_levels_and_matching_models_pass(
    flags: dict[str, Any], defaults: dict[str, Any]
) -> None:
    checked = OPTIONS.check_defaults(defaults)
    resolution = resolve_spec({"input": Path("x"), "schema": SCHEMA, **flags}, checked)
    expected = flags.get("provider") or defaults["provider"]
    assert resolution.spec.model.provider == expected


def test_a_provider_below_the_model_still_yields() -> None:
    defaults = OPTIONS.check_defaults({"provider": "anthropic"})
    spec = parse(
        ["--input", "x", "--schema", SCHEMA, "--model", "gpt-5.6-terra"], defaults
    )
    assert spec.model.provider == "openai"
    assert parse(to_argv(spec, defaults), defaults) == spec


def test_a_flag_provider_with_a_flag_model_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    (tmp_path / "list.txt").write_text("Bread, 2 lb, 4 d\n", encoding="utf-8")

    code, payload, err = _main(
        capsys,
        tmp_path,
        "{}",
        "--provider",
        "anthropic",
        "--model",
        "claude-x",
        input_name="list.txt",
    )

    assert code == 0, err
    assert payload["spec"]["model.provider"]["value"] == "anthropic"
    assert payload["spec"]["model.model"] == {"value": "claude-x", "source": "flag"}


def test_a_custom_endpoint_needs_a_model_at_its_level() -> None:
    base = {"input": Path("x"), "schema": SCHEMA}
    with pytest.raises(SpecError, match="--model"):
        resolve_spec({**base, "endpoint": "local"})
    defaults = OPTIONS.check_defaults({"model": "llama-x"})
    with pytest.raises(SpecError, match="--model"):
        resolve_spec({**base, "endpoint": "local"}, defaults)
    defaults = OPTIONS.check_defaults({"endpoint": "local", "model": "llama-x"})
    spec = resolve_spec(base, defaults).spec
    assert (spec.model.provider, spec.model.model) == ("custom", "llama-x")
