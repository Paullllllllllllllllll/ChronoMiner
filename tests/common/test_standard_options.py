"""The shared option rows take each tool's values and build the same rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer.common.options import Kind, OptionTable
from chronominer.common.standard_options import (
    MODEL_FIELDS,
    batch_cancel_options,
    batch_option,
    bounded_float,
    classifier_options,
    decision_options,
    dpi_value,
    image_options,
    jpeg_quality,
    model_options,
    nonempty_text,
    positive_int,
    render_options,
)


def _suggestions() -> tuple[str, ...]:
    return ("m1", "m2")


def _model(**extra: object) -> OptionTable:
    return OptionTable(
        model_options(
            providers=("openai", "custom"),
            suggestions=_suggestions,
            defaults={"model": "m1", "top_p": 1.0},
            **extra,  # type: ignore[arg-type]
        )
    )


def test_single_scope_rows_fill_the_model_block() -> None:
    table = _model()
    assert table.names == MODEL_FIELDS
    assert table["model"].default == "m1"
    assert table["model"].target == "model.model"
    assert table["provider"].default_help == "inferred from the model name"
    assert table["top_p"].help.endswith("sent only when supported")
    assert table["service_tier"].condition is not None
    assert table["service_tier"].condition.option == "provider"


def test_phased_rows_fall_back_to_the_unprefixed_row() -> None:
    table = OptionTable(
        model_options(
            providers=("openai",),
            suggestions=_suggestions,
            defaults={"verbosity": "medium"},
            phases=("a", "b"),
            phase_defaults={"b": {"verbosity": "low"}},
        )
    )
    assert table["verbosity"].default is None
    assert table["verbosity"].default_help == "per phase"
    assert table["a_verbosity"].default == "medium"
    assert table["b_verbosity"].default == "low"
    assert table["b_verbosity"].fallback == "verbosity"
    assert table["b_verbosity"].target == "b_model.verbosity"
    assert table["b_service_tier"].condition is not None
    assert table["b_service_tier"].condition.option == "b_provider"
    assert table["top_p"].default_help == "not sent"
    assert table["a_top_p"].help.endswith("sent only when given and supported")


def test_response_format_help_follows_the_choices() -> None:
    two = _model(response_formats=("schema", "json"))["response_format"]
    assert two.help == "schema (API-enforced) or json (prompted, validated)"
    three = _model()["response_format"]
    assert three.help.endswith("json (prompted, validated) or text (plain)")


def test_batch_is_a_settable_switch_or_a_flag() -> None:
    switch = batch_option("submit")
    flag = batch_option("submit", switch=False)
    assert (switch.kind, switch.settable) == (Kind.SWITCH, True)
    assert (flag.kind, flag.settable) == (Kind.FLAG, False)


def test_block_rows_build_valid_tables() -> None:
    OptionTable(image_options())
    cancel = OptionTable(batch_cancel_options(settings_target="settings"))
    assert cancel["settings"].target == "settings"
    render = OptionTable(
        render_options(
            input_help="in", formats=("txt",), dry_run_help="d", json_help="j"
        )
    )
    assert render["input"].default == Path(".")
    assert render["formats"].choices == ("txt",)


def test_classifier_rows_reuse_the_decision_rows_without_a_mode() -> None:
    table = OptionTable(classifier_options(force_help="ask again"))
    decision = OptionTable(decision_options("skip"))

    assert table.names == (
        "classifier",
        "force",
        "decision_model",
        "decision_provider",
        "decision_endpoint",
    )
    assert table["classifier"].default is None
    assert table["classifier"].default_help == "required"
    assert table["classifier"].target == "classify.classifier"
    assert table["force"].target == "classify.force"
    assert "decisions" not in table.names
    for name in ("decision_model", "decision_provider", "decision_endpoint"):
        assert table[name].condition is None
        assert decision[name].condition is not None
        assert table[name].target == f"classify.{name.removeprefix('decision_')}"
        assert table[name].default == decision[name].default


@pytest.mark.parametrize(
    ("convert", "raw", "expected"),
    [
        (positive_int, "3", 3),
        (dpi_value, "Native", "native"),
        (dpi_value, "300", 300),
        (jpeg_quality, "100", 100),
        (bounded_float(0.0, 1.0), "0.5", 0.5),
        (nonempty_text, " x ", " x "),
    ],
)
def test_converters_accept(convert: object, raw: str, expected: object) -> None:
    assert convert(raw) == expected  # type: ignore[operator]


@pytest.mark.parametrize(
    ("convert", "raw"),
    [
        (positive_int, "0"),
        (dpi_value, "-1"),
        (jpeg_quality, "101"),
        (bounded_float(0.0, 1.0), "nan"),
        (nonempty_text, "  "),
    ],
)
def test_converters_reject(convert: object, raw: str) -> None:
    with pytest.raises(ValueError):
        convert(raw)  # type: ignore[operator]
