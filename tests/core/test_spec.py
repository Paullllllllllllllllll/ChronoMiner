"""Tests for spec.py: code defaults, resolution rules, conditions, suggestions.

Round trips of the option tables live in ``test_spec_roundtrip.py``, schema
resolution in ``test_schema_resolution.py``.
"""

from __future__ import annotations

import errno
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chronominer.providers import infer_provider
from chronominer.spec import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_MODEL,
    MODEL_SUGGESTIONS,
    OPTIONS,
    BatchCancelSpec,
    ImageSpec,
    RunSpec,
    SpecError,
    effective_spec,
    inapplicable,
    model_suggestions,
    parse,
    parse_batch_cancel,
    parse_batch_check,
    parse_ranges_adjust,
    parse_repair,
    recommended_images,
    resolve_batch_cancel,
    resolve_ranges_adjust,
    resolve_repair,
    resolve_spec,
    to_argv,
)

SCHEMA = ["--schema", "WageRecords"]


def _code_defaults(path: str = "in.pdf") -> RunSpec:
    return resolve_spec({"input": Path(path), "schema": "WageRecords"}).spec


def test_the_effective_spec_names_the_source_of_every_option() -> None:
    defaults = OPTIONS.check_defaults(
        {"model": "gpt-5.6-terra", "concurrency": 40, "formats": ["csv"]}
    )
    flags = {
        "input": Path("x"),
        "schema": "WageRecords",
        "concurrency": 8,
        "layout": "mirror",
    }

    resolution = resolve_spec(flags, defaults)

    payload = effective_spec(resolution.spec, resolution.sources)
    assert payload["run.concurrency"] == {"value": 8, "source": "flag"}
    assert payload["model.model"] == {"value": "gpt-5.6-terra", "source": "settings"}
    assert payload["input.path"] == {"value": "x", "source": "flag"}
    assert payload["output.formats"] == {"value": ["csv"], "source": "settings"}
    assert payload["chunking.chunk_tokens"]["source"] == "default"
    assert len(payload) == len([o for o in OPTIONS if o.target is not None])


def test_to_argv_spells_out_values_the_settings_defaults_change() -> None:
    spec = _code_defaults()

    base = ["--input", "in.pdf", "--schema", "WageRecords"]

    def argv(**defaults: Any) -> list[str]:
        words = to_argv(spec, OPTIONS.check_defaults(defaults))
        assert words[:2] == base[:2] and words.count("WageRecords") == 1
        index = words.index("--schema")
        return words[2:index] + words[index + 2 :]

    assert to_argv(spec) == base
    assert argv(output="out") == ["--beside-input"]
    assert argv(context="ctx.txt") == ["--context", "auto"]
    assert argv(context_image="auto") == ["--context-image", "none"]
    assert argv(formats=["csv"]) == ["--formats", ""]
    assert argv(model="gpt-5.6-terra") == ["--model", DEFAULT_MODEL]
    assert argv(chunk_tokens=900) == ["--chunk-tokens", str(DEFAULT_CHUNK_TOKENS)]
    assert argv(endpoint="local") == ["--provider", "openai"]
    assert argv(batch=True) == ["--no-batch"]
    assert argv(provider="openai", concurrency=16, context="auto") == []


def test_an_automatic_context_image_is_spelled_auto() -> None:
    spec = parse(["--input", "x", *SCHEMA, "--context-image", "auto"])
    assert spec.prompt.context_image is None
    assert to_argv(spec)[-2:] == ["--context-image", "auto"]
    assert parse(to_argv(spec)) == spec


def test_relative_context_files_resolve_to_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "topics.txt").write_text("wages", encoding="utf-8")
    (tmp_path / "page.png").write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    spec = parse(["--input", "x", *SCHEMA, "--context", "topics.txt"])
    fallback = parse(["--input", "x", *SCHEMA, "--fallback-context", "topics.txt"])
    image = parse(["--input", "x", *SCHEMA, "--context-image", "page.png"])

    topics = str((tmp_path / "topics.txt").resolve())
    assert effective_spec(spec)["prompt.context"]["value"] == topics
    assert fallback.prompt.fallback_context == topics
    assert image.prompt.context_image == (tmp_path / "page.png").resolve()
    assert parse(["--input", "x", *SCHEMA, "--context", "wages"]).prompt.context == (
        "wages"
    )


def test_context_text_that_cannot_be_a_path_stays_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unknown_user, too_long = "~1850 Paris", "wage ledgers " * 30 + "1790s"
    expanduser, is_file = Path.expanduser, Path.is_file

    def fake_expanduser(path: Path) -> Path:
        if str(path) == unknown_user:
            raise RuntimeError("Could not determine home directory.")
        return expanduser(path)

    def fake_is_file(path: Path) -> bool:
        if str(path) == too_long:
            raise OSError(errno.ENAMETOOLONG, "File name too long")
        return is_file(path)

    monkeypatch.setattr(Path, "expanduser", fake_expanduser)
    monkeypatch.setattr(Path, "is_file", fake_is_file)

    for text in (unknown_user, too_long):
        spec = parse(["--input", "x", *SCHEMA, "--context", text])
        fallback = parse(["--input", "x", *SCHEMA, "--fallback-context", text])
        assert spec.prompt.context == text
        assert fallback.prompt.fallback_context == text


@pytest.mark.parametrize(
    ("defaults", "option"),
    [
        ({"max_output_tokens": 100}, "max_output_tokens"),
        ({"response_format": "json"}, "response_format"),
        ({"dpi": 300}, "dpi"),
        ({"grayscale": False}, "grayscale"),
    ],
)
def test_to_argv_rejects_a_default_without_an_unset_spelling(
    defaults: dict[str, Any], option: str
) -> None:
    with pytest.raises(SpecError, match=option):
        to_argv(_code_defaults(), OPTIONS.check_defaults(defaults))


def test_custom_provider_without_endpoint_is_an_error() -> None:
    with pytest.raises(SpecError, match="endpoint"):
        resolve_spec({"input": Path("x"), "schema": "s", "provider": "custom"})
    with pytest.raises(SpecError, match="endpoint"):
        resolve_repair({"provider": "custom"})
    with pytest.raises(SpecError, match="endpoint"):
        resolve_ranges_adjust({"input": Path("x"), "schema": "s", "provider": "custom"})


@pytest.mark.parametrize(
    ("flags", "defaults"),
    [
        ({"provider": "openai", "model": "anthropic:claude-x"}, {}),
        ({"model": "google:gemini-y"}, {"provider": "custom", "endpoint": "e"}),
        ({"provider": "openai", "model": "anthropic:claude-x"}, {"model": "gpt-5.5"}),
    ],
)
def test_a_provider_prefix_contradicting_the_provider_is_an_error(
    flags: dict[str, Any], defaults: dict[str, Any]
) -> None:
    with pytest.raises(SpecError, match="names provider"):
        resolve_spec(
            {"input": Path("x"), "schema": "s", **flags},
            OPTIONS.check_defaults(defaults),
        )


def test_a_provider_prefix_selects_the_provider() -> None:
    spec = parse(["--input", "x", *SCHEMA, "--model", "anthropic:claude-x"])
    assert (spec.model.provider, spec.model.model) == ("anthropic", "claude-x")
    defaults = OPTIONS.check_defaults({"provider": "openai"})
    spec = parse(["--input", "x", *SCHEMA, "--model", "claude-x"], defaults)
    assert spec.model.provider == "anthropic"


def test_a_dotted_anthropic_alias_resolves_to_its_api_id() -> None:
    spec = parse(["--input", "x", *SCHEMA, "--model", "claude-haiku-5.5"])
    assert (spec.model.provider, spec.model.model) == ("anthropic", "claude-haiku-5-5")
    assert parse(to_argv(spec)) == spec
    defaults = OPTIONS.check_defaults({"model": "claude-sonnet-5.5"})
    resolved = resolve_spec({"input": Path("x"), "schema": "s"}, defaults)
    assert resolved.spec.model.model == "claude-sonnet-5-5"
    repair = parse_repair(["--model", "anthropic:claude-opus-4.8"])
    assert repair.model.model == "claude-opus-4-8"
    routed = parse(["--input", "x", *SCHEMA, "--model", "anthropic/claude-sonnet-4.5"])
    assert routed.model.model == "anthropic/claude-sonnet-4.5"


def test_missing_input_is_an_error() -> None:
    with pytest.raises(SpecError, match="input"):
        resolve_spec({"schema": "s"})
    with pytest.raises(SpecError, match="input"):
        resolve_ranges_adjust({"schema": "s"})


@pytest.mark.parametrize(
    "argv",
    [
        ["--input", "x", *SCHEMA, "--formats", "txt"],
        ["--input", "x", *SCHEMA, "--formats", "json"],
        ["--input", "x", *SCHEMA, "--temperature", "2.5"],
        ["--input", "x", *SCHEMA, "--top-p", "nan"],
        ["--input", "x", *SCHEMA, "--jpeg-quality", "0"],
        ["--input", "x", *SCHEMA, "--dpi", "-3"],
        ["--input", "x", *SCHEMA, "--pages", "0"],
        ["--input", "x", *SCHEMA, "--pages", "7-3"],
        ["--input", "x", *SCHEMA, "--type", "mixed"],
        ["--input", "x", *SCHEMA, "--chunking", "auto-adjust"],
        ["--input", "x", *SCHEMA, "--chunk-tokens", "0"],
        ["--input", "x", *SCHEMA, "--context-window", "0"],
        ["--input", "x", *SCHEMA, "--verbosity", "concise"],
        ["--input", "x", *SCHEMA, "--layout", "hash"],
        ["--input", "x", *SCHEMA, "--method", "llm"],
        ["--input", "x", *SCHEMA, "--input-type", "pdf"],
        ["--input", "x", *SCHEMA, "--type", "image"],
        ["--input", "x", *SCHEMA, "--all", "--select", "1"],
        ["--input", "x", *SCHEMA, "--select", "1", "--files", "a.txt"],
    ],
)
def test_invalid_arguments_exit_with_status_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        parse(argv)
    assert info.value.code == 2


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [
                "--provider",
                "anthropic",
                "--model",
                "claude-x",
                "--service-tier",
                "flex",
            ],
            ["service_tier"],
        ),
        (["--provider", "openai", "--service-tier", "flex"], []),
        (["--context-window", "9"], ["context_window"]),
        (["--chunking", "adjust", "--context-window", "9"], []),
        (["--batch", "--files", "a.pdf", "--recursive"], []),
    ],
)
def test_condition_rules(argv: list[str], expected: list[str]) -> None:
    flags = OPTIONS.parse(["--input", "x", *SCHEMA, *argv])
    spec = resolve_spec(flags).spec
    assert inapplicable(spec, flags) == expected


@pytest.mark.parametrize("value", ["-1", "a", "5-2", ","])
def test_invalid_repair_indices_exit_with_status_2(value: str) -> None:
    with pytest.raises(SystemExit) as info:
        parse_repair(["--input", "out", "--indices", value])
    assert info.value.code == 2


def test_batch_check_keeps_the_run_formats_and_refuses_other_formats() -> None:
    check = parse_batch_check(["--input", "out", "--schema", "WageRecords"])
    assert (check.directory, check.schema) == (Path("out"), "WageRecords")
    assert check.formats is None
    defaults = OPTIONS.check_defaults({"formats": ["csv"]})
    assert parse_batch_check([], defaults).formats is None
    assert parse_batch_check(["--formats", "csv"], defaults).formats == ("csv",)
    for name in ("json", "sqlite", "docx"):
        with pytest.raises(SystemExit) as info:
            parse_batch_check(["--formats", name])
        assert info.value.code == 2


def test_batch_cancel_needs_ids_or_all_or_an_input() -> None:
    cancel = parse_batch_cancel(["--batch-ids", "b1,b2"])
    assert cancel == BatchCancelSpec(("b1", "b2"), False, False)
    assert parse_batch_cancel(["--all"]).all
    assert parse_batch_cancel(["--input", "out"]).directory == Path("out")
    with pytest.raises(SpecError, match="--batch-ids IDS or --all"):
        resolve_batch_cancel({})
    with pytest.raises(SystemExit):
        parse_batch_cancel(["--all", "--batch-ids", "b1"])


def test_ranges_adjust_needs_a_schema_and_reads_its_own_context() -> None:
    with pytest.raises(SpecError, match="schema is required"):
        parse_ranges_adjust(["--input", "t.txt"])
    defaults = OPTIONS.check_defaults({"context": "extract.txt", "context_window": 9})
    spec = parse_ranges_adjust(["--input", "t.txt", *SCHEMA, "--force"], defaults)
    assert spec.context is None
    assert spec.context_window == 9
    assert spec.run.force
    assert spec.model.model == DEFAULT_MODEL
    pages = parse_ranges_adjust(["--input", "t.txt", *SCHEMA, "--pages", "last:3"])
    assert pages.pages == "last:3"


def test_model_suggestions_imply_their_provider() -> None:
    for provider, names in MODEL_SUGGESTIONS.items():
        assert all(infer_provider(name) == provider for name in names)
    assert model_suggestions("openai")[0] == DEFAULT_MODEL
    assert model_suggestions("custom") == ()
    suggest = OPTIONS["model"].choice_provider
    assert OPTIONS["model"].choices is None
    assert suggest is not None
    assert tuple(suggest()) == model_suggestions()


def test_recommended_images_fill_only_the_unset_fields() -> None:
    recommended = recommended_images("openai")
    filled = ImageSpec(jpeg_quality=70).over(recommended)
    assert filled == replace(recommended, jpeg_quality=70)
    assert recommended_images("unknown") == recommended_images("custom")
    assert recommended == ImageSpec("native", "original", "jpeg", 95, False)


def test_to_argv_spells_only_changed_values() -> None:
    spec = _code_defaults()
    tiered = replace(spec, model=replace(spec.model, service_tier="priority"))
    assert to_argv(tiered) == [
        "--input",
        "in.pdf",
        "--schema",
        "WageRecords",
        "--service-tier",
        "priority",
    ]
