"""Round trips of every option table: ``parse(to_argv(spec, defaults)) == spec``.

Seeded random specs of the extract, ranges generate, ranges adjust, batch
repair, batch check, batch cancel and render tables, each with a
random settings defaults block and without one. The value pools include
values that begin with ``-``, which ``to_argv`` writes as ``--flag=value``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from chronominer.spec import (
    CHUNKING_MODES,
    CONTEXT_NONE,
    IMAGE_DETAILS,
    IMAGE_FORMATS,
    INPUT_TYPES,
    LAYOUTS,
    OPTIONS,
    OUTPUT_FORMATS,
    PROVIDERS,
    REASONING_EFFORTS,
    RESPONSE_FORMATS,
    SERVICE_TIERS,
    VERBOSITIES,
    BatchCancelSpec,
    BatchCheckSpec,
    ChunkingSpec,
    CommandSpec,
    ImageSpec,
    InputSpec,
    ModelSpec,
    OutputSpec,
    PromptSpec,
    RangesAdjustSpec,
    RangesControl,
    RangesGenerateSpec,
    RenderSpec,
    RepairControl,
    RepairSpec,
    RunControl,
    RunSpec,
    parse,
    parse_batch_cancel,
    parse_batch_check,
    parse_ranges_adjust,
    parse_ranges_generate,
    parse_render,
    parse_repair,
    to_argv,
)

SEEDS = range(200)
_PATHS = [Path("in.pdf"), Path("a b/c.txt"), Path("-in.pdf"), Path(".")]
_CONTEXTS: list[Any] = [None, CONTEXT_NONE, "ctx.txt", "-ctx.txt", "prices; wages"]
_FALLBACKS: list[Any] = [None, "notes.txt", "-notes", "wage ledgers"]
_IMAGES: list[Any] = [None, CONTEXT_NONE, Path("ctx.png").resolve()]
_MODELS = ["gpt-5.6-luna", "gpt-5.5", "claude-x", "gemini-y", "v/m", "-m"]
_SCHEMAS = ["WageRecords", "schemas/trade_records.json", "-s.json"]


def _formats(rng: random.Random) -> tuple[str, ...]:
    return tuple(f for f in OUTPUT_FORMATS if rng.random() < 0.5)


def _names(rng: random.Random) -> tuple[str, ...]:
    return rng.choice([(), ("a.pdf",), ("x y.txt", "-b.png"), ("-a",)])


def _flag(rng: random.Random) -> bool:
    return rng.random() < 0.5


def _model(rng: random.Random) -> ModelSpec:
    provider = rng.choice(PROVIDERS)
    return ModelSpec(
        provider=provider,
        model=rng.choice(_MODELS),
        endpoint=rng.choice(["local", "-remote"]) if provider == "custom" else None,
        reasoning_effort=rng.choice(REASONING_EFFORTS),
        verbosity=rng.choice(VERBOSITIES),
        max_output_tokens=rng.choice([None, 1, 4096]),
        temperature=rng.choice([0.0, 0.7, 1.0]),
        top_p=rng.choice([0.0, 1.0]),
        service_tier=rng.choice(SERVICE_TIERS),
        response_format=rng.choice([None, *RESPONSE_FORMATS]),
    )


def _images(rng: random.Random) -> ImageSpec:
    return ImageSpec(
        dpi=rng.choice([None, "native", 300]),
        detail=rng.choice([None, *IMAGE_DETAILS]),
        format=rng.choice([None, *IMAGE_FORMATS]),
        jpeg_quality=rng.choice([None, 1, 100]),
        grayscale=rng.choice([None, True, False]),
    )


def _extract(rng: random.Random) -> RunSpec:
    selection = rng.choice(["files", "select", "all", "none"])
    return RunSpec(
        input=InputSpec(
            path=rng.choice(_PATHS),
            type=rng.choice(INPUT_TYPES),
            files=_names(rng) if selection == "files" else (),
            recursive=_flag(rng),
            pages=rng.choice([None, "5", "-7", "3-", "1,3,5-8", "last:2"]),
            select=rng.choice(["1,3", "-x", "ledger"])
            if selection == "select"
            else None,
            all=selection == "all",
        ),
        output=OutputSpec(
            directory=rng.choice([None, Path("out"), Path("-o")]),
            layout=rng.choice(LAYOUTS),
            formats=_formats(rng),
        ),
        prompt=PromptSpec(
            schema=rng.choice(_SCHEMAS),
            context=rng.choice(_CONTEXTS),
            context_image=rng.choice(_IMAGES),
            fallback_context=rng.choice(_FALLBACKS),
        ),
        chunking=ChunkingSpec(
            mode=rng.choice(CHUNKING_MODES),
            chunk_tokens=rng.choice([500, 3500]),
            context_window=rng.choice([6, 20]),
        ),
        model=_model(rng),
        images=_images(rng),
        run=RunControl(
            batch=_flag(rng),
            force=_flag(rng),
            concurrency=rng.choice([1, 12]),
            dry_run=_flag(rng),
            json=_flag(rng),
        ),
    )


def _generate(rng: random.Random) -> RangesGenerateSpec:
    return RangesGenerateSpec(
        input=rng.choice(_PATHS),
        recursive=_flag(rng),
        chunk_tokens=rng.choice([500, 3500]),
        model=rng.choice(_MODELS),
        dry_run=_flag(rng),
        json=_flag(rng),
    )


def _adjust(rng: random.Random) -> RangesAdjustSpec:
    return RangesAdjustSpec(
        input=rng.choice(_PATHS),
        recursive=_flag(rng),
        schema=rng.choice(_SCHEMAS),
        context_window=rng.choice([6, 20]),
        context=rng.choice(_CONTEXTS),
        pages=rng.choice([None, "first:3", "2-4"]),
        model=_model(rng),
        run=RangesControl(
            force=_flag(rng),
            concurrency=rng.choice([1, 12]),
            dry_run=_flag(rng),
            json=_flag(rng),
        ),
        fallback_context=rng.choice(_FALLBACKS),
    )


def _repair(rng: random.Random) -> RepairSpec:
    return RepairSpec(
        input=rng.choice(_PATHS),
        schema=rng.choice([None, "WageRecords", "-s"]),
        indices=rng.choice([None, (0,), (3, 4, 5, 9)]),
        model=_model(rng),
        images=_images(rng),
        run=RepairControl(_flag(rng), _flag(rng), _flag(rng)),
    )


def _check(rng: random.Random) -> BatchCheckSpec:
    action = rng.choice([None, "adopt", "abandon"])
    return BatchCheckSpec(
        directory=rng.choice(_PATHS),
        batch_ids=_names(rng),
        formats=rng.choice([None, _formats(rng)]),
        json=_flag(rng),
        adopt=rng.choice(["b1", "-b2"]) if action == "adopt" else None,
        abandon=rng.choice(["s1", "-s2"]) if action == "abandon" else None,
        schema=rng.choice([None, "WageRecords", "-s"]),
    )


def _cancel(rng: random.Random) -> BatchCancelSpec:
    target = rng.choice(["ids", "all", "dir"])
    return BatchCancelSpec(
        batch_ids=rng.choice([("b1",), ("-b1", "b2")]) if target == "ids" else (),
        all=target == "all",
        json=_flag(rng),
        directory=rng.choice(_PATHS) if target == "dir" else None,
    )


def _render(rng: random.Random) -> RenderSpec:
    return RenderSpec(
        input=rng.choice(_PATHS),
        formats=rng.choice([None, _formats(rng)]),
        force=_flag(rng),
        dry_run=_flag(rng),
        json=_flag(rng),
    )


_DEFAULT_VALUES: dict[str, list[Any]] = {
    "type": list(INPUT_TYPES),
    "recursive": [True, False],
    "output": ["out", "-o"],
    "layout": list(LAYOUTS),
    "formats": [[], list(OUTPUT_FORMATS)],
    "schema": ["TradeRecords", "-s.json"],
    "context": ["auto", "none", "-ctx.txt", "prices; wages"],
    "fallback_context": ["none", "notes.txt", "wage ledgers"],
    "context_image": ["auto", "none"],
    "select": ["1-2", "ledger"],
    "chunking": list(CHUNKING_MODES),
    "chunk_tokens": [900],
    "context_window": [12],
    "provider": list(PROVIDERS),
    "model": ["gpt-5.6-terra", "claude-x", "gemini-y", "-m"],
    "endpoint": ["local", "-remote"],
    "reasoning_effort": list(REASONING_EFFORTS),
    "verbosity": list(VERBOSITIES),
    "max_output_tokens": [1000],
    "temperature": [0.3],
    "top_p": [0.5],
    "service_tier": list(SERVICE_TIERS),
    "response_format": list(RESPONSE_FORMATS),
    "dpi": ["native", 200],
    "image_detail": list(IMAGE_DETAILS),
    "image_format": list(IMAGE_FORMATS),
    "jpeg_quality": [80],
    "grayscale": [True, False],
    "batch": [True, False],
    "concurrency": [4],
}

# Settings defaults whose unset value has no spelling, by the field they set.
_UNSPELLABLE = {
    "select": ("input", "select"),
    "dpi": ("images", "dpi"),
    "image_detail": ("images", "detail"),
    "image_format": ("images", "format"),
    "jpeg_quality": ("images", "jpeg_quality"),
    "grayscale": ("images", "grayscale"),
    "max_output_tokens": ("model", "max_output_tokens"),
    "response_format": ("model", "response_format"),
}


def _defaults(rng: random.Random, spec: CommandSpec) -> dict[str, Any]:
    """Return a random checked defaults block that *spec* can undo as flags."""
    keys = rng.sample(sorted(_DEFAULT_VALUES), rng.randint(1, 12))
    raw: dict[str, Any] = {}
    for key in keys:
        part, field = _UNSPELLABLE.get(key, ("", ""))
        holder = getattr(spec, part, None)
        if hasattr(holder, field) and getattr(holder, field) is None:
            continue
        raw[key] = rng.choice(_DEFAULT_VALUES[key])
    return OPTIONS.check_defaults(raw)


type Parser = Callable[[list[str], Mapping[str, Any] | None], CommandSpec]

TABLES: dict[str, tuple[Callable[[random.Random], CommandSpec], Parser]] = {
    "extract": (_extract, lambda argv, defaults: parse(argv, defaults)),
    "ranges generate": (
        _generate,
        lambda argv, defaults: parse_ranges_generate(argv, defaults),
    ),
    "ranges adjust": (
        _adjust,
        lambda argv, defaults: parse_ranges_adjust(argv, defaults),
    ),
    "repair": (_repair, lambda argv, defaults: parse_repair(argv, defaults)),
    "batch check": (_check, lambda argv, defaults: parse_batch_check(argv, defaults)),
    "batch cancel": (_cancel, lambda argv, _: parse_batch_cancel(argv)),
    "render": (_render, lambda argv, _: parse_render(argv)),
}


@pytest.mark.parametrize("table", list(TABLES))
def test_every_table_round_trips_with_and_without_settings_defaults(
    table: str,
) -> None:
    generate, parse_table = TABLES[table]
    for seed in SEEDS:
        rng = random.Random(f"{table}-{seed}")
        spec = generate(rng)
        defaults = _defaults(rng, spec)

        assert parse_table(to_argv(spec), None) == spec, seed
        assert parse_table(to_argv(spec, defaults), defaults) == spec, seed


def test_a_value_beginning_with_a_dash_is_joined_to_its_flag() -> None:
    spec = parse(["--input", "x.pdf", "--schema=-s.json", "--model=-m"])
    assert (spec.prompt.schema, spec.model.model) == ("-s.json", "-m")
    argv = to_argv(spec)
    assert "--schema=-s.json" in argv
    assert "--model=-m" in argv
    assert parse(argv) == spec
    cancel = BatchCancelSpec(("-b1", "b2"), False, False)
    assert to_argv(cancel) == ["--batch-ids=-b1,b2"]
    assert parse_batch_cancel(to_argv(cancel)) == cancel
