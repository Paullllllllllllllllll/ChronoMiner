"""Tests for ext.py: pass-through hooks and the attempt context."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pytest

from chronominer import ext
from chronominer.common.usage import Usage
from chronominer.ext import Decision, Identity
from chronominer.settings import Endpoint, Settings
from chronominer.spec import (
    parse_batch_cancel,
    parse_ranges_adjust,
    parse_ranges_generate,
    resolve_repair,
    resolve_spec,
)

_SETTINGS = Settings(
    endpoints={"local": Endpoint("local", "http://127.0.0.1/v1/", "LOCAL_KEY")}
)


class _Builder:
    def __init__(self) -> None:
        self.identities: list[Identity] = []

    def __call__(self, identity: Identity) -> dict[str, Any]:
        self.identities.append(identity)
        return {"key_env": identity.api_key_env, "n": len(self.identities)}


def _status_error(status: int) -> Exception:
    exc = Exception(f"status {status}")
    exc.status_code = status  # type: ignore[attr-defined]
    return exc


def test_startup_hooks_pass_through() -> None:
    parser = argparse.ArgumentParser()
    ext.register_args(parser)
    assert set(vars(parser.parse_args([])).values()) <= {None}
    assert list(ext.wizard_steps()) == []
    spec = resolve_spec({"input": Path("x.pdf"), "schema": "WageRecords"}).spec
    repair = resolve_repair({"input": Path("out")}).spec
    ranges = (
        parse_ranges_generate(["--input", "x.txt"]),
        parse_ranges_adjust(["--input", "x.txt", "--schema", "WageRecords"]),
    )
    for each in (spec, repair, *ranges, parse_batch_cancel(["--all"])):
        ext.startup(each, settings=_SETTINGS, args=argparse.Namespace())
    assert ext.batch_allowed(spec, settings=_SETTINGS) is True
    assert ext.batch_allowed(repair, settings=_SETTINGS, args=argparse.Namespace())


def test_roles_are_extract_and_adjust_without_audio_clients() -> None:
    assert ext.ROLES == ("extract", "adjust")
    assert ext.CLIENT_KINDS == ("chat", "batch", "decision")
    with pytest.raises(ValueError, match="client kind"):
        ext.attempt(
            "extract",
            "openai",
            settings=_SETTINGS,
            build_client=_Builder(),
            kind="audio",
        )


def test_identity_kind_defaults_to_chat_and_rejects_unknown_kinds() -> None:
    chat = ext.attempt("extract", "openai", settings=_SETTINGS, build_client=_Builder())
    assert chat.identity.kind == "chat"
    for kind in ext.CLIENT_KINDS:
        made = ext.attempt(
            "extract",
            "openai",
            settings=_SETTINGS,
            build_client=_Builder(),
            kind=kind,
        )
        assert made.identity.kind == kind
    with pytest.raises(ValueError, match="client kind"):
        ext.attempt(
            "extract",
            "openai",
            settings=_SETTINGS,
            build_client=_Builder(),
            kind="speech",
        )


def test_identity_holds_the_key_variable_and_the_model() -> None:
    attempt = ext.attempt(
        "extract",
        "openai",
        100,
        settings=_SETTINGS,
        build_client=_Builder(),
        model="gpt-5.6-luna",
    )
    assert attempt.identity == Identity(
        "openai", "OPENAI_API_KEY", "extract", model="gpt-5.6-luna"
    )
    assert attempt.identity.group is None
    assert attempt.estimate == 100
    custom = ext.attempt(
        "adjust",
        "custom",
        settings=_SETTINGS,
        build_client=_Builder(),
        endpoint="local",
    )
    assert custom.identity.api_key_env == "LOCAL_KEY"
    assert custom.identity.model is None


def test_build_makes_the_client_without_entering() -> None:
    builder = _Builder()
    attempt = ext.attempt(
        "batch", "anthropic", settings=_SETTINGS, build_client=builder
    )

    client = attempt.build()

    assert client == {"key_env": "ANTHROPIC_API_KEY", "n": 1}
    assert attempt.client is client
    assert builder.identities == [attempt.identity]
    assert not attempt.released


def test_success_builds_the_client_commits_and_releases() -> None:
    builder = _Builder()

    async def run() -> ext.Attempt[dict[str, Any]]:
        async with ext.attempt(
            "extract", "openai", settings=_SETTINGS, build_client=builder
        ) as call:
            assert call.client == {"key_env": "OPENAI_API_KEY", "n": 1}
            assert not call.released
            call.commit(Usage(input_tokens=10, output_tokens=5, total_tokens=15))
            call.commit(None)
            return call

    call = asyncio.run(run())
    assert call.released
    assert call.committed == Usage(10, 5, 15)
    assert call.usage == Usage(10, 5, 15)
    assert builder.identities == [call.identity]


def test_refresh_rebuilds_the_client_for_the_same_identity() -> None:
    seen: list[Identity] = []

    def build(identity: Identity) -> int:
        seen.append(identity)
        return len(seen)

    async def run() -> None:
        async with ext.attempt(
            "adjust", "google", settings=_SETTINGS, build_client=build
        ) as call:
            first = call.identity
            assert call.client == 1
            assert await call.refresh() == 2
            assert call.client == 2
            assert call.identity is first
            with pytest.raises(AttributeError):
                call.identity = Identity("x", "Y", "z")  # type: ignore[misc]

    asyncio.run(run())
    assert len(set(seen)) == 1


def test_exception_releases_and_propagates() -> None:
    holder: list[ext.Attempt[Any]] = []

    async def run() -> None:
        async with ext.attempt(
            "extract", "openai", settings=_SETTINGS, build_client=_Builder()
        ) as call:
            holder.append(call)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run())
    assert holder[0].released


def test_cancellation_releases() -> None:
    holder: list[ext.Attempt[Any]] = []

    async def body(started: asyncio.Event) -> None:
        async with ext.attempt(
            "extract", "openai", settings=_SETTINGS, build_client=_Builder()
        ) as call:
            holder.append(call)
            started.set()
            await asyncio.sleep(3600)

    async def run() -> None:
        started = asyncio.Event()
        task = asyncio.create_task(body(started))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert holder[0].released


def test_failed_client_build_releases() -> None:
    def broken(identity: Identity) -> object:
        raise ValueError("no client")

    attempt = ext.attempt("extract", "openai", settings=_SETTINGS, build_client=broken)

    async def run() -> None:
        async with attempt:
            pass

    with pytest.raises(ValueError, match="no client"):
        asyncio.run(run())
    assert attempt.released


@pytest.mark.parametrize(
    ("error", "decision"),
    [
        (_status_error(429), Decision.RETRY),
        (_status_error(503), Decision.RETRY),
        (TimeoutError("timed out"), Decision.RETRY),
        (_status_error(400), Decision.FAIL),
        (_status_error(401), Decision.FAIL),
    ],
)
def test_on_error_returns_the_classifier_decision(
    error: Exception, decision: Decision
) -> None:
    attempt = ext.attempt(
        "extract", "openai", settings=_SETTINGS, build_client=_Builder()
    )
    assert attempt.on_error(error) is decision
    assert attempt.errors == [error]


def test_on_error_records_recovered_usage() -> None:
    error = _status_error(500)
    error.body = {  # type: ignore[attr-defined]
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    }
    attempt = ext.attempt(
        "extract", "openai", settings=_SETTINGS, build_client=_Builder()
    )
    assert attempt.on_error(error) is Decision.RETRY
    attempt.on_error(_status_error(502))
    attempt.commit(Usage(input_tokens=1, output_tokens=1, total_tokens=2))
    assert attempt.recovered == Usage(7, 3, 10)
    assert attempt.committed == Usage(1, 1, 2)
    assert attempt.usage == Usage(8, 4, 12)


def test_client_before_entry_is_an_error() -> None:
    attempt = ext.attempt(
        "extract", "openai", settings=_SETTINGS, build_client=_Builder()
    )
    with pytest.raises(RuntimeError):
        _ = attempt.client
