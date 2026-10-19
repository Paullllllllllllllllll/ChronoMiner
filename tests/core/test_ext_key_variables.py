"""Plan-time key checks ask ``ext.key_variables`` which variables serve a call.

Every test replaces the hook, so they hold under any extension build. A
check passes when the hook names no variable or any named one is set; the
error names the first. No call leaves the process: each front-end test
stops at the plan or skips every provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import chronominer.run as core
from chronominer import __main__ as entry
from chronominer import ext
from chronominer.batch.cancel import list_batches
from chronominer.batch.clients import BatchClients
from chronominer.batch.repair import RepairPlanError, plan_repair
from chronominer.settings import Settings
from tests.batch.test_repair_store import spec_for, store_document
from tests.characterization.inputs import make_scanned_pdf
from tests.conftest import DEFAULT_TEST_SCHEMA, make_settings, make_spec

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY")
ALTERNATIVE = "ALTERNATIVE_TEST_KEY"


class Variables:
    """Replace ``ext.key_variables``; record each call and return *names*."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...] = ()
    ) -> None:
        self.names = names
        self.calls: list[dict[str, Any]] = []

        def key_variables(
            provider: str,
            endpoint: str | None = None,
            *,
            kind: str = "chat",
            settings: Settings,
            args: argparse.Namespace | None = None,
        ) -> tuple[str, ...]:
            self.calls.append(
                {"provider": provider, "endpoint": endpoint, "kind": kind, "args": args}
            )
            return self.names

        monkeypatch.setattr(ext, "key_variables", key_variables)

    @property
    def kinds(self) -> list[str]:
        return [call["kind"] for call in self.calls]


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*KEYS, ALTERNATIVE):
        monkeypatch.delenv(name, raising=False)


# The plan of an extract run


def test_an_empty_tuple_lets_a_sync_run_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variables = Variables(monkeypatch)
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)
    args = argparse.Namespace(extra=True)

    run_plan = core.plan(make_spec(pdf, type="pdf"), make_settings(), args=args)

    assert len(run_plan.to_process) == 1
    assert variables.calls == [
        {"provider": "openai", "endpoint": None, "kind": "chat", "args": args}
    ]


def test_a_batch_run_asks_for_batch_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variables = Variables(monkeypatch)
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)
    args = argparse.Namespace(extra=True)

    core.plan(make_spec(pdf, type="pdf", batch=True), make_settings(), args=args)

    assert variables.kinds == ["batch"]
    assert variables.calls[0]["args"] is args


def test_an_alternative_variable_is_required_instead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Variables(monkeypatch, (ALTERNATIVE, "OTHER_TEST_KEY"))
    spec = make_spec(make_scanned_pdf(tmp_path / "scan.pdf", pages=1), type="pdf")
    monkeypatch.setenv("OPENAI_API_KEY", "settings-key")

    with pytest.raises(core.ConfigurationError, match=ALTERNATIVE):
        core.plan(spec, make_settings())

    monkeypatch.setenv("OTHER_TEST_KEY", "k")
    assert core.plan(spec, make_settings()).to_process


# The plan of a repair


@pytest.mark.parametrize(("flags", "kind"), [((), "chat"), (("batch",), "batch")])
async def test_an_empty_tuple_lets_a_repair_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flags: tuple[str, ...],
    kind: str,
) -> None:
    variables = Variables(monkeypatch)
    stored = await store_document(tmp_path, ["failed"])
    args = argparse.Namespace(extra=True)

    repair_plan = plan_repair(spec_for(stored.output, *flags), Settings(), args=args)

    assert repair_plan.units
    assert repair_plan.args is args
    assert variables.kinds == [kind]
    assert variables.calls[0]["args"] is args


async def test_a_repair_requires_the_alternative_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Variables(monkeypatch, (ALTERNATIVE,))
    monkeypatch.setenv("OPENAI_API_KEY", "settings-key")
    stored = await store_document(tmp_path, ["failed"])

    with pytest.raises(RepairPlanError, match=ALTERNATIVE):
        plan_repair(spec_for(stored.output), Settings())


# Batch clients


def test_has_key_asks_for_batch_keys_with_the_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = Variables(monkeypatch)
    args = argparse.Namespace(extra=True)
    clients = BatchClients(Settings(), "extract", args=args)

    assert clients.has_key("openai")
    variables.names = (ALTERNATIVE,)
    assert not clients.has_key("openai")
    monkeypatch.setenv(ALTERNATIVE, "k")
    assert clients.has_key("openai")
    assert variables.kinds == ["batch"] * 3
    assert all(call["args"] is args for call in variables.calls)


def test_a_listing_skips_providers_without_a_named_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = Variables(monkeypatch, (ALTERNATIVE,))
    args = argparse.Namespace(extra=True)

    listing = asyncio.run(list_batches(Settings(), ("openai", "google"), args=args))

    assert listing.skipped == ["openai", "google"]
    assert [call["provider"] for call in variables.calls] == ["openai", "google"]
    assert all(call["args"] is args for call in variables.calls)


# Front ends pass their parsed arguments


def _parsed(calls: list[dict[str, Any]]) -> argparse.Namespace:
    (args,) = {id(call["args"]): call["args"] for call in calls}.values()
    assert isinstance(args, argparse.Namespace)
    return args


@pytest.mark.parametrize(("extra", "kind"), [((), "chat"), (("--batch",), "batch")])
def test_the_extract_command_passes_kind_and_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, ...],
    kind: str,
) -> None:
    variables = Variables(monkeypatch, (ALTERNATIVE,))
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)
    argv = [
        "extract",
        "--input",
        str(pdf),
        "--schema",
        DEFAULT_TEST_SCHEMA,
        "--type",
        "pdf",
        *extra,
        "--json",
    ]

    assert entry.main([*argv, "--dry-run"]) == 0

    assert ALTERNATIVE in capsys.readouterr().err
    assert variables.kinds == [kind]
    assert Path(_parsed(variables.calls).input) == pdf


@pytest.mark.parametrize(("extra", "kind"), [((), "chat"), (("--batch",), "batch")])
async def test_the_repair_command_passes_kind_and_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, ...],
    kind: str,
) -> None:
    variables = Variables(monkeypatch, (ALTERNATIVE,))
    stored = await store_document(tmp_path, ["failed"])
    argv = [
        "repair",
        "--input",
        str(stored.output),
        *extra,
        "--dry-run",
        "--json",
    ]

    assert entry.main(argv) == 0

    assert ALTERNATIVE in capsys.readouterr().err
    assert variables.kinds == [kind]
    assert Path(_parsed(variables.calls).input) == stored.output


def test_the_cancel_command_passes_its_args_to_has_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    variables = Variables(monkeypatch, (ALTERNATIVE,))

    assert entry.main(["batch", "cancel", "--all", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert payload["cancelled"] == 0
    assert set(variables.kinds) == {"batch"}
    assert _parsed(variables.calls).all is True


def test_the_check_command_gives_its_batch_clients_the_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: list[argparse.Namespace | None] = []
    real = BatchClients.__init__

    def init(self: BatchClients, *args: Any, **kwargs: Any) -> None:
        real(self, *args, **kwargs)
        built.append(self.args)

    monkeypatch.setattr(BatchClients, "__init__", init)

    assert entry.main(["batch", "check", "--input", str(tmp_path), "--json"]) == 0

    (args,) = built
    assert isinstance(args, argparse.Namespace)
    assert Path(args.input) == tmp_path
