"""The client kind of each call site, client options and the batch consent.

Every client builder asks ``ext.client_options`` for the options of its
attempt identity and applies them; every batch consent receives the settings
and, where the front end has them, the parsed arguments. No call leaves the
process: the built clients are inspected, not used.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

import chronominer.batch.clients as clients_module
import chronominer.run as core
from chronominer import __main__ as entry
from chronominer import ext
from chronominer.batch.clients import BatchClients
from chronominer.batch.repair import RepairPlanError, plan_repair
from chronominer.ext import ClientOptions, Identity
from chronominer.llm.caller import ModelCaller
from chronominer.llm.client import build_chat_model
from chronominer.settings import Settings
from tests.batch.test_repair_store import spec_for, store_document
from tests.characterization.inputs import make_scanned_pdf
from tests.conftest import DEFAULT_TEST_SCHEMA, make_settings, make_spec
from tests.llm import helpers

BASE_URL = "http://proxy.invalid/v1"
OPTION_KEY = "option-key"


def _mark(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "marker": "hooked"}


OPTIONS_SET = ClientOptions(base_url=BASE_URL, api_key=OPTION_KEY, payload_hook=_mark)


class Kinds:
    """Wrap ``ext.attempt`` and record the identity of every attempt."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.identities: list[Identity] = []
        real = ext.attempt

        def attempt(*args: Any, **kwargs: Any) -> ext.Attempt[Any]:
            opened: ext.Attempt[Any] = real(*args, **kwargs)
            self.identities.append(opened.identity)
            return opened

        monkeypatch.setattr(ext, "attempt", attempt)

    @property
    def kinds(self) -> set[str]:
        return {identity.kind for identity in self.identities}


class Options:
    """Replace ``ext.client_options`` and record the identities it receives."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, options: ClientOptions = OPTIONS_SET
    ) -> None:
        self.identities: list[Identity] = []
        self.options = options

        def client_options(identity: Identity) -> ClientOptions:
            self.identities.append(identity)
            return self.options

        monkeypatch.setattr(ext, "client_options", client_options)


class Consent:
    """Replace ``ext.batch_allowed``; record its calls and refuse."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[Any, Settings, argparse.Namespace | None]] = []

        def batch_allowed(
            spec: Any,
            *,
            settings: Settings,
            args: argparse.Namespace | None = None,
        ) -> bool:
            self.calls.append((spec, settings, args))
            return False

        monkeypatch.setattr(ext, "batch_allowed", batch_allowed)


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_the_public_default_gives_no_options() -> None:
    identity = Identity("openai", "OPENAI_API_KEY", "extract")
    assert identity.kind == "chat"
    assert ext.client_options(identity) == ClientOptions()


# Kind per call site


def test_the_model_caller_builds_chat_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinds = Kinds(monkeypatch)
    helpers.install(monkeypatch, [{"a": 1}])

    ModelCaller("extract", helpers.model_spec(), helpers.env())

    assert kinds.kinds == {"chat"}


def test_batch_calls_build_batch_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds = Kinds(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        clients_module, "create_client", lambda provider, key, **_: (provider, key)
    )
    monkeypatch.setattr(clients_module, "create_backend", lambda *_: object())
    clients = BatchClients(Settings(), "extract")

    result = asyncio.run(clients.call("openai", lambda backend: "listed"))

    assert result == "listed"
    assert kinds.kinds == {"batch"}


# Builders applying the client options


def test_the_openai_chat_builder_applies_every_option(
    monkeypatch: pytest.MonkeyPatch, no_keys: None
) -> None:
    options = Options(monkeypatch)

    caller = ModelCaller("extract", helpers.model_spec(), helpers.env())

    model: Any = caller.model
    assert [identity.kind for identity in options.identities] == ["chat"]
    assert model.openai_api_key.get_secret_value() == OPTION_KEY
    assert model.openai_api_base == BASE_URL
    assert model.max_retries == 0
    assert model.model_name == helpers.model_spec().model
    assert model.http_async_client is not None
    payload = model._get_request_payload([HumanMessage(content="chunk")])
    assert payload["marker"] == "hooked"
    assert payload["model"] == helpers.model_spec().model


def test_the_chat_builder_without_options_keeps_the_plain_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    model: Any = build_chat_model(helpers.model_spec(), helpers.settings())

    assert type(model).__name__ == "ChatOpenAI"
    assert model.openai_api_key.get_secret_value() == "env-key"
    assert "marker" not in model._get_request_payload([HumanMessage(content="p")])


def test_a_custom_endpoint_takes_the_option_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model: Any = build_chat_model(
        helpers.custom_spec(),
        helpers.settings(),
        options=ClientOptions(base_url=BASE_URL, api_key=OPTION_KEY),
    )

    assert model.openai_api_base == BASE_URL
    assert model.openai_api_key.get_secret_value() == OPTION_KEY


def test_other_chat_builders_take_the_option_key(no_keys: None) -> None:
    spec = helpers.model_spec("claude-sonnet-4-6", "anthropic")
    model: Any = build_chat_model(
        spec, helpers.settings(), options=ClientOptions(api_key=OPTION_KEY)
    )

    assert model.anthropic_api_key.get_secret_value() == OPTION_KEY


def test_the_batch_builder_applies_only_the_key(
    monkeypatch: pytest.MonkeyPatch, no_keys: None
) -> None:
    options = Options(monkeypatch)
    built: list[tuple[str, str, dict[str, Any]]] = []

    def create_client(provider: str, key: str, **kwargs: Any) -> object:
        built.append((provider, key, kwargs))
        return object()

    monkeypatch.setattr(clients_module, "create_client", create_client)
    clients = BatchClients(Settings(), "extract")
    identity = Identity("openai", "OPENAI_API_KEY", "extract", kind="batch")

    clients._client(identity)

    assert options.identities == [identity]
    ((provider, key, kwargs),) = built
    assert (provider, key) == ("openai", OPTION_KEY)
    assert "base_url" not in kwargs


# Batch consent at each call site


def test_the_run_plan_passes_settings_and_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    consent = Consent(monkeypatch)
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)
    spec = make_spec(pdf, batch=True)
    settings = make_settings()
    args = argparse.Namespace(extra=True)

    with pytest.raises(core.ConfigurationError, match="Batch submission"):
        core.plan(spec, settings, args=args)

    assert consent.calls == [(spec, settings, args)]


def test_the_extract_command_passes_its_parsed_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    consent = Consent(monkeypatch)
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)

    code = entry.main(
        [
            "extract",
            "--input",
            str(pdf),
            "--schema",
            DEFAULT_TEST_SCHEMA,
            "--type",
            "pdf",
            "--batch",
        ]
    )

    assert code != 0
    ((_, _, args),) = consent.calls
    assert isinstance(args, argparse.Namespace)
    assert Path(args.input) == pdf


async def test_the_repair_plan_passes_settings_and_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    consent = Consent(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    stored = await store_document(tmp_path, ["failed"])
    spec = spec_for(stored.output, "batch")
    settings = Settings()
    args = argparse.Namespace(extra=True)

    with pytest.raises(RepairPlanError, match="Batch submission"):
        plan_repair(spec, settings, args=args)

    assert consent.calls == [(spec, settings, args)]
