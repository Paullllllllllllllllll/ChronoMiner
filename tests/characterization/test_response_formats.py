"""Characterization of enforced and prompted JSON on the extract command.

Each case extracts a twelve-record ledger through ``run_tool``, twice in fresh
roots, and compares the second run with ``golden/formats/<case>/``. A schema
is enforced when the model supports structured outputs: the request then
carries the ``json_schema`` binding (``structured`` in ``requests.json``),
or a ``function_calling`` binding for Llama through OpenRouter. The schema
text is always in the system prompt; the ``inject_schema_into_prompt: false``
override becomes ``--response-format schema``. A model without structured
outputs (``o3-deep-research``) gets prompted JSON and answers plain text,
which the tool parses. The remaining cases pin a fenced or prefixed answer,
truncated JSON, an empty answer and an answer that reports no content of the
requested type. Truncated JSON is not retried: the raw text lands in ``_output.json``
and its entries are missing from the CSV, while the run still exits 0. An
empty answer fails its chunk, and the run exits 1.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.conftest import CaseRoot, GoldenCheck
from tests.characterization.fakes import FakeLLM, record_unit
from tests.characterization.golden import compare, snapshot
from tests.characterization.inputs import make_text_file

pytestmark = pytest.mark.integration

RECORDS = 12
SCHEMA_HEADER = "The JSON schema:"
NO_INJECT = {"model_config": {"extraction_model": {"inject_schema_into_prompt": False}}}
ENFORCED_NAME = "ExtractionSchema"
"""Name every enforced schema carries, whatever the schema file says."""
NO_CONTENT = {"contains_no_content_of_requested_type": True, "entries": []}
PROMPTED_MODEL = "o3-deep-research"
"""An OpenAI model without structured outputs (its model page says so)."""

Script = Callable[[FakeLLM], None]
Check = Callable[[RunResult], None]


@dataclass(frozen=True)
class FormatCase:
    """One response-format case; ``flags`` and ``config`` go to ``run_tool``."""

    name: str
    flags: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    script: Script | None = None
    check: Check | None = None


def _ledger(root: Path) -> Path:
    return make_text_file(root / "docs" / "ledger.txt", RECORDS)


def _system_text(call: Mapping[str, Any]) -> str:
    message = call["messages"][0]
    assert message["role"] == "system"
    return "".join(block.get("text", "") for block in message["blocks"])


def _records(result: RunResult) -> list[dict[str, Any]]:
    path = result.output_json(result.root / "docs" / "ledger.txt")
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _csv(result: RunResult) -> str:
    path = result.output_json(result.root / "docs" / "ledger.txt")
    return path.with_suffix(".csv").read_text(encoding="utf-8-sig")


def _enforced(result: RunResult) -> None:
    for call in result.calls:
        structured = call["structured"]
        assert structured is not None
        assert structured["method"] == "json_schema"
        assert structured["schema"]["name"] == ENFORCED_NAME


def _prompted(result: RunResult) -> None:
    assert [call["structured"] for call in result.calls] == [None] * len(result.calls)
    assert SCHEMA_HEADER in _system_text(result.calls[0])


def _check_enforced_injected(result: RunResult) -> None:
    _enforced(result)
    assert SCHEMA_HEADER in _system_text(result.calls[0])
    assert result.summary is not None
    assert (result.summary["items_complete"], result.summary["items_failed"]) == (
        1,
        0,
    )


def _check_prompted_openai(result: RunResult) -> None:
    _prompted(result)
    assert {call["model"] for call in result.calls} == {PROMPTED_MODEL}


def _check_enforced_openrouter(result: RunResult) -> None:
    for call in result.calls:
        structured = call["structured"]
        assert structured is not None
        assert structured["method"] == "function_calling"
        assert structured["schema"]["title"] == ENFORCED_NAME
    assert {model["provider"] for model in result.models} == {"openrouter"}


def _check_prompted_forms(result: RunResult) -> None:
    _prompted(result)
    assert [call["attempt"] for call in result.calls] == [1] * len(result.calls)


def _check_single_attempts(result: RunResult) -> None:
    assert [call["attempt"] for call in result.calls] == [1] * len(result.calls)


def _check_malformed(result: RunResult) -> None:
    """An answer without a JSON object fails its chunk and the item."""
    _check_single_attempts(result)
    assert result.exit_code == 1
    assert result.summary is not None
    assert result.summary["items_failed"] == 1
    assert "record-3" not in _csv(result)
    assert "record-0" in _csv(result)


def _check_empty(result: RunResult) -> None:
    _check_single_attempts(result)
    assert result.exit_code == 1
    assert result.summary is not None
    assert result.summary["items_failed"] == 1


def _check_no_content(result: RunResult) -> None:
    _enforced(result)
    assert len(_records(result)) == len(result.calls)


def _script_prompted_forms(llm: FakeLLM) -> None:
    llm.script(record_unit(0), form="fenced")
    llm.script(record_unit(3), form="preamble")


def _script_malformed(llm: FakeLLM) -> None:
    llm.script(record_unit(3), form="malformed")


def _script_empty(llm: FakeLLM) -> None:
    llm.script(record_unit(6), form="empty")


def _script_no_content(llm: FakeLLM) -> None:
    llm.script(record_unit(3), payload=NO_CONTENT)


CASES: tuple[FormatCase, ...] = (
    FormatCase("enforced_schema_injected", check=_check_enforced_injected),
    FormatCase(
        "enforced_schema_not_injected",
        config=NO_INJECT,
        check=_check_enforced_injected,
    ),
    FormatCase(
        "prompted_openai_deep_research",
        flags={"model": PROMPTED_MODEL},
        check=_check_prompted_openai,
    ),
    FormatCase(
        "enforced_openrouter_llama",
        flags={"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
        check=_check_enforced_openrouter,
    ),
    FormatCase(
        "prompted_fenced_and_preamble",
        flags={"model": PROMPTED_MODEL},
        script=_script_prompted_forms,
        check=_check_prompted_forms,
    ),
    FormatCase(
        "enforced_malformed_json",
        script=_script_malformed,
        check=_check_malformed,
    ),
    FormatCase(
        "prompted_malformed_json",
        flags={"model": PROMPTED_MODEL},
        script=_script_malformed,
        check=_check_malformed,
    ),
    FormatCase(
        "enforced_empty_answer",
        script=_script_empty,
        check=_check_empty,
    ),
    FormatCase(
        "enforced_no_content_of_requested_type",
        script=_script_no_content,
        check=_check_no_content,
    ),
)


def _run(case: FormatCase, llm: FakeLLM, root: Path) -> RunResult:
    llm.units.clear()
    if case.script is not None:
        case.script(llm)
    return run_tool(
        "extract",
        llm=llm,
        root=root,
        config=case.config,
        input=_ledger(root),
        output=root / "out",
        **dict(case.flags),
    )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_response_format_case(
    case: FormatCase,
    fake_llm: FakeLLM,
    case_root: CaseRoot,
    golden: GoldenCheck,
) -> None:
    first = snapshot(_run(case, fake_llm, case_root("run1")))
    result = _run(case, fake_llm, case_root("run2"))
    problems = compare(first, snapshot(result))
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)

    assert result.calls
    assert {call["role"] for call in result.calls} == {"extract"}
    if case.check is not None:
        case.check(result)
    golden(f"formats/{case.name}", result)


def test_case_names_are_unique() -> None:
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))
