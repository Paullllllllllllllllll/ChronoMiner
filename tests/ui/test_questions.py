"""The shared model, response and image questions and their review lines."""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from chronominer import spec
from chronominer.common.testing import KEEP, ScriptedPrompter, ScriptError
from chronominer.common.wizard import (
    BACK,
    DONE,
    SETTINGS_LEGEND,
    Nav,
    Review,
    State,
    StepContext,
    render_review,
)
from chronominer.spec import (
    DEFAULT_CHUNK_TOKENS,
    OPTIONS,
    RANGES_ADJUST_OPTIONS,
    RANGES_GENERATE_OPTIONS,
    REPAIR_OPTIONS,
)
from chronominer.ui.questions import (
    OTHER,
    PENDING_SCHEMA,
    ImageQuestions,
    ModelQuestions,
    Question,
    ResponseQuestions,
    chunk_tokens_question,
    current_model,
    images_line,
    model_line,
    resolution_of,
    response_line,
    support_of,
)
from chronominer.ui.session import Session, new_state
from tests.ui.conftest import ENDPOINT, ENDPOINT_URL, SessionMaker

Group = Callable[[StepContext, Session], Any]

EXAMPLE_SCHEMAS = (
    "BibliographicCatalogueEntries",
    "CensusHouseholds",
    "CorrespondenceMetadata",
    "ParishRegisterEvents",
    "PriceListEntries",
    "ProbateInventories",
    "TradeRecords",
    "WageRecords",
)


@dataclasses.dataclass
class Asked:
    result: Nav
    state: State
    prompter: ScriptedPrompter

    @property
    def lines(self) -> list[str]:
        return self.prompter.lines


def _ask(
    group: Group,
    session: Session,
    answers: Sequence[Any],
    *,
    table: Any = OPTIONS,
    given: Mapping[str, Any] | None = None,
) -> Asked:
    state = new_state(session, table)
    for name, value in (given or {}).items():
        state.set(name, value)
    prompter = ScriptedPrompter(answers)
    with prompter:
        questions: list[Question] = group(StepContext(state, prompter), session)
        result = StepContext.sequence(questions)
    return Asked(result, state, prompter)


def _model(ctx: StepContext, session: Session) -> list[Question]:
    return ModelQuestions(ctx, session).questions()


def _response(ctx: StepContext, session: Session) -> list[Question]:
    return ResponseQuestions(ctx, session).questions()


def _images(ctx: StepContext, session: Session) -> list[Question]:
    return ImageQuestions(ctx, session).questions()


def _effective(asked: Asked, session: Session) -> Any:
    model = current_model(asked.state, session)
    assert model is not None
    return model


def test_provider_and_model_with_recommended_tuning(session: Session) -> None:
    asked = _ask(
        _model, session, ["anthropic", OTHER, "claude-haiku-4-5", "medium", KEEP]
    )
    assert asked.result is DONE
    model = _effective(asked, session)
    assert (model.provider, model.model) == ("anthropic", "claude-haiku-4-5")
    assert model.reasoning_effort == "medium"
    assert asked.state.answers == {
        "model": "claude-haiku-4-5",
        "reasoning_effort": "medium",
    }
    assert "? Verbosity:" not in asked.lines


def test_back_returns_to_the_provider(session: Session) -> None:
    asked = _ask(
        _model,
        session,
        ["anthropic", BACK, "openai", "gpt-5.6-terra", KEEP, "recommended"],
    )
    assert asked.lines.count("? Provider:") == 2
    model = _effective(asked, session)
    assert (model.provider, model.model) == ("openai", "gpt-5.6-terra")


def test_back_on_the_first_question_leaves_the_group(session: Session) -> None:
    asked = _ask(_model, session, [BACK])
    assert asked.result is BACK
    assert asked.state.answers == {}


def test_a_typed_model_switches_to_the_provider_it_implies(session: Session) -> None:
    asked = _ask(
        _model, session, ["openai", OTHER, "claude-sonnet-5", KEEP, "recommended"]
    )
    assert "  provider anthropic, as the model name implies" in asked.lines
    model = _effective(asked, session)
    assert (model.provider, model.model) == ("anthropic", "claude-sonnet-5")
    assert "? Reasoning effort:" in asked.lines


@pytest.mark.parametrize(
    ("model", "asked"),
    [
        ("claude-opus-5-5", True),
        ("claude-sonnet-5-5", True),
        ("claude-haiku-5-5", True),
        ("claude-3-7-sonnet", False),
    ],
)
def test_the_effort_is_asked_for_claude_models_that_take_it(
    session: Session, model: str, asked: bool
) -> None:
    effort = [KEEP] if asked else []
    answers = ["anthropic", OTHER, model, *effort, "recommended"]
    lines = _ask(_model, session, answers).lines
    assert ("? Reasoning effort:" in lines) is asked


def test_a_named_endpoint_sets_provider_and_model_together(session: Session) -> None:
    asked = _ask(_model, session, [f"endpoint:{ENDPOINT}", "llama-3", "recommended"])
    assert f"   Endpoint {ENDPOINT}  {ENDPOINT_URL}" in asked.lines
    model = _effective(asked, session)
    assert (model.provider, model.endpoint, model.model) == (
        "custom",
        ENDPOINT,
        "llama-3",
    )
    assert asked.state.answers == {
        "model": "llama-3",
        "provider": "custom",
        "endpoint": ENDPOINT,
    }


def test_customized_tuning_asks_only_what_the_core_sends(session: Session) -> None:
    reasoning = _ask(
        _model,
        session,
        ["openai", "gpt-5.6-luna", KEEP, "customize", "low", "", "priority"],
    )
    assert reasoning.state.value("verbosity") == "low"
    assert reasoning.state.value("service_tier") == "priority"
    assert reasoning.state.value("max_output_tokens") is None
    assert "? Temperature (0.0 to 2.0):" not in reasoning.lines
    sampling = _ask(
        _model,
        session,
        ["openai", OTHER, "gpt-4o", "customize", "2000", "0.2", "0.9", KEEP],
    )
    assert "? Reasoning effort:" not in sampling.lines
    assert "? Verbosity:" not in sampling.lines
    model = _effective(sampling, session)
    assert (model.max_output_tokens, model.temperature, model.top_p) == (
        2000,
        0.2,
        0.9,
    )


def test_recommended_tuning_clears_customized_values(session: Session) -> None:
    asked = _ask(
        _model,
        session,
        ["openai", "gpt-5.6-luna", KEEP, "recommended"],
        given={"verbosity": "low", "service_tier": "priority"},
    )
    assert " > Customize" in asked.lines
    assert asked.state.value("verbosity") == "medium"
    assert asked.state.value("service_tier") == "default"


def test_no_unset_reset_where_a_settings_default_applies(
    make_session: SessionMaker,
) -> None:
    answers = ["openai", "gpt-5.6-luna", KEEP, "customize", KEEP, KEEP, KEEP]
    free = _ask(_model, make_session(), answers)
    assert "? Max output tokens (empty = the model's limit):" in free.lines
    session = make_session({"max_output_tokens": 4000})
    asked = _ask(_model, session, answers)
    assert "? Max output tokens: [4000 (s)] 4000" in asked.lines
    assert not any("empty =" in line for line in asked.lines)
    assert asked.state.from_settings("max_output_tokens")
    with pytest.raises(ScriptError, match="rejected"):
        _ask(_model, session, ["openai", "gpt-5.6-luna", KEEP, "customize", KEEP, ""])


def test_settings_defaults_are_preselected_and_marked(
    make_session: SessionMaker,
) -> None:
    session = make_session({"model": "gpt-5.6-terra", "reasoning_effort": "low"})
    asked = _ask(_model, session, [KEEP, KEEP, KEEP, "recommended"])
    assert " > OpenAI (s)" in asked.lines
    assert " > gpt-5.6-terra (s)" in asked.lines
    assert " > low (s)" in asked.lines
    assert asked.state.answers == {}
    assert asked.state.from_settings("model")


@pytest.mark.parametrize("table", [REPAIR_OPTIONS, RANGES_ADJUST_OPTIONS])
def test_model_questions_work_on_the_repair_and_adjust_tables(
    session: Session, table: Any
) -> None:
    asked = _ask(
        _model,
        session,
        ["google", "gemini-3.7-flash", KEEP, "recommended"],
        table=table,
    )
    model = _effective(asked, session)
    assert (model.provider, model.model) == ("google", "gemini-3.7-flash")


def _choices_after(lines: Sequence[str], question: str) -> list[str]:
    """Return the choice lines shown under the first *question*."""
    start = lines.index(question) + 1
    shown: list[str] = []
    for line in lines[start:]:
        if not line.startswith(("   ", " > ")):
            break
        shown.append(line[3:])
    return shown


def _example_schemas(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offer only the example schemas, whatever ``schemas/`` holds."""
    folder.mkdir(parents=True)
    for path in spec.SCHEMAS_DIR.glob("*.json"):
        if json.loads(path.read_text(encoding="utf-8")).get("name") in (
            EXAMPLE_SCHEMAS
        ):
            shutil.copyfile(path, folder / path.name)
    monkeypatch.setattr(spec, "SCHEMAS_DIR", folder)


def test_response_questions_default_schema_and_context_choices(
    session: Session,
    image_folder: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _example_schemas(tmp_path / "schemas", monkeypatch)
    image = next(image_folder.iterdir())
    asked = _ask(
        _response,
        session,
        [KEEP, "PriceListEntries", "none", "file", f'"{image}"'],
    )
    assert " > Automatic  schema for this model" in asked.lines
    assert "   Schema (API-enforced)" in asked.lines
    assert _choices_after(asked.lines, "? Schema:") == [
        *EXAMPLE_SCHEMAS,
        "A schema file...",
        "Back",
    ]
    assert asked.state.value("response_format") is None
    assert asked.state.value("schema") == "PriceListEntries"
    assert asked.state.value("context") == "none"
    assert asked.state.value("context_image") == image
    assert _choices_after(asked.lines, "? Extraction context:") == [
        "Automatic (sidecar)  the item's {stem}_extract_context.txt",
        "Topics as text",
        "A text file",
        "None",
        "Back",
    ]
    assert "? Context for items without a sidecar:" not in asked.lines
    assert "   Automatic  the item's {stem}_extract_context image" in asked.lines


def test_context_takes_topics_as_text_and_a_fallback(session: Session) -> None:
    asked = _ask(
        _response,
        session,
        [KEEP, "WageRecords", "auto", "text", "wage ledgers", KEEP],
    )
    assert asked.result is DONE
    assert asked.state.value("context") is None
    assert asked.state.value("fallback_context") == "wage ledgers"
    assert "Topics:" in asked.prompter.transcript
    topics = _ask(_response, session, [KEEP, "WageRecords", "text", "prices", KEEP])
    assert topics.state.value("context") == "prices"
    assert "? Context for items without a sidecar:" not in topics.lines


def test_schema_is_offered_only_when_the_model_supports_structured_output(
    session: Session,
) -> None:
    given = {"model": "o1-mini"}
    asked = _ask(
        _response, session, [KEEP, "WageRecords", "auto", KEEP, KEEP], given=given
    )
    assert _choices_after(asked.lines, "? Response format:") == [
        "Automatic  json for this model",
        "JSON (prompted, validated)",
        "Back",
    ]
    with pytest.raises(ScriptError, match="is not one of"):
        _ask(_response, session, ["schema"], given=given)


def test_back_on_every_select_returns_to_the_previous_question(
    session: Session,
) -> None:
    answers = [
        "json",
        BACK,
        KEEP,
        "TradeRecords",
        BACK,
        "TradeRecords",
        "none",
        BACK,
        "auto",
        KEEP,
        KEEP,
    ]
    asked = _ask(_response, session, answers)
    assert asked.result is DONE
    assert asked.lines.count("? Response format:") == 2
    assert asked.lines.count("? Schema:") == 3
    assert asked.lines.count("? Extraction context:") == 3
    assert asked.lines.count("? Context for items without a sidecar:") == 1
    assert asked.lines.count("? Context image:") == 2
    selects = sum(line.startswith("? ") for line in asked.lines)
    assert asked.lines.count("   Back") == selects
    assert asked.state.answers == {"response_format": "json", "schema": "TradeRecords"}
    first = _ask(_response, session, [BACK])
    assert first.result is BACK


def test_a_schema_file_is_checked_and_a_quoted_path_accepted(
    session: Session, tmp_path: Path
) -> None:
    schema = tmp_path / "custom_schema.json"
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    answers = ["schema", OTHER, f'"{schema}"', "auto", KEEP, KEEP]
    asked = _ask(_response, session, answers)
    assert asked.state.value("schema") == str(schema)
    with pytest.raises(ScriptError, match="no such file"):
        _ask(_response, session, ["json", OTHER, str(tmp_path / "missing.json")])
    broken = tmp_path / "broken.json"
    broken.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ScriptError, match="must hold a JSON object"):
        _ask(_response, session, ["json", OTHER, f"'{broken}'"])


def test_response_format_offers_no_automatic_over_a_settings_default(
    make_session: SessionMaker,
) -> None:
    session = make_session({"response_format": "json"})
    asked = _ask(_response, session, [KEEP, "CensusHouseholds", KEEP, KEEP, KEEP])
    assert not any("Automatic" in line for line in asked.lines[:6])
    assert " > JSON (prompted, validated) (s)" in asked.lines
    assert asked.state.from_settings("response_format")


def test_a_settings_schema_is_preselected_and_marked(
    make_session: SessionMaker,
) -> None:
    session = make_session({"schema": "WageRecords"})
    asked = _ask(_response, session, [KEEP, KEEP, KEEP, KEEP, KEEP])
    assert " > WageRecords (s)" in asked.lines
    assert asked.state.from_settings("schema")


def test_the_adjust_table_asks_the_adjust_context_and_no_context_image(
    session: Session,
) -> None:
    asked = _ask(
        _response,
        session,
        [KEEP, "ProbateInventories", "auto", KEEP],
        table=RANGES_ADJUST_OPTIONS,
    )
    assert "? Adjust context:" in asked.lines
    assert " > Automatic (sidecar)  the item's {stem}_adjust_context.txt" in (
        asked.lines
    )
    assert "? Context image:" not in asked.lines
    assert asked.state.value("schema") == "ProbateInventories"


@pytest.mark.parametrize("table", [OPTIONS, RANGES_GENERATE_OPTIONS])
def test_chunk_tokens_take_a_positive_integer(session: Session, table: Any) -> None:
    def group(ctx: StepContext, session: Session) -> list[Question]:
        return [chunk_tokens_question(ctx)]

    kept = _ask(group, session, [KEEP], table=table)
    assert f"? Tokens per chunk: [{DEFAULT_CHUNK_TOKENS}] 3500" in kept.lines
    assert kept.state.answers == {}
    asked = _ask(group, session, ["2000"], table=table)
    assert asked.state.value("chunk_tokens") == 2000
    with pytest.raises(ScriptError, match="rejected"):
        _ask(group, session, ["0"], table=table)
    skipped = _ask(group, session, [], table=REPAIR_OPTIONS)
    assert skipped.result is DONE


def test_recommended_images_follow_the_model_family(session: Session) -> None:
    asked = _ask(_images, session, [KEEP])
    assert (
        " > Recommended for openai models  native DPI, detail original, JPEG 95, color"
    ) in asked.lines
    google = _ask(
        _images,
        session,
        ["customize", "200", "high", "png", False],
        given={"model": "gemini-3.7-flash"},
    )
    assert " > Recommended (high)" in google.lines
    assert "   original" not in google.lines
    assert "? JPEG quality (1 to 100, empty = 95):" not in google.lines
    values = [google.state.value(name) for name in ("dpi", "image_format")]
    assert values == [200, "png"]
    assert google.state.value("grayscale") is False


def test_image_resets_are_withheld_over_settings_defaults(
    make_session: SessionMaker,
) -> None:
    session = make_session({"dpi": 300, "image_detail": "high"})
    asked = _ask(_images, session, ["customize", KEEP, KEEP, KEEP, KEEP, KEEP])
    assert "? DPI (native or a number): [300 (s)] 300" in asked.lines
    assert not any(line.startswith("   Recommended (") for line in asked.lines[:12])
    assert " > high (s)" in asked.lines
    note = _ask(_images, session, [KEEP]).lines
    assert any("300 DPI (s), detail high (s)" in line for line in note)


def test_model_line_marks_a_settings_default_model(make_session: SessionMaker) -> None:
    session = make_session({"model": "gpt-5.6-terra"})
    resolved = resolution_of(new_state(session, OPTIONS), session)
    spec = resolved.spec
    line = model_line(spec.model, resolved.sources, session)
    screen = render_review(Review([line]), width=80)
    assert SETTINGS_LEGEND in screen.splitlines()[0]
    assert (
        "  model   openai / gpt-5.6-terra (s) | reasoning xhigh | tier default"
        in screen
    )


def test_a_state_without_schema_resolves_with_the_pending_schema(
    session: Session,
) -> None:
    resolved = resolution_of(new_state(session, OPTIONS), session)
    assert resolved.spec.prompt.schema == PENDING_SCHEMA
    adjust = resolution_of(new_state(session, RANGES_ADJUST_OPTIONS), session)
    assert adjust.spec.schema == PENDING_SCHEMA
    repair = resolution_of(new_state(session, REPAIR_OPTIONS), session)
    assert repair.spec.schema is None


def test_response_and_images_lines(make_session: SessionMaker) -> None:
    session = make_session({"schema": "PriceListEntries"})
    resolved = resolution_of(new_state(session, OPTIONS), session)
    spec = resolved.spec
    screen = render_review(
        Review(
            [
                response_line(spec.model, spec.prompt, resolved.sources, session),
                images_line(spec.model, spec.images, resolved.sources),
            ]
        )
    ).splitlines()
    assert screen[1] == (
        "  response   automatic (schema) | PriceListEntries (s) | context auto | "
        "no context image"
    )
    assert screen[2] == (
        "  images     native DPI | detail original | JPEG 95 | color | "
        "recommended for openai"
    )


def test_support_of_a_custom_endpoint_takes_its_declaration(
    session: Session,
) -> None:
    state = new_state(session, OPTIONS)
    state.set("model", "llama-3")
    state.set("provider", "custom")
    state.set("endpoint", ENDPOINT)
    model = current_model(state, session)
    assert model is not None
    support = support_of(model, session.settings)
    assert support.schema and support.temperature
    assert not (support.reasoning or support.service_tier)
    unknown = dataclasses.replace(model, endpoint="nowhere")
    assert not support_of(unknown, session.settings).schema
