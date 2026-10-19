"""Fake provider classes for the characterization tests.

``FakeLLM.constructor`` builds stand-ins for the provider classes the tool
instantiates for synchronous calls: ``ChatOpenAI`` (OpenAI, OpenRouter and
custom endpoints through ``base_url``), ``ChatAnthropic`` and
``ChatGoogleGenerativeAI``. The synchronous path builds no SDK client of its
own; the batch backends' SDK clients are faked in ``fakes_batch``. Each
stand-in records its constructor kwargs and returns a ``ScriptedChatModel``,
so the real client factory, request assembly, structured-output routing,
retries and parsing all run.

A request is matched to a unit by content, never by call order: the page
marker of its last image (``page-N``, see ``inputs.decode_marker``), else the
first record heading of its user text (``record-N``, see
``inputs.records_in``). A boundary-adjustment request (role ``adjust``) is
keyed ``adjust-N`` by the record heading its answer moves to. A schema the
provider enforces is recorded in the form of its ``json_schema``
structured-output call, whether it travels as an invoke kwarg or through
``with_structured_output`` (see ``_enforcement``), so enforced and prompted
JSON differ in ``calls``.

Answers by role: an extraction answer is schema JSON built from the enforced
schema or, for prompted JSON, from ``FakeLLM.schema`` (``example_payload``);
a boundary answer moves the range start to the next record heading
(``boundary_payload``). A ``UnitScript`` replaces the payload, the answer form
(fenced, preamble, malformed, empty) or single attempts (raise an error,
return raw text), so a test scripts any case without editing this module::

    fake_llm.script(record_unit(0), payload={"entries": []})
    fake_llm.reply(page_unit(1), 1, Reply(transient_error()))
    fake_llm.script(adjust_unit(3), always=Reply(text="not json"))
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from PIL import Image

from chronominer.common.testing import (
    FakeRequest,
    ScriptedChatModel,
    ScriptedStructured,
    StructuredCall,
)
from tests.characterization.inputs import RECORD_RE, decode_marker, records_in

CUSTOM_KEY_ENV = "CM_FAKE_CUSTOM_API_KEY"
DUMMY_KEYS = {
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "GOOGLE_API_KEY": "test-google-key",
    "OPENROUTER_API_KEY": "test-openrouter-key",
    CUSTOM_KEY_ENV: "test-custom-key",
}
_KEY_NAMES = {value: name for name, value in DUMMY_KEYS.items()}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_CLASS_PROVIDERS = {"ChatAnthropic": "anthropic", "ChatGoogleGenerativeAI": "google"}

PROVIDER_CLASSES: tuple[tuple[str, str], ...] = (
    ("langchain_openai", "ChatOpenAI"),
    ("langchain_anthropic", "ChatAnthropic"),
    ("langchain_google_genai", "ChatGoogleGenerativeAI"),
)
"""(public module, class name) of every provider class the fake replaces.

``seams.replace`` swaps each class wherever the tool binds it.
"""

SDK_STRUCTURED_METHODS: dict[str, str] = {
    "ChatOpenAI": "json_schema",
    "ChatAnthropic": "function_calling",
    "ChatGoogleGenerativeAI": "json_schema",
}
"""Default ``with_structured_output`` method of each provider class."""

USAGE: dict[str, int] = {
    "input_tokens": 1500,
    "output_tokens": 400,
    "total_tokens": 1900,
}

Role = Literal["extract", "adjust"]
EXTRACT: Role = "extract"
ADJUST: Role = "adjust"

BOUNDARY_SENTINEL = "<<<CURRENT_CHUNK_START>>>"
"""Line the boundary-adjustment request inserts before the current range start."""

Form = Literal["schema", "fenced", "preamble", "malformed", "empty"]
"""Answer forms of a unit script.

``schema`` is bare JSON, ``fenced`` JSON in a json code fence, ``preamble``
JSON after one sentence of prose, ``malformed`` truncated JSON and ``empty``
an empty message.
"""

PREAMBLE = "Here are the extracted records:"


def page_unit(index: int) -> str:
    """Return the unit of the page with marker *index* (0-based)."""
    return f"page-{index}"


def record_unit(index: int) -> str:
    """Return the unit of a text request whose first record is *index*."""
    return f"record-{index}"


def adjust_unit(index: int) -> str:
    """Return the unit of a boundary request that moves to record *index*."""
    return f"adjust-{index}"


_UNIT_RE = re.compile(r"^(?P<kind>[a-z]+)-(?P<number>\d+)$")


def unit_number(unit: str | None) -> int | None:
    """Return the index a unit names, or None."""
    match = _UNIT_RE.match(unit or "")
    return int(match.group("number")) if match else None


def _unit_sort_key(unit: str | None) -> tuple[int, str, int]:
    match = _UNIT_RE.match(unit or "")
    if match is None:
        return 1, str(unit or ""), 0
    return 0, match.group("kind"), int(match.group("number"))


class FakeAPIError(Exception):
    """Provider error carrying an HTTP status code, as the SDK errors do."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def transient_error() -> FakeAPIError:
    """Return a 503 error, which the retry loops retry."""
    return FakeAPIError(503, "Error code: 503 - service unavailable")


def permanent_error() -> FakeAPIError:
    """Return a 400 error, which the retry loops do not retry."""
    return FakeAPIError(400, "Error code: 400 - invalid request")


@dataclass(frozen=True)
class Reply:
    """Replace one answer: raise, return raw text, or answer in another form.

    ``exception`` is raised from the model call. Otherwise ``text`` is the
    raw message content, or ``form`` picks the content built from the unit
    script. ``refusal`` lands in ``additional_kwargs["refusal"]`` and
    ``response_metadata`` replaces the default metadata.
    """

    exception: BaseException | None = None
    text: str | None = None
    form: Form | None = None
    refusal: str | None = None
    response_metadata: Mapping[str, Any] | None = None


@dataclass
class UnitScript:
    """Scripted answers for one unit.

    ``payload`` replaces the generated answer object. ``replies`` maps an
    attempt number (counted per unit and run from 1) to a ``Reply``;
    ``always`` answers every attempt without an entry there. A transient
    error then success is ``replies={1: Reply(transient_error())}``; a
    permanent failure is ``always=Reply(permanent_error())``.
    """

    payload: Mapping[str, Any] | None = None
    form: Form = "schema"
    replies: dict[int, Reply] = field(default_factory=dict)
    always: Reply | None = None


def render(payload: Mapping[str, Any], form: Form) -> str:
    """Return the message content that carries *payload* in *form*."""
    if form == "empty":
        return ""
    body = json.dumps(payload, ensure_ascii=False)
    if form == "malformed":
        return body[: max(1, len(body) // 2)]
    if form == "fenced":
        return f"```json\n{body}\n```"
    if form == "preamble":
        return f"{PREAMBLE}\n\n{body}"
    return body


def _example_value(node: Any, name: str, number: int, label: str) -> Any:
    if not isinstance(node, Mapping):
        return None
    for key in ("anyOf", "oneOf"):
        options = node.get(key)
        if isinstance(options, list) and options:
            chosen = next(
                (
                    o
                    for o in options
                    if isinstance(o, Mapping) and o.get("type") != "null"
                ),
                options[0],
            )
            return _example_value(chosen, name, number, label)
    if isinstance(node.get("enum"), list) and node["enum"]:
        return node["enum"][0]
    kind = node.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object" or (kind is None and "properties" in node):
        properties = node.get("properties") or {}
        return {
            str(key): _example_value(value, str(key), number, label)
            for key, value in properties.items()
        }
    if kind == "array":
        return [_example_value(node.get("items") or {}, name, number, label)]
    if kind == "string":
        return f"{name} {label}"
    if kind == "integer":
        return number + 1
    if kind == "number":
        return number + 1.5
    if kind == "boolean":
        return False
    return None


def example_payload(schema: Mapping[str, Any] | None, unit: str | None) -> Any:
    """Return a schema-shaped answer whose values name *unit*.

    Strings read ``"<field> <unit>"``, numbers derive from the unit's index,
    booleans are false, arrays hold one item and nullable fields take their
    non-null type. *schema* may be a schema file (with ``schema``) or a bare
    JSON schema; without one the answer is an empty object.
    """
    if not schema:
        return {}
    inner = schema.get("schema") if isinstance(schema.get("schema"), Mapping) else None
    label = unit or "unit"
    return _example_value(inner or schema, "value", unit_number(unit) or 0, label)


def boundary_payload(text: str) -> tuple[str | None, dict[str, Any]]:
    """Return the unit and answer of a boundary request over *text*.

    The answer keeps the range start when a record heading follows the
    sentinel directly, moves it to the next heading after the sentinel, or,
    failing that, to the nearest heading before it. Without any heading the
    start is kept. The unit names the heading's record (``adjust-N``).
    """
    lines = text.splitlines()
    try:
        at = next(i for i, line in enumerate(lines) if BOUNDARY_SENTINEL in line)
    except StopIteration:
        at = 0
    after = [(i, line) for i, line in enumerate(lines) if i > at]
    before = [(i, line) for i, line in enumerate(lines) if i < at]
    on_target = bool(after) and RECORD_RE.search(after[0][1]) is not None
    heading = next((line for _, line in after if RECORD_RE.search(line)), None)
    if heading is None:
        heading = next(
            (line for _, line in reversed(before) if RECORD_RE.search(line)), None
        )
    payload: dict[str, Any] = {
        "contains_no_semantic_boundary": False,
        "needs_more_context": False,
        "boundary_already_on_target": on_target or heading is None,
        "certainty": 95,
        "semantic_marker": "" if on_target or heading is None else heading.strip(),
    }
    found = records_in(heading or "")
    return (adjust_unit(found[0]) if found else None), payload


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def summarize_schema(schema: Any) -> dict[str, Any]:
    """Return a compact, stable description of a structured-output schema."""
    if not isinstance(schema, Mapping):
        return {"value": repr(schema)}
    inner = schema.get("schema", schema)
    properties = inner.get("properties") if isinstance(inner, Mapping) else None
    canonical = json.dumps(schema, sort_keys=True, ensure_ascii=False, default=str)
    summary: dict[str, Any] = {
        "properties": sorted(properties) if isinstance(properties, Mapping) else None,
        "digest": _sha(canonical),
    }
    for key in ("type", "name", "strict", "title"):
        if key in schema:
            summary[key] = schema[key]
    return summary


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return f"<{type(value).__name__}>"


def describe_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return invoke or bound kwargs with schema payloads summarized."""
    described: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "text" and isinstance(value, Mapping) and "format" in value:
            text = dict(value)
            text["format"] = summarize_schema(text["format"])
            described[key] = _jsonable(text)
        elif key == "response_format" and isinstance(value, Mapping):
            fmt = dict(value)
            if isinstance(fmt.get("json_schema"), Mapping):
                fmt["json_schema"] = summarize_schema(fmt["json_schema"])
            described[key] = _jsonable(fmt)
        elif key == "response_schema":
            described[key] = summarize_schema(value)
        else:
            described[key] = _jsonable(value)
    return described


def _decode_data_url(url: str) -> tuple[str | None, bytes | None]:
    if not url.startswith("data:") or "," not in url:
        return None, None
    header, data = url.split(",", 1)
    mime = header[5:].split(";", 1)[0] or None
    try:
        return mime, base64.b64decode(data)
    except (binascii.Error, ValueError):
        return mime, None


def _b64(raw: Any) -> bytes | None:
    if raw is None:
        return None
    try:
        return base64.b64decode(str(raw))
    except (binascii.Error, ValueError):
        return None


def _image_from_block(block: Mapping[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    kind = block.get("type")
    if kind == "image_url":
        image_url = block.get("image_url")
        url = image_url.get("url", "") if isinstance(image_url, Mapping) else image_url
        mime, data = _decode_data_url(str(url or ""))
        info: dict[str, Any] = {"type": "image_url", "mime": mime}
        if isinstance(image_url, Mapping) and "detail" in image_url:
            info["detail"] = image_url["detail"]
        return info, data
    source = block.get("source")
    if kind == "image" and isinstance(source, Mapping):
        return {
            "type": "image",
            "source_type": source.get("type"),
            "mime": source.get("media_type"),
        }, _b64(source.get("data"))
    if kind == "image" and "base64" in block:
        info = {"type": "image", "mime": block.get("mime_type")}
        if "media_resolution" in block:
            info["media_resolution"] = block["media_resolution"]
        return info, _b64(block.get("base64"))
    extra = {key: _jsonable(value) for key, value in block.items() if key != "type"}
    return {"type": str(kind), **extra}, None


_IMAGE_BLOCKS = frozenset({"image_url", "image"})


@dataclass
class _Shapes:
    messages: list[dict[str, Any]]
    page: int | None
    system: str
    user: str


def _message_shapes(messages: list[BaseMessage]) -> _Shapes:
    """Return the message digests, the last image's marker, system and user text."""
    shapes: list[dict[str, Any]] = []
    page: int | None = None
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        content = message.content
        entry: dict[str, Any] = {"role": message.type}
        blocks: list[dict[str, Any]] = []
        parts = [content] if isinstance(content, str) else list(content)
        texts = system_parts if message.type == "system" else user_parts
        for block in parts:
            if isinstance(block, str):
                blocks.append({"type": "text", "text": block})
                texts.append(block)
                continue
            if block.get("type") in ("text", "input_text"):
                text = str(block.get("text", ""))
                blocks.append({"type": "text", "text": text})
                texts.append(text)
                continue
            info, data = _image_from_block(block)
            if data is not None and info["type"] in _IMAGE_BLOCKS:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    info["width"], info["height"] = image.size
                    page = decode_marker(image)
            blocks.append(info)
        entry["blocks"] = blocks
        shapes.append(entry)
    return _Shapes(shapes, page, "\n".join(system_parts), "\n".join(user_parts))


def redact(value: Any) -> Any:
    """Return a JSON-ready copy with dummy keys named by their variable."""
    if isinstance(value, httpx.Timeout):
        return {"httpx.Timeout": value.as_dict()}
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    if isinstance(value, Mapping):
        return {str(key): redact(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    if isinstance(value, str) and value in _KEY_NAMES:
        return f"<env:{_KEY_NAMES[value]}>"
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return f"<{type(value).__name__}>"


def provider_of(constructor: str, base_url: Any) -> str:
    """Return the provider a constructor call addresses."""
    if constructor in _CLASS_PROVIDERS:
        return _CLASS_PROVIDERS[constructor]
    if base_url is None:
        return "openai"
    if str(base_url).rstrip("/") == OPENROUTER_BASE_URL:
        return "openrouter"
    return "custom"


@dataclass
class Construction:
    """One provider-class constructor call."""

    constructor: str
    kwargs: dict[str, Any]
    provider: str
    model: str

    def record(self) -> dict[str, Any]:
        """Return the golden entry: provider, constructor and redacted kwargs."""
        return {
            "provider": self.provider,
            "constructor": self.constructor,
            "kwargs": redact(self.kwargs),
        }


class SDKDefaultsChatModel(ScriptedChatModel):
    """Scripted model with its provider class's structured-output defaults.

    A call that omits ``method`` uses ``default_method``.
    """

    default_method: str = "function_calling"

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        method: str | None = None,
        **kwargs: Any,
    ) -> ScriptedStructured:
        chosen = method if method is not None else self.default_method
        return ScriptedStructured(self, StructuredCall(schema, chosen, include_raw))


def _describe_structured(request: FakeRequest) -> dict[str, Any] | None:
    call = request.structured
    if call is None:
        return None
    return {
        "method": call.method,
        "include_raw": call.include_raw,
        "schema": summarize_schema(call.schema),
    }


def _json_schema_call(schema: Any) -> dict[str, Any]:
    return {
        "method": "json_schema",
        "include_raw": True,
        "schema": summarize_schema(schema),
    }


def _enforcement(
    kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, Any]:
    """Return invoke kwargs without an enforced schema, its call and the schema.

    A schema sent as an invoke kwarg (OpenAI ``text.format``, Chat
    Completions ``response_format``, Gemini ``response_mime_type`` with
    ``response_schema``) reaches the provider as the same request field that
    ``with_structured_output(schema, method="json_schema")`` fills, so it is
    recorded as that call: ``text.format`` and ``response_format`` without
    their ``type`` wrapper, the Gemini schema as given. The third value is
    the JSON schema the answer must follow.
    """
    rest = dict(kwargs)
    text = rest.get("text")
    if isinstance(text, Mapping):
        fmt = text.get("format")
        if isinstance(fmt, Mapping) and fmt.get("type") == "json_schema":
            others = {key: value for key, value in text.items() if key != "format"}
            if others:
                rest["text"] = others
            else:
                del rest["text"]
            schema = {key: value for key, value in fmt.items() if key != "type"}
            return rest, _json_schema_call(schema), schema.get("schema")
    fmt = rest.get("response_format")
    if isinstance(fmt, Mapping) and fmt.get("type") == "json_schema":
        del rest["response_format"]
        wrapped = fmt.get("json_schema")
        inner = wrapped.get("schema") if isinstance(wrapped, Mapping) else None
        return rest, _json_schema_call(wrapped), inner
    if (
        rest.get("response_mime_type") == "application/json"
        and "response_schema" in rest
    ):
        schema = rest.pop("response_schema")
        del rest["response_mime_type"]
        return rest, _json_schema_call(schema), schema
    return rest, None, None


def _structured_schema(call: StructuredCall | None) -> Any:
    if call is None or not isinstance(call.schema, Mapping):
        return None
    inner = call.schema.get("schema")
    return inner if isinstance(inner, Mapping) else call.schema


def is_boundary_request(system: str, schema: Any) -> bool:
    """Return True for a boundary-adjustment request."""
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    if isinstance(properties, Mapping) and "semantic_marker" in properties:
        return True
    return "semantic_marker" in system


def text_unit(text: str) -> str | None:
    """Return the unit of a text request: its first record heading."""
    found = records_in(text)
    return record_unit(found[0]) if found else None


class FakeLLM:
    """Provider-class stand-ins and the scripts, records and routing behind them.

    ``units`` holds the unit scripts (a missing one answers by default) and
    ``schema`` the schema file of the run (``run_tool`` sets it), which
    shapes prompted-JSON answers and batch answers without an enforced
    schema. ``calls`` holds one digest per request with the full text of
    every text block; ``constructions`` one entry per constructed client.
    With ``sdk_defaults`` set, clients constructed afterwards are
    ``SDKDefaultsChatModel`` instances with their class's default method
    from ``SDK_STRUCTURED_METHODS``.
    """

    def __init__(
        self,
        units: Mapping[str, UnitScript] | None = None,
        *,
        schema: Mapping[str, Any] | None = None,
        sdk_defaults: bool = False,
    ) -> None:
        self.units: dict[str, UnitScript] = dict(units or {})
        self.schema: Mapping[str, Any] | None = schema
        self.sdk_defaults = sdk_defaults
        self.calls: list[dict[str, Any]] = []
        self.constructions: list[Construction] = []
        self._lock = threading.Lock()
        self._attempts: dict[str | None, int] = {}

    # -- configuration ---------------------------------------------------------
    def unit(self, key: str) -> UnitScript:
        """Return (creating an empty one if needed) the script of a unit."""
        with self._lock:
            script = self.units.get(key)
            if script is None:
                script = UnitScript()
                self.units[key] = script
            return script

    def script(self, key: str, **changes: Any) -> UnitScript:
        """Replace fields of unit *key*'s script and return it."""
        script = self.unit(key)
        for name, value in changes.items():
            if not hasattr(script, name):
                raise AttributeError(f"UnitScript has no field {name!r}")
            setattr(script, name, value)
        return script

    def reply(self, key: str, attempt: int, reply: Reply) -> None:
        """Script answer *attempt* (from 1) for unit *key*."""
        self.unit(key).replies[attempt] = reply

    def reset_records(self) -> None:
        """Forget recorded calls, constructions and attempt counters."""
        with self._lock:
            self.calls.clear()
            self.constructions.clear()
            self._attempts.clear()

    # -- answers ---------------------------------------------------------------
    def payload_for(
        self, role: Role, unit: str | None, text: str, schema: Any = None
    ) -> Any:
        """Return the answer object of a request: scripted or generated."""
        script = self.units.get(unit) if unit is not None else None
        if script is not None and script.payload is not None:
            return copy.deepcopy(dict(script.payload))
        if role == ADJUST:
            return boundary_payload(text)[1]
        return example_payload(schema if schema else self.schema, unit)

    def scripted_reply(
        self, unit: str | None, attempt: int
    ) -> tuple[Reply | None, Form]:
        """Return the reply scripted for *attempt* of *unit* and the unit's form."""
        script = self.units.get(unit) if unit is not None else None
        if script is None:
            return None, "schema"
        return script.replies.get(attempt, script.always), script.form

    # -- provider classes ------------------------------------------------------
    def constructor(self, name: str) -> Callable[..., ScriptedChatModel]:
        """Return a stand-in for the provider class *name*."""

        def construct(*args: Any, **kwargs: Any) -> ScriptedChatModel:
            if args:
                raise TypeError(f"{name} stand-in takes keyword arguments only")
            construction = Construction(
                constructor=name,
                kwargs=kwargs,
                provider=provider_of(name, kwargs.get("base_url")),
                model=str(kwargs.get("model", kwargs.get("model_name", ""))),
            )
            with self._lock:
                self.constructions.append(construction)
            if self.sdk_defaults:
                return SDKDefaultsChatModel(
                    responder=self.respond,
                    recorder=self.record,
                    tag=construction,
                    default_method=SDK_STRUCTURED_METHODS.get(name, "function_calling"),
                )
            return ScriptedChatModel(
                responder=self.respond, recorder=self.record, tag=construction
            )

        return construct

    @property
    def models(self) -> list[dict[str, Any]]:
        """Return the golden entry of every recorded construction."""
        with self._lock:
            return [construction.record() for construction in self.constructions]

    # -- answering -------------------------------------------------------------
    def record(self, request: FakeRequest) -> None:
        """Record the digest of one request; note its role, unit and attempt.

        A schema enforced through invoke kwargs is recorded as its
        structured-output call (see ``_enforcement``).
        """
        construction: Construction = request.tag
        shapes = _message_shapes(request.messages)
        kwargs, enforced, enforced_schema = _enforcement(request.kwargs)
        structured = _describe_structured(request) or enforced
        schema = _structured_schema(request.structured) or enforced_schema
        role: Role = ADJUST if is_boundary_request(shapes.system, schema) else EXTRACT
        if role == ADJUST:
            unit = boundary_payload(shapes.user)[0]
        elif shapes.page is not None:
            unit = page_unit(shapes.page)
        else:
            unit = text_unit(shapes.user)
        with self._lock:
            attempt = self._attempts.get(unit, 0) + 1
            self._attempts[unit] = attempt
            self.calls.append(
                {
                    "role": role,
                    "unit": unit,
                    "attempt": attempt,
                    "provider": construction.provider,
                    "model": construction.model,
                    "kwargs": describe_kwargs(kwargs),
                    "bound": describe_kwargs(request.bound),
                    "structured": structured,
                    "messages": shapes.messages,
                }
            )
        request.notes.update(
            role=role, unit=unit, attempt=attempt, schema=schema, text=shapes.user
        )

    def respond(self, request: FakeRequest) -> AIMessage:
        """Return the scripted answer to a recorded request."""
        construction: Construction = request.tag
        notes = request.notes
        reply, form = self.scripted_reply(notes["unit"], notes["attempt"])
        metadata: dict[str, Any] = {
            "model_name": construction.model,
            "status": "completed",
        }
        additional: dict[str, Any] = {}
        if reply is not None:
            if reply.exception is not None:
                raise reply.exception
            if reply.response_metadata is not None:
                metadata = dict(reply.response_metadata)
            if reply.refusal is not None:
                additional["refusal"] = reply.refusal
            if reply.form is not None:
                form = reply.form
        if reply is not None and reply.text is not None:
            content = reply.text
        else:
            payload = self.payload_for(
                notes["role"], notes["unit"], notes["text"], notes["schema"]
            )
            content = render(payload, form)
        return AIMessage(
            content=content,
            usage_metadata=dict(USAGE),
            response_metadata=metadata,
            additional_kwargs=additional,
        )

    def sorted_calls(self) -> list[dict[str, Any]]:
        """Return the call digests in a concurrency-independent order."""

        def key(call: dict[str, Any]) -> tuple[str, tuple[int, str, int], int]:
            return call["role"], _unit_sort_key(call["unit"]), call["attempt"]

        with self._lock:
            return sorted((copy.deepcopy(call) for call in self.calls), key=key)


__all__ = [
    "ADJUST",
    "BOUNDARY_SENTINEL",
    "CUSTOM_KEY_ENV",
    "DUMMY_KEYS",
    "EXTRACT",
    "PROVIDER_CLASSES",
    "USAGE",
    "Construction",
    "FakeAPIError",
    "FakeLLM",
    "Form",
    "Reply",
    "UnitScript",
    "adjust_unit",
    "boundary_payload",
    "example_payload",
    "is_boundary_request",
    "page_unit",
    "permanent_error",
    "record_unit",
    "redact",
    "render",
    "summarize_schema",
    "text_unit",
    "transient_error",
    "unit_number",
]
