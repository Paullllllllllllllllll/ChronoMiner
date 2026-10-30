"""The run specifications and the option tables that build them.

``OPTIONS`` holds the extract options: the CLI parser, the settings
``defaults`` block and the equivalent command line derive from it. The
ranges, repair and batch check tables reuse its schema, chunking, model,
image and format options under the same names, so a settings default of
those options applies there too. Image options left unset take the
provider's recommended value (:func:`recommended_images`).

``--schema`` names a schema file or a schema of the repository's
``schemas/`` folder; :func:`resolve_schema` finds it. ``--classifier`` of
the classify commands names a classifier file of ``classifiers/`` the same
way (:func:`~chronominer.common.classifiers.resolve_classifier`). A spec
keeps the argument as given.

A run that calls a model refuses a provider set at a higher precedence level
than a model of another provider. List options given more than once
accumulate (:class:`~chronominer.common.options.Option`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal, cast

from chronominer.common.classifiers import load_classifier
from chronominer.common.image_presets import (
    DetailCheck,
    DetailRules,
    ImagePresets,
    anthropic_preset,
    custom_preset,
    google_preset,
    openai_preset,
)
from chronominer.common.options import (
    Condition,
    Kind,
    Option,
    OptionTable,
    Resolved,
    Source,
)
from chronominer.common.polling import DEFAULT_POLL_MINUTES, wait_options
from chronominer.common.selection import parse_page_range
from chronominer.common.standard_options import (
    DEFAULT_DECISION_MODEL,
    DRY_RUN_HELP,
    IMAGE_DETAILS,
    IMAGE_FORMATS,
    JSON_HELP,
    REASONING_EFFORTS,
    SERVICE_TIERS,
    VERBOSITIES,
    Dpi,
    abandon_option,
    adopt_option,
    batch_cancel_options,
    batch_ids_option,
    batch_option,
    check_input_option,
    classifier_options,
    concurrency_option,
    decision_options,
    dry_run_option,
    force_option,
    image_options,
    json_option,
    model_options,
    nonempty_text,
    output_options,
    positive_int,
    render_options,
    settings_option,
)
from chronominer.providers import (
    PROVIDERS,
    api_model_id,
    implied_provider,
    infer_provider,
    split_provider_prefix,
)

__all__ = [
    "BATCH_CANCEL_OPTIONS",
    "BATCH_CHECK_OPTIONS",
    "CHUNKING_MODES",
    "CLASSIFIERS_DIR",
    "CLASSIFY_CHUNKS_OPTIONS",
    "CLASSIFY_DOCUMENTS_OPTIONS",
    "CLASSIFY_ENTRIES_OPTIONS",
    "CLASSIFY_UNITS",
    "CONTEXT_AUTO",
    "CONTEXT_NONE",
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_FORMATS",
    "DEFAULT_MODEL",
    "DROP_OPTIONS",
    "IMAGE_DETAILS",
    "IMAGE_FORMATS",
    "IMAGE_PRESETS",
    "INPUT_TYPES",
    "LAYOUTS",
    "MODEL_SUGGESTIONS",
    "OPTIONS",
    "OUTPUT_FORMATS",
    "PROVIDERS",
    "PROVIDER_TABLES",
    "RANGES_ADJUST_OPTIONS",
    "RANGES_GENERATE_OPTIONS",
    "REASONING_EFFORTS",
    "RENDER_OPTIONS",
    "REPAIR_OPTIONS",
    "RESPONSE_FORMATS",
    "SCHEMAS_DIR",
    "SERVICE_TIERS",
    "VERBOSITIES",
    "WAIT_HOURS",
    "BatchCancelSpec",
    "BatchCheckSpec",
    "ChunkingSpec",
    "ClassifierChoice",
    "ClassifyChunking",
    "ClassifyControl",
    "ClassifyEntriesSpec",
    "ClassifyItemsSpec",
    "ClassifyOutput",
    "ClassifySpec",
    "CommandSpec",
    "ContextChoice",
    "ContextText",
    "Dpi",
    "DecisionSpec",
    "DropSpec",
    "ImageSpec",
    "InputSpec",
    "ModelSpec",
    "OutputSpec",
    "PromptSpec",
    "RangesAdjustSpec",
    "RangesControl",
    "RangesGenerateSpec",
    "RenderSpec",
    "RepairControl",
    "RepairSpec",
    "Resolution",
    "ResolvedSchema",
    "RunControl",
    "RunSpec",
    "SpecError",
    "classifier_names",
    "condition_warnings",
    "effective_spec",
    "inapplicable",
    "item_type",
    "model_suggestions",
    "parse",
    "parse_batch_cancel",
    "parse_batch_check",
    "parse_drop",
    "parse_ranges_adjust",
    "parse_ranges_generate",
    "parse_render",
    "parse_repair",
    "recommended_images",
    "resolve_batch_cancel",
    "resolve_batch_check",
    "resolve_classify_entries",
    "resolve_classify_items",
    "resolve_drop",
    "resolve_ranges_adjust",
    "resolve_ranges_generate",
    "resolve_render",
    "resolve_repair",
    "resolve_schema",
    "resolve_spec",
    "schema_names",
    "to_argv",
]

logger = logging.getLogger(__name__)

INPUT_TYPES = ("auto", "text", "images", "pdf")
"""The ``--type`` values; ``images`` selects the image files (item type image)."""
CHUNKING_MODES = ("auto", "line-ranges", "adjust")
LAYOUTS = ("flat", "mirror")
OUTPUT_FORMATS = ("csv",)
DEFAULT_FORMATS: tuple[str, ...] = ()
RESPONSE_FORMATS = ("schema", "json")

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_VERBOSITY = "medium"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_SERVICE_TIER = "default"
DEFAULT_CONCURRENCY = 16
DEFAULT_CHUNK_TOKENS = 3500
DEFAULT_CONTEXT_WINDOW = 6
WAIT_HOURS = 50.0
"""Default ``--wait-hours`` of ``batch check``: the longest provider batch
window (Gemini, 48 hours) plus time to collect the results."""

CONTEXT_AUTO = "auto"
CONTEXT_NONE: Final = "none"

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"
"""The repository's schema folder; public and private schemas share it."""

CLASSIFIERS_DIR = Path(__file__).resolve().parent.parent / "classifiers"
"""The repository's classifier folder, read like :data:`SCHEMAS_DIR`."""

MODEL_SUGGESTIONS: Mapping[str, tuple[str, ...]] = {
    "openai": (
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.5",
    ),
    "anthropic": ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
    "google": ("gemini-3.7-flash", "gemini-3.1-pro", "gemini-3.5-flash"),
    "openrouter": (
        "openai/gpt-5.6-luna",
        "anthropic/claude-sonnet-5-5",
        "google/gemini-3.1-pro",
    ),
}

type ContextChoice = Path | Literal["none"] | None
"""A context image, :data:`CONTEXT_NONE`, or None for the automatic lookup."""

type ContextText = str | None
"""Context as given: topics as text or a path to a text file,
:data:`CONTEXT_NONE`, or None for the automatic lookup."""

_ITEM_TYPES: Final = {"images": "image"}


def item_type(kind: str) -> str:
    """Return the item type a ``--type`` value selects (``images``: image)."""
    return _ITEM_TYPES.get(kind, kind)


class SpecError(ValueError):
    """The options do not form a valid run specification."""


@dataclass(frozen=True)
class InputSpec:
    """The input path, its type and the selection of items and chunks.

    ``type`` is auto, text, images or pdf. A folder holding several items
    needs one selection: ``select`` (numbers, ranges or a name search),
    ``all`` or ``files`` (names inside the folder). ``pages`` selects chunks
    (pages of a PDF, the one chunk of an image, line ranges of a text).
    """

    path: Path
    type: str
    files: tuple[str, ...]
    recursive: bool
    pages: str | None
    select: str | None = None
    all: bool = False


@dataclass(frozen=True)
class OutputSpec:
    """Where and what to write; ``directory`` None means beside the input.

    ``formats`` are the rendered formats besides ``{stem}_output.json``; the
    database and the JSON output are always written.
    """

    directory: Path | None
    layout: str
    formats: tuple[str, ...]


@dataclass(frozen=True)
class PromptSpec:
    """Schema and context of the extraction calls.

    ``schema`` is the ``--schema`` argument as given (see
    :func:`resolve_schema`). ``context`` is topics as text or a path to a
    text file; None looks up the input's sidecar file, and
    ``fallback_context`` (text or a path) serves an input without one.
    ``context_image`` None looks up the input's sidecar image;
    :data:`CONTEXT_NONE` sends no context.
    """

    schema: str
    context: ContextText
    context_image: ContextChoice
    fallback_context: ContextText = None


@dataclass(frozen=True)
class ChunkingSpec:
    """How text inputs are cut into chunks.

    ``mode`` auto cuts by ``chunk_tokens``; line-ranges reads the input's
    ``{stem}_line_ranges.txt`` (written by ``chunk_tokens`` when missing);
    adjust also moves the range boundaries with the model, looking at
    ``context_window`` lines around each boundary.
    """

    mode: str
    chunk_tokens: int
    context_window: int


@dataclass(frozen=True)
class ModelSpec:
    """The model of the extraction and adjust calls.

    ``provider`` is a provider name once resolved, and ``model`` carries no
    ``provider:`` prefix. ``endpoint`` names a custom endpoint from the
    settings. ``max_output_tokens`` None takes the model's output limit.
    ``response_format`` None means schema when the model supports it, else
    json.
    """

    provider: str
    model: str
    endpoint: str | None
    reasoning_effort: str
    verbosity: str
    max_output_tokens: int | None
    temperature: float
    top_p: float
    service_tier: str
    response_format: str | None


@dataclass(frozen=True)
class ImageSpec:
    """Page image settings; None takes the provider's recommended value."""

    dpi: Dpi | None = None
    detail: str | None = None
    format: str | None = None
    jpeg_quality: int | None = None
    grayscale: bool | None = None

    def over(self, base: ImageSpec) -> ImageSpec:
        """Return these values with unset fields taken from ``base``."""
        return ImageSpec(
            *(
                getattr(self, f.name)
                if getattr(self, f.name) is not None
                else getattr(base, f.name)
                for f in fields(self)
            )
        )


@dataclass(frozen=True)
class RunControl:
    """Batch, resume, concurrency and agent-contract switches."""

    batch: bool
    force: bool
    concurrency: int
    dry_run: bool
    json: bool


@dataclass(frozen=True)
class DecisionSpec:
    """The decision-model pre-pass: mode, model id, dialect and endpoint.

    ``provider`` None resolves the dialect from the model id.
    """

    mode: str
    model: str
    provider: str | None
    endpoint: str | None


@dataclass(frozen=True)
class RunSpec:
    """Everything one extract run needs besides the machine settings."""

    input: InputSpec
    output: OutputSpec
    prompt: PromptSpec
    chunking: ChunkingSpec
    model: ModelSpec
    images: ImageSpec
    run: RunControl
    decision: DecisionSpec = field(
        default_factory=lambda: DecisionSpec("off", DEFAULT_DECISION_MODEL, None, None)
    )


@dataclass(frozen=True)
class RangesGenerateSpec:
    """Write ``{stem}_line_ranges.txt`` for the text inputs at ``input``.

    ``model`` names the model whose tokenizer counts ``chunk_tokens``.
    """

    input: Path
    recursive: bool
    chunk_tokens: int
    model: str
    dry_run: bool
    json: bool


@dataclass(frozen=True)
class RangesControl:
    """Resume, concurrency and agent-contract switches of a ranges adjust."""

    force: bool
    concurrency: int
    dry_run: bool
    json: bool


@dataclass(frozen=True)
class RangesAdjustSpec:
    """Move the line-range boundaries of the text inputs at ``input``.

    ``schema`` names the entries whose boundaries the model looks for;
    ``context`` is the adjust context (None looks up the input's sidecar,
    then takes ``fallback_context``); ``pages`` selects line ranges.
    """

    input: Path
    recursive: bool
    schema: str
    context_window: int
    context: ContextText
    pages: str | None
    model: ModelSpec
    run: RangesControl
    fallback_context: ContextText = None


@dataclass(frozen=True)
class RepairControl:
    """Batch and agent-contract switches of a repair."""

    batch: bool
    dry_run: bool
    json: bool


@dataclass(frozen=True)
class RepairSpec:
    """Run the failed chunks of the documents found at ``input`` again.

    ``schema`` limits the repair to documents extracted with that schema;
    ``indices`` to the given 0-based chunk indices among the failed ones.
    ``skipped`` selects the chunks the decision pre-pass skipped instead.
    """

    input: Path
    schema: str | None
    indices: tuple[int, ...] | None
    model: ModelSpec
    images: ImageSpec
    run: RepairControl
    skipped: bool = False


@dataclass(frozen=True)
class BatchCheckSpec:
    """Check and finalize batch submissions found below ``directory``.

    ``schema`` limits the check to documents extracted with that schema.
    ``formats`` None renders each finalized document in the formats of the
    run that submitted it. ``adopt`` names a provider batch to record for
    an unresolved submission; ``abandon`` names an unresolved or submitted
    submission whose units return to pending. With ``wait`` the check repeats every
    ``poll_minutes`` until no batch work is pending; no check starts after
    ``wait_hours``.
    """

    directory: Path
    batch_ids: tuple[str, ...]
    formats: tuple[str, ...] | None
    json: bool
    adopt: str | None = None
    abandon: str | None = None
    schema: str | None = None
    wait: bool = False
    poll_minutes: float = DEFAULT_POLL_MINUTES
    wait_hours: float = WAIT_HOURS


@dataclass(frozen=True)
class BatchCancelSpec:
    """The options of ``batch cancel``.

    The command cancels the named batches, every unfinished batch with
    ``all``, or the open submissions of the databases below ``directory``.
    """

    batch_ids: tuple[str, ...]
    all: bool
    json: bool
    directory: Path | None = None


@dataclass(frozen=True)
class RenderSpec:
    """The options of ``render``; ``formats`` None renders the registered ones."""

    input: Path
    formats: tuple[str, ...] | None = None
    force: bool = False
    dry_run: bool = False
    json: bool = False


@dataclass(frozen=True)
class DropSpec:
    """The options of ``drop``: the store and exactly one document selection."""

    input: Path
    select: str | None = None
    files: tuple[str, ...] = ()
    all: bool = False
    dry_run: bool = False
    json: bool = False


@dataclass(frozen=True)
class ClassifierChoice:
    """The classifier of a classify command and its decision model.

    ``classifier`` is the ``--classifier`` argument as given;
    ``provider`` None resolves the dialect from the model id. ``force``
    asks again instead of reusing stored answers.
    """

    classifier: str
    force: bool
    model: str
    provider: str | None
    endpoint: str | None


@dataclass(frozen=True)
class ClassifyControl:
    """The agent-contract switches of a classify command."""

    dry_run: bool
    json: bool


@dataclass(frozen=True)
class ClassifyOutput:
    """The output root whose store a classify command reads and writes."""

    directory: Path | None
    layout: str


@dataclass(frozen=True)
class ClassifyChunking:
    """How a classify command cuts text items the store does not hold."""

    mode: str
    chunk_tokens: int


@dataclass(frozen=True)
class ClassifyItemsSpec:
    """``classify chunks`` or ``classify documents`` over input items.

    ``unit`` is ``chunk`` or ``document``. ``model`` names the model whose
    tokenizer counts ``chunk_tokens`` and whose image presets size the
    pages of items the store does not hold.
    """

    unit: str
    input: InputSpec
    output: ClassifyOutput
    chunking: ClassifyChunking
    model: str
    images: ImageSpec
    classify: ClassifierChoice
    run: ClassifyControl


@dataclass(frozen=True)
class ClassifyEntriesSpec:
    """``classify entries`` over the stores at or below ``input``."""

    input: Path
    classify: ClassifierChoice
    run: ClassifyControl


type ClassifySpec = ClassifyItemsSpec | ClassifyEntriesSpec

type CommandSpec = (
    RunSpec
    | RangesGenerateSpec
    | RangesAdjustSpec
    | RepairSpec
    | BatchCheckSpec
    | BatchCancelSpec
    | RenderSpec
    | DropSpec
    | ClassifyItemsSpec
    | ClassifyEntriesSpec
)


@dataclass(frozen=True)
class ResolvedSchema:
    """A schema file: its ``name`` field (else the file stem), path and JSON."""

    name: str
    path: Path
    schema: Mapping[str, Any]


def _read_schema(path: Path) -> ResolvedSchema:
    """Return the schema in *path*.

    Raises:
        SpecError: The file cannot be read or holds no JSON object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read schema {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"schema {path} must hold a JSON object")
    name = data.get("name")
    return ResolvedSchema(
        name=name if isinstance(name, str) and name.strip() else path.stem,
        path=path,
        schema=data,
    )


def _folder_schemas(schemas_dir: Path) -> list[ResolvedSchema]:
    """Return the readable schemas of *schemas_dir*, sorted by file name."""
    if not schemas_dir.is_dir():
        return []
    found: list[ResolvedSchema] = []
    for path in sorted(schemas_dir.glob("*.json")):
        try:
            found.append(_read_schema(path))
        except SpecError as exc:
            logger.warning("Skipping schema file: %s", exc)
    return found


def schema_names(schemas_dir: Path | None = None) -> tuple[str, ...]:
    """Return the names of the schemas in *schemas_dir* (the repo's by default)."""
    folder = SCHEMAS_DIR if schemas_dir is None else schemas_dir
    return tuple(schema.name for schema in _folder_schemas(folder))


def resolve_schema(argument: str, schemas_dir: Path | None = None) -> ResolvedSchema:
    """Return the schema a ``--schema`` argument names.

    The argument is, in this order: an existing file path; a file
    ``<argument>.json`` (or ``<argument>`` ending in ``.json``) in
    *schemas_dir*; a schema in *schemas_dir* whose ``name`` field equals
    the argument. *schemas_dir* defaults to :data:`SCHEMAS_DIR`.

    Raises:
        SpecError: No schema matches, or the matching file is not a JSON
            object; the message names the available schemas.
    """
    folder = SCHEMAS_DIR if schemas_dir is None else schemas_dir
    text = argument.strip()
    if not text:
        raise SpecError("--schema needs a schema name or path")
    given = Path(text).expanduser()
    if given.is_file():
        return _read_schema(given)
    names = [f"{text}.json"]
    if text.lower().endswith(".json"):
        names.insert(0, text)
    for name in names:
        candidate = folder / name
        if Path(name).name == name and candidate.is_file():
            return _read_schema(candidate)
    available = _folder_schemas(folder)
    for schema in available:
        if schema.name == text:
            return schema
    listing = ", ".join(sorted(schema.name for schema in available)) or "none"
    raise SpecError(
        f"unknown schema {argument!r}: not a file, and no schema in {folder} "
        f"has that file or name; available: {listing}"
    )


def classifier_names(folder: Path | None = None) -> tuple[str, ...]:
    """Return the names of the valid classifiers in *folder* (the repo's)."""
    folder = CLASSIFIERS_DIR if folder is None else folder
    if not folder.is_dir():
        return ()
    names: list[str] = []
    for path in sorted(folder.glob("*.json")):
        try:
            names.append(load_classifier(path).name)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping classifier file: %s", exc)
    return tuple(names)


IMAGE_PRESETS = ImagePresets(
    families={
        "openai": openai_preset(),
        "anthropic": anthropic_preset(),
        "google": google_preset(),
        "custom": custom_preset(),
    },
    detail=DetailRules(
        checks={
            "openai": DetailCheck(
                kept=frozenset({"low", "high", "auto"}), original=True
            ),
            "openrouter": DetailCheck(kept=frozenset({"low", "high", "auto"})),
        },
    ),
)
"""The image presets of each model family: the values the wizard recommends
and the request sections of ``extract.images``. Every family sends colour
JPEG at quality 95. An item
whose pages are stored keeps the stored values of every image option the
run leaves unset. OpenAI and OpenRouter keep low, high and auto, and OpenAI
models that take it original; any other detail becomes high."""

_RECOMMENDED_IMAGES: Mapping[str, ImageSpec] = {
    family: ImageSpec(**asdict(preset.recommended))
    for family, preset in IMAGE_PRESETS.families.items()
}


def recommended_images(family: str) -> ImageSpec:
    """Return the recommended image settings of a model family.

    ``family`` is openai, anthropic, google or custom; OpenRouter models use
    the family of the underlying model, unknown families the custom values.
    """
    return _RECOMMENDED_IMAGES.get(family, _RECOMMENDED_IMAGES["custom"])


def model_suggestions(provider: str | None = None) -> tuple[str, ...]:
    """Return suggested model names of *provider*, or of every provider.

    Suggestions are not enforced; a custom endpoint has none.
    """
    if provider is None:
        names = (name for names in MODEL_SUGGESTIONS.values() for name in names)
        return tuple(dict.fromkeys(names))
    return MODEL_SUGGESTIONS.get(provider, ())


def _pages(value: str) -> str:
    parse_page_range(value)
    return value.strip()


def _text_or_file(value: str) -> str:
    """Return *value*, or its absolute path when it names an existing file."""
    try:
        path = Path(nonempty_text(value)).expanduser()
        is_file = path.is_file()
    except (OSError, RuntimeError):
        return value
    return str(path.resolve()) if is_file else value


def _context(value: str) -> ContextText:
    keyword = value.strip().lower()
    if keyword == CONTEXT_AUTO:
        return None
    if keyword == CONTEXT_NONE:
        return CONTEXT_NONE
    return _text_or_file(value)


def _fallback_context(value: str) -> ContextText:
    if value.strip().lower() == CONTEXT_NONE:
        return None
    return _text_or_file(value)


def _context_image(value: str) -> ContextChoice:
    keyword = value.strip().lower()
    if keyword == CONTEXT_AUTO:
        return None
    if keyword == CONTEXT_NONE:
        return CONTEXT_NONE
    return Path(nonempty_text(value)).expanduser().resolve()


def _parse_indices(value: str) -> tuple[int, ...]:
    """Parse 0-based indices such as ``0,5,12`` or ``3-7``."""
    indices: set[int] = set()
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            continue
        if part.startswith("-"):
            raise ValueError(f"invalid index {part!r}: indices are not negative")
        start, separator, end = part.partition("-")
        try:
            low = int(start)
            high = int(end) if separator else low
        except ValueError:
            raise ValueError(f"invalid index or range {part!r}") from None
        if low > high:
            raise ValueError(f"range start exceeds its end in {part!r}")
        indices.update(range(low, high + 1))
    if not indices:
        raise ValueError("no indices given")
    return tuple(sorted(indices))


def _format_indices(indices: Iterable[int]) -> str:
    """Return sorted indices with consecutive runs written as ranges."""
    runs: list[list[int]] = []
    for index in sorted(set(indices)):
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def _indices(value: str) -> str:
    return _format_indices(_parse_indices(value))


_ADJUST = Condition("chunking", frozenset({"adjust"}))


def _prompt_options() -> list[Option]:
    return [
        Option(
            "schema",
            "--schema",
            (
                "extraction schema: a file path, a file name in schemas/ "
                "without .json, or the name field of a schema there"
            ),
            group="prompt",
            type=nonempty_text,
            metavar="NAME|PATH",
            choice_provider=schema_names,
            default_help="required",
            target="prompt.schema",
        ),
        Option(
            "context",
            "--context",
            (
                "extraction context: topics as text or a path to a text file "
                "(overrides sidecars), auto (the input's "
                "{stem}_extract_context.txt, else the folder's context file) or "
                "none"
            ),
            group="prompt",
            type=_context,
            metavar="TEXT|PATH|auto|none",
            default_help=CONTEXT_AUTO,
            target="prompt.context",
        ),
        Option(
            "fallback_context",
            "--fallback-context",
            (
                "extraction context for inputs without a sidecar under "
                "--context auto: topics as text, a path to a text file, or none"
            ),
            group="prompt",
            type=_fallback_context,
            metavar="TEXT|PATH|none",
            default_help=CONTEXT_NONE,
            target="prompt.fallback_context",
        ),
        Option(
            "context_image",
            "--context-image",
            (
                "image sent with every chunk: an image file, auto (the "
                "input's {stem}_extract_context image) or none"
            ),
            group="prompt",
            type=_context_image,
            metavar="PATH|auto|none",
            default=CONTEXT_NONE,
            target="prompt.context_image",
        ),
    ]


def _chunking_options() -> list[Option]:
    return [
        Option(
            "chunking",
            "--chunking",
            (
                "text chunking: auto (by --chunk-tokens), line-ranges (the "
                "input's {stem}_line_ranges.txt) or adjust (line ranges whose "
                "boundaries the model moves first); adjust rewrites that file "
                "beside the input, not under --output"
            ),
            group="chunking",
            choices=CHUNKING_MODES,
            default="auto",
            target="chunking.mode",
        ),
        Option(
            "chunk_tokens",
            "--chunk-tokens",
            "target tokens per text chunk, also for line ranges written anew",
            group="chunking",
            type=positive_int,
            metavar="N",
            default=DEFAULT_CHUNK_TOKENS,
            target="chunking.chunk_tokens",
        ),
        Option(
            "context_window",
            "--context-window",
            "lines around each range boundary sent to the model",
            group="chunking",
            type=positive_int,
            metavar="N",
            default=DEFAULT_CONTEXT_WINDOW,
            condition=_ADJUST,
            target="chunking.context_window",
        ),
    ]


def _model_options() -> list[Option]:
    return model_options(
        providers=PROVIDERS,
        suggestions=model_suggestions,
        defaults={
            "model": DEFAULT_MODEL,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "verbosity": DEFAULT_VERBOSITY,
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "service_tier": DEFAULT_SERVICE_TIER,
        },
        response_formats=RESPONSE_FORMATS,
    )


def _decision_options() -> list[Option]:
    return decision_options(
        "decision-model pre-pass: shadow records each chunk's relevance to the "
        "schema; apply also skips the chunks below the validated threshold"
    )


def _image_options() -> list[Option]:
    return image_options()


_GROUP_TITLES = {
    "input": "input",
    "output": "output",
    "prompt": "schema and context",
    "chunking": "chunking",
    "model": "model",
    "decision model": "decision model",
    "images": "images",
    "selection": "chunks to repair",
    "batch": "batches",
    "classify": "classifier",
    "run": "run control",
}


def _input_options() -> list[Option]:
    return [
        Option(
            "input",
            "--input",
            "text (.txt, .md), image or PDF file, or a folder of these",
            group="input",
            type=Path,
            metavar="PATH",
            target="input.path",
            settable=False,
        ),
        Option(
            "type",
            "--type",
            (
                "input type; auto takes the text, image and PDF files of a "
                "folder and detects a file's type by its extension"
            ),
            group="input",
            choices=INPUT_TYPES,
            default="auto",
            target="input.type",
        ),
        Option(
            "select",
            "--select",
            "select items by number (1,3), range (1-5) or name search",
            group="input",
            type=nonempty_text,
            metavar="EXPR",
            exclusive="selection",
            target="input.select",
        ),
        Option(
            "all",
            "--all",
            "process every discovered item",
            group="input",
            kind=Kind.FLAG,
            default=False,
            exclusive="selection",
            target="input.all",
        ),
        Option(
            "files",
            "--files",
            "comma-separated names of files inside the input folder",
            group="input",
            kind=Kind.LIST,
            type=nonempty_text,
            metavar="LIST",
            default=(),
            exclusive="selection",
            target="input.files",
            settable=False,
        ),
        Option(
            "recursive",
            "--recursive",
            "search the subfolders of the input folder",
            group="input",
            kind=Kind.SWITCH,
            default=False,
            target="input.recursive",
        ),
        Option(
            "pages",
            "--pages",
            (
                "chunks (PDF pages, line ranges): 5, first:5, last:5, 3-7, 3-, "
                "-7 or 1,3,5-8"
            ),
            group="input",
            type=_pages,
            metavar="RANGE",
            default_help="all",
            target="input.pages",
            settable=False,
        ),
    ]


def _output_options() -> list[Option]:
    return [
        *output_options(),
        Option(
            "layout",
            "--layout",
            (
                "output folder layout: flat (every output in the output "
                "folder) or mirror (the input folder tree)"
            ),
            group="output",
            choices=LAYOUTS,
            default="flat",
            target="output.layout",
        ),
        Option(
            "formats",
            "--formats",
            (
                "comma-separated formats rendered besides {stem}_output.json "
                "and chronominer.sqlite"
            ),
            group="output",
            kind=Kind.LIST,
            choices=OUTPUT_FORMATS,
            default=DEFAULT_FORMATS,
            default_help="none",
            metavar="LIST",
            target="output.formats",
        ),
    ]


def _run_options() -> list[Option]:
    return [
        batch_option("submit the chunks to the provider's batch API"),
        force_option(
            "extract inputs again from the start: every stored unit of the "
            "input is reset, also outside --pages; stored responses are kept"
        ),
        concurrency_option(DEFAULT_CONCURRENCY),
        dry_run_option("run.dry_run"),
        json_option("run.json"),
        settings_option(),
    ]


OPTIONS = OptionTable(
    [
        *_input_options(),
        *_output_options(),
        *_prompt_options(),
        *_chunking_options(),
        *_model_options(),
        *_decision_options(),
        *_image_options(),
        *_run_options(),
    ],
    group_titles=_GROUP_TITLES,
)

_MODEL_NAMES = tuple(option.name for option in _model_options())
_IMAGE_NAMES = tuple(option.name for option in _image_options())


def _input_option(help: str, *, metavar: str = "PATH", **extra: Any) -> Option:
    return Option(
        "input",
        "--input",
        help,
        group="input",
        type=Path,
        metavar=metavar,
        target="input",
        settable=False,
        **extra,
    )


RANGES_GENERATE_OPTIONS = OptionTable(
    [
        _input_option("text file, or a folder of text files, to write line ranges for"),
        replace(OPTIONS["recursive"], target="recursive"),
        replace(OPTIONS["chunk_tokens"], target="chunk_tokens"),
        replace(
            OPTIONS["model"],
            help="model whose tokenizer counts the chunk tokens",
            target="model",
        ),
        dry_run_option("dry_run"),
        json_option("json"),
        settings_option(),
    ],
    group_titles=_GROUP_TITLES,
)

RANGES_ADJUST_OPTIONS = OptionTable(
    [
        _input_option(
            "text file, or a folder of text files, whose line ranges to adjust"
        ),
        replace(OPTIONS["recursive"], target="recursive"),
        replace(
            OPTIONS["schema"],
            help=(
                "schema of the entries whose boundaries the model looks for: "
                "a file path, a file name in schemas/ or a schema's name field"
            ),
            target="schema",
        ),
        replace(OPTIONS["context_window"], condition=None, target="context_window"),
        replace(
            OPTIONS["context"],
            help=(
                "adjust context: topics as text or a path to a text file "
                "(overrides sidecars), auto (the input's "
                "{stem}_adjust_context.txt, else the folder's context file) or "
                "none"
            ),
            target="context",
            settable=False,
        ),
        replace(
            OPTIONS["fallback_context"],
            help=(
                "adjust context for inputs without a sidecar under --context "
                "auto: topics as text, a path to a text file, or none"
            ),
            target="fallback_context",
            settable=False,
        ),
        replace(
            OPTIONS["pages"],
            help="line ranges: 5, first:5, last:5, 3-7, 3-, -7 or 1,3,5-8",
            target="pages",
        ),
        *(OPTIONS[name] for name in _MODEL_NAMES),
        replace(
            OPTIONS["force"],
            help="adjust the line ranges again: the stored decisions are reset",
        ),
        OPTIONS["concurrency"],
        dry_run_option("run.dry_run"),
        json_option("run.json"),
        settings_option(),
    ],
    group_titles=_GROUP_TITLES,
)


def _schema_filter(help: str) -> Option:
    return Option(
        "schema",
        "--schema",
        help,
        group="batch",
        type=nonempty_text,
        metavar="NAME",
        default_help="every schema",
        target="schema",
        settable=False,
    )


REPAIR_OPTIONS = OptionTable(
    [
        _input_option(
            (
                "folder, chronominer.sqlite or input whose failed chunks to run "
                "again; subfolders included"
            ),
            default=Path("."),
            default_help="the current directory",
        ),
        _schema_filter("repair only documents extracted with this schema"),
        Option(
            "indices",
            "--indices",
            "0-based chunk indices (order in the document) among the failed "
            "chunks: 0,5,12 or 3-7",
            group="selection",
            type=_indices,
            metavar="LIST",
            default_help="every failed chunk",
            settable=False,
        ),
        Option(
            "skipped",
            "--skipped",
            "extract the chunks the decision pre-pass skipped as irrelevant "
            "instead of the failed ones",
            group="selection",
            kind=Kind.FLAG,
            default=False,
            settable=False,
            target="skipped",
        ),
        *(OPTIONS[name] for name in (*_MODEL_NAMES, *_IMAGE_NAMES)),
        batch_option("submit the repairs to the provider's batch API"),
        dry_run_option("run.dry_run"),
        json_option("run.json"),
        settings_option(),
    ],
    group_titles=_GROUP_TITLES,
)


BATCH_CHECK_OPTIONS = OptionTable(
    [
        check_input_option(),
        _schema_filter("check only documents extracted with this schema"),
        batch_ids_option(
            "comma-separated batch IDs to check", default_help="every batch"
        ),
        adopt_option(),
        abandon_option(
            "give up an unresolved or submitted submission: its chunks return to "
            "pending and its uploaded files are deleted; a submitted batch is "
            "not cancelled"
        ),
        *wait_options(group="batch", default_hours=WAIT_HOURS),
        replace(
            OPTIONS["formats"],
            default=None,
            default_help="the formats of the run that submitted each document",
            target="formats",
            settable=False,
        ),
        json_option("json"),
        settings_option(),
    ],
    group_titles=_GROUP_TITLES,
)

BATCH_CANCEL_OPTIONS = OptionTable(
    batch_cancel_options(),
    group_titles=_GROUP_TITLES,
)


RENDER_OPTIONS = OptionTable(
    render_options(
        input_help="folder or chronominer.sqlite to render",
        formats=OUTPUT_FORMATS,
        formats_help="comma-separated formats to render besides {stem}_output.json",
        dry_run_help=DRY_RUN_HELP,
        json_help=JSON_HELP,
    ),
    group_titles=_GROUP_TITLES,
)

DROP_OPTIONS = OptionTable(
    [
        _input_option("output folder holding chronominer.sqlite, or the database"),
        replace(
            OPTIONS["select"],
            help="select documents by number (1,3), range (1-5) or name search",
            target="select",
            settable=False,
        ),
        replace(
            OPTIONS["all"],
            help="drop every stored document",
            target="all",
            settable=False,
        ),
        replace(
            OPTIONS["files"],
            help="comma-separated document names or input file names",
            target="files",
        ),
        dry_run_option("dry_run"),
        json_option("json"),
    ],
    group_titles=_GROUP_TITLES,
)

CLASSIFY_UNITS: Final = ("chunk", "entry", "document")
"""The units of ``classify chunks``, ``classify entries`` and ``classify documents``."""


def _classifier_rows(unit: str) -> list[Option]:
    """Return the classifier and decision-model rows, then ``--force``."""
    rows = classifier_options(
        force_help=(
            f"ask about every {unit} again instead of reusing the stored answers"
        ),
        choice_provider=classifier_names,
    )
    return sorted(rows, key=lambda option: option.name == "force")


def _classify_items_table(unit: str) -> OptionTable:
    """Return the table of ``classify chunks`` or ``classify documents``."""
    pages = [OPTIONS["pages"]] if unit == "chunk" else []
    return OptionTable(
        [
            replace(
                OPTIONS["input"],
                help="text (.txt, .md), image or PDF file, or a folder of these",
            ),
            *(OPTIONS[name] for name in ("type", "select", "all", "files")),
            OPTIONS["recursive"],
            *pages,
            replace(
                OPTIONS["output"],
                help="output folder whose chronominer.sqlite holds the answers",
            ),
            replace(
                OPTIONS["beside_input"],
                help="use the chronominer.sqlite beside each input",
            ),
            OPTIONS["layout"],
            replace(
                OPTIONS["chunking"],
                help=(
                    "text chunking of items the store does not hold: auto (by "
                    "--chunk-tokens) or line-ranges (the input's "
                    "{stem}_line_ranges.txt, which must exist); adjust is refused"
                ),
            ),
            replace(
                OPTIONS["chunk_tokens"],
                help="target tokens per text chunk under --chunking auto",
            ),
            replace(
                OPTIONS["model"],
                help=(
                    "model whose tokenizer counts the chunk tokens and whose image "
                    "presets size the pages of items the store does not hold"
                ),
                target="model",
            ),
            *(OPTIONS[name] for name in _IMAGE_NAMES),
            *_classifier_rows(unit),
            dry_run_option("run.dry_run"),
            json_option("run.json"),
            settings_option(),
        ],
        group_titles=_GROUP_TITLES,
    )


CLASSIFY_CHUNKS_OPTIONS = _classify_items_table("chunk")
CLASSIFY_DOCUMENTS_OPTIONS = _classify_items_table("document")
CLASSIFY_ENTRIES_OPTIONS = OptionTable(
    [
        _input_option(
            "output folder or chronominer.sqlite whose entries to classify; "
            "subfolders included"
        ),
        *_classifier_rows("entry"),
        dry_run_option("run.dry_run"),
        json_option("run.json"),
        settings_option(),
    ],
    group_titles=_GROUP_TITLES,
)

PROVIDER_TABLES: Final = (
    OPTIONS,
    RANGES_ADJUST_OPTIONS,
    REPAIR_OPTIONS,
    BATCH_CHECK_OPTIONS,
    BATCH_CANCEL_OPTIONS,
    CLASSIFY_CHUNKS_OPTIONS,
    CLASSIFY_ENTRIES_OPTIONS,
    CLASSIFY_DOCUMENTS_OPTIONS,
)
"""The tables of the commands that call providers; ``ui`` starts the first five."""


def _parse_table(table: OptionTable, argv: Sequence[str]) -> dict[str, Any]:
    """Parse *argv* with *table*; argparse exits with status 2 on errors."""
    return table.parse(argv)


@dataclass(frozen=True)
class Resolution[S]:
    """The effective spec and the source of each field (dotted path)."""

    spec: S
    sources: Mapping[str, Source]


def effective_spec(
    spec: CommandSpec, sources: Mapping[str, Source] | None = None
) -> dict[str, dict[str, Any]]:
    """Return every field of *spec* (dotted path) with its value and source.

    Values are JSON values; fields missing from *sources* were given by the
    caller and count as flags.
    """
    known = sources or {}
    return {
        path: {
            "value": _jsonable(value),
            "source": known.get(path, Source.FLAG).value,
        }
        for path, value in _flatten(spec).items()
    }


def inapplicable(spec: RunSpec, given: Iterable[str]) -> list[str]:
    """Return the *given* option names whose condition fails for *spec*.

    Such options do not affect the run; a front end reports them.
    """
    flat = _flatten(spec)
    values = {
        option.name: flat[option.target]
        for option in OPTIONS
        if option.target is not None
    }
    wanted = set(given)
    return [name for name in OPTIONS.inapplicable(values) if name in wanted]


def condition_warnings(
    table: OptionTable,
    flags: Mapping[str, Any],
    defaults: Mapping[str, Any],
    spec: CommandSpec,
) -> list[str]:
    """Return one warning per given option whose condition does not hold.

    *flags* are the options given on the command line, *defaults* the
    settings defaults block. The warning names the flag or settings default
    that supplied the value; values from code defaults never warn.
    """
    flat = effective_spec(spec)
    values = {
        option.name: flat[option.target]["value"]
        for option in table
        if option.target is not None and option.target in flat
    }
    offenders: dict[tuple[str, str], tuple[Condition, list[str]]] = {}
    for option in table:
        condition = option.condition
        if condition is None or option.target is None or condition.holds(values):
            continue
        origin = _origin(table, option, (("flag", flags), ("settings", defaults)))
        if origin is None:
            continue
        rule = table[origin[1]].condition or condition
        subject = condition.option.replace("_", " ")
        reason = f"the {subject} is {values.get(condition.option)}"
        offenders.setdefault(origin, (rule, []))[1].append(reason)
    messages = []
    for (level, name), (rule, reasons) in offenders.items():
        flag = table[name].flag
        label = flag if level == "flag" else f"the settings default {name} ({flag})"
        messages.append(
            f"{label} applies {rule.describe()} and has no effect here: "
            + ", ".join(reasons)
        )
    return messages


def _origin(
    table: OptionTable,
    option: Option,
    levels: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[str, str] | None:
    """Return the level and option name that supply *option*, if not a default."""
    chain = [option.name]
    while (fallback := table[chain[-1]].fallback) is not None:
        if fallback in chain:
            break
        chain.append(fallback)
    for level, given in levels:
        for name in chain:
            if name in given:
                return level, name
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        path = f"{prefix}{f.name}"
        if is_dataclass(value):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def _part(values: Mapping[str, Any], name: str) -> dict[str, Any]:
    prefix = name + "."
    return {
        key.removeprefix(prefix): value
        for key, value in values.items()
        if key.startswith(prefix)
    }


def _applicable(
    table: OptionTable, defaults: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return the settings defaults that set settable options of *table*."""
    return {
        name: value
        for name, value in (defaults or {}).items()
        if name in table and table[name].settable
    }


def _resolve_table(
    table: OptionTable, flags: Mapping[str, Any], defaults: Mapping[str, Any] | None
) -> dict[str, Resolved]:
    unknown = set(flags) - set(table.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    return table.resolve(flags, _applicable(table, defaults))


def _settle_endpoint(resolved: dict[str, Resolved]) -> None:
    """Make a named endpoint imply provider custom, by precedence."""
    provider = resolved["provider"]
    endpoint = resolved["endpoint"]
    if endpoint.value is not None and provider.value != "custom":
        if endpoint.source.rank >= provider.source.rank:
            resolved["provider"] = Resolved("custom", endpoint.source)
        else:
            resolved["endpoint"] = Resolved(None, provider.source)


_LEVELS: Mapping[Source, str] = {
    Source.FLAG: "a flag",
    Source.SETTINGS: "the settings defaults",
    Source.DEFAULT: "the code default",
}


def _check_model_level(resolved: Mapping[str, Resolved]) -> None:
    """Refuse a provider set above a model that is not one of its models.

    A custom provider needs a model at its endpoint's level or above; any
    other provider refuses a lower-level model that implies another provider.
    """
    provider = resolved["provider"]
    model = resolved["model"]
    if provider.value == "custom":
        endpoint = resolved["endpoint"]
        if model.source.rank < endpoint.source.rank:
            raise SpecError(
                f"--endpoint {endpoint.value} (from {_LEVELS[endpoint.source]}) "
                f"needs a model of that endpoint, but the model {model.value!r} "
                f"comes from {_LEVELS[model.source]}; name it with --model NAME"
            )
        return
    if provider.source.rank <= model.source.rank:
        return
    implied = implied_provider(model.value)
    if implied is not None and implied != provider.value:
        raise SpecError(
            f"--provider {provider.value} (from {_LEVELS[provider.source]}) "
            f"cannot run the model {model.value!r} of provider {implied} "
            f"(from {_LEVELS[model.source]}); name a model of provider "
            f"{provider.value} with --model NAME"
        )


def _settle_provider(resolved: dict[str, Resolved]) -> None:
    """Settle the provider from the model name and strip a provider prefix.

    An unset provider is inferred and takes the model's source. A provider
    set below the model's level yields to the provider the name implies; a
    prefix contradicting a provider set at the model's level or above, or a
    custom provider, is an error. The model name then becomes the id the
    provider's API knows (:func:`~chronominer.providers.api_model_id`).
    """
    _settle_endpoint(resolved)
    provider = resolved["provider"]
    model = resolved["model"]
    prefixed, name = split_provider_prefix(model.value)
    if prefixed is not None:
        resolved["model"] = Resolved(name, model.source)
    if provider.value is None:
        inferred = prefixed or infer_provider(name)
        resolved["provider"] = Resolved(inferred, model.source)
    elif prefixed is not None and prefixed != provider.value:
        if provider.value == "custom" or provider.source.rank >= model.source.rank:
            raise SpecError(
                f"model {model.value!r} names provider {prefixed}, "
                f"but the provider is {provider.value}"
            )
        resolved["provider"] = Resolved(prefixed, model.source)
    elif provider.value != "custom" and provider.source.rank < model.source.rank:
        implied = implied_provider(name)
        if implied is not None and implied != provider.value:
            resolved["provider"] = Resolved(implied, model.source)
    if resolved["provider"].value == "custom" and resolved["endpoint"].value is None:
        raise SpecError(
            "provider custom needs --endpoint NAME naming an endpoint from the settings"
        )
    model = resolved["model"]
    api_id = api_model_id(resolved["provider"].value, model.value)
    if api_id != model.value:
        resolved["model"] = Resolved(api_id, model.source)


def _settle_model(resolved: dict[str, Resolved]) -> None:
    """Settle the provider of a run that calls a model, then check its level."""
    _settle_provider(resolved)
    _check_model_level(resolved)


def _model_spec(values: Mapping[str, Any]) -> ModelSpec:
    return ModelSpec(**_part(values, "model"))


def _require_schema(value: str | None) -> None:
    if value is None:
        raise SpecError("a schema is required (--schema NAME|PATH)")


def resolve_spec(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[RunSpec]:
    """Resolve extract options: flag, then settings default, then code default.

    ``flags`` are converted values keyed by option name, as returned by
    ``OPTIONS.parse``; ``defaults`` is a checked settings defaults block.

    Raises:
        SpecError: No input path, no schema, a custom provider without an
            endpoint, a model prefix that contradicts the provider, or a
            provider set above a model of another provider.
    """
    resolved = _resolve_table(OPTIONS, flags, defaults)
    _settle_model(resolved)
    values, sources = OPTIONS.by_targets(resolved)
    if values["input.path"] is None:
        raise SpecError("an input path is required (--input PATH)")
    _require_schema(values["prompt.schema"])
    spec = RunSpec(
        input=InputSpec(**_part(values, "input")),
        output=OutputSpec(**_part(values, "output")),
        prompt=PromptSpec(**_part(values, "prompt")),
        chunking=ChunkingSpec(**_part(values, "chunking")),
        model=_model_spec(values),
        images=ImageSpec(**_part(values, "images")),
        run=RunControl(**_part(values, "run")),
        decision=DecisionSpec(**_part(values, "decision")),
    )
    return Resolution(spec, sources)


def resolve_ranges_generate(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[RangesGenerateSpec]:
    """Resolve ranges generate options; settings defaults apply to shared options.

    Raises:
        SpecError: No input path.
    """
    resolved = _resolve_table(RANGES_GENERATE_OPTIONS, flags, defaults)
    values, sources = RANGES_GENERATE_OPTIONS.by_targets(resolved)
    if values["input"] is None:
        raise SpecError("an input path is required (--input PATH)")
    return Resolution(RangesGenerateSpec(**values), sources)


def resolve_ranges_adjust(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[RangesAdjustSpec]:
    """Resolve ranges adjust options; settings defaults apply to shared options.

    Raises:
        SpecError: No input path, no schema, or a provider rule broken as in
            :func:`resolve_spec`.
    """
    resolved = _resolve_table(RANGES_ADJUST_OPTIONS, flags, defaults)
    _settle_model(resolved)
    values, sources = RANGES_ADJUST_OPTIONS.by_targets(resolved)
    if values["input"] is None:
        raise SpecError("an input path is required (--input PATH)")
    _require_schema(values["schema"])
    spec = RangesAdjustSpec(
        input=values["input"],
        recursive=values["recursive"],
        schema=values["schema"],
        context_window=values["context_window"],
        context=values["context"],
        fallback_context=values["fallback_context"],
        pages=values["pages"],
        model=_model_spec(values),
        run=RangesControl(**_part(values, "run")),
    )
    return Resolution(spec, sources)


def resolve_repair(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[RepairSpec]:
    """Resolve repair options; settings defaults apply to the shared options.

    ``--indices`` narrows the failed chunks to the given chunk indices.

    Raises:
        SpecError: A provider rule broken as in :func:`resolve_spec`.
    """
    resolved = _resolve_table(REPAIR_OPTIONS, flags, defaults)
    _settle_model(resolved)
    values, sources = REPAIR_OPTIONS.by_targets(resolved)
    indices = flags.get("indices")
    sources = {
        **sources,
        "indices": Source.FLAG if "indices" in flags else Source.DEFAULT,
    }
    spec = RepairSpec(
        input=values["input"],
        schema=values["schema"],
        indices=None if indices is None else _parse_indices(indices),
        model=_model_spec(values),
        images=ImageSpec(**_part(values, "images")),
        run=RepairControl(**_part(values, "run")),
        skipped=values["skipped"],
    )
    return Resolution(spec, sources)


def resolve_batch_check(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[BatchCheckSpec]:
    """Resolve batch check options; no option takes a settings default.

    Raises:
        SpecError: ``--poll-minutes`` or ``--wait-hours`` without ``--wait``.
    """
    if not flags.get("wait"):
        for name in ("poll_minutes", "wait_hours"):
            if name in flags:
                raise SpecError(f"{BATCH_CHECK_OPTIONS[name].flag} needs --wait")
    resolved = _resolve_table(BATCH_CHECK_OPTIONS, flags, defaults)
    values, sources = BATCH_CHECK_OPTIONS.by_targets(resolved)
    return Resolution(BatchCheckSpec(**values), sources)


def resolve_batch_cancel(flags: Mapping[str, Any]) -> Resolution[BatchCancelSpec]:
    """Resolve batch cancel options.

    Raises:
        SpecError: None of batch IDs, ``--all`` and ``--input`` is given.
    """
    resolved = _resolve_table(BATCH_CANCEL_OPTIONS, flags, None)
    values, sources = BATCH_CANCEL_OPTIONS.by_targets(resolved)
    if not values["batch_ids"] and not values["all"] and values["directory"] is None:
        raise SpecError("batch cancel needs --batch-ids IDS or --all, or --input DIR")
    return Resolution(BatchCancelSpec(**values), sources)


def resolve_render(flags: Mapping[str, Any]) -> Resolution[RenderSpec]:
    """Resolve render options; no option takes a settings default."""
    resolved = _resolve_table(RENDER_OPTIONS, flags, None)
    values, sources = RENDER_OPTIONS.by_targets(resolved)
    return Resolution(RenderSpec(**values), sources)


def resolve_drop(flags: Mapping[str, Any]) -> Resolution[DropSpec]:
    """Resolve drop options; no option takes a settings default.

    Raises:
        SpecError: No input path, or not exactly one of ``--select``,
            ``--files`` and ``--all``.
    """
    resolved = _resolve_table(DROP_OPTIONS, flags, None)
    values, sources = DROP_OPTIONS.by_targets(resolved)
    if values["input"] is None:
        raise SpecError("an input path is required (--input PATH)")
    chosen = (values["select"] is not None, bool(values["files"]), values["all"])
    if sum(chosen) != 1:
        raise SpecError("give exactly one of --select, --files and --all")
    return Resolution(DropSpec(**values), sources)


def _require_classifier(value: str | None) -> None:
    if value is None:
        raise SpecError("a classifier is required (--classifier NAME|PATH)")


def resolve_classify_items(
    unit: str, flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[ClassifyItemsSpec]:
    """Resolve ``classify chunks`` (*unit* chunk) or ``classify documents`` options.

    Settings defaults apply to the options shared with extract.

    Raises:
        SpecError: No input path, no classifier, or ``--chunking adjust``,
            which would move line ranges with the model.
    """
    table = CLASSIFY_CHUNKS_OPTIONS if unit == "chunk" else CLASSIFY_DOCUMENTS_OPTIONS
    resolved = _resolve_table(table, flags, defaults)
    values, sources = table.by_targets(resolved)
    if values["input.path"] is None:
        raise SpecError("an input path is required (--input PATH)")
    _require_classifier(values["classify.classifier"])
    if values["chunking.mode"] == "adjust":
        raise SpecError(
            "classify moves no line ranges, so --chunking adjust is refused; run "
            "'chronominer ranges adjust' first, then classify with --chunking "
            "line-ranges"
        )
    spec = ClassifyItemsSpec(
        unit=unit,
        input=InputSpec(**{"pages": None, **_part(values, "input")}),
        output=ClassifyOutput(**_part(values, "output")),
        chunking=ClassifyChunking(**_part(values, "chunking")),
        model=values["model"],
        images=ImageSpec(**_part(values, "images")),
        classify=ClassifierChoice(**_part(values, "classify")),
        run=ClassifyControl(**_part(values, "run")),
    )
    return Resolution(spec, sources)


def resolve_classify_entries(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[ClassifyEntriesSpec]:
    """Resolve ``classify entries`` options; settings defaults apply to shared ones.

    Raises:
        SpecError: No input path or no classifier.
    """
    resolved = _resolve_table(CLASSIFY_ENTRIES_OPTIONS, flags, defaults)
    values, sources = CLASSIFY_ENTRIES_OPTIONS.by_targets(resolved)
    if values["input"] is None:
        raise SpecError("an input path is required (--input PATH)")
    _require_classifier(values["classify.classifier"])
    spec = ClassifyEntriesSpec(
        input=values["input"],
        classify=ClassifierChoice(**_part(values, "classify")),
        run=ClassifyControl(**_part(values, "run")),
    )
    return Resolution(spec, sources)


def parse(argv: Sequence[str], defaults: Mapping[str, Any] | None = None) -> RunSpec:
    """Parse extract arguments; argparse exits with status 2 on errors."""
    return resolve_spec(_parse_table(OPTIONS, argv), defaults).spec


def parse_ranges_generate(
    argv: Sequence[str], defaults: Mapping[str, Any] | None = None
) -> RangesGenerateSpec:
    """Parse ranges generate arguments; argparse exits with status 2 on errors."""
    flags = _parse_table(RANGES_GENERATE_OPTIONS, argv)
    return resolve_ranges_generate(flags, defaults).spec


def parse_ranges_adjust(
    argv: Sequence[str], defaults: Mapping[str, Any] | None = None
) -> RangesAdjustSpec:
    """Parse ranges adjust arguments; argparse exits with status 2 on errors."""
    flags = _parse_table(RANGES_ADJUST_OPTIONS, argv)
    return resolve_ranges_adjust(flags, defaults).spec


def parse_repair(
    argv: Sequence[str], defaults: Mapping[str, Any] | None = None
) -> RepairSpec:
    """Parse repair arguments; argparse exits with status 2 on errors."""
    return resolve_repair(_parse_table(REPAIR_OPTIONS, argv), defaults).spec


def parse_batch_check(
    argv: Sequence[str], defaults: Mapping[str, Any] | None = None
) -> BatchCheckSpec:
    """Parse batch check arguments; argparse exits with status 2 on errors."""
    return resolve_batch_check(_parse_table(BATCH_CHECK_OPTIONS, argv), defaults).spec


def parse_batch_cancel(argv: Sequence[str]) -> BatchCancelSpec:
    """Parse batch cancel arguments; argparse exits with status 2 on errors."""
    return resolve_batch_cancel(_parse_table(BATCH_CANCEL_OPTIONS, argv)).spec


def parse_render(argv: Sequence[str]) -> RenderSpec:
    """Parse render arguments; argparse exits with status 2 on errors."""
    return resolve_render(_parse_table(RENDER_OPTIONS, argv)).spec


def parse_drop(argv: Sequence[str]) -> DropSpec:
    """Parse drop arguments; argparse exits with status 2 on errors."""
    return resolve_drop(_parse_table(DROP_OPTIONS, argv)).spec


_UNSET = object()

# Unset values with a spelling; other options cannot be reset to None.
_UNSET_SPELLINGS = {
    "context": CONTEXT_AUTO,
    "fallback_context": CONTEXT_NONE,
    "context_image": CONTEXT_AUTO,
}


def _spell(given: dict[str, Any], option: Option, value: Any) -> None:
    """Add *option* with *value* to *given*, or raise when it has no spelling."""
    if option.name == "output" and value is None:
        given["beside_input"] = True
    elif option.name in _UNSET_SPELLINGS or (
        value is not None and (option.kind is not Kind.FLAG or value)
    ):
        given[option.name] = value
    else:
        raise SpecError(
            f"the settings default of {option.name} cannot be undone on the "
            f"command line: {option.flag} has no spelling for {value!r}"
        )


def _alternative_given(
    table: OptionTable, option: Option, given: Mapping[str, Any]
) -> bool:
    """Whether an alternative of *option* is given, which resets *option*."""
    return option.exclusive is not None and any(
        other.exclusive == option.exclusive and other.name in given for other in table
    )


def _spell_values(
    table: OptionTable, flat: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, Any]:
    """Spell every value that differs from its code or its settings default.

    The input is always spelled; the provider is left to :func:`_spell_provider`.
    """
    targeted = [
        option
        for option in table
        if option.target is not None and option.name != "provider"
    ]
    given: dict[str, Any] = {}
    for option in targeted:
        value = flat[option.target or ""]
        if (option.name == "input" and value is not None) or value != option.default:
            _spell(given, option, value)
    for option in targeted:
        if (
            option.name in given
            or option.name == "endpoint"
            or _alternative_given(table, option, given)
        ):
            continue
        value = flat[option.target or ""]
        setting = defaults.get(option.name, _UNSET)
        if setting is not _UNSET and setting != value:
            _spell(given, option, value)
    return given


def _spell_provider(
    given: dict[str, Any],
    model: ModelSpec,
    defaults: Mapping[str, Any],
    resolve: Callable[[Mapping[str, Any]], ModelSpec],
) -> None:
    """Spell the provider unless the model name and the settings yield it.

    When the spelled provider would outrank an unspelled model, the model is
    spelled too.
    """
    if model.provider == infer_provider(model.model):
        if not defaults:
            return
        if _yields(given, model, resolve):
            return
    given["provider"] = model.provider
    if "model" not in given and not _yields(given, model, resolve):
        given["model"] = model.model


def _yields(
    given: Mapping[str, Any],
    model: ModelSpec,
    resolve: Callable[[Mapping[str, Any]], ModelSpec],
) -> bool:
    """Whether *given* resolves to the provider, endpoint and model of *model*."""
    try:
        resolved = resolve(given)
    except SpecError:
        return False
    return (resolved.provider, resolved.endpoint, resolved.model) == (
        model.provider,
        model.endpoint,
        model.model,
    )


def _with_keywords(given: dict[str, Any]) -> dict[str, Any]:
    for name, keyword in _UNSET_SPELLINGS.items():
        if name in given and given[name] is None:
            given[name] = keyword
    return given


def to_argv(spec: CommandSpec, defaults: Mapping[str, Any] | None = None) -> list[str]:
    """Return arguments that reproduce ``spec`` with and without ``defaults``.

    ``defaults`` is a checked settings defaults block. Every value that
    differs from its code default is spelled out, and so is every value the
    defaults would change; a provider the model name implies is omitted
    unless the defaults would change it. A value that begins with ``-`` is
    written as ``--flag=value``.

    Raises:
        SpecError: The defaults change a value that has no spelling, such
            as an unset ``--max-output-tokens``.
    """
    table = _TABLES[type(spec)]
    applicable = _applicable(table, defaults)
    given = _spell_values(table, _flatten(spec), applicable)
    if isinstance(spec, RepairSpec) and spec.indices is not None:
        given["indices"] = _format_indices(spec.indices)
    resolver = _MODEL_RESOLVERS.get(type(spec))
    if resolver is not None:
        model = cast(RunSpec | RangesAdjustSpec | RepairSpec, spec).model
        _spell_provider(
            given, model, applicable, lambda trial: resolver(trial, applicable)
        )
    return _argv(table, _with_keywords(given))


def _run_model(trial: Mapping[str, Any], defaults: Mapping[str, Any]) -> ModelSpec:
    return resolve_spec(trial, defaults).spec.model


def _adjust_model(trial: Mapping[str, Any], defaults: Mapping[str, Any]) -> ModelSpec:
    return resolve_ranges_adjust(trial, defaults).spec.model


def _repair_model(trial: Mapping[str, Any], defaults: Mapping[str, Any]) -> ModelSpec:
    return resolve_repair(trial, defaults).spec.model


_MODEL_RESOLVERS: Mapping[
    type, Callable[[Mapping[str, Any], Mapping[str, Any]], ModelSpec]
] = {
    RunSpec: _run_model,
    RangesAdjustSpec: _adjust_model,
    RepairSpec: _repair_model,
}

_TABLES: Mapping[type, OptionTable] = {
    RunSpec: OPTIONS,
    RangesGenerateSpec: RANGES_GENERATE_OPTIONS,
    RangesAdjustSpec: RANGES_ADJUST_OPTIONS,
    RepairSpec: REPAIR_OPTIONS,
    BatchCheckSpec: BATCH_CHECK_OPTIONS,
    BatchCancelSpec: BATCH_CANCEL_OPTIONS,
    RenderSpec: RENDER_OPTIONS,
    DropSpec: DROP_OPTIONS,
}


def _argv(table: OptionTable, given: Mapping[str, Any]) -> list[str]:
    """Return the tokens of *given* in table order; see :func:`to_argv`."""
    unknown = set(given) - set(table.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    tokens: list[str] = []
    for option in table:
        if option.name not in given:
            continue
        tokens.extend(option.format(given[option.name]))
    return tokens
