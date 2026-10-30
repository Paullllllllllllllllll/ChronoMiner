"""Standard option rows: the run options every tool declares alike.

The builders return :class:`~.options.Option` rows for the model block, the
page-image block, the output location, run control (``--force``,
``--concurrency``, ``--batch``, ``--dry-run``, ``--json``, ``--settings``),
batch check and cancel, ``render``, the decision-model block and a classify
command (``--classifier`` with the decision-model rows). Each tool passes
its own defaults, help texts and conditions, and arranges the rows into its
option tables; the rows themselves are declared once here. The value converters the rows
use are public, so a tool's own rows convert values the same way.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .options import Condition, Kind, Option

__all__ = [
    "DECISION_MODES",
    "DECISION_PROVIDERS",
    "DEFAULT_DECISION_MODEL",
    "DRY_RUN_HELP",
    "IMAGE_DETAILS",
    "IMAGE_FORMATS",
    "JSON_HELP",
    "MODEL_FIELDS",
    "REASONING_EFFORTS",
    "RECOMMENDED",
    "RESPONSE_FORMATS",
    "SERVICE_TIERS",
    "SETTINGS_HELP",
    "VERBOSITIES",
    "Dpi",
    "abandon_option",
    "adopt_option",
    "batch_cancel_options",
    "batch_ids_option",
    "batch_option",
    "bounded_float",
    "check_input_option",
    "classifier_options",
    "concurrency_option",
    "decision_options",
    "dpi_value",
    "dry_run_option",
    "flag_option",
    "force_option",
    "image_options",
    "jpeg_quality",
    "json_option",
    "model_options",
    "nonempty_text",
    "output_options",
    "positive_int",
    "render_options",
    "settings_option",
]

DECISION_MODES = ("off", "shadow", "apply")
DECISION_PROVIDERS = ("openai", "openrouter")
DEFAULT_DECISION_MODEL = "typesafe/jev-1.14"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
VERBOSITIES = ("low", "medium", "high")
SERVICE_TIERS = ("auto", "default", "flex", "priority")
RESPONSE_FORMATS = ("schema", "json", "text")
IMAGE_DETAILS = ("low", "high", "auto", "original")
IMAGE_FORMATS = ("jpeg", "png")

MODEL_FIELDS = (
    "provider",
    "model",
    "endpoint",
    "reasoning_effort",
    "verbosity",
    "max_output_tokens",
    "temperature",
    "top_p",
    "service_tier",
    "response_format",
)
"""The fields of the model block, in table order."""

RECOMMENDED = "the provider's recommended value"
"""The shown default of an image option left unset."""

DRY_RUN_HELP = "plan without API calls or writes"
JSON_HELP = "print one JSON summary line on stdout"
SETTINGS_HELP = "alternate settings file"

Dpi = int | Literal["native"]


def positive_int(value: str) -> int:
    """Convert an integer above zero."""
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("must be > 0")
    return parsed


def bounded_float(low: float, high: float) -> Callable[[str], float]:
    """Return a converter of finite numbers from *low* to *high*."""

    def convert(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not low <= parsed <= high:
            raise ValueError(f"must be between {low} and {high}")
        return parsed

    return convert


def dpi_value(value: str) -> Dpi:
    """Convert ``native`` or a positive DPI value."""
    if value.strip().lower() == "native":
        return "native"
    return positive_int(value)


def jpeg_quality(value: str) -> int:
    """Convert a JPEG quality from 1 to 100."""
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise ValueError("must be between 1 and 100")
    return parsed


def nonempty_text(value: str) -> str:
    """Return *value* unchanged unless it is blank."""
    if not value.strip():
        raise ValueError("must not be empty")
    return value


_RESPONSE_FORMAT_HELP = {
    "schema": "schema (API-enforced)",
    "json": "json (prompted, validated)",
    "text": "text (plain)",
}

_MODEL_DEFAULT_HELP = {
    "provider": "inferred from the model name",
    "endpoint": "none",
    "max_output_tokens": "the model's output limit",
    "response_format": "schema when the model supports it, else json",
}


def _response_format_help(formats: Sequence[str]) -> str:
    parts = [_RESPONSE_FORMAT_HELP[name] for name in formats]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " or " + parts[-1]


def _model_rows(
    providers: Sequence[str],
    suggestions: Callable[[], Sequence[str]],
    response_formats: Sequence[str],
    top_p_sent_always: bool,
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Return the help text and the argparse fields of each model field."""
    top_p_when = "supported" if top_p_sent_always else "given and supported"
    return {
        "provider": ("model provider", {"choices": tuple(providers)}),
        "model": (
            "model name; a provider: prefix (anthropic:NAME) selects the provider",
            {
                "type": nonempty_text,
                "metavar": "NAME",
                "choice_provider": suggestions,
            },
        ),
        "endpoint": (
            "named custom endpoint from the settings (implies provider custom)",
            {"type": nonempty_text, "metavar": "NAME"},
        ),
        "reasoning_effort": ("reasoning effort", {"choices": REASONING_EFFORTS}),
        "verbosity": ("output verbosity", {"choices": VERBOSITIES}),
        "max_output_tokens": (
            "maximum output tokens",
            {"type": positive_int, "metavar": "N"},
        ),
        "temperature": (
            "sampling temperature, 0.0 to 2.0",
            {"type": bounded_float(0.0, 2.0), "metavar": "T"},
        ),
        "top_p": (
            f"nucleus sampling, 0.0 to 1.0; sent only when {top_p_when}",
            {"type": bounded_float(0.0, 1.0), "metavar": "P"},
        ),
        "service_tier": ("OpenAI service tier", {"choices": SERVICE_TIERS}),
        "response_format": (
            _response_format_help(response_formats),
            {"choices": tuple(response_formats)},
        ),
    }


def model_options(
    *,
    providers: Sequence[str],
    suggestions: Callable[[], Sequence[str]],
    defaults: Mapping[str, Any],
    response_formats: Sequence[str] = RESPONSE_FORMATS,
    default_help: Mapping[str, str] | None = None,
    phases: Sequence[str] = (),
    phase_defaults: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Option]:
    """Return the model block: provider through response format.

    *defaults* holds the code default of each field (a missing field has
    none); *default_help* replaces the shown default of a field. ``--top-p``
    without a default is sent only when given. ``--service-tier`` applies to
    the provider openai only.

    Without *phases* the rows fill ``model.<field>`` in the group ``model``.
    With *phases* the unprefixed rows carry no default and fill no field;
    each phase gets ``--<phase>-<field>`` rows in the group ``<phase> model``
    that fall back to the unprefixed row and fill ``<phase>_model.<field>``.
    *phase_defaults* replaces field defaults per phase.
    """
    top_p_unset = defaults.get("top_p") is None
    rows = _model_rows(providers, suggestions, response_formats, not top_p_unset)
    shown = dict(_MODEL_DEFAULT_HELP)
    if top_p_unset:
        shown["top_p"] = "not sent"
    shown.update(default_help or {})
    if not phases:
        return [
            Option(
                name,
                "--" + name.replace("_", "-"),
                help,
                group="model",
                default=defaults.get(name),
                default_help=shown.get(name),
                condition=_tier_condition(name, "provider"),
                target=f"model.{name}",
                **extra,
            )
            for name, (help, extra) in rows.items()
        ]
    options = [
        Option(
            name,
            "--" + name.replace("_", "-"),
            help,
            group="model",
            default_help=shown.get(name, "per phase"),
            condition=_tier_condition(name, "provider"),
            **extra,
        )
        for name, (help, extra) in rows.items()
    ]
    for phase in phases:
        own = (phase_defaults or {}).get(phase, {})
        options.extend(
            Option(
                f"{phase}_{name}",
                f"--{phase}-{name.replace('_', '-')}",
                help,
                group=f"{phase} model",
                default=own.get(name, defaults.get(name)),
                default_help=shown.get(name),
                condition=_tier_condition(name, f"{phase}_provider"),
                fallback=name,
                target=f"{phase}_model.{name}",
                **extra,
            )
            for name, (help, extra) in rows.items()
        )
    return options


def _tier_condition(name: str, provider: str) -> Condition | None:
    if name != "service_tier":
        return None
    return Condition(provider, frozenset({"openai"}))


def image_options(
    *, dpi_help: str = "render resolution: native or a DPI value"
) -> list[Option]:
    """Return the page-image block; unset values take the provider's preset.

    The rows fill ``images.dpi``, ``images.detail``, ``images.format``,
    ``images.jpeg_quality`` and ``images.grayscale``.
    """

    def image(name: str, help: str, target: str, **extra: Any) -> Option:
        return Option(
            name,
            "--" + name.replace("_", "-"),
            help,
            group="images",
            default_help=RECOMMENDED,
            target=f"images.{target}",
            **extra,
        )

    return [
        image("dpi", dpi_help, "dpi", type=dpi_value, metavar="native|N"),
        image(
            "image_detail",
            "image detail sent to the model",
            "detail",
            choices=IMAGE_DETAILS,
        ),
        image("image_format", "page image format", "format", choices=IMAGE_FORMATS),
        image(
            "jpeg_quality",
            "JPEG quality, 1 to 100",
            "jpeg_quality",
            type=jpeg_quality,
            metavar="N",
        ),
        image(
            "grayscale",
            "convert page images to grayscale",
            "grayscale",
            kind=Kind.SWITCH,
        ),
    ]


def output_options() -> list[Option]:
    """Return ``--output DIR`` and its alternative ``--beside-input``."""
    return [
        Option(
            "output",
            "--output",
            "output folder",
            group="output",
            type=Path,
            metavar="DIR",
            default_help="beside the input",
            exclusive="output_location",
            target="output.directory",
        ),
        Option(
            "beside_input",
            "--beside-input",
            "write outputs beside each input",
            group="output",
            kind=Kind.FLAG,
            default=False,
            exclusive="output_location",
            settable=False,
        ),
    ]


def flag_option(
    name: str,
    help: str,
    *,
    group: str = "run",
    target: str | None = None,
    exclusive: str | None = None,
) -> Option:
    """Return an off-by-default ``store_true`` flag that no settings set."""
    return Option(
        name,
        "--" + name.replace("_", "-"),
        help,
        group=group,
        kind=Kind.FLAG,
        default=False,
        exclusive=exclusive,
        target=target,
        settable=False,
    )


def force_option(help: str, *, target: str = "run.force") -> Option:
    """Return ``--force``; the tool's help says what is redone."""
    return flag_option("force", help, target=target)


def concurrency_option(
    default: int,
    *,
    help: str = "parallel requests",
    target: str = "run.concurrency",
) -> Option:
    """Return ``--concurrency N`` with the tool's default."""
    return Option(
        "concurrency",
        "--concurrency",
        help,
        group="run",
        type=positive_int,
        metavar="N",
        default=default,
        target=target,
    )


def batch_option(
    help: str,
    *,
    switch: bool = True,
    condition: Condition | None = None,
    target: str = "run.batch",
) -> Option:
    """Return ``--batch``: a settable switch, or a flag (``switch`` False)."""
    return Option(
        "batch",
        "--batch",
        help,
        group="run",
        kind=Kind.SWITCH if switch else Kind.FLAG,
        default=False,
        condition=condition,
        target=target,
        settable=switch,
    )


def dry_run_option(target: str, help: str = DRY_RUN_HELP) -> Option:
    """Return ``--dry-run`` filling *target*."""
    return flag_option("dry_run", help, target=target)


def json_option(target: str, help: str = JSON_HELP) -> Option:
    """Return ``--json`` filling *target*."""
    return flag_option("json", help, target=target)


def settings_option(*, help: str = SETTINGS_HELP, target: str | None = None) -> Option:
    """Return ``--settings FILE``."""
    return Option(
        "settings",
        "--settings",
        help,
        group="run",
        type=Path,
        metavar="FILE",
        default_help="config/settings.yaml",
        target=target,
        settable=False,
    )


def check_input_option() -> Option:
    """Return the ``--input DIR`` of ``batch check``."""
    return Option(
        "input",
        "--input",
        "folder searched for batch submissions, subfolders included",
        group="batch",
        type=Path,
        metavar="DIR",
        default=Path("."),
        default_help="the current directory",
        target="directory",
        settable=False,
    )


def batch_ids_option(
    help: str,
    *,
    exclusive: str | None = None,
    default_help: str | None = None,
) -> Option:
    """Return ``--batch-ids IDS``, a comma-separated list."""
    return Option(
        "batch_ids",
        "--batch-ids",
        help,
        group="batch",
        kind=Kind.LIST,
        type=nonempty_text,
        metavar="IDS",
        default=(),
        default_help=default_help,
        exclusive=exclusive,
        target="batch_ids",
        settable=False,
    )


def adopt_option() -> Option:
    """Return ``--adopt BATCH_ID`` of ``batch check``."""
    return Option(
        "adopt",
        "--adopt",
        "record this provider batch for the unresolved submission it belongs "
        "to, then finalize it",
        group="batch",
        metavar="BATCH_ID",
        exclusive="unresolved_action",
        target="adopt",
        settable=False,
    )


def abandon_option(help: str) -> Option:
    """Return ``--abandon SUBMISSION_ID``; the tool's help names its units."""
    return Option(
        "abandon",
        "--abandon",
        help,
        group="batch",
        metavar="SUBMISSION_ID",
        exclusive="unresolved_action",
        target="abandon",
        settable=False,
    )


def batch_cancel_options(*, settings_target: str | None = None) -> list[Option]:
    """Return the option rows of ``batch cancel``.

    The rows are the exactly-one target group, ``--json`` and ``--settings``.
    """
    return [
        batch_ids_option(
            "comma-separated batch IDs to cancel", exclusive="cancel_target"
        ),
        flag_option(
            "all",
            "cancel every unfinished batch of every provider",
            group="batch",
            target="all",
            exclusive="cancel_target",
        ),
        Option(
            "input",
            "--input",
            "cancel the open batch submissions recorded in the databases below "
            "this folder",
            group="batch",
            type=Path,
            metavar="DIR",
            exclusive="cancel_target",
            target="directory",
            settable=False,
        ),
        json_option("json"),
        settings_option(target=settings_target),
    ]


def render_options(
    *,
    input_help: str,
    formats: Sequence[str],
    formats_help: str = "comma-separated formats to render",
    dry_run_help: str,
    json_help: str,
) -> list[Option]:
    """Return the rows of ``render``; no row takes a settings default."""
    return [
        Option(
            "input",
            "--input",
            input_help,
            group="input",
            type=Path,
            metavar="PATH",
            default=Path("."),
            target="input",
            settable=False,
        ),
        Option(
            "formats",
            "--formats",
            formats_help,
            group="output",
            kind=Kind.LIST,
            choices=tuple(formats),
            metavar="LIST",
            default_help="the formats each document was rendered in",
            target="formats",
            settable=False,
        ),
        flag_option("force", "replace files changed outside the tool", target="force"),
        dry_run_option("dry_run", dry_run_help),
        json_option("json", json_help),
    ]


def decision_options(
    mode_help: str,
    *,
    target: str = "decision",
    default_model: str = DEFAULT_DECISION_MODEL,
    suggestions: Callable[[], Sequence[str]] | None = None,
) -> list[Option]:
    """Return the decision-model block: mode, model, provider and endpoint.

    ``--decisions`` (``off``, ``shadow``, ``apply``) fills ``<target>.mode``;
    *mode_help* says what ``apply`` does in the tool. The other rows fill
    ``<target>.model``, ``<target>.provider`` and ``<target>.endpoint`` and
    apply only while the mode is not ``off``. They never fall back to the
    chat model's rows. A provider left unset is resolved from the model id
    (``vendor/model`` ids go to OpenRouter).
    """
    active = Condition("decisions", frozenset({"shadow", "apply"}))
    return [
        Option(
            "decisions",
            "--decisions",
            mode_help,
            group="decision model",
            choices=DECISION_MODES,
            default="off",
            target=f"{target}.mode",
        ),
        *_decision_model_rows(target, default_model, suggestions, active),
    ]


def classifier_options(
    *,
    force_help: str,
    target: str = "classify",
    choice_provider: Callable[[], Sequence[str]] | None = None,
    default_model: str = DEFAULT_DECISION_MODEL,
    suggestions: Callable[[], Sequence[str]] | None = None,
) -> list[Option]:
    """Return the rows of a classify command: classifier, force, decision model.

    ``--classifier NAME|PATH`` is required (no default; the tool refuses a
    run without it) and fills ``<target>.classifier``; *choice_provider*
    lists the classifier names found. ``--force`` fills ``<target>.force``;
    *force_help* says what is asked again. The decision-model rows are those
    of :func:`decision_options` without the mode and its condition.
    """
    return [
        Option(
            "classifier",
            "--classifier",
            "classifier: a file path, a file name in the classifier folder "
            "without .json, or the name field of a classifier there",
            group="classify",
            type=nonempty_text,
            metavar="NAME|PATH",
            choice_provider=choice_provider,
            default_help="required",
            target=f"{target}.classifier",
            settable=False,
        ),
        force_option(force_help, target=f"{target}.force"),
        *_decision_model_rows(target, default_model, suggestions, None),
    ]


def _decision_model_rows(
    target: str,
    default_model: str,
    suggestions: Callable[[], Sequence[str]] | None,
    condition: Condition | None,
) -> list[Option]:
    """Return the decision model, provider and endpoint rows under *condition*."""
    group = "decision model"
    return [
        Option(
            "decision_model",
            "--decision-model",
            "decision model id",
            group=group,
            type=nonempty_text,
            choice_provider=suggestions,
            default=default_model,
            metavar="ID",
            condition=condition,
            target=f"{target}.model",
        ),
        Option(
            "decision_provider",
            "--decision-provider",
            "decision API dialect",
            group=group,
            choices=DECISION_PROVIDERS,
            default_help="from the model id",
            condition=condition,
            target=f"{target}.provider",
        ),
        Option(
            "decision_endpoint",
            "--decision-endpoint",
            "decision endpoint URL",
            group=group,
            type=nonempty_text,
            default_help="the provider's",
            metavar="URL",
            condition=condition,
            target=f"{target}.endpoint",
        ),
    ]
