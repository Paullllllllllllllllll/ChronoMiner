"""The steps of a new extraction in the guided UI.

Steps in order: input, input type and items, chunking, pages, output and
formats, model, prompt and context (response format, schema, context and
context image), page images, and the run options. Each :class:`Step` names
the options it may set and applies only to the inputs that use them:
chunking to text items, pages to text and PDF items, page images to image
and PDF items. Answers act like command-line flags over the settings
defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from chronominer.batch.repair import batch_refusal
from chronominer.common.batch import supports_batch
from chronominer.common.selection import (
    is_numeric_selection,
    parse_page_range,
    select_items,
)
from chronominer.common.wizard import (
    BACK,
    DONE,
    Choice,
    Nav,
    State,
    Step,
    StepContext,
)
from chronominer.discovery import SourceItem, natural_sort_key
from chronominer.spec import INPUT_TYPES, item_type
from chronominer.ui.questions import (
    IMAGE_OPTIONS,
    MODEL_OPTIONS,
    ImageQuestions,
    ModelQuestions,
    Question,
    ResponseQuestions,
    chunk_tokens_question,
    resolution_of,
)
from chronominer.ui.session import (
    Session,
    absolute,
    has_settings_default,
    inventory_of,
    plural,
)

__all__ = [
    "CHUNKING_TITLES",
    "RESPONSE_STEP_OPTIONS",
    "RUN_FLAGS",
    "RUN_OPTIONS",
    "SELECTION_OPTIONS",
    "STEP_NAMES",
    "TYPE_TITLES",
    "batch_available",
    "build_steps",
    "file_names",
    "offered_chunking",
    "offered_types",
    "ranges_available",
    "run_types",
    "unit_word",
]

STEP_NAMES: Final = (
    "input",
    "type",
    "chunking",
    "pages",
    "output",
    "model",
    "response",
    "images",
    "run",
)
RESPONSE_STEP_OPTIONS: Final = (
    "response_format",
    "schema",
    "context",
    "fallback_context",
    "context_image",
)
RUN_FLAGS: Final = ("force", "dry_run")
RUN_OPTIONS: Final = ("batch", *RUN_FLAGS, "concurrency")
SELECTION_OPTIONS: Final = ("all", "select", "files")
"""The options that choose the items of a folder; one answer sets one."""

TYPE_TITLES: Final = {
    "auto": "Automatic (each file by its type)",
    "text": "Text files",
    "images": "Images",
    "pdf": "PDFs",
}
_ALL: Final = "all"
_NUMBERS: Final = "numbers"
_SEARCH: Final = "search"
_LIST: Final = "list"
_LIST_LIMIT: Final = 50
"""The longest numbered item list shown before a number or range answer."""
CHUNKING_TITLES: Final = {
    "auto": ("Automatic", "cut by tokens per chunk"),
    "line-ranges": ("Line ranges", "each item's _line_ranges.txt"),
    "adjust": ("Adjusted line ranges", "the model moves the boundaries first"),
}
_RUN_FLAG_TITLES = {
    "force": "Extract complete items again (--force)",
    "dry_run": "Dry run: show the plan, no API calls or writes",
}
_IMAGE_TYPES = frozenset({"image", "pdf"})
_CHUNKED_TYPES = frozenset({"text", "pdf"})


def _input(state: State) -> Path | None:
    value = state.value("input")
    return None if value is None else absolute(value)


def _existing_input(text: str) -> str | None:
    if not text.strip():
        return "enter a file or a folder"
    if not absolute(text).exists():
        return "no such file or folder"
    return None


def _relative(item: SourceItem, root: Path) -> str:
    try:
        return item.path.relative_to(root).as_posix()
    except ValueError:
        return item.name


def _typed_indices(state: State) -> list[int]:
    """Return the inventory indices of the items of the chosen type.

    They are in the order the run discovers them and numbers them for
    ``--select``: natural order of their paths below the input.
    """
    location = _input(state)
    if location is None:
        return []
    inventory = inventory_of(state, location)
    kind = str(state.value("type"))
    wanted = item_type(kind)
    indices = [
        index
        for index, item in enumerate(inventory.items)
        if kind == "auto" or item.type == wanted
    ]
    return sorted(
        indices,
        key=lambda index: natural_sort_key(_relative(inventory.items[index], location)),
    )


def _run_indices(state: State) -> list[int]:
    """Return the inventory indices of the items the answers select."""
    location = _input(state)
    if location is None:
        return []
    inventory = inventory_of(state, location)
    indices = _typed_indices(state)
    if not location.is_dir():
        return indices
    files = set(state.value("files") or ())
    if files:
        return [i for i in indices if _relative(inventory.items[i], location) in files]
    expr = state.value("select")
    if expr:
        names = [_relative(inventory.items[index], location) for index in indices]
        return select_items(names, str(expr)).pick(indices)
    return indices


def run_types(state: State) -> set[str]:
    """Return the types of the items the run would process."""
    location = _input(state)
    if location is None:
        return set()
    inventory = inventory_of(state, location)
    return {inventory.items[index].type for index in _run_indices(state)}


def ranges_available(state: State) -> bool:
    """Whether every text item of the run has a ``_line_ranges.txt``."""
    location = _input(state)
    if location is None:
        return False
    inventory = inventory_of(state, location)
    texts = [i for i in _run_indices(state) if inventory.items[i].type == "text"]
    return bool(texts) and all(inventory.has_ranges(index) for index in texts)


def offered_chunking(state: State) -> list[str]:
    """Return the chunking modes the text items can run with.

    ``line-ranges`` and ``adjust`` write a missing ``_line_ranges.txt`` from
    the tokens per chunk first, so every mode runs on every text item.
    """
    return list(CHUNKING_TITLES)


def unit_word(state: State) -> str:
    """Return what ``--pages`` counts for the items of the run."""
    types = run_types(state) & _CHUNKED_TYPES
    if types == {"pdf"}:
        return "pages"
    if types == {"text"} and state.value("chunking") != "auto":
        return "line ranges"
    return "chunks"


class _InputQuestions:
    """The questions of the input step: path, subfolders, a check."""

    def __init__(self, ctx: StepContext) -> None:
        self.ctx = ctx
        self.state = ctx.state

    def questions(self) -> list[Question]:
        return [self.path, self.recursive, self.found]

    def _show_found(self, location: Path) -> None:
        inventory = inventory_of(self.state, location)
        self.ctx.prompter.show(f"  found {inventory.describe()}")

    def path(self) -> Nav:
        state = self.state
        current = _input(state)
        answer = self.ctx.prompter.path(
            "Input (file or folder):",
            default="" if current is None else str(current),
            validate=_existing_input,
        )
        if answer is BACK:
            return BACK
        location = absolute(answer)
        if location != current:
            _clear_selection(state)
        state.set("input", location)
        if not location.is_dir():
            state.clear("recursive")
        self._show_found(location)
        _keep_runnable_type(state)
        return DONE

    def recursive(self) -> Nav | None:
        location = _input(self.state)
        if location is None or not location.is_dir():
            return None
        before = bool(self.state.value("recursive"))
        result = self.ctx.select_option(
            "recursive",
            "Search the subfolders too?",
            [
                Choice(False, "No, this folder only"),
                Choice(True, "Yes, every subfolder"),
            ],
        )
        if result is DONE and bool(self.state.value("recursive")) != before:
            _clear_selection(self.state)
            self._show_found(location)
            _keep_runnable_type(self.state)
        return result

    def _empty(self) -> str | None:
        """Return why the input holds nothing to process, or None."""
        location = _input(self.state)
        if location is None or inventory_of(self.state, location).items:
            return None
        hint = ""
        if location.is_dir() and not self.state.value("recursive"):
            hint = " or search the subfolders"
        return f"  nothing to process; choose another input{hint}"

    def found(self) -> Nav | None:
        """Ask the path again, in place, while the input holds nothing."""
        message = self._empty()
        if message is None:
            return None
        while message is not None:
            self.ctx.prompter.show(message)
            for question in (self.path, self.recursive):
                if question() is BACK:
                    return BACK
            message = self._empty()
        return DONE


def _ask_input(ctx: StepContext) -> Nav:
    return StepContext.sequence(_InputQuestions(ctx).questions())


def offered_types(state: State) -> list[str]:
    """Return the input types the input can run as: auto, then those it holds."""
    location = _input(state)
    if location is None:
        return ["auto"]
    held = {item.type for item in inventory_of(state, location).items}
    types = [kind for kind in INPUT_TYPES if kind != "auto" and item_type(kind) in held]
    return ["auto", *types] if types else []


def _keep_runnable_type(state: State) -> None:
    """Fall back to auto when the input does not hold the chosen type."""
    offered = offered_types(state)
    if offered and state.value("type") not in offered:
        state.set("type", "auto")


def _clear_selection(state: State) -> None:
    """Forget which items were chosen; the next answer chooses again."""
    for name in SELECTION_OPTIONS:
        state.clear(name)


def file_names(state: State) -> list[str]:
    """Return the names of the chosen type's items in the input folder.

    They are in the order ``--select`` numbers them; a file input has none.
    """
    location = _input(state)
    if location is None or not location.is_dir():
        return []
    inventory = inventory_of(state, location)
    return [_relative(inventory.items[i], location) for i in _typed_indices(state)]


def _selection_mode(state: State) -> str:
    if state.value("files"):
        return _LIST
    expr = state.value("select")
    if expr:
        return _NUMBERS if is_numeric_selection(str(expr)) else _SEARCH
    return _ALL


class _TypeQuestions:
    """The questions of the input type step: the type, then which items.

    A folder of several items asks ``Which items?``: all of them
    (``--all``), by number or range or by name search (``--select``), or
    from the list (``--files``).
    """

    def __init__(self, ctx: StepContext) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.how: str | None = None

    def questions(self) -> list[Question]:
        return [self.kind, self.which, self.expression, self.pick]

    def kind(self) -> Nav:
        state = self.state
        location = _input(state)
        inventory = inventory_of(state, location) if location is not None else None
        choices = []
        for kind in offered_types(state):
            note = None
            if inventory is not None and kind != "auto":
                wanted = item_type(kind)
                count = sum(1 for item in inventory.items if item.type == wanted)
                note = plural(count, "item")
            choices.append(Choice(kind, TYPE_TITLES[kind], note))
        before = state.value("type")
        result = self.ctx.select_option("type", "Input type:", choices)
        if result is DONE and state.value("type") != before:
            _clear_selection(state)
        return result

    def which(self) -> Nav | None:
        state = self.state
        names = file_names(state)
        if len(names) < 2:
            _clear_selection(state)
            return None
        answer = self.ctx.prompter.select(
            "Which items?",
            [
                Choice(_ALL, f"All {len(names)}"),
                Choice(_NUMBERS, "By number or range", "such as 1,3 or 2-5"),
                Choice(_SEARCH, "By name search", "every name holding the text"),
                Choice(_LIST, "Choose from the list"),
            ],
            default=_selection_mode(state),
        )
        if answer is BACK:
            return BACK
        self.how = answer
        if answer == _ALL:
            _clear_selection(state)
            state.set("all", True)
        return DONE

    def expression(self) -> Nav | None:
        if self.how not in (_NUMBERS, _SEARCH):
            return None
        state = self.state
        names = file_names(state)
        numbers = self.how == _NUMBERS
        if numbers:
            for number, name in enumerate(names[:_LIST_LIMIT], start=1):
                self.ctx.prompter.show(f"  {number:>4}  {name}")
            if len(names) > _LIST_LIMIT:
                self.ctx.prompter.show(f"  ... and {len(names) - _LIST_LIMIT} more")
        current = state.value("select")
        if current is not None and is_numeric_selection(str(current)) != numbers:
            current = None

        def validate(text: str) -> str | None:
            if not text.strip():
                return "enter numbers or ranges" if numbers else "enter a name part"
            if numbers and not is_numeric_selection(text):
                return "enter numbers or ranges, such as 1,3 or 2-5"
            if not numbers and is_numeric_selection(text):
                return "digits alone are item numbers; choose By number or range"
            selection = select_items(names, text)
            if selection.unmatched:
                return f"no item {', '.join(selection.unmatched)} of 1 to {len(names)}"
            if not selection.indices:
                return "no item matches"
            return None

        answer = self.ctx.prompter.text(
            "Items (numbers or ranges):" if numbers else "Name search:",
            default="" if current is None else str(current),
            validate=validate,
        )
        if answer is BACK:
            return BACK
        _clear_selection(state)
        state.set("select", answer.strip())
        count = len(select_items(names, answer).indices)
        self.ctx.prompter.show(f"  {plural(count, 'item')} chosen")
        return DONE

    def pick(self) -> Nav | None:
        if self.how != _LIST:
            return None
        state = self.state
        names = file_names(state)
        current = [name for name in state.value("files") or () if name in names]
        while True:
            answer = self.ctx.prompter.checkbox(
                "Files:", [Choice(name, name) for name in names], checked=current
            )
            if answer is BACK:
                return BACK
            if answer:
                _clear_selection(state)
                state.set("files", tuple(answer))
                return DONE
            self.ctx.prompter.show("  select at least one file")


def _ask_type(ctx: StepContext) -> Nav:
    return StepContext.sequence(_TypeQuestions(ctx).questions())


def _ask_chunking(ctx: StepContext) -> Nav:
    state = ctx.state
    ask_tokens = chunk_tokens_question(ctx)

    def mode() -> Nav:
        choices = []
        for name in offered_chunking(state):
            title, note = CHUNKING_TITLES[name]
            choices.append(Choice(name, title, note))
        return ctx.select_option("chunking", "Text chunking:", choices)

    def tokens() -> Nav | None:
        if state.value("chunking") != "auto" and ranges_available(state):
            state.clear("chunk_tokens")
            return None
        return ask_tokens()

    def window() -> Nav | None:
        if state.value("chunking") != "adjust":
            state.clear("context_window")
            return None
        return ctx.text_option("context_window", "Lines around each boundary:")

    return StepContext.sequence([mode, tokens, window])


def _page_range(text: str) -> str | None:
    try:
        parse_page_range(text)
    except ValueError as exc:
        return str(exc)
    return None


def _ask_pages(ctx: StepContext) -> Nav:
    state = ctx.state
    chosen: dict[str, str] = {}

    def mode() -> Nav:
        unit = unit_word(state)
        answer = ctx.prompter.select(
            f"Which {unit}?",
            [
                Choice("all", f"All {unit}"),
                Choice(
                    "range",
                    "A range",
                    "such as 5, first:5, last:5, 3-7, 3-, -7 or 1,3,5-8",
                ),
            ],
            default="range" if state.value("pages") else "all",
        )
        if answer is BACK:
            return BACK
        chosen["mode"] = answer
        if answer == "all":
            state.clear("pages")
        return DONE

    def text() -> Nav | None:
        if chosen.get("mode") != "range":
            return None
        answer = ctx.prompter.text(
            "Range:", default=state.value("pages") or "", validate=_page_range
        )
        if answer is BACK:
            return BACK
        state.set("pages", answer.strip())
        return DONE

    return StepContext.sequence([mode, text])


def _folder(text: str) -> str | None:
    if not text.strip():
        return "enter a folder"
    if absolute(text).is_file():
        return "this is a file; enter a folder"
    return None


_FORMAT_TITLES = {"csv": "CSV (_output.csv)"}


def _ask_output(ctx: StepContext) -> Nav:
    state = ctx.state
    chosen: dict[str, str] = {}

    def where() -> Nav:
        folder = state.value("output") is not None
        answer = ctx.prompter.select(
            "Where should the outputs go?",
            [
                Choice("beside", "Beside each input"),
                Choice("folder", "Into one folder", "created when missing"),
            ],
            default="folder" if folder else "beside",
            marker=state.marker("output") if folder else None,
        )
        if answer is BACK:
            return BACK
        chosen["where"] = answer
        if answer == "folder":
            state.clear("beside_input")
        elif has_settings_default(state, "output"):
            state.set("beside_input", True)
        else:
            state.clear("output")
            state.clear("beside_input")
        return DONE

    def folder() -> Nav | None:
        if chosen.get("where") != "folder":
            return None
        current = state.value("output")
        answer = ctx.prompter.path(
            "Output folder:",
            default="" if current is None else str(current),
            validate=_folder,
            only_directories=True,
            marker=state.marker("output"),
        )
        if answer is BACK:
            return BACK
        state.set("output", absolute(answer))
        return DONE

    def layout() -> Nav | None:
        location = _input(state)
        if chosen.get("where") != "folder" or location is None or not location.is_dir():
            state.clear("layout")
            return None
        return ctx.select_option(
            "layout",
            "Folder layout:",
            [
                Choice("flat", "Every output in the output folder"),
                Choice("mirror", "Mirror the input folders"),
            ],
        )

    def formats() -> Nav:
        return ctx.checkbox_option(
            "formats",
            "Also write (chronominer.sqlite and _output.json are always written):",
            [Choice(name, title) for name, title in _FORMAT_TITLES.items()],
        )

    return StepContext.sequence([where, folder, layout, formats])


def _model_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        return StepContext.sequence(ModelQuestions(ctx, session).questions())

    return Step("model", "Model", ask=ask, sets=MODEL_OPTIONS)


def _response_step(session: Session) -> Step:
    """The prompt-and-context step: response format, schema and context."""

    def ask(ctx: StepContext) -> Nav:
        return StepContext.sequence(ResponseQuestions(ctx, session).questions())

    return Step("response", "Prompt and context", ask=ask, sets=RESPONSE_STEP_OPTIONS)


def _images_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        return StepContext.sequence(ImageQuestions(ctx, session).questions())

    return Step(
        "images",
        "Page images",
        ask=ask,
        sets=IMAGE_OPTIONS,
        applies=lambda state: bool(run_types(state) & _IMAGE_TYPES),
    )


def batch_available(state: State, session: Session) -> bool:
    """Whether the run may use the batch API: provider and extension."""
    try:
        spec = resolution_of(state, session).spec
    except ValueError:
        return False
    return (
        supports_batch(spec.model.provider)
        and batch_refusal(spec, settings=session.settings, args=session.args) is None
    )


def _run_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        state = ctx.state

        def submission() -> Nav | None:
            if not batch_available(state, session):
                state.set("batch", False)
                return None
            return ctx.select_option(
                "batch",
                "Submission:",
                [
                    Choice(False, "Synchronous", "results now"),
                    Choice(True, "Batch API", "collected later with batch check"),
                ],
            )

        def flags() -> Nav:
            choices = [
                Choice(name, _RUN_FLAG_TITLES[name], state.marker(name))
                for name in RUN_FLAGS
            ]
            answer = ctx.prompter.checkbox(
                "Run options:",
                choices,
                checked=[name for name in RUN_FLAGS if state.value(name)],
            )
            if answer is BACK:
                return BACK
            for name in RUN_FLAGS:
                state.set(name, name in answer)
            return DONE

        def concurrency() -> Nav | None:
            if state.value("batch"):
                return None
            return ctx.text_option("concurrency", "Parallel requests:")

        return StepContext.sequence([submission, flags, concurrency])

    return Step("run", "Run options", ask=ask, sets=RUN_OPTIONS)


def build_steps(session: Session) -> list[Step]:
    """Return the extract steps in order, task and extension steps excluded."""
    return [
        Step("input", "Input", ask=_ask_input, sets=("input", "recursive")),
        Step(
            "type",
            "Input type and items",
            ask=_ask_type,
            sets=("type", *SELECTION_OPTIONS),
        ),
        Step(
            "chunking",
            "Text chunking",
            ask=_ask_chunking,
            sets=("chunking", "chunk_tokens", "context_window"),
            applies=lambda state: "text" in run_types(state),
        ),
        Step(
            "pages",
            "Pages and chunks",
            ask=_ask_pages,
            sets=("pages",),
            applies=lambda state: bool(run_types(state) & _CHUNKED_TYPES),
        ),
        Step(
            "output",
            "Output and formats",
            ask=_ask_output,
            sets=("output", "beside_input", "layout", "formats"),
        ),
        _model_step(session),
        _response_step(session),
        _images_step(session),
        _run_step(session),
    ]
