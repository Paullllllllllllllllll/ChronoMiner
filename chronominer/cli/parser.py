"""Argument parsing: the top-level parser and one parser per subcommand."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, NoReturn

from chronominer import ext
from chronominer.common.options import HelpFormatter, OptionTable
from chronominer.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    DROP_OPTIONS,
    OPTIONS,
    PROVIDER_TABLES,
    RANGES_ADJUST_OPTIONS,
    RANGES_GENERATE_OPTIONS,
    RENDER_OPTIONS,
    REPAIR_OPTIONS,
)

__all__ = [
    "SUBCOMMANDS",
    "UI_DESCRIPTION",
    "Parser",
    "UsageError",
    "add_ui_command",
    "build_parser",
]

PROG = "chronominer"

DESCRIPTION = (
    "Extract structured data from text files, images and PDFs with language "
    "models, driven by a JSON schema."
)

TOP_EPILOG = "Run 'chronominer <command> --help' for the options of a command."

PRECEDENCE = """\
Precedence: a flag wins over the settings defaults: block, which wins over the \
code default. --settings FILE reads another settings file instead of \
config/settings.yaml.

"""

JSON_CONTRACT = """
With --json, stdout carries exactly one JSON summary line on every exit; \
progress, warnings and errors go to stderr.
"""


def _exit_codes(*rows: tuple[str, str]) -> str:
    """Return an exit-code block: one ``code  meaning`` row each, then 130."""
    lines = ["Exit codes:"]
    for code, meaning in (*rows, ("130", "interrupted (Ctrl+C or Ctrl+Break)")):
        lines.append(f"  {code:<4} {meaning}")
    return "\n".join(lines) + "\n"


EXTRACT_EPILOG = (
    PRECEDENCE
    + _exit_codes(
        (
            "0",
            "every item completed or, under --batch, was submitted, or\n"
            "       nothing was left to do",
        ),
        (
            "1",
            "an item failed (refused items included), or another run\n"
            "       leases one (a killed run keeps its lease until it expires)",
        ),
        (
            "2",
            "usage or configuration error, no item found, several items\n"
            "       without --all, --select or --files, or a --select that\n"
            "       matches nothing",
        ),
    )
    + JSON_CONTRACT
    + """
Examples:
  Extract the price lists of one text file:
    chronominer extract --schema price_list_entries --input list.txt
      --output out
  Every text, image and PDF file of a folder, with a CSV view:
    chronominer extract --schema wage_records --input sources --all
      --output out --formats csv
  The first three images of a folder:
    chronominer extract --schema census_households --input scans
      --type images --select 1-3
  Text chunks from the line ranges beside each input:
    chronominer extract --schema trade_records --input ledger.txt
      --chunking line-ranges
  Plan a run without API calls and print the summary as JSON:
    chronominer extract --schema price_list_entries --input sources --all
      --dry-run --json
"""
)

RANGES_GENERATE_EPILOG = (
    PRECEDENCE
    + _exit_codes(
        ("0", "every text file got its line ranges"),
        ("1", "a text file failed"),
        ("2", "usage or configuration error, or no text file found"),
    )
    + JSON_CONTRACT
    + """
Examples:
  Write {stem}_line_ranges.txt beside every text file of a folder:
    chronominer ranges generate --input texts --chunk-tokens 3000
"""
)

RANGES_ADJUST_EPILOG = (
    PRECEDENCE
    + _exit_codes(
        ("0", "every text file was adjusted"),
        ("1", "a text file failed"),
        (
            "2",
            "usage or configuration error (an unknown schema, an unset key\n"
            "       variable), or no text file with a line-ranges file",
        ),
    )
    + JSON_CONTRACT
    + """
Examples:
  Move the range boundaries of one file to entry boundaries:
    chronominer ranges adjust --input ledger.txt --schema trade_records
  Adjust again from the start, with more lines around each boundary:
    chronominer ranges adjust --input texts --schema wage_records
      --context-window 10 --force
"""
)

REPAIR_EPILOG = (
    PRECEDENCE
    + _exit_codes(
        ("0", "every targeted chunk was repaired, or nothing was left to do"),
        ("1", "a targeted chunk stayed unrepaired, or the repair failed"),
        (
            "2",
            "usage or configuration error: a missing input, an input no\n"
            "       database records, a database the store refuses, no key or\n"
            "       batch API, or a model without image input for page chunks",
        ),
    )
    + JSON_CONTRACT
    + """
Examples:
  Run the failed chunks below an output folder again:
    chronominer repair --input out
  Repair two chunks of one schema's documents through the batch API:
    chronominer repair --input out --schema wage_records --indices 0,3
      --batch
"""
)

BATCH_CHECK_EPILOG = (
    """\
Without --formats each finalized document is rendered in the formats of the \
run that submitted it. With --wait the check repeats every --poll-minutes \
until no batch work is pending; no check starts after --wait-hours, but a \
running one finishes.

"""
    + _exit_codes(
        ("0", "nothing failed"),
        (
            "1",
            "a submission failed or is unresolved; with --wait also when the\n"
            "       deadline passed with work pending or a check failed with an\n"
            "       error that is not retried",
        ),
        (
            "2",
            "usage error, a missing folder, a database the store refuses, or\n"
            "       an --adopt or --abandon that names nothing to act on",
        ),
    )
    + JSON_CONTRACT
)

BATCH_CANCEL_EPILOG = (
    _exit_codes(
        ("0", "every batch was cancelled, or none was left"),
        ("1", "a cancellation or a provider listing failed"),
        ("2", "usage error, a missing folder or a database the store refuses"),
    )
    + JSON_CONTRACT
)

RENDER_EPILOG = (
    """\
Without --formats each document is rendered in the formats it was rendered \
in before. A file changed outside the tool is left as it is unless --force.

"""
    + _exit_codes(
        ("0", "every file holds its rendering or was left as edited"),
        ("1", "a file could not be replaced, or another run holds a document"),
        ("2", "usage error, or a database the store refuses"),
    )
    + JSON_CONTRACT
    + """
Examples:
  Render every document below a folder again:
    chronominer render --input out
  Write the CSV views too, replacing files edited by hand:
    chronominer render --input out --formats csv --force
"""
)

DROP_EPILOG = (
    """\
drop deletes each selected document with its chunks, line ranges, responses \
and entries, and the rendered files whose contents still match the recorded \
hash; a file changed outside the tool is kept and named. A document that \
another run leases, or whose chunks an open or unresolved batch holds, is \
refused. The next extraction processes a dropped document from the start.

"""
    + _exit_codes(
        ("0", "every selected document was dropped (or, with --dry-run, would be)"),
        ("1", "a document was refused: a live run or a batch holds it"),
        (
            "2",
            "usage error, no store at --input, or a --select or --files that\n"
            "       matches no stored document",
        ),
    )
    + JSON_CONTRACT
    + """
Examples:
  List what dropping two documents would delete:
    chronominer drop --input out --select 1-2 --dry-run
  Drop one document by its input file name:
    chronominer drop --input out --files ledger.txt
"""
)

UI_DESCRIPTION = (
    "Guided run: choose a task (extract, ranges generate, ranges adjust, batch "
    "check, batch cancel, repair, render or drop), answer its steps "
    "with Back navigation, then review the values, marked where they come from "
    "the settings defaults, with the equivalent command, and run it, copy the "
    "command or quit. Needs an interactive terminal."
)

UI_EPILOG = (
    "Exit codes are those of the chosen task's command. For scripts and "
    "agents, use 'chronominer extract' and the other commands with flags."
)


class UsageError(Exception):
    """The command line is invalid; *usage* is the parser's usage text."""

    def __init__(self, message: str, *, prog: str, usage: str) -> None:
        super().__init__(message)
        self.prog = prog
        self.usage = usage


class Parser(argparse.ArgumentParser):
    """An argument parser that raises :class:`UsageError` instead of exiting.

    Subcommand parsers created from it share the class, so every parse
    error reaches the caller, which can still write the JSON summary.
    """

    def error(self, message: str) -> NoReturn:
        raise UsageError(message, prog=self.prog, usage=self.format_usage())


# The action ``add_subparsers`` returns; its class is private to argparse.
Commands = Any


def _add_command(
    commands: Commands,
    name: str,
    table: OptionTable,
    *,
    help: str,
    description: str,
    epilog: str,
    usage: str | None = None,
) -> argparse.ArgumentParser:
    """Add a subcommand; the extension's arguments join the commands that call
    providers (:data:`~chronominer.spec.PROVIDER_TABLES`)."""
    parser: argparse.ArgumentParser = commands.add_parser(
        name,
        help=help,
        description=description,
        epilog=epilog,
        usage=usage,
        formatter_class=HelpFormatter,
    )
    table.add_arguments(parser)
    if any(table is provider for provider in PROVIDER_TABLES):
        ext.register_args(parser)
    return parser


def _add_group(
    commands: Commands, name: str, *, help: str, description: str, dest: str
) -> tuple[argparse.ArgumentParser, Commands]:
    """Add a command that only holds subcommands; return it and its actions."""
    group: argparse.ArgumentParser = commands.add_parser(
        name, help=help, description=description, formatter_class=HelpFormatter
    )
    actions = group.add_subparsers(dest=dest, metavar="COMMAND", title="commands")
    group.set_defaults(group_parser=group)
    return group, actions


def add_extract_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the extract subcommand."""
    return _add_command(
        commands,
        "extract",
        OPTIONS,
        help="extract structured entries from texts, images and PDFs",
        description=(
            "Extract the entries a schema describes from text files, images "
            "and PDFs; resumes earlier runs unless --force is given."
        ),
        epilog=EXTRACT_EPILOG,
        usage="%(prog)s --schema NAME|PATH --input PATH [options]",
    )


def add_ranges_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the ranges subcommand with its generate and adjust subcommands."""
    ranges, actions = _add_group(
        commands,
        "ranges",
        help="write or adjust the line ranges of text inputs",
        description=(
            "Write the {stem}_line_ranges.txt files of text inputs, or move "
            "their boundaries to entry boundaries with a model."
        ),
        dest="ranges_command",
    )
    _add_command(
        actions,
        "generate",
        RANGES_GENERATE_OPTIONS,
        help="write token-based line ranges",
        description=(
            "Cut each text file into line ranges of about --chunk-tokens "
            "tokens and write them to {stem}_line_ranges.txt beside it."
        ),
        epilog=RANGES_GENERATE_EPILOG,
        usage="%(prog)s --input PATH [options]",
    )
    _add_command(
        actions,
        "adjust",
        RANGES_ADJUST_OPTIONS,
        help="move line-range boundaries to entry boundaries",
        description=(
            "Ask the model where the entries of a schema begin near each "
            "range boundary, store the decisions and render "
            "{stem}_line_ranges.txt from them again."
        ),
        epilog=RANGES_ADJUST_EPILOG,
        usage="%(prog)s --input PATH --schema NAME|PATH [options]",
    )
    return ranges


def add_batch_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the batch subcommand with its check and cancel subcommands."""
    batch, actions = _add_group(
        commands,
        "batch",
        help="check or cancel batch submissions",
        description="Check and finalize, or cancel, batch submissions.",
        dest="batch_command",
    )
    _add_command(
        actions,
        "check",
        BATCH_CHECK_OPTIONS,
        help="check batches and write the outputs of finished ones",
        description=(
            "Check the batch submissions found below a folder and write the "
            "outputs of every finished batch."
        ),
        epilog=BATCH_CHECK_EPILOG,
    )
    _add_command(
        actions,
        "cancel",
        BATCH_CANCEL_OPTIONS,
        help="cancel batches",
        description=(
            "Cancel the named batches, every unfinished batch, or the open batch "
            "submissions below a folder."
        ),
        epilog=BATCH_CANCEL_EPILOG,
    )
    return batch


def add_repair_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the repair subcommand."""
    return _add_command(
        commands,
        "repair",
        REPAIR_OPTIONS,
        help="run the failed chunks of extractions again",
        description=(
            "Find the failed chunks of the documents at a file, folder or "
            "database, run them again, store the new results in the database "
            "and render the documents' outputs from it again."
        ),
        epilog=REPAIR_EPILOG,
        usage="%(prog)s [--input PATH] [options]",
    )


def add_render_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the render subcommand."""
    return _add_command(
        commands,
        "render",
        RENDER_OPTIONS,
        help="render the JSON, CSV and line-range views from the store again",
        description=(
            "Render the files of every document in the databases at or below "
            "a folder from the store, without model calls."
        ),
        epilog=RENDER_EPILOG,
    )


def add_drop_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the drop subcommand."""
    return _add_command(
        commands,
        "drop",
        DROP_OPTIONS,
        help="delete stored documents so that the next run starts them anew",
        description=(
            "Delete the selected documents of a store with their rows and "
            "rendered files, without model calls."
        ),
        epilog=DROP_EPILOG,
        usage="%(prog)s --input PATH (--select EXPR | --files LIST | --all)",
    )


def add_ui_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the ui subcommand: the settings option and the extension arguments."""
    ui: argparse.ArgumentParser = commands.add_parser(
        "ui",
        help="guided run in an interactive terminal",
        description=UI_DESCRIPTION,
        epilog=UI_EPILOG,
        formatter_class=HelpFormatter,
    )
    OPTIONS.subset(("settings",)).add_arguments(ui)
    ext.register_args(ui)
    return ui


SUBCOMMANDS: tuple[Callable[[Commands], argparse.ArgumentParser], ...] = (
    add_extract_command,
    add_ranges_command,
    add_batch_command,
    add_repair_command,
    add_render_command,
    add_drop_command,
    add_ui_command,
)


def build_parser() -> Parser:
    """Return the top-level parser with every subcommand."""
    parser = Parser(
        prog=PROG,
        description=DESCRIPTION,
        epilog=TOP_EPILOG,
        formatter_class=HelpFormatter,
    )
    commands = parser.add_subparsers(
        dest="command", metavar="COMMAND", title="commands"
    )
    for add_command in SUBCOMMANDS:
        add_command(commands)
    return parser
