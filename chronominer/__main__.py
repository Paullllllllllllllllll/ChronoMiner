"""Console entry point: dispatch the subcommand; bare ``chronominer`` prints help.

On Windows, Ctrl+Break (``SIGBREAK``) is handled like Ctrl+C
(:mod:`chronominer.common.interrupt`), so an interrupted run releases its
leases and records its end.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from chronominer.cli.parser import UsageError, build_parser
from chronominer.common.interrupt import install_break_handler

# Subcommand path -> (module, function) that runs it.
_HANDLERS: dict[tuple[str, ...], tuple[str, str]] = {
    ("extract",): ("chronominer.cli.extract", "execute"),
    ("ranges", "generate"): ("chronominer.cli.ranges", "execute_generate"),
    ("ranges", "adjust"): ("chronominer.cli.ranges", "execute_adjust"),
    ("batch", "check"): ("chronominer.cli.batch", "execute_check"),
    ("batch", "cancel"): ("chronominer.cli.batch", "execute_cancel"),
    ("repair",): ("chronominer.cli.repair", "execute_repair"),
    ("render",): ("chronominer.cli.store", "execute_render"),
    ("drop",): ("chronominer.cli.store", "execute_drop"),
    ("ui",): ("chronominer.ui.app", "execute"),
}

# Commands that only hold subcommands, with the attribute naming the chosen one.
_GROUPS = {"ranges": "ranges_command", "batch": "batch_command"}


def _command(namespace: argparse.Namespace) -> tuple[str, ...]:
    group = _GROUPS.get(namespace.command)
    if group is not None:
        return (namespace.command, getattr(namespace, group) or "")
    return (namespace.command or "",)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the subcommand in *argv* (``sys.argv[1:]`` when None)."""
    install_break_handler()
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args:
        parser.print_help()
        return 0
    try:
        namespace = parser.parse_args(args)
    except UsageError as error:
        from chronominer.cli.command import report_usage_error

        return report_usage_error(error, args)
    handler = _HANDLERS.get(_command(namespace))
    if handler is None:
        help_parser = getattr(namespace, "group_parser", None) or parser
        help_parser.print_help()
        return 0
    module_name, function = handler
    execute = getattr(importlib.import_module(module_name), function)
    return int(execute(namespace))


if __name__ == "__main__":
    raise SystemExit(main())
