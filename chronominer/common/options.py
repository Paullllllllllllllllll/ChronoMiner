"""Option tables: one declaration per run option.

An :class:`OptionTable` generates the argparse arguments of one subcommand,
reads the given values back, writes values as an equivalent argument list,
validates a settings ``defaults`` block and resolves each option with its
source: a flag, then a settings default, then the code default.

Options with a ``fallback`` take the value of another option at the same
precedence level (``--review-model`` falls back to ``--model``).
Options that share an ``exclusive`` name are alternatives: only those set at
the highest precedence level keep their value.
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "Condition",
    "HelpFormatter",
    "Kind",
    "Option",
    "OptionError",
    "OptionTable",
    "Resolved",
    "Source",
]


class HelpFormatter(argparse.HelpFormatter):
    """argparse layout that never splits words at hyphens.

    Option help is wrapped without breaking a flag or a hyphenated value
    across lines. Descriptions and epilogs keep their line breaks: blank
    lines and indented lines stay as written, longer unindented lines wrap.
    """

    def _split_lines(self, text: str, width: int) -> list[str]:
        text = self._whitespace_matcher.sub(" ", text).strip()
        return _wrap(text, width)

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        lines: list[str] = []
        for line in text.splitlines():
            if not line.strip():
                lines.append("")
            elif line[0].isspace():
                lines.append(indent + line.rstrip())
            else:
                lines.extend(_wrap(line.strip(), width, indent))
        return "\n".join(lines)


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    return textwrap.wrap(
        text,
        width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_on_hyphens=False,
        break_long_words=False,
    )


class OptionError(ValueError):
    """An option table, value or defaults block is invalid."""


class Source(StrEnum):
    """Where a resolved value came from."""

    FLAG = "flag"
    SETTINGS = "settings"
    DEFAULT = "default"

    @property
    def rank(self) -> int:
        """Precedence: a higher rank wins."""
        return {"flag": 2, "settings": 1, "default": 0}[self.value]


class Kind(StrEnum):
    """How an option appears on the command line."""

    VALUE = "value"
    SWITCH = "switch"
    FLAG = "flag"
    LIST = "list"


@dataclass(frozen=True)
class Condition:
    """The option applies only while another option has one of ``values``."""

    option: str
    values: frozenset[str]

    def holds(self, values: Mapping[str, Any]) -> bool:
        """Whether the condition holds for the given option values."""
        return values.get(self.option) in self.values

    def describe(self) -> str:
        """Return a short help text, such as ``only for provider openai``."""
        names = ", ".join(sorted(self.values))
        return f"only for {self.option.replace('_', '-')} {names}"


@dataclass(frozen=True)
class Option:
    """Metadata of one run option.

    ``kind`` decides the argparse form: VALUE takes an argument, SWITCH adds
    ``--name/--no-name``, FLAG is ``store_true``, LIST takes a comma-separated
    value and accumulates when given more than once. VALUE and LIST accept
    ``--flag value`` and ``--flag=value`` alike. ``type`` converts one string;
    ``choices`` restricts the converted values; ``choice_provider`` lists
    suggestions that are not enforced (for example model names). ``target`` is
    the dotted field path the value fills in the caller's run specification;
    ``settable`` allows the option in a settings defaults block.
    """

    name: str
    flag: str
    help: str
    group: str = "options"
    kind: Kind = Kind.VALUE
    type: Callable[[str], Any] = str
    choices: tuple[Any, ...] | None = None
    choice_provider: Callable[[], Sequence[str]] | None = None
    default: Any = None
    default_help: str | None = None
    metavar: str | None = None
    condition: Condition | None = None
    fallback: str | None = None
    exclusive: str | None = None
    target: str | None = None
    settable: bool = True

    def convert(self, raw: Any) -> Any:
        """Convert a command-line string or a native settings value.

        Raises:
            OptionError: The value has the wrong type or is not a choice.
        """
        if self.kind in (Kind.SWITCH, Kind.FLAG):
            if isinstance(raw, bool):
                return raw
            raise OptionError(f"{self.flag}: expected true or false, got {raw!r}")
        if self.kind is Kind.LIST:
            return self._convert_list(raw)
        return self._convert_one(raw)

    def _convert_one(self, raw: Any) -> Any:
        if isinstance(raw, bool | list | tuple | dict):
            raise OptionError(f"{self.flag}: invalid value {raw!r}")
        try:
            value = self.type(raw if isinstance(raw, str) else str(raw))
        except (ValueError, TypeError, argparse.ArgumentTypeError) as exc:
            raise OptionError(f"{self.flag}: {exc}") from exc
        if self.choices is not None and value not in self.choices:
            allowed = ", ".join(str(choice) for choice in self.choices)
            raise OptionError(f"{self.flag}: {value!r} is not one of {allowed}")
        return value

    def _convert_list(self, raw: Any) -> tuple[Any, ...]:
        if isinstance(raw, str):
            parts: list[Any] = [part.strip() for part in raw.split(",")]
        elif isinstance(raw, list | tuple):
            parts = list(raw)
        else:
            raise OptionError(f"{self.flag}: expected a list, got {raw!r}")
        return self._ordered([self._convert_one(part) for part in parts if part != ""])

    def _ordered(self, values: Sequence[Any]) -> tuple[Any, ...]:
        """Return converted list items without duplicates, in choice order."""
        if self.choices is not None:
            return tuple(choice for choice in self.choices if choice in values)
        return tuple(dict.fromkeys(values))

    def format(self, value: Any) -> list[str]:
        """Return the argument tokens that set ``value``.

        A value that begins with ``-`` is written as ``--flag=value``, since
        argparse would read it as a flag on its own.
        """
        if self.kind is Kind.SWITCH:
            return [self.flag if value else self.negative_flag]
        if self.kind is Kind.FLAG:
            return [self.flag] if value else []
        if self.kind is Kind.LIST:
            text = ",".join(str(item) for item in value)
        else:
            text = str(value)
        if text.startswith("-"):
            return [f"{self.flag}={text}"]
        return [self.flag, text]

    @property
    def negative_flag(self) -> str:
        """The ``--no-`` form of a switch."""
        return "--no-" + self.flag.removeprefix("--")

    def help_text(self) -> str:
        """Return the help line with the default and the condition."""
        parts = [self.help]
        if self.choices is not None and self.kind is Kind.LIST:
            parts.append("from " + ", ".join(str(choice) for choice in self.choices))
        default = self.default_help
        if default is None and self.default is not None and self.kind is not Kind.FLAG:
            default = self._default_text()
        if default:
            parts.append(f"default: {default}")
        if self.condition is not None:
            parts.append(self.condition.describe())
        return "; ".join(parts)

    def _default_text(self) -> str:
        if self.kind is Kind.SWITCH:
            return "on" if self.default else "off"
        if self.kind is Kind.LIST:
            return ",".join(str(item) for item in self.default)
        return str(self.default)

    def _argparse_type(self, raw: str) -> Any:
        try:
            return self.convert(raw)
        except OptionError as exc:
            message = str(exc).removeprefix(f"{self.flag}: ")
            raise argparse.ArgumentTypeError(message) from exc


@dataclass(frozen=True)
class Resolved:
    """A resolved value and its source."""

    value: Any
    source: Source


class OptionTable:
    """An ordered, validated set of options for one subcommand."""

    def __init__(
        self,
        options: Iterable[Option],
        *,
        group_titles: Mapping[str, str] | None = None,
    ) -> None:
        self._options: dict[str, Option] = {}
        flags: set[str] = set()
        for option in options:
            if option.name in self._options:
                raise OptionError(f"duplicate option name {option.name!r}")
            if option.flag in flags:
                raise OptionError(f"duplicate flag {option.flag!r}")
            if not option.flag.startswith("--"):
                raise OptionError(f"flag {option.flag!r} must start with --")
            flags.add(option.flag)
            self._options[option.name] = option
        for option in self._options.values():
            for ref in (option.fallback, getattr(option.condition, "option", None)):
                if ref is not None and ref not in self._options:
                    raise OptionError(f"{option.name}: unknown option {ref!r}")
        self._group_titles = dict(group_titles or {})

    def __iter__(self) -> Iterator[Option]:
        return iter(self._options.values())

    def __len__(self) -> int:
        return len(self._options)

    def __contains__(self, name: object) -> bool:
        return name in self._options

    def __getitem__(self, name: str) -> Option:
        return self._options[name]

    @property
    def names(self) -> tuple[str, ...]:
        """Option names in table order."""
        return tuple(self._options)

    def subset(self, names: Iterable[str]) -> OptionTable:
        """Return a table of the named options in table order.

        Raises:
            OptionError: A name is unknown, or a kept option's fallback or
                condition names an option left out.
        """
        wanted = set(names)
        unknown = wanted - set(self._options)
        if unknown:
            raise OptionError(f"unknown options: {', '.join(sorted(unknown))}")
        return OptionTable(
            (option for name, option in self._options.items() if name in wanted),
            group_titles=self._group_titles,
        )

    def by_target(self, target: str) -> Option:
        """Return the option that fills ``target``."""
        for option in self._options.values():
            if option.target == target:
                return option
        raise KeyError(target)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add every option to ``parser``; absent options leave no attribute."""
        groups: dict[str, Any] = {}
        exclusive: dict[str, Any] = {}
        for option in self._options.values():
            if option.group not in groups:
                title = self._group_titles.get(option.group, option.group)
                groups[option.group] = parser.add_argument_group(title)
            container = groups[option.group]
            if option.exclusive is not None:
                if option.exclusive not in exclusive:
                    exclusive[option.exclusive] = (
                        container.add_mutually_exclusive_group()
                    )
                container = exclusive[option.exclusive]
            _add_argument(container, option)

    def build_parser(
        self,
        prog: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
    ) -> argparse.ArgumentParser:
        """Return a new parser holding the table's arguments."""
        parser = argparse.ArgumentParser(
            prog=prog,
            description=description,
            epilog=epilog,
            formatter_class=HelpFormatter,
        )
        self.add_arguments(parser)
        return parser

    def values(self, namespace: argparse.Namespace) -> dict[str, Any]:
        """Return the options given on the command line, by name."""
        given = vars(namespace)
        return {name: given[name] for name in self._options if name in given}

    def parse(self, argv: Sequence[str]) -> dict[str, Any]:
        """Parse ``argv`` with a fresh parser; exits with status 2 on errors."""
        return self.values(self.build_parser().parse_args(list(argv)))

    def to_argv(self, values: Mapping[str, Any]) -> list[str]:
        """Return argument tokens for ``values`` in table order."""
        unknown = set(values) - set(self._options)
        if unknown:
            raise OptionError(f"unknown options: {', '.join(sorted(unknown))}")
        argv: list[str] = []
        for name, option in self._options.items():
            if name in values:
                argv.extend(option.format(values[name]))
        return argv

    def check_defaults(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a settings defaults block and convert its values.

        Raises:
            OptionError: A key is not a settable option name, a value does
                not convert, or two alternatives are both set.
        """
        checked: dict[str, Any] = {}
        for key, value in raw.items():
            option = self._options.get(str(key))
            if option is None or not option.settable:
                raise OptionError(f"unknown option in defaults: {key!r}")
            checked[option.name] = option.convert(value)
        self._check_exclusive(checked, "defaults")
        return checked

    def _check_exclusive(self, values: Mapping[str, Any], where: str) -> None:
        seen: dict[str, str] = {}
        for name in values:
            group = self._options[name].exclusive
            if group is None:
                continue
            if group in seen:
                raise OptionError(
                    f"{where}: {seen[group]} and {name} exclude each other"
                )
            seen[group] = name

    def resolve(
        self,
        flags: Mapping[str, Any],
        defaults: Mapping[str, Any] | None = None,
    ) -> dict[str, Resolved]:
        """Resolve every option: flag, then settings default, then code default.

        Each level is searched for the option itself first and then along its
        fallback chain. Among alternatives, members set below the highest
        level revert to their code default and take the winner's source.
        """
        levels = ((Source.FLAG, flags), (Source.SETTINGS, defaults or {}))
        resolved: dict[str, Resolved] = {}
        for name, option in self._options.items():
            resolved[name] = self._resolve_one(option, levels)
        for group in {o.exclusive for o in self._options.values() if o.exclusive}:
            members = [o for o in self._options.values() if o.exclusive == group]
            top = max(resolved[o.name].source.rank for o in members)
            winner = next(
                resolved[o.name].source
                for o in members
                if resolved[o.name].source.rank == top
            )
            for option in members:
                if resolved[option.name].source.rank < top:
                    resolved[option.name] = Resolved(option.default, winner)
        return resolved

    def _resolve_one(
        self,
        option: Option,
        levels: Sequence[tuple[Source, Mapping[str, Any]]],
    ) -> Resolved:
        chain = self._chain(option)
        for source, given in levels:
            for name in chain:
                if name in given:
                    return Resolved(given[name], source)
        return Resolved(option.default, Source.DEFAULT)

    def _chain(self, option: Option) -> list[str]:
        chain = [option.name]
        current = option
        while current.fallback is not None and current.fallback not in chain:
            chain.append(current.fallback)
            current = self._options[current.fallback]
        return chain

    def by_targets(
        self, resolved: Mapping[str, Resolved]
    ) -> tuple[dict[str, Any], dict[str, Source]]:
        """Return values and sources keyed by target for targeted options."""
        values: dict[str, Any] = {}
        sources: dict[str, Source] = {}
        for name, option in self._options.items():
            if option.target is None or name not in resolved:
                continue
            values[option.target] = resolved[name].value
            sources[option.target] = resolved[name].source
        return values, sources

    def inapplicable(self, values: Mapping[str, Any]) -> list[str]:
        """Return the names of set options whose condition does not hold."""
        return [
            name
            for name, option in self._options.items()
            if option.condition is not None
            and values.get(name) is not None
            and not option.condition.holds(values)
        ]


class _Accumulate(argparse.Action):
    """Store a list option, merged with the values of its earlier mentions."""

    def __init__(self, *args: Any, option: Option, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.option = option

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        earlier = getattr(namespace, self.dest, None)
        if earlier is not None:
            values = self.option._ordered((*earlier, *values))
        setattr(namespace, self.dest, values)


def _add_argument(container: Any, option: Option) -> None:
    common: dict[str, Any] = {
        "dest": option.name,
        "help": option.help_text().replace("%", "%%"),
        "default": argparse.SUPPRESS,
    }
    if option.kind is Kind.SWITCH:
        container.add_argument(
            option.flag, action=argparse.BooleanOptionalAction, **common
        )
    elif option.kind is Kind.FLAG:
        container.add_argument(option.flag, action="store_true", **common)
    else:
        metavar = option.metavar
        if metavar is None and option.kind is Kind.VALUE and option.choices:
            metavar = "{" + ",".join(str(choice) for choice in option.choices) + "}"
        if option.kind is Kind.LIST:
            common.update(action=_Accumulate, option=option)
        container.add_argument(
            option.flag,
            type=option._argparse_type,
            metavar=metavar or option.name.upper(),
            **common,
        )
