from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional

import pytest

import modules.cli.execution_framework as ef


class _DummyUI:
    def __init__(self, logger: logging.Logger, use_colors: bool = True):
        self.logger = logger
        self.use_colors = use_colors
        self.banner_shown = False
        self.messages: List[tuple[str, str]] = []

    def display_banner(self) -> None:
        self.banner_shown = True

    def print_info(self, msg: str) -> None:
        self.messages.append(("info", msg))

    def print_warning(self, msg: str) -> None:
        self.messages.append(("warning", msg))

    def print_error(self, msg: str) -> None:
        self.messages.append(("error", msg))

    def print_success(self, msg: str) -> None:
        self.messages.append(("success", msg))


def _patch_common(monkeypatch: pytest.MonkeyPatch, *, interactive: bool) -> None:
    monkeypatch.setattr(ef, "UserInterface", _DummyUI)

    def _setup_logger(_: str) -> logging.Logger:
        logger = logging.getLogger("test.execution_framework")
        logger.addHandler(logging.NullHandler())
        return logger

    monkeypatch.setattr(ef, "setup_logger", _setup_logger)
    monkeypatch.setattr(ef, "should_use_interactive_mode", lambda _cfg: interactive)
    monkeypatch.setattr(
        ef,
        "load_core_resources",
        lambda: (object(), {"paths": True}, {"model": True}, {"chunk": True}, {"schemas": True}),
    )


def test_dualmodescript_execute_interactive_calls_run_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch, interactive=True)

    class _Script(ef.DualModeScript):
        def __init__(self) -> None:
            super().__init__("test")
            self.interactive_called = False
            self.cli_called = False

        def create_argument_parser(self) -> argparse.ArgumentParser:
            raise AssertionError("create_argument_parser should not be called in interactive mode")

        def run_interactive(self) -> None:
            self.interactive_called = True

        def run_cli(self, args: argparse.Namespace) -> None:
            self.cli_called = True

    s = _Script()
    s.execute()

    assert s.is_interactive is True
    assert s.ui is not None
    assert isinstance(s.ui, _DummyUI)
    assert s.ui.banner_shown is True
    assert s.interactive_called is True
    assert s.cli_called is False


def test_dualmodescript_execute_cli_calls_run_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch, interactive=False)

    class _Script(ef.DualModeScript):
        def __init__(self) -> None:
            super().__init__("test")
            self.received_args: Optional[argparse.Namespace] = None

        def create_argument_parser(self) -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(add_help=False)
            parser.parse_args = lambda: argparse.Namespace(foo="bar")
            return parser

        def run_interactive(self) -> None:
            raise AssertionError("run_interactive should not be called in CLI mode")

        def run_cli(self, args: argparse.Namespace) -> None:
            self.received_args = args

    s = _Script()
    s.execute()

    assert s.is_interactive is False
    assert s.ui is None
    assert s.received_args is not None
    assert s.received_args.foo == "bar"


def test_dualmodescript_keyboardinterrupt_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch, interactive=True)

    class _Script(ef.DualModeScript):
        def create_argument_parser(self) -> argparse.ArgumentParser:
            raise AssertionError("not used")

        def run_interactive(self) -> None:
            raise KeyboardInterrupt

        def run_cli(self, args: argparse.Namespace) -> None:
            raise AssertionError("not used")

    with pytest.raises(SystemExit) as exc:
        _Script("test").execute()

    assert exc.value.code == 0


def test_create_simple_dual_mode_executor_passes_config_to_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Dict[str, Any] = {}

    def parser_factory() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.parse_args = lambda: argparse.Namespace(x=1)
        return parser

    def interactive_runner(ui: _DummyUI, config: Dict[str, Any]) -> None:
        seen["mode"] = "interactive"
        seen["ui"] = ui
        seen["config"] = config

    def cli_runner(args: argparse.Namespace, config: Dict[str, Any]) -> None:
        seen["mode"] = "cli"
        seen["args"] = args
        seen["config"] = config

    main = ef.create_simple_dual_mode_executor(
        script_name="simple",
        parser_factory=parser_factory,
        interactive_runner=interactive_runner,
        cli_runner=cli_runner,
    )

    _patch_common(monkeypatch, interactive=False)
    main()

    assert seen["mode"] == "cli"
    assert seen["args"].x == 1
    assert set(seen["config"].keys()) == {"paths", "model", "chunking", "schemas"}


@pytest.mark.asyncio
async def test_async_dual_mode_script_execute_async_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch, interactive=False)

    class _Script(ef.AsyncDualModeScript):
        def __init__(self) -> None:
            super().__init__("test")
            self.received_args: Optional[argparse.Namespace] = None

        def create_argument_parser(self) -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(add_help=False)
            parser.parse_args = lambda: argparse.Namespace(foo="bar")
            return parser

        async def run_interactive(self) -> None:
            raise AssertionError("not used")

        async def run_cli(self, args: argparse.Namespace) -> None:
            self.received_args = args

    s = _Script()
    await s._execute_async()

    assert s.received_args is not None
    assert s.received_args.foo == "bar"
