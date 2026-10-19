"""Pytest fixtures and configuration for ChronoMiner tests.

The session is hermetic. ``pytest_configure`` strips provider keys from the
environment, points the home directories at a temp dir and seeds a private
tiktoken cache. The autouse ``_hermetic`` fixture then redirects every
per-user path into the test's temp area (home, temp dir, the default
settings file and the logs folder of ``chronominer.settings``), so no real
``config/settings.yaml`` is ever read. The fixture blocks non-loopback
connections, launches of the tool itself and of clipboard programs, fails
any test that writes outside the pytest base temp directory, and removes the
log handlers a test installed.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from chronominer.common import testing
from chronominer.common.testing import (
    XDG_VARS,
    LaunchEntry,
    WriteGuard,
    home_env,
    is_loopback_address,
    is_secret_env_var,
)
from chronominer.settings import Settings
from chronominer.spec import RunSpec, resolve_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAUNCH_ENTRY = LaunchEntry(
    programs=("chronominer",),
    scripts=("chronominer/__main__.py",),
    modules=("chronominer",),
)

DEFAULT_TEST_SCHEMA = "WageRecords"
"""The schema of :func:`make_spec` unless a test names another."""

_SESSION_ENV_BACKUP: dict[str, str | None] = {}

__all__ = [
    "CLIPBOARD_PROGRAMS",
    "DEFAULT_TEST_SCHEMA",
    "PROJECT_ROOT",
    "ClipboardLaunchRefusedError",
    "NetworkBlockedError",
    "ToolLaunchRefusedError",
    "WriteGuard",
    "home_env",
    "is_loopback_address",
    "is_secret_env_var",
    "launches_clipboard",
    "launches_tool",
    "make_job",
    "make_settings",
    "make_spec",
]


class NetworkBlockedError(RuntimeError):
    """Raised when a test connects to a non-loopback address."""


class ToolLaunchRefusedError(RuntimeError):
    """Raised when a test launches a ChronoMiner entry point."""


class ClipboardLaunchRefusedError(RuntimeError):
    """Raised when a test starts a clipboard program."""


CLIPBOARD_PROGRAMS = frozenset({"clip", "pbcopy", "xclip", "xsel", "wl-copy"})
"""Programs that would overwrite the user's clipboard."""


def make_spec(input_path: Path | str = "doc.pdf", **flags: Any) -> RunSpec:
    """Return an extract spec from option values keyed by option name.

    The schema is :data:`DEFAULT_TEST_SCHEMA` unless *flags* name one; a
    folder's items are all selected (``--all``) unless *flags* select some.
    """
    flags.setdefault("schema", DEFAULT_TEST_SCHEMA)
    if not {"select", "files", "all"} & set(flags):
        flags["all"] = True
    return resolve_spec({"input": Path(input_path), **flags}).spec


def make_settings(**changes: Any) -> Settings:
    """Return the code-default settings with *changes* applied."""
    return dataclasses.replace(Settings(), **changes)


def make_job(
    spec: RunSpec,
    *,
    output_dir: Path | None,
    input_root: Path,
    settings: Settings | None = None,
    events: Any = None,
    **changes: Any,
) -> Any:
    """Return the extraction job of *spec*; *changes* set further job fields."""
    from chronominer.events import NullEvents
    from chronominer.llm.caller import CallEnv

    job = importlib.import_module("chronominer.extract").Job
    values = settings if settings is not None else Settings()
    return job(
        spec=spec,
        settings=values,
        events=events if events is not None else NullEvents(),
        env=CallEnv.create(values),
        output_dir=output_dir,
        input_root=input_root,
        **changes,
    )


def _session_dir(config: pytest.Config, name: str) -> Path:
    factory: pytest.TempPathFactory | None = getattr(config, "_tmp_path_factory", None)
    if factory is not None:
        return factory.mktemp(name)
    return Path(tempfile.mkdtemp(prefix=f"chronominer-{name}-"))


def _set_session_env(name: str, value: str | None) -> None:
    _SESSION_ENV_BACKUP.setdefault(name, os.environ.get(name))
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _tiktoken_sources() -> list[Path]:
    """Return the existing tiktoken cache directories of the real profile."""
    candidates = [
        os.environ.get("TIKTOKEN_CACHE_DIR"),
        os.environ.get("DATA_GYM_CACHE_DIR"),
        os.path.join(tempfile.gettempdir(), "data-gym-cache"),
    ]
    return [Path(c) for c in candidates if c and Path(c).is_dir()]


def _seed_tiktoken_cache(config: pytest.Config) -> None:
    """Copy the cached encodings into a session dir; never download them."""
    cache = _session_dir(config, "tiktoken-cache")
    for source in _tiktoken_sources():
        for entry in source.iterdir():
            target = cache / entry.name
            if entry.is_file() and not target.exists():
                shutil.copyfile(entry, target)
    _set_session_env("TIKTOKEN_CACHE_DIR", str(cache))
    _set_session_env("DATA_GYM_CACHE_DIR", None)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Drop key and heartbeat variables, redirect the homes, seed tiktoken."""
    for name in [name for name in os.environ if is_secret_env_var(name)]:
        _set_session_env(name, None)
    _set_session_env("CHRONO_HEARTBEAT_FILE", None)
    _seed_tiktoken_cache(config)
    session_home = _session_dir(config, "session-home")
    env = home_env(session_home)
    xdg_names = {*XDG_VARS, *(n for n in os.environ if n.startswith("XDG_"))}
    env.update(dict.fromkeys(xdg_names, str(session_home)))
    for name, value in env.items():
        _set_session_env(name, value)
    _install_write_guard()


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore the environment changed in ``pytest_configure``."""
    for name, value in _SESSION_ENV_BACKUP.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _SESSION_ENV_BACKUP.clear()


_WRITE_GUARD = WriteGuard()
_WRITE_GUARD_INSTALLED = False


def _install_write_guard() -> None:
    global _WRITE_GUARD_INSTALLED
    if not _WRITE_GUARD_INSTALLED:
        sys.addaudithook(_WRITE_GUARD.audit)
        _WRITE_GUARD_INSTALLED = True


def launches_tool(args: Any, executable: Any = None) -> bool:
    """Return True when a Popen argv would start a ChronoMiner entry point."""
    return testing.launches_entry(args, executable, LAUNCH_ENTRY)


def _program_name(program: Any) -> str:
    """Return the lower-case file name of *program* without ``.exe``."""
    name = Path(os.fsdecode(program)).name.lower()
    return name.removesuffix(".exe")


def launches_clipboard(args: Any, executable: Any = None) -> bool:
    """Return True when a Popen argv would start a clipboard program."""
    if isinstance(args, str | bytes | os.PathLike):
        words = os.fsdecode(args).split()
        first: Any = words[0].strip("\"'") if words else ""
    else:
        first = next(iter(args), "")
    programs = [first] if executable is None else [first, executable]
    return any(_program_name(program) in CLIPBOARD_PROGRAMS for program in programs)


def _guard_network_and_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo
    real_popen_init = subprocess.Popen.__init__

    def getaddrinfo(host: Any, port: Any, *a: Any, **kw: Any) -> Any:
        # asyncio's Windows proactor loop connects through ConnectEx, which
        # bypasses socket.connect; every connection still resolves its host here.
        name = host.decode() if isinstance(host, bytes) else host
        if name not in (None, "localhost") and not is_loopback_address((name, 0)):
            raise NetworkBlockedError(f"Network access blocked in tests: {host}")
        return real_getaddrinfo(host, port, *a, **kw)

    def connect(sock: socket.socket, address: Any) -> None:
        if not is_loopback_address(address):
            raise NetworkBlockedError(f"Network access blocked in tests: {address}")
        real_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        if not is_loopback_address(address):
            raise NetworkBlockedError(f"Network access blocked in tests: {address}")
        return real_connect_ex(sock, address)

    def popen_init(self: subprocess.Popen[Any], args: Any, *a: Any, **kw: Any) -> None:
        executable = kw.get("executable", a[1] if len(a) > 1 else None)
        if launches_tool(args, executable):
            raise ToolLaunchRefusedError(
                f"Tests must not launch ChronoMiner as a subprocess: {args!r}"
            )
        if launches_clipboard(args, executable):
            raise ClipboardLaunchRefusedError(
                f"Tests must not start a clipboard program: {args!r}"
            )
        real_popen_init(self, args, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(subprocess.Popen, "__init__", popen_init)


_LOGGER_NAMES = ("", "chronominer")

LoggerState = dict[str, tuple[list[logging.Handler], int]]


def _all_loggers() -> dict[str, logging.Logger]:
    loggers = {
        name: logger
        for name, logger in logging.Logger.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }
    for name in _LOGGER_NAMES:
        loggers[name] = logging.getLogger(name)
    return loggers


def _logger_state() -> LoggerState:
    return {
        name: (list(logger.handlers), logger.level)
        for name, logger in _all_loggers().items()
    }


def _restore_loggers(state: LoggerState) -> None:
    """Close and drop the handlers a test added; reset the base levels.

    Pytest's own capture handlers are swapped per test phase and left alone.
    An open log file would block the removal of the temp dir on Windows.
    """
    for name, logger in _all_loggers().items():
        handlers, level = state.get(name, ([], logger.level))
        for handler in list(logger.handlers):
            if handler in handlers or type(handler).__module__.startswith("_pytest"):
                continue
            logger.removeHandler(handler)
            handler.close()
        if name in _LOGGER_NAMES:
            logger.setLevel(level)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT


@pytest.fixture(autouse=True, scope="session")
def _ensure_repo_on_syspath() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tool_area(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the per-test directory for config, logs and state.

    It lies beside ``tmp_path``, not inside it, so a test that lists
    ``tmp_path`` sees only its own files.
    """
    return tmp_path_factory.mktemp("cm")


@pytest.fixture(autouse=True)
def _hermetic(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    tool_area: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[WriteGuard]:
    """Redirect per-user paths, settings and logs."""
    import chronominer.settings as settings_module

    home = tmp_path / "home"
    temp = tmp_path / "tmp"
    for directory in (home, temp):
        directory.mkdir()
    for name, value in home_env(home).items():
        monkeypatch.setenv(name, value)
    for name in XDG_VARS:
        monkeypatch.setenv(name, str(home))
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(temp))
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    monkeypatch.setattr(
        settings_module,
        "DEFAULT_SETTINGS_PATH",
        tmp_path / "config" / "settings.yaml",
    )
    monkeypatch.setattr(settings_module, "DEFAULT_LOGS_DIR", tmp_path / "logs")

    _guard_network_and_launches(monkeypatch)

    logger_state = _logger_state()

    _WRITE_GUARD.start([tmp_path_factory.getbasetemp()])
    try:
        yield _WRITE_GUARD
    finally:
        leaked = _WRITE_GUARD.stop()
        _restore_loggers(logger_state)
    if leaked:
        listing = "\n  ".join(leaked)
        pytest.fail(
            f"Test wrote outside the pytest temp directory:\n  {listing}",
            pytrace=False,
        )


@pytest.fixture
def write_guard(_hermetic: WriteGuard) -> WriteGuard:
    """Return the write guard so a test can consume expected violations."""
    return _hermetic


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path]:
    """Return a directory under ``tmp_path`` for test outputs."""
    out_dir = tmp_path / "temp_dir"
    out_dir.mkdir()
    yield out_dir


@pytest.fixture
def temp_input_dir(temp_dir: Path) -> Path:
    """Create a temporary input directory structure."""
    input_dir = temp_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir


@pytest.fixture
def temp_output_dir(temp_dir: Path) -> Path:
    """Create a temporary output directory structure."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_image_folder(temp_input_dir: Path) -> Path:
    """Create a sample image folder with placeholder images."""
    img_folder = temp_input_dir / "test_images"
    img_folder.mkdir(parents=True, exist_ok=True)

    # Placeholder image files: empty files with image extensions.
    for i in range(3):
        (img_folder / f"page_{i + 1:03d}.png").write_bytes(b"")

    return img_folder


@pytest.fixture
def sample_pdf_file(temp_input_dir: Path) -> Path:
    """Create a placeholder PDF file for testing."""
    pdf_path = temp_input_dir / "test_document.pdf"
    # A PDF header only: not a valid PDF, but enough for path tests.
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return pdf_path


@pytest.fixture
def sample_text_file(temp_input_dir: Path) -> Path:
    """Create a sample text input for testing."""
    txt_path = temp_input_dir / "sample_wage_book.txt"
    txt_path.write_text(
        "Wage book of the town mason, 1791\n\n"
        "Johann Meier, journeyman, 12 days at 30 kr., 6 fl.\n"
        "Hans Keller, apprentice, 12 days at 15 kr., 3 fl.\n\n"
        "Paid on Saturday, 14 May.\n",
        encoding="utf-8",
    )
    return txt_path


@pytest.fixture
def mock_env_no_api_keys() -> Generator[None]:
    """Mock environment with no API keys set."""
    env_copy = {k: v for k, v in os.environ.items() if not is_secret_env_var(k)}
    with patch.dict(os.environ, env_copy, clear=True):
        yield


@pytest.fixture
def mock_env_with_openai_key() -> Generator[None]:
    """Mock environment with OpenAI API key set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"}):
        yield
