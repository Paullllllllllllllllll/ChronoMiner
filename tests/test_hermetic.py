"""Tests for the hermetic harness in tests/conftest.py."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import (
    CLIPBOARD_PROGRAMS,
    PROJECT_ROOT,
    ClipboardLaunchRefusedError,
    NetworkBlockedError,
    ToolLaunchRefusedError,
    WriteGuard,
    is_loopback_address,
    is_secret_env_var,
    launches_clipboard,
    launches_tool,
)


def _under(path: Path | str, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


class TestPathRedirection:
    def test_home_is_under_tmp_path(self, tmp_path: Path) -> None:
        assert _under(Path.home(), tmp_path)
        assert _under(os.path.expanduser("~"), tmp_path)

    def test_tempdir_is_under_tmp_path(self, tmp_path: Path) -> None:
        assert _under(tempfile.gettempdir(), tmp_path)

    def test_log_file_is_under_tmp_path(self, tmp_path: Path) -> None:
        from chronominer.settings import Settings

        assert _under(Settings().log_file(), tmp_path / "logs")


class TestSettingsRedirection:
    def test_default_settings_path_is_under_tmp_path(self, tmp_path: Path) -> None:
        from chronominer import settings

        assert _under(settings.DEFAULT_SETTINGS_PATH, tmp_path)

    def test_a_real_settings_file_is_never_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronominer import settings

        opened: list[Path] = []
        real_read = Path.read_text

        def spy(self: Path, *args: Any, **kwargs: Any) -> Any:
            opened.append(self)
            return real_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", spy)
        loaded = settings.load()

        repository_file = PROJECT_ROOT / "config" / "settings.yaml"
        assert repository_file not in opened
        assert loaded.source in (None, settings.EXAMPLE_SETTINGS_PATH)


class TestNetworkGuard:
    def test_non_loopback_connect_raises(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with pytest.raises(NetworkBlockedError):
                sock.connect(("192.0.2.1", 443))
            with pytest.raises(NetworkBlockedError):
                sock.connect_ex(("192.0.2.1", 443))

    def test_name_resolution_is_blocked(self) -> None:
        with pytest.raises(NetworkBlockedError):
            socket.getaddrinfo("api.openai.com", 443)

    def test_async_http_client_is_blocked(self) -> None:
        """The Windows proactor loop bypasses socket.connect; resolution does not."""
        import asyncio

        import httpx

        async def fetch() -> None:
            async with httpx.AsyncClient() as client:
                await client.get("https://api.openai.com/v1/models")

        with pytest.raises(NetworkBlockedError):
            asyncio.run(fetch())

    def test_loopback_socketpair_works(self) -> None:
        left, right = socket.socketpair()
        with left, right:
            left.sendall(b"ping")
            assert right.recv(4) == b"ping"

    def test_loopback_classification(self) -> None:
        assert is_loopback_address(("127.0.0.1", 80))
        assert is_loopback_address(("::1", 80, 0, 0))
        assert is_loopback_address(("localhost", 80))
        assert not is_loopback_address(("api.example.org", 443))
        assert not is_loopback_address(("10.0.0.1", 443))


class TestWriteGuard:
    def test_write_to_repo_root_is_recorded(self, write_guard: WriteGuard) -> None:
        probe = PROJECT_ROOT / f".hermetic-probe-{uuid.uuid4().hex}.txt"
        try:
            probe.write_text("probe", encoding="utf-8")
        finally:
            probe.unlink(missing_ok=True)

        recorded = write_guard.consume(probe)

        expected = os.path.normcase(str(probe))
        assert expected in recorded
        assert write_guard.violations == []

    def test_write_under_tmp_path_is_not_recorded(
        self, tmp_path: Path, write_guard: WriteGuard
    ) -> None:
        target = tmp_path / "sub" / "out.txt"
        target.parent.mkdir()
        target.write_text("ok", encoding="utf-8")
        target.rename(tmp_path / "moved.txt")

        assert write_guard.violations == []


class TestEnvironment:
    def test_no_key_variables(self) -> None:
        assert [name for name in os.environ if is_secret_env_var(name)] == []

    def test_key_variable_classification(self) -> None:
        assert is_secret_env_var("OPENAI_API_KEY")
        assert is_secret_env_var("custom_api_key_var")
        assert not is_secret_env_var("PYTHONIOENCODING")

    @pytest.mark.usefixtures("mock_env_with_openai_key")
    def test_stubbed_key_is_a_test_value(self) -> None:
        assert "test" in os.environ["OPENAI_API_KEY"]


class TestLaunchGuard:
    def test_entry_point_launch_is_refused(self) -> None:
        script = PROJECT_ROOT / "chronominer" / "__main__.py"
        with pytest.raises(ToolLaunchRefusedError):
            subprocess.run([sys.executable, str(script), "--help"], check=False)

    def test_the_script_is_refused(self) -> None:
        windows_path = "chronominer\\__main__.py"
        assert launches_tool([sys.executable, "chronominer/__main__.py"])
        assert launches_tool([sys.executable, windows_path, "extract"])
        assert launches_tool([sys.executable, "__main__.py"])
        assert launches_tool("python __main__.py")

    def test_the_module_is_refused(self) -> None:
        assert launches_tool([sys.executable, "-m", "chronominer"])
        assert launches_tool([sys.executable, "-mchronominer"])
        assert launches_tool("python -m chronominer --help")

    def test_program_and_runpy_forms_are_refused(self) -> None:
        assert launches_tool(["chronominer", "extract"])
        assert launches_tool(["chronominer.exe", "--help"])
        assert launches_tool(["uv", "run", "chronominer"], "chronominer")
        assert launches_tool(
            [
                sys.executable,
                "-c",
                "import runpy; runpy.run_path('chronominer/__main__.py')",
            ]
        )

    def test_unrelated_commands_are_allowed(self) -> None:
        assert not launches_tool([sys.executable, "-c", "print(1)"])
        assert not launches_tool(
            [sys.executable, str(PROJECT_ROOT / "chronominer" / "llm" / "x.py")]
        )
        assert not launches_tool(["ffmpeg", "-i", "in.wav", "out.mp3"])


class TestClipboardGuard:
    @pytest.mark.parametrize("platform", ["win32", "darwin"])
    def test_copy_to_clipboard_returns_false_and_starts_nothing(
        self, platform: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronominer.common.wizard import copy_to_clipboard

        started: list[Any] = []

        def execute_child(self: subprocess.Popen[Any], args: Any, *a: Any) -> None:
            started.append(args)
            raise AssertionError(f"a program was started: {args!r}")

        monkeypatch.setattr(subprocess.Popen, "_execute_child", execute_child)

        assert copy_to_clipboard("x", platform=platform) is False
        assert started == []

    def test_clipboard_launch_is_refused(self) -> None:
        with pytest.raises(ClipboardLaunchRefusedError):
            subprocess.run(["clip"], input=b"x", check=False)

    def test_clipboard_programs_are_recognized(self) -> None:
        for program in CLIPBOARD_PROGRAMS:
            assert launches_clipboard([program])
        assert launches_clipboard(["C:\\Windows\\System32\\clip.exe"])
        assert launches_clipboard("xclip -selection clipboard")
        assert launches_clipboard(["anything"], "/usr/bin/wl-copy")
        assert not launches_clipboard(["ffmpeg", "-i", "clip.wav"])
        assert not launches_clipboard([sys.executable, "-c", "print(1)"])
