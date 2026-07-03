"""Tests for the vendored cross-tool shared token ledger.

This test file is vendored byte-identically alongside
``shared_ledger.py`` into every participating repo. It locates the
module dynamically (repo layouts differ) and pins its content hash so
that any divergent local edit fails the drift test in that repo.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

# Content hash of shared_ledger.py with newlines normalized to LF.
# Update ONLY when intentionally releasing a new ledger module version,
# then re-copy module + tests to all sibling repos.
EXPECTED_SHA256 = "1f6c9a5703d463750d05e47ed25b9a96fda24d88e948fe674fcffb697d541bc8"

_SKIP_DIRS = {".venv", ".git", "scratch", "backup", "node_modules", ".mypy_cache"}


def _repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("pyproject.toml not found above test file")


def _find_module_file() -> Path:
    root = _repo_root()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if "shared_ledger.py" in filenames:
            return Path(dirpath) / "shared_ledger.py"
    raise RuntimeError("shared_ledger.py not found in repo")


MODULE_FILE = _find_module_file()


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "shared_ledger_under_test", MODULE_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = _load_module()


def _make(tmp_path: Path, tool: str = "chronominer", **kwargs: object) -> object:
    return sl.SharedTokenLedger(tool, ledger_dir=tmp_path, **kwargs)


class TestBasicMerging:
    def test_sync_creates_ledger_and_returns_combined(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        assert ledger.sync(1000) == 1000
        data = json.loads((tmp_path / sl.LEDGER_FILENAME).read_text(encoding="utf-8"))
        assert data["schema_version"] == sl.LEDGER_SCHEMA_VERSION
        assert data["tools"]["chronominer"] == 1000

    def test_sync_accumulates_deltas(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.sync(100)
        ledger.sync(250)
        assert ledger.sync(50) == 400

    def test_two_tools_sum_combined(self, tmp_path: Path) -> None:
        miner = _make(tmp_path, "chronominer")
        scriber = _make(tmp_path, "chronotranscriber")
        miner.sync(300)
        assert scriber.sync(200) == 500
        assert miner.read_breakdown() == {
            "chronominer": 300,
            "chronotranscriber": 200,
        }

    def test_negative_delta_clamped_to_zero(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.sync(100)
        assert ledger.sync(-50) == 100

    def test_tool_name_required(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            sl.SharedTokenLedger("  ", ledger_dir=tmp_path)


class TestSeeding:
    def test_seed_adopts_higher_legacy_count(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.sync(100)
        assert ledger.seed(150) == 150

    def test_seed_never_lowers_existing_field(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.sync(200)
        assert ledger.seed(80) == 200

    def test_repeated_seeding_does_not_double_count(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.seed(500)
        assert ledger.seed(500) == 500


class TestRolloverAndRecovery:
    def test_day_rollover_resets_all_fields(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        stale = {
            "schema_version": sl.LEDGER_SCHEMA_VERSION,
            "date": yesterday,
            "tools": {"chronominer": 9999, "autoexcerpter": 12345},
            "last_updated": "irrelevant",
        }
        (tmp_path / sl.LEDGER_FILENAME).write_text(json.dumps(stale), encoding="utf-8")
        assert ledger.sync(10) == 10

    def test_corrupt_json_recovers_fresh(self, tmp_path: Path) -> None:
        (tmp_path / sl.LEDGER_FILENAME).write_text("{not json", encoding="utf-8")
        ledger = _make(tmp_path)
        assert ledger.sync(42) == 42

    def test_read_combined_stale_date_is_zero(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        stale = {
            "schema_version": sl.LEDGER_SCHEMA_VERSION,
            "date": "2000-01-01",
            "tools": {"chronominer": 777},
            "last_updated": "irrelevant",
        }
        (tmp_path / sl.LEDGER_FILENAME).write_text(json.dumps(stale), encoding="utf-8")
        assert ledger.read_combined() == 0

    def test_read_combined_missing_file_is_none(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        assert ledger.read_combined() is None


class TestDegradation:
    def test_lock_timeout_degrades_to_none(self, tmp_path: Path) -> None:
        holder = _make(tmp_path)
        blocked = _make(tmp_path, lock_timeout=0.2)
        handle = holder._acquire_lock()
        assert handle is not None
        try:
            start = time.monotonic()
            assert blocked.sync(10) is None
            assert time.monotonic() - start < 3.0
        finally:
            sl._unlock_handle(handle)
            handle.close()
        # After release, the blocked ledger works again.
        assert blocked.sync(10) == 10

    def test_degradation_warns_once(self, tmp_path: Path, caplog: object) -> None:
        holder = _make(tmp_path)
        blocked = _make(tmp_path, lock_timeout=0.05)
        handle = holder._acquire_lock()
        assert handle is not None
        try:
            blocked.sync(1)
            blocked.sync(1)
        finally:
            sl._unlock_handle(handle)
            handle.close()
        assert blocked._warned_degraded is True


class TestAtomicWrites:
    def test_no_temp_files_left_behind(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        for _ in range(5):
            ledger.sync(10)
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_temp_name_contains_pid(self, tmp_path: Path, monkeypatch: object) -> None:
        ledger = _make(tmp_path)
        captured: list[str] = []
        original_replace = Path.replace

        def spy(self: Path, target: object) -> object:
            if self.suffix == ".tmp":
                captured.append(self.name)
            return original_replace(self, target)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "replace", spy)
        ledger.sync(5)
        assert captured and str(os.getpid()) in captured[0]


class TestThreadSafety:
    def test_concurrent_threads_lose_no_updates(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(25):
                    assert ledger.sync(10) is not None
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not errors
        assert ledger.read_breakdown() == {"chronominer": 8 * 25 * 10}


_SUBPROCESS_SCRIPT = """
import importlib.util
import random
import sys
import time

module_file, ledger_dir, tool, rounds, delta = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
)
spec = importlib.util.spec_from_file_location("sl", module_file)
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)
ledger = sl.SharedTokenLedger(tool, ledger_dir=ledger_dir, lock_timeout=30.0)
for _ in range(rounds):
    result = ledger.sync(delta)
    assert result is not None, "sync degraded under contention"
    time.sleep(random.uniform(0, 0.002))
"""


@pytest.mark.slow
class TestMultiprocessStress:
    def test_no_lost_updates_across_processes(self, tmp_path: Path) -> None:
        rounds, delta = 40, 7
        jobs = [
            ("alpha",),
            ("alpha",),
            ("beta",),
            ("beta",),
        ]
        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _SUBPROCESS_SCRIPT,
                    str(MODULE_FILE),
                    str(tmp_path),
                    tool,
                    str(rounds),
                    str(delta),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for (tool,) in jobs
        ]
        for proc in procs:
            _out, err = proc.communicate(timeout=120)
            assert proc.returncode == 0, err.decode(errors="replace")
        ledger = sl.SharedTokenLedger("alpha", ledger_dir=tmp_path)
        breakdown = ledger.read_breakdown()
        expected_per_tool = 2 * rounds * delta
        assert breakdown == {
            "alpha": expected_per_tool,
            "beta": expected_per_tool,
        }
        assert ledger.read_combined() == 2 * expected_per_tool


class TestModuleDrift:
    def test_module_checksum_pinned(self) -> None:
        raw = MODULE_FILE.read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(raw).hexdigest()
        assert actual == EXPECTED_SHA256, (
            "shared_ledger.py has drifted from the pinned vendored version. "
            "If the change is intentional, bump LEDGER_MODULE_VERSION, update "
            "EXPECTED_SHA256 here, and re-copy module + tests to all sibling "
            f"repos. Actual hash: {actual}"
        )

    def test_module_version_matches(self) -> None:
        assert sl.LEDGER_MODULE_VERSION == "1.0.0"
