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
EXPECTED_SHA256 = "00679a7f517cbdbe5a41050deaf787b2fe1e4eeda77175e4a5c830ae3aca3fd8"

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
        yesterday = (
            datetime.strptime(sl._today(), "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")
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

    def test_non_numeric_own_field_does_not_crash(self, tmp_path: Path) -> None:
        # A syntactically valid ledger with a non-numeric own-tool value must
        # degrade (return None) rather than raise ValueError up the never-crash
        # call path.
        poisoned = {
            "schema_version": sl.LEDGER_SCHEMA_VERSION,
            "date": sl._today(),
            "tools": {"chronominer": "12a"},
            "last_updated": "irrelevant",
        }
        (tmp_path / sl.LEDGER_FILENAME).write_text(
            json.dumps(poisoned), encoding="utf-8"
        )
        ledger = _make(tmp_path)
        # Merge coerces the bad field to 0 and proceeds; no exception escapes.
        assert ledger.sync(10) == 10

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


class TestPoolDerivation:
    def test_exact_matches(self) -> None:
        assert sl.derive_pool("openai", "gpt-5.6-sol") == sl.POOL_LARGE
        assert sl.derive_pool("openai", "gpt-5.6-terra") == sl.POOL_SMALL
        assert sl.derive_pool("openai", "gpt-5-mini") == sl.POOL_SMALL
        assert sl.derive_pool("openai", "gpt-5") == sl.POOL_LARGE
        assert sl.derive_pool("openai", "o1") == sl.POOL_LARGE
        assert sl.derive_pool("openai", "o1-mini") == sl.POOL_SMALL

    def test_longest_prefix_wins(self) -> None:
        # "gpt-4.1-mini" must not be claimed by the shorter "gpt-4.1".
        assert sl.derive_pool("openai", "gpt-4.1-mini") == sl.POOL_SMALL
        assert sl.derive_pool("openai", "gpt-4.1") == sl.POOL_LARGE
        assert sl.derive_pool("openai", "gpt-5.1-codex-mini") == sl.POOL_SMALL

    def test_dated_snapshot_matches_at_separator(self) -> None:
        assert sl.derive_pool("openai", "gpt-4o-2024-08-06") == sl.POOL_LARGE
        assert sl.derive_pool("openai", "gpt-4o-mini-2024-07-18") == sl.POOL_SMALL

    def test_separator_boundary_prevents_partial_match(self) -> None:
        # "gpt-5.55" must not match "gpt-5.5"; it falls through to "gpt-5"
        # at the "." boundary and lands in the large pool.
        assert sl.derive_pool("openai", "gpt-5.55") == sl.POOL_LARGE

    def test_non_openai_and_unknown_are_none(self) -> None:
        assert sl.derive_pool("anthropic", "claude-sonnet-5") is None
        assert sl.derive_pool("custom", "gpt-4o") is None
        assert sl.derive_pool("openai", "some-local-model") is None
        assert sl.derive_pool("openai", None) is None
        assert sl.derive_pool(None, "gpt-4o") is None

    def test_router_style_prefix_stripped(self) -> None:
        assert sl.derive_pool("openai", "openai/gpt-4o") == sl.POOL_LARGE


class TestConfiguredPools:
    """Config-defined pools override the built-ins per provider."""

    def test_custom_provider_pools(self) -> None:
        pools = sl.compile_pools(
            {"myhost": {"standard": ["my-model", "my-model-large"]}}
        )
        assert sl.derive_pool("myhost", "my-model-v2", pools) == "standard"
        assert sl.derive_pool("myhost", "my-model-large", pools) == "standard"
        assert sl.derive_pool("myhost", "other", pools) is None

    def test_configured_provider_replaces_builtin_table(self) -> None:
        # A configured openai table takes precedence over the built-ins.
        pools = sl.compile_pools({"openai": {"tiny": ["gpt-4o-mini"]}})
        assert sl.derive_pool("openai", "gpt-4o-mini", pools) == "tiny"
        # gpt-4o is not in the configured table; no built-in fallback for a
        # provider the config covers.
        assert sl.derive_pool("openai", "gpt-4o", pools) is None

    def test_uncovered_provider_falls_back_to_builtin(self) -> None:
        pools = sl.compile_pools({"myhost": {"standard": ["my-model"]}})
        assert sl.derive_pool("openai", "gpt-4o", pools) == sl.POOL_LARGE

    def test_longest_prefix_wins_in_configured_pools(self) -> None:
        pools = sl.compile_pools({"p": {"a": ["base"], "b": ["base-pro"]}})
        assert sl.derive_pool("p", "base-pro-2", pools) == "b"
        assert sl.derive_pool("p", "base-2", pools) == "a"

    def test_compile_pools_drops_malformed_entries(self) -> None:
        pools = sl.compile_pools(
            {
                "": {"a": ["x"]},
                "p": {"": ["x"], "a": "not-a-list", "b": [1, "", "ok"]},
                "q": "not-a-mapping",
            }
        )
        assert pools == {"p": (("ok", "b"),)}

    def test_compile_pools_normalizes_case(self) -> None:
        pools = sl.compile_pools({"OpenAI": {"a": ["GPT-X"]}})
        assert sl.derive_pool("openai", "gpt-x-1", pools) == "a"


def _bucket(
    provider: str = "openai",
    key_env: str = "OPENAI_API_KEY",
    pool: str | None = "small",
) -> object:
    return sl.BucketKey(provider, key_env, pool)


class TestUsageRows:
    def test_sync_usage_creates_rows_and_tool_total(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        snap = ledger.sync_usage({_bucket(): 1000})
        assert snap is not None
        assert snap.combined == 1000
        assert snap.own_total == 1000
        assert snap.buckets == {_bucket(): 1000}
        data = json.loads((tmp_path / sl.LEDGER_FILENAME).read_text(encoding="utf-8"))
        assert data["schema_version"] == 2
        assert data["tools"]["chronominer"] == 1000
        assert data["usage"] == [
            {
                "tool": "chronominer",
                "provider": "openai",
                "key_env": "OPENAI_API_KEY",
                "pool": "small",
                "tokens": 1000,
            }
        ]

    def test_tool_total_equals_row_sum(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.sync_usage({_bucket(): 300})
        ledger.sync_usage({_bucket(key_env="OPENAI_API_KEY_2"): 200})
        ledger.sync(50)  # legacy un-stamped delta
        data = json.loads((tmp_path / sl.LEDGER_FILENAME).read_text(encoding="utf-8"))
        row_sum = sum(r["tokens"] for r in data["usage"])
        assert data["tools"]["chronominer"] == row_sum == 550

    def test_legacy_sync_lands_in_unattributed_row(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.sync(120)
        snap = ledger.read_usage()
        assert snap is not None
        assert snap.own_buckets == {sl.UNATTRIBUTED_BUCKET: 120}
        # Unattributed rows never feed per-key enforcement aggregates.
        assert snap.buckets == {}
        assert snap.combined == 120

    def test_buckets_aggregate_across_tools(self, tmp_path: Path) -> None:
        miner = _make(tmp_path, "chronominer")
        scriber = _make(tmp_path, "chronotranscriber")
        miner.sync_usage({_bucket(): 300})
        snap = scriber.sync_usage({_bucket(): 200})
        assert snap is not None
        assert snap.buckets == {_bucket(): 500}
        assert snap.own_buckets == {_bucket(): 200}
        assert snap.combined == 500

    def test_negative_and_zero_deltas_ignored(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        snap = ledger.sync_usage({_bucket(): -50, _bucket(pool="large"): 0})
        assert snap is not None
        assert snap.combined == 0
        assert snap.buckets == {}

    def test_seed_usage_max_semantics_never_double_counts(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.sync_usage({_bucket(): 400})
        snap = ledger.seed_usage(500, {_bucket(): 450})
        assert snap is not None
        assert snap.own_buckets[_bucket()] == 450
        # Residual of the total lands in unattributed via reconciliation.
        assert snap.own_buckets[sl.UNATTRIBUTED_BUCKET] == 50
        assert snap.own_total == 500
        # Repeat: nothing changes.
        snap2 = ledger.seed_usage(500, {_bucket(): 450})
        assert snap2 is not None
        assert snap2.own_total == 500
        assert snap2.own_buckets[_bucket()] == 450

    def test_read_usage_stale_date_is_empty(self, tmp_path: Path) -> None:
        stale = {
            "schema_version": 2,
            "date": "2000-01-01",
            "tools": {"chronominer": 777},
            "usage": [],
            "last_updated": "irrelevant",
        }
        (tmp_path / sl.LEDGER_FILENAME).write_text(json.dumps(stale), encoding="utf-8")
        ledger = _make(tmp_path)
        snap = ledger.read_usage()
        assert snap is not None
        assert snap.combined == 0 and snap.buckets == {}

    def test_read_usage_missing_file_is_none(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        assert ledger.read_usage() is None

    def test_malformed_rows_dropped_without_crash(self, tmp_path: Path) -> None:
        poisoned = {
            "schema_version": 2,
            "date": sl._today(),
            "tools": {"chronominer": 100},
            "usage": [
                "not a dict",
                {"tool": "", "provider": "openai", "key_env": "K", "tokens": 5},
                {
                    "tool": "chronominer",
                    "provider": "openai",
                    "key_env": "K",
                    "pool": 7,
                    "tokens": 5,
                },
                {
                    "tool": "chronominer",
                    "provider": "openai",
                    "key_env": "OPENAI_API_KEY",
                    "pool": "small",
                    "tokens": "12a",
                },
            ],
            "last_updated": "irrelevant",
        }
        (tmp_path / sl.LEDGER_FILENAME).write_text(
            json.dumps(poisoned), encoding="utf-8"
        )
        ledger = _make(tmp_path)
        snap = ledger.sync_usage({_bucket(): 10})
        assert snap is not None
        # The malformed rows vanish; the surviving coerced-to-0 row plus the
        # new delta plus reconciliation keep total == row sum.
        assert snap.own_total == 110
        assert snap.own_buckets[_bucket()] == 10
        assert snap.own_buckets[sl.UNATTRIBUTED_BUCKET] == 100


class TestV1AdoptionAndMixedWriters:
    def test_v1_ledger_adopted_via_unattributed(self, tmp_path: Path) -> None:
        v1 = {
            "schema_version": 1,
            "date": sl._today(),
            "tools": {"chronominer": 19_420_572, "autoexcerpter": 1000},
            "last_updated": "irrelevant",
        }
        (tmp_path / sl.LEDGER_FILENAME).write_text(json.dumps(v1), encoding="utf-8")
        ledger = _make(tmp_path)
        snap = ledger.sync_usage({_bucket(): 10})
        assert snap is not None
        # The day's combined count is never lost.
        assert snap.combined == 19_420_572 + 1000 + 10
        assert snap.own_total == 19_420_582
        assert snap.own_buckets[sl.UNATTRIBUTED_BUCKET] == 19_420_572
        assert snap.own_buckets[_bucket()] == 10
        # The other tool's total is untouched (no rows invented for it).
        breakdown = ledger.read_breakdown()
        assert breakdown is not None and breakdown["autoexcerpter"] == 1000

    def test_v1_writer_drift_reconciled_on_next_v2_write(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.sync_usage({_bucket(): 100})
        # Simulate a v1 writer: bump the tool total without touching usage.
        path = tmp_path / sl.LEDGER_FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        data["tools"]["chronominer"] += 40
        data["schema_version"] = 1  # v1 also stamps the version back
        path.write_text(json.dumps(data), encoding="utf-8")
        snap = ledger.sync_usage({_bucket(): 10})
        assert snap is not None
        assert snap.own_total == 150
        assert snap.own_buckets[_bucket()] == 110
        assert snap.own_buckets[sl.UNATTRIBUTED_BUCKET] == 40

    def test_hand_edited_rows_exceeding_total_raise_total(self, tmp_path: Path) -> None:
        ledger = _make(tmp_path)
        ledger.sync_usage({_bucket(): 100})
        path = tmp_path / sl.LEDGER_FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        data["tools"]["chronominer"] = 30  # inconsistent hand edit
        path.write_text(json.dumps(data), encoding="utf-8")
        snap = ledger.sync_usage({_bucket(): 0})
        assert snap is not None
        # Usage is never silently discarded: total is raised to the row sum.
        assert snap.own_total == 100


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
        assert sl.LEDGER_MODULE_VERSION == "2.1.0"


class TestResetBoundary:
    """The budget day rolls over at 00:01 UTC, not exact UTC midnight."""

    def test_day_key_before_buffer_is_previous_day(self, monkeypatch: object) -> None:
        from datetime import UTC
        from datetime import datetime as real_datetime

        class _FrozenDateTime(real_datetime):
            @classmethod
            def now(cls, tz: object = None) -> real_datetime:
                return real_datetime(2026, 7, 5, 0, 0, 30, tzinfo=UTC)

        monkeypatch.setattr(sl, "datetime", _FrozenDateTime)
        assert sl._today() == "2026-07-04"

    def test_day_key_after_buffer_is_new_day(self, monkeypatch: object) -> None:
        from datetime import UTC
        from datetime import datetime as real_datetime

        class _FrozenDateTime(real_datetime):
            @classmethod
            def now(cls, tz: object = None) -> real_datetime:
                return real_datetime(2026, 7, 5, 0, 1, 30, tzinfo=UTC)

        monkeypatch.setattr(sl, "datetime", _FrozenDateTime)
        assert sl._today() == "2026-07-05"
