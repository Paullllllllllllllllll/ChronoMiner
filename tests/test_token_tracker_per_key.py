"""Tests for per-(provider key, pool) token accounting and enforcement.

Exercises DailyTokenTracker's schema-v2 bucket stamping and the two-gate
enforcement (per-key pool cap primary, combined cap secondary), plus the
scope switch, private-state bucket persistence, degraded per-bucket
preservation, and config parsing. The shared-ledger contract itself is
covered by tests/test_shared_ledger.py (do not touch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.infra.shared_ledger import (
    DEFAULT_POOL_CAPS,
    UNATTRIBUTED_BUCKET,
    BucketKey,
)
from modules.infra.token_tracker import (
    DailyTokenTracker,
    _parse_pool_caps_config,
)

_OPENAI = "openai"
_KEY1 = "OPENAI_API_KEY"
_KEY2 = "OPENAI_API_KEY_2"
_SMALL_MODEL = "gpt-4o-mini"  # derives the OpenAI "small" pool
_LARGE_MODEL = "gpt-4o"  # derives the OpenAI "large" pool


def _tracker(tmp_path: Path, **kw) -> DailyTokenTracker:
    defaults = dict(
        daily_limit=10**9,
        enabled=True,
        state_file=tmp_path / "s.json",
        chunk_estimate_seed=10,
        estimate_smoothing=0.0,
        per_key_pool_caps_enabled=True,
        pool_caps={(_OPENAI, "small"): 100, (_OPENAI, "large"): 50},
    )
    defaults.update(kw)
    return DailyTokenTracker(**defaults)  # type: ignore[arg-type]


class TestBucketStamping:
    @pytest.mark.unit
    def test_add_tokens_stamps_the_serving_bucket(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path)
        t.add_tokens(30, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        bucket = BucketKey(_OPENAI, _KEY1, "small")
        assert t._bucket_used_today[bucket] == 30
        # The plain sum stays the classic total.
        assert t._tokens_used_today == 30

    @pytest.mark.unit
    def test_unstamped_add_lands_in_unattributed(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path)
        t.add_tokens(30)  # no stamp
        assert t._bucket_used_today[UNATTRIBUTED_BUCKET] == 30

    @pytest.mark.unit
    def test_partial_stamp_is_unattributed(self, tmp_path: Path) -> None:
        # Missing key_env -> not stamped, even with a provider/model.
        t = _tracker(tmp_path)
        t.add_tokens(30, provider=_OPENAI, model=_SMALL_MODEL)
        assert t._bucket_used_today[UNATTRIBUTED_BUCKET] == 30


class TestPerKeyPoolGate:
    @pytest.mark.unit
    def test_cap_blocks_one_key_while_the_other_admits(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path)
        # Exhaust key 1's small pool (cap 100).
        t.add_tokens(100, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)

        # Key 1 is blocked...
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
            is None
        )
        # ...but key 2's pool is untouched, so it admits.
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY2, model=_SMALL_MODEL) == 10
        )

    @pytest.mark.unit
    def test_separate_pools_on_the_same_key_are_independent(
        self, tmp_path: Path
    ) -> None:
        t = _tracker(tmp_path)
        t.add_tokens(50, provider=_OPENAI, key_env=_KEY1, model=_LARGE_MODEL)
        # Large pool for key 1 is now full (cap 50); small pool is free.
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_LARGE_MODEL)
            is None
        )
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL) == 10
        )

    @pytest.mark.unit
    def test_reservations_count_against_the_pool_cap(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path)
        assert (
            t.try_reserve(90, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL) == 90
        )
        # 90 reserved of a 100 cap; a 20 no longer fits.
        assert (
            t.try_reserve(20, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
            is None
        )
        # 10 fits exactly.
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL) == 10
        )

    @pytest.mark.unit
    def test_defaults_used_when_no_overrides(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path, pool_caps={})
        stats = t.get_stats()
        assert stats["pool_caps"]["defaults"] == DEFAULT_POOL_CAPS
        assert stats["pool_caps"]["overrides"] == {}
        assert stats["per_key_pool_caps_enabled"] is True
        # Default caps enforce even without configured overrides.
        bucket = BucketKey(_OPENAI, _KEY1, "large")
        assert t._pool_cap_for(bucket) == DEFAULT_POOL_CAPS["large"]

    @pytest.mark.unit
    def test_disabled_per_key_caps_never_block_on_pool(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path, per_key_pool_caps_enabled=False)
        t.add_tokens(1000, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        # No per-key cap; combined limit is huge -> admitted.
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL) == 10
        )


class TestPoolNoneNeverBlocked:
    @pytest.mark.unit
    def test_local_endpoint_never_blocked_under_pooled_scope(
        self, tmp_path: Path
    ) -> None:
        # Combined cap 100, already exceeded, but a pool-None call is exempt.
        t = _tracker(tmp_path, daily_limit=100, scope="pooled")
        t.add_tokens(500, provider="local", key_env="LOCAL_KEY", model="llama-3")
        assert (
            t.try_reserve(10, provider="local", key_env="LOCAL_KEY", model="llama-3")
            == 10
        )
        assert (
            t.is_limit_reached(provider="local", key_env="LOCAL_KEY", model="llama-3")
            is False
        )

    @pytest.mark.unit
    def test_unstamped_call_still_combined_gated_under_pooled(
        self, tmp_path: Path
    ) -> None:
        # The pool-None exemption applies to STAMPED calls; an unstamped call
        # keeps the legacy combined-only semantics.
        t = _tracker(tmp_path, daily_limit=100, scope="pooled")
        t.add_tokens(500)  # unstamped -> combined 500 over 100
        assert t.try_reserve(10) is None
        assert t.is_limit_reached() is True

    @pytest.mark.unit
    def test_scope_all_restores_legacy_blocking(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path, daily_limit=100, scope="all")
        t.add_tokens(500, provider="local", key_env="LOCAL_KEY", model="llama-3")
        # scope=all -> even the pool-None call is blocked by the combined cap.
        assert (
            t.try_reserve(10, provider="local", key_env="LOCAL_KEY", model="llama-3")
            is None
        )
        assert (
            t.is_limit_reached(provider="local", key_env="LOCAL_KEY", model="llama-3")
            is True
        )

    @pytest.mark.unit
    def test_pooled_call_still_hits_combined_gate(self, tmp_path: Path) -> None:
        # A pooled (OpenAI) call under a low combined cap is blocked by the
        # secondary combined gate even though the per-key pool has headroom.
        t = _tracker(tmp_path, daily_limit=100, pool_caps={(_OPENAI, "small"): 10**9})
        t.add_tokens(100, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
            is None
        )


class TestWaitLoopBucketHandling:
    @pytest.mark.unit
    def test_would_block_reflects_last_denied_bucket(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path)
        t.add_tokens(100, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        # Denial records the blocked bucket.
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
            is None
        )
        # The stampless wait predicate honours it (per-key block, not combined).
        assert t.would_block_next_page() is True

    @pytest.mark.unit
    def test_rebind_active_key_unblocks_without_restart(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path)
        t.add_tokens(100, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
            is None
        )
        assert t.would_block_next_page() is True

        # A mid-wait remap to key 2 switches the active bucket, which has
        # headroom, so the wait predicate clears.
        assert t.rebind_active_key_env(_OPENAI, _KEY2) is True
        assert t.would_block_next_page() is False

    @pytest.mark.unit
    def test_describe_pool_block_names_the_key(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path)
        t.add_tokens(100, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        msg = t.describe_pool_block()
        assert msg is not None
        assert _KEY1 in msg
        assert "small" in msg

    @pytest.mark.unit
    def test_describe_pool_block_none_without_pool_block(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path, daily_limit=100)
        t.add_tokens(100)  # combined block, unstamped -> no per-key pool block
        t.try_reserve(10)
        assert t.describe_pool_block() is None


class TestPrivateStatePersistence:
    @pytest.mark.unit
    def test_buckets_persist_across_instances(self, tmp_path: Path) -> None:
        state = tmp_path / "state.json"
        t1 = _tracker(tmp_path, state_file=state)
        t1.add_tokens(40, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        t1.add_tokens(15, provider=_OPENAI, key_env=_KEY2, model=_SMALL_MODEL)
        t1.flush()

        data = json.loads(state.read_text(encoding="utf-8"))
        assert data["tokens_used"] == 55  # legacy int kept for compatibility
        assert data["buckets"]["openai|OPENAI_API_KEY|small"] == 40
        assert data["buckets"]["openai|OPENAI_API_KEY_2|small"] == 15

        t2 = _tracker(tmp_path, state_file=state)
        assert t2._bucket_used_today[BucketKey(_OPENAI, _KEY1, "small")] == 40
        assert t2._bucket_used_today[BucketKey(_OPENAI, _KEY2, "small")] == 15

    @pytest.mark.unit
    def test_legacy_state_without_buckets_adopts_unattributed(
        self, tmp_path: Path
    ) -> None:
        from modules.infra.shared_ledger import _today

        state = tmp_path / "state.json"
        state.write_text(
            json.dumps({"date": _today(), "tokens_used": 320}),
            encoding="utf-8",
        )
        t = _tracker(tmp_path, state_file=state)
        assert t._bucket_used_today == {UNATTRIBUTED_BUCKET: 320}
        assert t._tokens_used_today == 320


class TestDegradedPerBucketPreservation:
    @pytest.mark.unit
    def test_degraded_keeps_per_bucket_deltas_then_lands(self, tmp_path: Path) -> None:
        from tests.test_token_tracker_shared_budget import _FakeLedger

        t = DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=1,
            estimate_smoothing=0.0,
            shared_enabled=True,
            shared_ledger_dir=tmp_path / "ledger",
        )
        fake = _FakeLedger()
        with t._lock:
            t._ledger = fake
            t._seeded = False
            t._combined_total = 0
            t._bucket_totals = {}
            t._unsynced_deltas = {}
            t._ledger_degraded = False
            t._last_ledger_sync_monotonic = 0.0

        # Degraded seed fails; two stamped buckets accumulate their deltas.
        t.add_tokens(100, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        assert t._ledger_degraded is True
        t.add_tokens(50, provider=_OPENAI, key_env=_KEY2, model=_SMALL_MODEL)

        b1 = BucketKey(_OPENAI, _KEY1, "small")
        b2 = BucketKey(_OPENAI, _KEY2, "small")
        assert t._unsynced_deltas[b1] == 100
        assert t._unsynced_deltas[b2] == 50
        # Standalone per-bucket enforcement still works while degraded.
        assert t._bucket_used_today[b1] == 100

        # Recovery lands the full accumulated amount.
        fake.fail = False
        t.sync_ledger_now()
        assert fake.field == 150
        assert t._ledger_degraded is False


class TestConfigParsing:
    @pytest.mark.unit
    def test_absent_block_keeps_caps_enabled_with_defaults(self) -> None:
        scope, enabled, caps, pools = _parse_pool_caps_config({"daily_tokens": 100})
        assert scope == "pooled"
        assert enabled is True
        assert caps == {}
        assert pools == {}

    @pytest.mark.unit
    def test_enabled_false_turns_caps_off(self) -> None:
        _scope, enabled, _caps, _pools = _parse_pool_caps_config(
            {"per_key_pool_caps": {"enabled": False}}
        )
        assert enabled is False

    @pytest.mark.unit
    def test_scope_all_parsed(self) -> None:
        scope, _enabled, _caps, _pools = _parse_pool_caps_config({"scope": "all"})
        assert scope == "all"

    @pytest.mark.unit
    def test_invalid_scope_falls_back_to_pooled(self) -> None:
        scope, _enabled, _caps, _pools = _parse_pool_caps_config({"scope": "nonsense"})
        assert scope == "pooled"

    @pytest.mark.unit
    def test_bare_int_entries_parsed_with_underscores(self) -> None:
        # The bare-int form stays a pure cap: no custom pool definition.
        _scope, _enabled, caps, pools = _parse_pool_caps_config(
            {
                "per_key_pool_caps": {
                    "enabled": True,
                    "openai": {"small": "9_750_000", "large": 975000},
                }
            }
        )
        assert caps == {
            ("openai", "small"): 9_750_000,
            ("openai", "large"): 975_000,
        }
        assert pools == {}

    @pytest.mark.unit
    def test_mapping_form_parses_cap_and_models(self) -> None:
        _scope, _enabled, caps, pools = _parse_pool_caps_config(
            {
                "per_key_pool_caps": {
                    "enabled": True,
                    "openai": {"small": 9_750_000, "large": {"cap": "975_000"}},
                    "myhost": {"standard": {"cap": 5_000_000, "models": ["my-model"]}},
                }
            }
        )
        assert caps == {
            ("openai", "small"): 9_750_000,
            ("openai", "large"): 975_000,
            ("myhost", "standard"): 5_000_000,
        }
        assert pools == {"myhost": {"standard": ("my-model",)}}

    @pytest.mark.unit
    def test_capless_models_entry_defines_pool_without_cap(self) -> None:
        _scope, _enabled, caps, pools = _parse_pool_caps_config(
            {
                "per_key_pool_caps": {
                    "myhost": {"standard": {"models": ["my-model", "other"]}}
                }
            }
        )
        assert caps == {}
        assert pools == {"myhost": {"standard": ("my-model", "other")}}

    @pytest.mark.unit
    def test_bare_daily_tokens_unaffected(self) -> None:
        # A bare-integer daily_tokens keeps its combined-cap meaning; parsing the
        # pool-cap settings does not touch it.
        cfg = {"daily_tokens": 5_000_000, "enabled": True}
        scope, enabled, caps, pools = _parse_pool_caps_config(cfg)
        assert (scope, enabled, caps, pools) == ("pooled", True, {}, {})
        assert cfg["daily_tokens"] == 5_000_000


class TestCustomPoolDefinitions:
    """Configured pools: arbitrary providers/labels, prefix-matched models."""

    @pytest.mark.unit
    def test_custom_provider_pool_is_derived_capped_and_enforced(
        self, tmp_path: Path
    ) -> None:
        t = _tracker(
            tmp_path,
            pool_caps={("myhost", "standard"): 60},
            provider_pools={"myhost": {"standard": ("my-model",)}},
        )
        # Prefix match at a separator boundary: "my-model-v2" -> "standard".
        t.add_tokens(60, provider="myhost", key_env="MYHOST_KEY", model="my-model-v2")
        bucket = BucketKey("myhost", "MYHOST_KEY", "standard")
        assert t._bucket_used_today[bucket] == 60
        # The cap (60) is exhausted for this key...
        assert (
            t.try_reserve(
                10, provider="myhost", key_env="MYHOST_KEY", model="my-model-v2"
            )
            is None
        )
        # ...while a second key on the same pool still admits.
        assert (
            t.try_reserve(
                10, provider="myhost", key_env="MYHOST_KEY_2", model="my-model"
            )
            == 10
        )

    @pytest.mark.unit
    def test_configured_openai_pools_replace_builtins(self, tmp_path: Path) -> None:
        t = _tracker(
            tmp_path,
            pool_caps={},
            provider_pools={"openai": {"tiny": (_SMALL_MODEL,)}},
        )
        # The configured mapping REPLACES the built-ins for openai: the listed
        # model derives the custom label...
        t.add_tokens(5, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        assert BucketKey(_OPENAI, _KEY1, "tiny") in t._bucket_used_today
        # ...and a model only in the built-in tables now derives pool None.
        t.add_tokens(5, provider=_OPENAI, key_env=_KEY1, model=_LARGE_MODEL)
        assert BucketKey(_OPENAI, _KEY1, None) in t._bucket_used_today

    @pytest.mark.unit
    def test_uncovered_provider_keeps_builtin_pools(self, tmp_path: Path) -> None:
        t = _tracker(
            tmp_path,
            provider_pools={"myhost": {"standard": ("my-model",)}},
        )
        t.add_tokens(5, provider=_OPENAI, key_env=_KEY1, model=_LARGE_MODEL)
        assert BucketKey(_OPENAI, _KEY1, "large") in t._bucket_used_today

    @pytest.mark.unit
    def test_capless_custom_pool_is_tracked_but_uncapped(self, tmp_path: Path) -> None:
        # models list without a cap: the pool derives and is tracked, but no
        # per-key cap gates it (label unknown to DEFAULT_POOL_CAPS -> None).
        t = _tracker(
            tmp_path,
            pool_caps={},
            provider_pools={"myhost": {"standard": ("my-model",)}},
        )
        t.add_tokens(10**8, provider="myhost", key_env="MYHOST_KEY", model="my-model")
        bucket = BucketKey("myhost", "MYHOST_KEY", "standard")
        assert t._bucket_used_today[bucket] == 10**8
        assert t._pool_cap_for(bucket) is None
        assert (
            t.try_reserve(10, provider="myhost", key_env="MYHOST_KEY", model="my-model")
            == 10
        )

    @pytest.mark.unit
    def test_set_pool_config_updates_live(self, tmp_path: Path) -> None:
        # The wait loop's live refresh path: raising a cap mid-wait unblocks.
        t = _tracker(tmp_path)
        t.add_tokens(100, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL)
            is None
        )
        t.set_pool_config(
            "pooled", True, {(_OPENAI, "small"): 1_000, (_OPENAI, "large"): 50}, {}
        )
        assert (
            t.try_reserve(10, provider=_OPENAI, key_env=_KEY1, model=_SMALL_MODEL) == 10
        )
