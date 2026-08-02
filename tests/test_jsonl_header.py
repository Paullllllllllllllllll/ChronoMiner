"""Tests for JSONL header utilities: fingerprint, build, read, validate, stats."""

import json
from pathlib import Path

from modules.infra.jsonl import (
    build_jsonl_header,
    compute_ranges_fingerprint,
    compute_stats_from_jsonl,
    extract_completed_ids,
    is_jsonl_adjustment_complete,
    read_jsonl_header,
    update_jsonl_header,
    validate_jsonl_header,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MATCHING_CONFIG = {
    "normalize_whitespace": True,
    "case_sensitive": False,
    "normalize_diacritics": True,
    "strip_punctuation": False,
    "allow_substring_match": True,
    "min_substring_length": 3,
}

_RETRY_CONFIG = {
    "certainty_threshold": 70,
    "max_low_certainty_retries": 15,
    "max_marker_mismatch_retries": 15,
    "max_context_expansion_attempts": 6,
    "delete_ranges_with_no_content": True,
    "scan_range_multiplier": 2,
    "max_gap_between_ranges": 2,
}


def _write_line_ranges(path: Path, ranges: list[tuple[int, int]]) -> None:
    path.write_text(
        "\n".join(f"({s}, {e})" for s, e in ranges) + "\n",
        encoding="utf-8",
    )


def _make_range_record(
    stem: str,
    idx: int,
    *,
    original: tuple,
    adjusted: tuple,
    delete: bool = False,
    llm_calls: int = 2,
    boundary_on_target: bool = False,
) -> dict:
    return {
        "custom_id": f"{stem}-range-{idx}",
        "response": {
            "body": {
                "original_range": list(original),
                "adjusted_range": list(adjusted),
                "should_delete": delete,
                "decision": {
                    "contains_no_semantic_boundary": delete,
                    "needs_more_context": False,
                    "boundary_already_on_target": boundary_on_target,
                    "certainty": 90,
                    "semantic_marker": None if delete else "Recipe",
                },
                "attempts": [],
                "total_llm_calls": llm_calls,
            }
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# compute_ranges_fingerprint
# ---------------------------------------------------------------------------


class TestComputeRangesFingerprint:
    def test_deterministic(self, tmp_path: Path) -> None:
        lr = tmp_path / "doc_line_ranges.txt"
        _write_line_ranges(lr, [(1, 100), (101, 200)])
        fp1 = compute_ranges_fingerprint(lr)
        fp2 = compute_ranges_fingerprint(lr)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_changes_on_content_change(self, tmp_path: Path) -> None:
        lr = tmp_path / "doc_line_ranges.txt"
        _write_line_ranges(lr, [(1, 100), (101, 200)])
        fp1 = compute_ranges_fingerprint(lr)
        _write_line_ranges(lr, [(1, 100), (101, 250)])
        fp2 = compute_ranges_fingerprint(lr)
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# build_jsonl_header
# ---------------------------------------------------------------------------


class TestBuildJsonlHeader:
    def test_structure(self) -> None:
        header_rec = build_jsonl_header(
            ranges_fingerprint="abc123",
            total_ranges=10,
            boundary_type="RecipeBoundary",
            model_name="gpt-5-mini",
            context_window=6,
            matching_config=_MATCHING_CONFIG,
            retry_config=_RETRY_CONFIG,
            prompt_hash="deadbeef",
            context_path="/some/path.txt",
        )
        assert "jsonl_header" in header_rec
        h = header_rec["jsonl_header"]
        assert h["version"] == 2
        assert h["ranges_fingerprint"] == "abc123"
        assert h["total_ranges"] == 10
        assert h["model_name"] == "gpt-5-mini"
        assert "created_at" in h

    def test_no_custom_id(self) -> None:
        header_rec = build_jsonl_header(
            ranges_fingerprint="x",
            total_ranges=1,
            boundary_type="B",
            model_name="m",
            context_window=6,
        )
        assert "custom_id" not in header_rec

    def test_context_hash_stored(self) -> None:
        header_rec = build_jsonl_header(
            ranges_fingerprint="x",
            total_ranges=1,
            boundary_type="B",
            model_name="m",
            context_window=6,
            context_path="/some/path.txt",
            context_hash="ctx1",
        )
        assert header_rec["jsonl_header"]["context_hash"] == "ctx1"

    def test_context_hash_defaults_to_none_but_key_present(self) -> None:
        header_rec = build_jsonl_header(
            ranges_fingerprint="x",
            total_ranges=1,
            boundary_type="B",
            model_name="m",
            context_window=6,
        )
        h = header_rec["jsonl_header"]
        assert "context_hash" in h
        assert h["context_hash"] is None

    def test_version_unchanged_with_context_hash(self) -> None:
        """Adding context_hash must not bump the hard-equality header version:
        a bump would invalidate every in-flight resume artifact."""
        header_rec = build_jsonl_header(
            ranges_fingerprint="x",
            total_ranges=1,
            boundary_type="B",
            model_name="m",
            context_window=6,
            context_hash="ctx1",
        )
        assert header_rec["jsonl_header"]["version"] == 2


# ---------------------------------------------------------------------------
# read_jsonl_header
# ---------------------------------------------------------------------------


class TestReadJsonlHeader:
    def test_reads_first_line_only(self, tmp_path: Path) -> None:
        header_rec = build_jsonl_header(
            ranges_fingerprint="fp",
            total_ranges=5,
            boundary_type="B",
            model_name="m",
            context_window=6,
        )
        range_rec = _make_range_record("doc", 1, original=(1, 100), adjusted=(1, 100))
        jsonl = tmp_path / "temp.jsonl"
        _write_jsonl(jsonl, [header_rec, range_rec])

        header = read_jsonl_header(jsonl)
        assert header is not None
        assert header["ranges_fingerprint"] == "fp"
        assert header["total_ranges"] == 5

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert read_jsonl_header(tmp_path / "missing.jsonl") is None

    def test_returns_none_for_legacy_jsonl(self, tmp_path: Path) -> None:
        range_rec = _make_range_record("doc", 1, original=(1, 100), adjusted=(1, 100))
        jsonl = tmp_path / "temp.jsonl"
        _write_jsonl(jsonl, [range_rec])
        assert read_jsonl_header(jsonl) is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "temp.jsonl"
        jsonl.write_text("", encoding="utf-8")
        assert read_jsonl_header(jsonl) is None

    def test_reads_header_through_utf8_bom(self, tmp_path: Path) -> None:
        # A BOM (file re-saved in Notepad) must not hide the header: an
        # unreadable header makes resume look absent and re-bills the run.
        header_rec = build_jsonl_header(
            ranges_fingerprint="fp",
            total_ranges=5,
            boundary_type="B",
            model_name="m",
            context_window=6,
        )
        range_rec = _make_range_record("doc", 1, original=(1, 100), adjusted=(1, 100))
        jsonl = tmp_path / "bom_temp.jsonl"
        jsonl.write_bytes(
            b"\xef\xbb\xbf"
            + "".join(
                json.dumps(rec, ensure_ascii=False) + "\n"
                for rec in (header_rec, range_rec)
            ).encode("utf-8")
        )

        header = read_jsonl_header(jsonl)
        assert header is not None
        assert header["ranges_fingerprint"] == "fp"

        # Records past the header stay readable, and a header update rewrites
        # the file as plain UTF-8 without losing either record.
        assert extract_completed_ids(jsonl) == {1}
        assert update_jsonl_header(jsonl, {"completed_at": "2026-01-01T00:00:00"})
        assert not jsonl.read_bytes().startswith(b"\xef\xbb\xbf")

        updated = read_jsonl_header(jsonl)
        assert updated is not None
        assert updated["completed_at"] == "2026-01-01T00:00:00"
        assert updated["ranges_fingerprint"] == "fp"
        assert extract_completed_ids(jsonl) == {1}


# ---------------------------------------------------------------------------
# validate_jsonl_header
# ---------------------------------------------------------------------------


class TestValidateJsonlHeader:
    def _sample_header(self, **overrides) -> dict:
        defaults = {
            "version": 2,
            "ranges_fingerprint": "abc",
            "total_ranges": 10,
            "boundary_type": "B",
            "model_name": "m",
            "context_window": 6,
            "matching_config": _MATCHING_CONFIG,
            "retry_config": _RETRY_CONFIG,
            "prompt_hash": "phash",
            "context_path": None,
            "created_at": "2026-01-01T00:00:00Z",
        }
        defaults.update(overrides)
        return defaults

    def test_matching_returns_true(self) -> None:
        h = self._sample_header()
        assert (
            validate_jsonl_header(
                h,
                ranges_fingerprint="abc",
                boundary_type="B",
                model_name="m",
                context_window=6,
                matching_config=_MATCHING_CONFIG,
                retry_config=_RETRY_CONFIG,
                prompt_hash="phash",
            )
            is True
        )

    def test_fingerprint_mismatch(self) -> None:
        h = self._sample_header()
        assert (
            validate_jsonl_header(
                h,
                ranges_fingerprint="DIFFERENT",
                boundary_type="B",
                model_name="m",
                context_window=6,
            )
            is False
        )

    def test_model_mismatch(self) -> None:
        h = self._sample_header()
        assert (
            validate_jsonl_header(
                h,
                ranges_fingerprint="abc",
                boundary_type="B",
                model_name="OTHER_MODEL",
                context_window=6,
            )
            is False
        )

    def test_context_window_mismatch(self) -> None:
        h = self._sample_header()
        assert (
            validate_jsonl_header(
                h,
                ranges_fingerprint="abc",
                boundary_type="B",
                model_name="m",
                context_window=99,
            )
            is False
        )

    def test_version_mismatch_rejected(self) -> None:
        """Version-1 headers (pre-absolute-index era) are always stale."""
        h = self._sample_header(version=1)
        assert (
            validate_jsonl_header(
                h,
                ranges_fingerprint="abc",
                boundary_type="B",
                model_name="m",
                context_window=6,
            )
            is False
        )

    def test_prompt_hash_ignored_when_caller_none(self) -> None:
        h = self._sample_header(prompt_hash="stored")
        assert (
            validate_jsonl_header(
                h,
                ranges_fingerprint="abc",
                boundary_type="B",
                model_name="m",
                context_window=6,
                prompt_hash=None,
            )
            is True
        )

    def _validate_with_context(self, header: dict, context_hash: str | None) -> bool:
        return validate_jsonl_header(
            header,
            ranges_fingerprint="abc",
            boundary_type="B",
            model_name="m",
            context_window=6,
            matching_config=_MATCHING_CONFIG,
            retry_config=_RETRY_CONFIG,
            prompt_hash="phash",
            context_hash=context_hash,
        )

    def test_context_hash_match(self) -> None:
        h = self._sample_header(context_hash="ctx1")
        assert self._validate_with_context(h, "ctx1") is True

    def test_context_hash_mismatch(self) -> None:
        h = self._sample_header(context_hash="ctx1")
        assert self._validate_with_context(h, "ctx2") is False

    def test_legacy_header_without_context_hash_is_wildcard(self) -> None:
        """Headers written before context hashing existed keep resuming."""
        h = self._sample_header()
        assert "context_hash" not in h
        assert self._validate_with_context(h, "ctx1") is True

    def test_context_added_after_first_run_invalidates(self) -> None:
        """Header 'none' (no context then) vs a real hash now -> stale."""
        h = self._sample_header(context_hash="none")
        assert self._validate_with_context(h, "a" * 64) is False

    def test_context_removed_after_first_run_invalidates(self) -> None:
        """Header hash vs 'none' now (context deleted) -> stale."""
        h = self._sample_header(context_hash="a" * 64)
        assert self._validate_with_context(h, "none") is False

    def test_context_hash_ignored_when_caller_none(self) -> None:
        h = self._sample_header(context_hash="ctx1")
        assert self._validate_with_context(h, None) is True


# ---------------------------------------------------------------------------
# compute_stats_from_jsonl
# ---------------------------------------------------------------------------


class TestComputeStatsFromJsonl:
    def test_correct_counts(self, tmp_path: Path) -> None:
        header = build_jsonl_header(
            ranges_fingerprint="x",
            total_ranges=5,
            boundary_type="B",
            model_name="m",
            context_window=6,
        )
        records = [
            header,
            _make_range_record(
                "d", 1, original=(1, 100), adjusted=(5, 100), llm_calls=2
            ),
            _make_range_record(
                "d", 2, original=(101, 200), adjusted=(101, 200), llm_calls=1
            ),
            _make_range_record(
                "d",
                3,
                original=(201, 300),
                adjusted=(201, 300),
                delete=True,
                llm_calls=3,
            ),
            _make_range_record(
                "d", 4, original=(301, 400), adjusted=(310, 400), llm_calls=2
            ),
            _make_range_record(
                "d", 5, original=(401, 500), adjusted=(401, 500), llm_calls=1
            ),
        ]
        jsonl = tmp_path / "temp.jsonl"
        _write_jsonl(jsonl, records)

        stats = compute_stats_from_jsonl(jsonl)
        assert stats["total_ranges"] == 5
        assert stats["ranges_adjusted"] == 2  # ranges 1 and 4
        assert stats["ranges_deleted"] == 1  # range 3
        assert stats["ranges_kept_original"] == 2  # ranges 2 and 5
        assert stats["ranges_already_on_target"] == 0
        assert stats["total_llm_calls"] == 9

    def test_counts_already_on_target(self, tmp_path: Path) -> None:
        header = build_jsonl_header(
            ranges_fingerprint="fp",
            total_ranges=3,
            boundary_type="B",
            model_name="m",
            context_window=6,
        )
        records = [
            header,
            _make_range_record(
                "d", 1, original=(1, 100), adjusted=(5, 100), llm_calls=2
            ),
            _make_range_record(
                "d",
                2,
                original=(101, 200),
                adjusted=(101, 200),
                llm_calls=1,
                boundary_on_target=True,
            ),
            _make_range_record(
                "d",
                3,
                original=(201, 300),
                adjusted=(201, 300),
                llm_calls=1,
                boundary_on_target=True,
            ),
        ]
        jsonl = tmp_path / "temp.jsonl"
        _write_jsonl(jsonl, records)

        stats = compute_stats_from_jsonl(jsonl)
        assert stats["total_ranges"] == 3
        assert stats["ranges_adjusted"] == 1
        assert stats["ranges_kept_original"] == 2
        assert stats["ranges_already_on_target"] == 2

    def test_header_not_counted(self, tmp_path: Path) -> None:
        header = build_jsonl_header(
            ranges_fingerprint="x",
            total_ranges=1,
            boundary_type="B",
            model_name="m",
            context_window=6,
        )
        jsonl = tmp_path / "temp.jsonl"
        _write_jsonl(jsonl, [header])

        stats = compute_stats_from_jsonl(jsonl)
        assert stats["total_ranges"] == 0


# ---------------------------------------------------------------------------
# extract_completed_ids with header
# ---------------------------------------------------------------------------


class TestExtractCompletedIdsSkipsHeader:
    def test_header_not_counted(self, tmp_path: Path) -> None:
        import re

        header = build_jsonl_header(
            ranges_fingerprint="x",
            total_ranges=2,
            boundary_type="B",
            model_name="m",
            context_window=6,
        )
        records = [
            header,
            _make_range_record("d", 1, original=(1, 100), adjusted=(1, 100)),
            _make_range_record("d", 2, original=(101, 200), adjusted=(101, 200)),
        ]
        jsonl = tmp_path / "temp.jsonl"
        _write_jsonl(jsonl, records)

        pattern = re.compile(r"-range-(\d+)$")
        ids = extract_completed_ids(jsonl, id_pattern=pattern)
        assert ids == {1, 2}


# ---------------------------------------------------------------------------
# is_jsonl_adjustment_complete: context_hash clause
# ---------------------------------------------------------------------------


class TestAdjustmentCompleteContextHash:
    """The completed-run skip check must invalidate on an adjust-context change.

    Every fixture writes a header with ``completed_at`` and a matching
    ``final_ranges_fingerprint`` so the context clause is the only variable.
    """

    _FINGERPRINT = "fp-final"

    def _setup(self, tmp_path: Path, **header_overrides) -> Path:
        lr_file = tmp_path / "doc_line_ranges.txt"
        _write_line_ranges(lr_file, [(1, 100)])

        header: dict = {
            "version": 2,
            "ranges_fingerprint": self._FINGERPRINT,
            "final_ranges_fingerprint": self._FINGERPRINT,
            "total_ranges": 1,
            "boundary_type": "B",
            "model_name": "m",
            "context_window": 6,
            "matching_config": _MATCHING_CONFIG,
            "retry_config": _RETRY_CONFIG,
            "prompt_hash": "phash",
            "context_path": None,
            "created_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T01:00:00Z",
        }
        header.update(header_overrides)
        _write_jsonl(
            tmp_path / "doc_line_ranges_adjust_temp.jsonl", [{"jsonl_header": header}]
        )
        return lr_file

    def _check(self, lr_file: Path, context_hash: str | None) -> bool:
        return is_jsonl_adjustment_complete(
            lr_file,
            boundary_type="B",
            context_window=6,
            model_name="m",
            matching_config=_MATCHING_CONFIG,
            retry_config=_RETRY_CONFIG,
            ranges_fingerprint=self._FINGERPRINT,
            prompt_hash="phash",
            context_hash=context_hash,
        )

    def test_full_match_including_context_hash(self, tmp_path: Path) -> None:
        """No over-invalidation: an unchanged context still counts as complete."""
        lr_file = self._setup(tmp_path, context_hash="ctx1")
        assert self._check(lr_file, "ctx1") is True

    def test_context_hash_mismatch(self, tmp_path: Path) -> None:
        lr_file = self._setup(tmp_path, context_hash="ctx1")
        assert self._check(lr_file, "ctx2") is False

    def test_legacy_header_without_context_hash_is_wildcard(
        self, tmp_path: Path
    ) -> None:
        lr_file = self._setup(tmp_path)
        header = read_jsonl_header(tmp_path / "doc_line_ranges_adjust_temp.jsonl")
        assert header is not None and "context_hash" not in header
        assert self._check(lr_file, "ctx1") is True

    def test_context_added_after_first_run_invalidates(self, tmp_path: Path) -> None:
        lr_file = self._setup(tmp_path, context_hash="none")
        assert self._check(lr_file, "a" * 64) is False

    def test_context_removed_after_first_run_invalidates(self, tmp_path: Path) -> None:
        lr_file = self._setup(tmp_path, context_hash="a" * 64)
        assert self._check(lr_file, "none") is False

    def test_context_hash_ignored_when_caller_none(self, tmp_path: Path) -> None:
        lr_file = self._setup(tmp_path, context_hash="ctx1")
        assert self._check(lr_file, None) is True
