import httpx
import pytest

from modules.llm.transient_errors import (
    ChunkTimeoutError,
    is_connection_error,
    is_timeout_error,
    resolve_chunk_timeout,
)


class APITimeoutError(Exception):
    """Stand-in for an SDK timeout type raised without an httpx cause."""


@pytest.mark.unit
def test_bare_read_timeout_is_a_timeout():
    assert is_timeout_error(httpx.ReadTimeout("slow"))


@pytest.mark.unit
def test_connect_timeout_is_a_timeout():
    assert is_timeout_error(httpx.ConnectTimeout("slow"))


@pytest.mark.unit
def test_wrapped_read_timeout_is_a_timeout():
    try:
        try:
            raise httpx.ReadTimeout("slow")
        except httpx.ReadTimeout as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as exc:
        assert is_timeout_error(exc)


@pytest.mark.unit
def test_sdk_timeout_matched_by_class_name():
    assert is_timeout_error(APITimeoutError("no httpx cause"))


@pytest.mark.unit
def test_builtin_timeout_error_is_a_timeout():
    assert is_timeout_error(TimeoutError())


@pytest.mark.unit
def test_connect_error_is_not_a_timeout():
    assert not is_timeout_error(httpx.ConnectError("refused"))


@pytest.mark.unit
def test_unrelated_error_is_not_a_timeout():
    assert not is_timeout_error(ValueError("nope"))


@pytest.mark.unit
def test_cause_cycle_terminates():
    first = ValueError("first")
    second = ValueError("second")
    first.__cause__ = second
    second.__cause__ = first
    assert not is_timeout_error(first)


@pytest.mark.unit
def test_timeout_beyond_depth_bound_is_not_found():
    deepest: BaseException = httpx.ReadTimeout("slow")
    current: BaseException = deepest
    for _ in range(25):
        wrapper = RuntimeError("wrapped")
        wrapper.__cause__ = current
        current = wrapper
    assert not is_timeout_error(current)


@pytest.mark.unit
def test_connect_error_is_a_connection_error():
    assert is_connection_error(httpx.ConnectError("refused"))


@pytest.mark.unit
def test_read_timeout_is_a_connection_error():
    assert is_connection_error(httpx.ReadTimeout("slow"))


@pytest.mark.unit
def test_wrapped_connect_error_is_a_connection_error():
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = httpx.ConnectError("refused")
    assert is_connection_error(wrapper)


@pytest.mark.unit
def test_unrelated_error_is_not_a_connection_error():
    assert not is_connection_error(ValueError("nope"))


@pytest.mark.unit
def test_chunk_timeout_error_carries_label_and_seconds():
    exc = ChunkTimeoutError("chunk 3/12", 2100.0)
    assert isinstance(exc, Exception)
    assert exc.label == "chunk 3/12"
    assert exc.seconds == 2100.0
    assert "chunk 3/12" in str(exc)
    assert "2100" in str(exc)


@pytest.mark.unit
def test_auto_ceiling_from_total_and_attempts():
    assert resolve_chunk_timeout({"total": 900}, timeout_attempts=3) == 3000.0


@pytest.mark.unit
def test_explicit_auto_matches_implicit_auto():
    cfg = {"total": 900, "chunk_timeout": "auto"}
    assert resolve_chunk_timeout(cfg, timeout_attempts=3) == 3000.0


@pytest.mark.unit
def test_garbage_total_falls_back_to_default():
    cfg = {"total": "soon", "chunk_timeout": "auto"}
    assert resolve_chunk_timeout(cfg, timeout_attempts=3) == 2100.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw", [False, None, 0, "off", "none", "disabled", "OFF", " Disabled "]
)
def test_disabled_values_yield_no_ceiling(raw):
    cfg = {"total": 900, "chunk_timeout": raw}
    assert resolve_chunk_timeout(cfg, timeout_attempts=3) is None


@pytest.mark.unit
def test_explicit_number_is_used_verbatim():
    cfg = {"total": 900, "chunk_timeout": 1234.5}
    assert resolve_chunk_timeout(cfg, timeout_attempts=3) == 1234.5


@pytest.mark.unit
def test_numeric_string_is_parsed():
    cfg = {"total": 900, "chunk_timeout": "600"}
    assert resolve_chunk_timeout(cfg, timeout_attempts=3) == 600.0


@pytest.mark.unit
def test_negative_number_disables_the_ceiling():
    cfg = {"total": 900, "chunk_timeout": -5}
    assert resolve_chunk_timeout(cfg, timeout_attempts=3) is None


@pytest.mark.unit
def test_true_selects_auto():
    cfg = {"total": 900, "chunk_timeout": True}
    assert resolve_chunk_timeout(cfg, timeout_attempts=3) == 3000.0


@pytest.mark.unit
def test_unparseable_string_selects_auto():
    cfg = {"total": 900, "chunk_timeout": "soonish"}
    assert resolve_chunk_timeout(cfg, timeout_attempts=3) == 3000.0


@pytest.mark.unit
def test_unsupported_type_disables_the_ceiling():
    cfg = {"total": 900, "chunk_timeout": ["nope"]}
    assert resolve_chunk_timeout(cfg, timeout_attempts=3) is None


@pytest.mark.unit
def test_missing_config_uses_defaults():
    assert resolve_chunk_timeout(None, timeout_attempts=3) == 2100.0
