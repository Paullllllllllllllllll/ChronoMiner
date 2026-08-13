import httpx
import pytest

from modules.llm.http_timeouts import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    build_httpx_timeout,
)


@pytest.mark.unit
def test_none_read_timeout_returns_none():
    assert build_httpx_timeout(None) is None


@pytest.mark.unit
def test_read_phase_keeps_configured_budget():
    timeout = build_httpx_timeout(900.0)
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 900.0
    assert timeout.connect == DEFAULT_CONNECT_TIMEOUT
    assert timeout.write == DEFAULT_WRITE_TIMEOUT
    assert timeout.pool == DEFAULT_POOL_TIMEOUT


@pytest.mark.unit
def test_overrides_are_honored():
    timeout = build_httpx_timeout(900.0, connect=5, write=60, pool=15)
    assert timeout is not None
    assert timeout.read == 900.0
    assert timeout.connect == 5.0
    assert timeout.write == 60.0
    assert timeout.pool == 15.0


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, "nonsense", None, True, False, [5]])
def test_invalid_overrides_fall_back_to_defaults(bad):
    timeout = build_httpx_timeout(900.0, connect=bad, write=bad, pool=bad)
    assert timeout is not None
    assert timeout.connect == DEFAULT_CONNECT_TIMEOUT
    assert timeout.write == DEFAULT_WRITE_TIMEOUT
    assert timeout.pool == DEFAULT_POOL_TIMEOUT


@pytest.mark.unit
def test_timeout_object_is_unhashable():
    """Pin httpx.Timeout's unhashability.

    langchain-openai shares HTTP clients through an ``lru_cache`` keyed
    on the constructor arguments, so an unhashable timeout bypasses that
    cache and each chat model builds its own client. If this assertion
    ever fails, httpx changed its client-sharing semantics and the
    client-cache note in the module docstring needs revisiting.
    """
    timeout = build_httpx_timeout(900.0)
    with pytest.raises(TypeError):
        hash(timeout)
