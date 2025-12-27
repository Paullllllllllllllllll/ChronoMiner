import asyncio
import pytest
from modules.core.concurrency import run_concurrent_tasks


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_concurrent_tasks_basic():
    async def sample_coro(x):
        await asyncio.sleep(0.01)
        return x * 2
    
    args_list = [(1,), (2,), (3,), (4,)]
    results = await run_concurrent_tasks(sample_coro, args_list, concurrency_limit=2)
    
    assert len(results) == 4
    assert results == [2, 4, 6, 8]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_concurrent_tasks_with_delay():
    async def sample_coro(x):
        return x + 1
    
    args_list = [(1,), (2,)]
    results = await run_concurrent_tasks(sample_coro, args_list, concurrency_limit=2, delay=0.01)
    
    assert len(results) == 2
    assert results == [2, 3]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_concurrent_tasks_concurrency_limit():
    execution_times = []
    
    async def sample_coro(x):
        execution_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.05)
        return x
    
    args_list = [(i,) for i in range(4)]
    await run_concurrent_tasks(sample_coro, args_list, concurrency_limit=2)
    
    assert len(execution_times) == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_concurrent_tasks_handles_exceptions():
    async def failing_coro(x):
        if x == 2:
            raise ValueError("Test error")
        return x * 2
    
    args_list = [(1,), (2,), (3,)]
    results = await run_concurrent_tasks(failing_coro, args_list, concurrency_limit=3)
    
    assert len(results) == 3
    assert results[0] == 2
    assert results[1] is None
    assert results[2] == 6


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_concurrent_tasks_empty_args_list():
    async def sample_coro(x):
        return x
    
    results = await run_concurrent_tasks(sample_coro, [], concurrency_limit=5)
    
    assert len(results) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_concurrent_tasks_single_task():
    async def sample_coro(x, y):
        return x + y
    
    args_list = [(5, 3)]
    results = await run_concurrent_tasks(sample_coro, args_list, concurrency_limit=1)
    
    assert len(results) == 1
    assert results[0] == 8


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_concurrent_tasks_multiple_args():
    async def sample_coro(x, y, z):
        return x + y + z
    
    args_list = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    results = await run_concurrent_tasks(sample_coro, args_list, concurrency_limit=3)
    
    assert len(results) == 3
    assert results == [6, 15, 24]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_concurrent_tasks_all_fail():
    async def failing_coro(x):
        raise RuntimeError("Always fails")
    
    args_list = [(1,), (2,), (3,)]
    results = await run_concurrent_tasks(failing_coro, args_list, concurrency_limit=2)
    
    assert len(results) == 3
    assert all(r is None for r in results)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_concurrent_tasks_high_concurrency():
    async def sample_coro(x):
        await asyncio.sleep(0.001)
        return x ** 2
    
    args_list = [(i,) for i in range(20)]
    results = await run_concurrent_tasks(sample_coro, args_list, concurrency_limit=10)
    
    assert len(results) == 20
    assert results == [i ** 2 for i in range(20)]
