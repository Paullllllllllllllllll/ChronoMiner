"""Child process and shared fixtures for the store tests.

Usage, from the repository root:

    python tests/common/store_child.py acquire DB NAME
        Open the store, begin a run, print READY and wait for a line on
        stdin; then try to lease document NAME and print ``ACQUIRED <gen>``
        or ``REFUSED``, wait for another line and exit.

The module also defines the injectable clock the tests import.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chronominer.common.store import Store  # noqa: E402

TOOL = "tool"


class FakeClock:
    """An injectable clock in epoch seconds that tests advance by hand."""

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def acquire(db: Path, name: str) -> None:
    async with Store(db, TOOL) as store:
        await store.begin_run("acquire")
        doc_id = await store.ensure_document(name)
        print("READY", flush=True)  # noqa: T201 - the parent reads stdout
        await asyncio.to_thread(sys.stdin.readline)
        lease = await store.acquire(doc_id)
        print(  # noqa: T201 - the parent reads stdout
            "REFUSED" if lease is None else f"ACQUIRED {lease.generation}", flush=True
        )
        await asyncio.to_thread(sys.stdin.readline)


def main(argv: list[str]) -> None:
    mode, db = argv[0], Path(argv[1])
    if mode == "acquire":
        asyncio.run(acquire(db, argv[2]))
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
