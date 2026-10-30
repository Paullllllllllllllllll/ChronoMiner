"""Child process of the killed-submission test.

Usage, from the repository root:

    python tests/batch/submit_child.py ROOT STATE LEASE_TTL

Opens the store in ROOT with a lease time of LEASE_TTL seconds and submits
two page chunks of document ``doc`` to the OpenAI batch API through the fake
backend of ``tests/batch/conftest.py``. Right after the fake provider has
accepted the create, the child writes the provider's batches to the JSON
file STATE, prints READY and sleeps until the parent kills it, so the store
never learns the batch id. The OpenAI key variable must be set.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chronominer.batch.submit import submit_batch
from chronominer.common.store import Store
from chronominer.output.store import (
    COMPONENTS,
    TOOL,
    database_path,
)
from chronominer.settings import Settings
from tests.batch.conftest import (
    World,
    install_world,
    leased_document,
    page_images,
    run_spec,
    save_world,
)


async def submit(root: Path, state: Path, lease_ttl: float) -> None:
    world = World()
    install_world(world, SimpleNamespace(setattr=setattr))

    def accepted(batch_id: str) -> None:
        save_world(world, state)
        print("READY", flush=True)  # noqa: T201 - the parent reads stdout
        time.sleep(300)

    world.hooks["accepted"] = accepted
    store = Store(database_path(root), TOOL, components=COMPONENTS, lease_ttl=lease_ttl)
    async with store:
        await store.begin_run("extract")
        handle = await leased_document(store)
        await submit_batch(
            page_images(2),
            document=handle,
            spec=run_spec(root),
            settings=Settings(),
        )


def main(argv: list[str]) -> None:
    asyncio.run(submit(Path(argv[0]), Path(argv[1]), float(argv[2])))


if __name__ == "__main__":
    main(sys.argv[1:])
