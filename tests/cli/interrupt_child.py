"""Child process of the Ctrl+Break test.

Usage, from the repository root:

    python tests/cli/interrupt_child.py PDF OUT HOME

Runs ``chronominer extract --schema WageRecords --input PDF --output OUT
--concurrency 1 --json`` through the console entry point, with a scripted
model that answers every chunk with :data:`ANSWER`. The settings file and
the log folder default to HOME, as the test fixtures set them in the parent,
so the repository's files are neither read nor written. After the first chunk is
stored the child prints READY and waits, so the parent can send Ctrl+Break
while the document is leased. The OpenAI key variable must be set. The exit
code is the command's.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chronominer.settings as settings_module  # noqa: E402
from chronominer.__main__ import main  # noqa: E402
from chronominer.extract.units import DocumentHandle  # noqa: E402
from tests.llm import helpers  # noqa: E402

ANSWER: dict[str, Any] = {
    "contains_no_content_of_requested_type": False,
    "entries": [
        {
            "person": "John Carter",
            "occupation": "mason",
            "wage": 2.5,
            "period": "per day",
            "currency": "shillings",
            "date": "1791",
            "employer": None,
            "source_page": None,
            "notes": None,
        }
    ],
}

_real_write = DocumentHandle.write


async def _write_then_wait(*args: Any, **kwargs: Any) -> None:
    await _real_write(*args, **kwargs)
    print("READY", flush=True)  # noqa: T201 - the parent reads stdout
    await asyncio.sleep(300)


def run(argv: list[str]) -> int:
    pdf, out, home = argv
    settings_module.DEFAULT_SETTINGS_PATH = Path(home) / "config" / "settings.yaml"
    settings_module.DEFAULT_LOGS_DIR = Path(home) / "logs"
    helpers.install(SimpleNamespace(setattr=setattr), [ANSWER])  # type: ignore[arg-type]
    DocumentHandle.write = _write_then_wait  # type: ignore[method-assign]
    return main(
        [
            "extract",
            "--schema",
            "WageRecords",
            "--input",
            pdf,
            "--output",
            out,
            "--concurrency",
            "1",
            "--json",
        ]
    )


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
