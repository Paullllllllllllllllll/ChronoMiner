"""Drop stored documents (the ``drop`` subcommand).

:func:`drop_root` finds the output root of a ``--input`` (an output folder
holding ``chronominer.sqlite``, or the database itself);
:func:`stored_documents` lists the documents to select from, and
:func:`drop_documents` deletes the chosen ones through
:meth:`~chronominer.common.store.Store.drop`, or reports what it would delete.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from chronominer.common.store import DocumentRow, DropReport
from chronominer.output.store import DATABASE_NAME, open_store, read_store

__all__ = [
    "KEPT_NOTE",
    "DropInputError",
    "drop_documents",
    "drop_root",
    "stored_documents",
]

KEPT_NOTE = (
    "A kept _output.json, or a kept CSV a run renders, makes the next extraction "
    "list its document as unstored and skip it; move the file away to extract the "
    "document from the start."
)
"""What a file kept by a drop means for the next run."""


class DropInputError(ValueError):
    """The ``--input`` of a drop names no store."""


def drop_root(path: Path) -> Path:
    """Return the output root whose store *path* names.

    Raises:
        DropInputError: *path* does not exist, is a file other than the
            database, or is a folder without one.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise DropInputError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.name != DATABASE_NAME:
            raise DropInputError(f"{path} is not a {DATABASE_NAME} file")
        return path.parent
    if not (path / DATABASE_NAME).is_file():
        raise DropInputError(f"No {DATABASE_NAME} in {path}")
    return path


def stored_documents(root: Path) -> list[DocumentRow]:
    """Return the documents of *root*'s store, ordered by name.

    Raises:
        DropInputError: The database holds no store (an empty file).
        StoreError: The store refuses the database.
    """
    store = read_store(root)
    if store is None:
        raise DropInputError(f"{root / DATABASE_NAME} holds no store")
    return store.documents()


async def drop_documents(
    root: Path, documents: Sequence[DocumentRow], *, dry_run: bool
) -> DropReport:
    """Drop *documents* from *root*'s store; a dry run changes nothing."""
    ids = [document.doc_id for document in documents]
    if dry_run:
        store = read_store(root)
        if store is None:
            raise DropInputError(f"{root / DATABASE_NAME} holds no store")
        return await store.drop(ids, dry_run=True)
    async with open_store(root) as store:
        return await store.drop(ids)
