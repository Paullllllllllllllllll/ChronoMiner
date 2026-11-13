"""The record of a run that stopped because a provider serves no further calls.

The retry ladder raises :class:`~chronominer.common.retry.RunStopped`; the
run loop of each command turns it into a :class:`RunStop`, which the run's
result carries to the front ends: the exit code
:attr:`~chronominer.common.contract.ExitCode.STOPPED`, the error line of
:meth:`RunStop.message` and the ``stopped`` field of the JSON summary.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RunStop"]


@dataclass(frozen=True)
class RunStop:
    """Why a run stopped and what it left unprocessed.

    *name* is the item in work when the run stopped, None when none was;
    *units* counts its *noun* units left unprocessed, None when not counted;
    *further* counts the items the run did not start. *hint* says how the
    unprocessed work resumes.
    """

    reason: str
    name: str | None = None
    units: int | None = None
    further: int = 0
    noun: str = "unit"
    hint: str = "a rerun resumes them"

    @property
    def items(self) -> int:
        """The items left unprocessed, the one in work included."""
        return self.further + (0 if self.name is None else 1)

    def message(self) -> str:
        """Return the line that reports the stop."""
        parts: list[str] = []
        if self.name is not None:
            parts.append(
                f"'{self.name}'"
                if self.units is None
                else f"{self.units} {self.noun}(s) of '{self.name}'"
            )
        if self.further or not parts:
            parts.append(f"{self.further} further item(s)")
        return (
            f"Run stopped ({self.reason}): {' and '.join(parts)} were left "
            f"unprocessed; {self.hint}."
        )
