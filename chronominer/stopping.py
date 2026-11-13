"""The record of a run that stopped because a provider serves no further calls.

The retry ladder raises :class:`~chronominer.common.retry.RunStopped`; the
run loop of each command turns it into a :class:`RunStop`, which the run's
result carries to the front ends: the exit code
:attr:`~chronominer.common.contract.ExitCode.STOPPED`, the error line of
:meth:`RunStop.message` and the ``stopped`` field of the JSON summary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["RunStop"]


@dataclass(frozen=True)
class RunStop:
    """Why a run stopped and what it left unprocessed.

    *name* is the item in work when the run stopped, None when none was;
    *units* counts its *noun* units left unprocessed, None when not counted;
    *further* counts the items the run did not start. When several items
    were in work, *in_work* counts them, *name* is None and *units* is
    their sum. *hint* says how the unprocessed work resumes.
    """

    reason: str
    name: str | None = None
    units: int | None = None
    further: int = 0
    noun: str = "unit"
    hint: str = "a rerun resumes them"
    in_work: int = 0

    @classmethod
    def of(
        cls,
        reason: str,
        stopped: Sequence[tuple[str, int | None]],
        *,
        further: int,
        noun: str = "unit",
    ) -> RunStop:
        """Return the stop of a run that left *stopped* items in work.

        *stopped* holds each item's name and its units left, in item order.
        """
        if len(stopped) == 1:
            ((name, units),) = stopped
            return cls(reason, name, units, further=further, noun=noun)
        counts = [units for _, units in stopped]
        total = None if None in counts else sum(n for n in counts if n is not None)
        return cls(
            reason, None, total, further=further, noun=noun, in_work=len(stopped)
        )

    @property
    def items(self) -> int:
        """The items left unprocessed, the ones in work included."""
        if self.in_work:
            return self.further + self.in_work
        return self.further + (0 if self.name is None else 1)

    def message(self) -> str:
        """Return the line that reports the stop."""
        parts: list[str] = []
        if self.in_work:
            work = f"{self.in_work} item(s) in work"
            parts.append(
                work if self.units is None else f"{self.units} {self.noun}(s) of {work}"
            )
        elif self.name is not None:
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
