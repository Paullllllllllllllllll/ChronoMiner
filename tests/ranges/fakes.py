"""A scripted boundary caller and the ledgers of the range tests.

:class:`FakeCaller` stands in for ``ModelCaller``: it answers each boundary
request from a function of the user text, by default the characterization
fake's answer (move the start to the next record heading), and records the
requests. An answer may be an exception to raise.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from chronominer.common.structured import StructuredRequest, StructuredResult
from tests.characterization.fakes import boundary_payload

Answer = Callable[[str, str], Any]
"""``(user text, label) -> payload dict or exception``."""


def default_answer(text: str, label: str) -> Any:
    """Answer like the characterization fake: next record heading."""
    return boundary_payload(text)[1]


def boundary(
    *,
    marker: str = "",
    on_target: bool = False,
    no_content: bool = False,
    more_context: bool = False,
    certainty: int = 95,
) -> dict[str, Any]:
    """Return an answer in the boundary schema."""
    return {
        "contains_no_semantic_boundary": no_content,
        "needs_more_context": more_context,
        "boundary_already_on_target": on_target,
        "certainty": certainty,
        "semantic_marker": marker,
    }


@dataclass
class FakeCaller:
    """Answer boundary requests from *answer*; record them in *requests*."""

    answer: Answer = default_answer
    provider: str = "openai"
    requests: list[tuple[str, str, str]] = field(default_factory=list)
    systems: list[Any] = field(default_factory=list)
    """The system message content of every request, as sent."""

    async def call(
        self, messages: Sequence[Any], request: StructuredRequest, *, label: str
    ) -> StructuredResult:
        system, user = str(messages[0].content), str(messages[1].content)
        self.requests.append((system, user, label))
        self.systems.append(messages[0].content)
        outcome = self.answer(user, label)
        if isinstance(outcome, BaseException):
            raise outcome
        text = outcome if isinstance(outcome, str) else json.dumps(outcome)
        data = outcome if isinstance(outcome, dict) else None
        return StructuredResult(
            response_format=request.response_format,
            text=text,
            data=data,
            error=None if data is not None else "invalid JSON",
            response=None,
            responses=(),
            enforced=True,
        )

    @property
    def users(self) -> list[str]:
        """Return the user text of every request."""
        return [user for _, user, _ in self.requests]
