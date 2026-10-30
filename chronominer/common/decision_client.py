"""Send decision requests over an injected ``httpx`` client.

:class:`DecisionClient` posts one :func:`~.decisions.request_body` to the
dialect's endpoint and returns the parsed answers. It does not retry:
``retry.call_with_retry`` is the single retry authority and classifies the
``httpx`` errors this client lets through. The caller owns the HTTP client, so
the capture hooks installed on it record every call. The key travels only in
the Authorization header and appears in no repr.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from .decisions import (
    ENDPOINTS,
    DecisionResult,
    Dialect,
    MalformedAnswer,
    Part,
    Question,
    parse_response,
    request_body,
)

__all__ = ["DecisionClient", "DecisionStatusError"]


class DecisionStatusError(httpx.HTTPStatusError):
    """An error status from a decision endpoint, readable by the retry ladder."""

    @property
    def status_code(self) -> int:
        """The response's HTTP status."""
        return self.response.status_code


class DecisionClient:
    """Ask a decision model in one dialect.

    *endpoint* replaces the dialect's default URL; *timeout* (seconds) is sent
    with every request, None keeps the HTTP client's own.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        dialect: Dialect,
        api_key: str,
        endpoint: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("a decision client needs an API key")
        self._http = http
        self.dialect: Dialect = dialect
        self.endpoint = endpoint or ENDPOINTS[dialect]
        self._api_key = api_key
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"DecisionClient(dialect={self.dialect!r}, endpoint={self.endpoint!r})"

    async def decide(
        self, model: str, parts: Sequence[Part], questions: Sequence[Question]
    ) -> DecisionResult:
        """Return the model's answers to *questions* about *parts*.

        Raises:
            DecisionStatusError: The endpoint answered with an error status.
            httpx.TransportError: The request did not complete.
            MalformedAnswer: The response does not answer the questions.
            ValueError: The request itself is invalid (see ``request_body``).
        """
        body = request_body(self.dialect, model, parts, questions)
        response = await self._http.post(
            self.endpoint,
            json=body,
            headers=self._headers,
            timeout=(
                httpx.USE_CLIENT_DEFAULT if self._timeout is None else self._timeout
            ),
        )
        if response.is_error:
            raise DecisionStatusError(
                f"decision endpoint answered {response.status_code} "
                f"{response.reason_phrase}",
                request=response.request,
                response=response,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise MalformedAnswer("the response is not JSON") from exc
        try:
            return parse_response(self.dialect, payload, questions)
        except MalformedAnswer as exc:  # the text may echo the response
            raise MalformedAnswer(
                str(exc).replace(self._api_key, "***"), exc.usage
            ) from None
