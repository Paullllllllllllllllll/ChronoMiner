"""The decision model of a run: its HTTP client, its clients and its runner.

:class:`DecisionModel` serves one decision model id, dialect and endpoint
through the shared :class:`~chronominer.common.decision_runner.DecisionRunner`.
Each try of a request opens an attempt of the extension seam
(:func:`chronominer.ext.attempt`, client kind ``decision``) under the
caller's role, behind the dialect's rate limiter and the settings' retry
budgets; its usage, failed attempts included, counts toward the run's totals
under that role. The relevance pre-pass (role ``decide``) and the classify
commands (role ``classify``) use it.
"""

from __future__ import annotations

import contextlib
import os
from typing import Final

import httpx

from chronominer import ext
from chronominer.common.capture import async_capture_hooks
from chronominer.common.decision_client import DecisionClient
from chronominer.common.decision_runner import DecisionAttempt, DecisionRunner, Seam
from chronominer.common.decisions import Dialect, dialect_for
from chronominer.common.usage import Usage
from chronominer.llm.caller import CallEnv

__all__ = ["KIND", "DecisionModel"]

KIND: Final = "decision"
"""The client kind of every decision-model call."""


class DecisionModel:
    """One decision model of a run, asked through :attr:`runner`.

    Raises:
        OSError: None of the key variables of the decision provider is set.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str | None,
        endpoint: str | None,
        env: CallEnv,
        role: str,
    ) -> None:
        settings = env.settings
        self.model = model
        self.endpoint = endpoint
        self.env = env
        self.role = role
        self.dialect: Dialect = dialect_for(model, provider)
        variables = ext.key_variables(self.dialect, kind=KIND, settings=settings)
        if variables and not any(os.environ.get(name) for name in variables):
            raise OSError(f"the decision model needs the key variable {variables[0]}")
        self._http = httpx.AsyncClient(
            timeout=settings.decisions.timeout_s, event_hooks=async_capture_hooks()
        )
        self._clients: dict[str, DecisionClient] = {}
        seam = Seam(
            attempt=self._attempt,
            record=self._record,
            retry=settings.retry,
            gate=env.limiters.get(self.dialect),
        )
        self.runner = DecisionRunner(seam, model=model, dialect=self.dialect)

    def client_for(self, identity: ext.Identity) -> DecisionClient:
        """Return the client for an attempt identity, building it if new.

        The client takes the extension's client options of the identity.

        Raises:
            OSError: The identity's key variable is unset.
        """
        client = self._clients.get(identity.api_key_env)
        if client is None:
            options = ext.client_options(identity)
            key = options.api_key or os.environ.get(identity.api_key_env, "").strip()
            if not key:
                raise OSError(
                    f"the decision model needs the key variable {identity.api_key_env}"
                )
            client = DecisionClient(
                self._http,
                dialect=self.dialect,
                api_key=key,
                endpoint=options.base_url or self.endpoint,
            )
            self._clients[identity.api_key_env] = client
        return client

    def _attempt(self) -> DecisionAttempt:
        return ext.attempt(
            self.role,
            self.dialect,
            None,
            settings=self.env.settings,
            build_client=self.client_for,
            endpoint=None,
            model=self.model,
            kind=KIND,
        )

    def _record(self, usage: Usage) -> None:
        self.env.record(self.role, usage)

    async def aclose(self) -> None:
        """Close the HTTP client. Never raises."""
        with contextlib.suppress(Exception):
            await self._http.aclose()
