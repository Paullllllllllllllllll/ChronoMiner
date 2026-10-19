"""SDK clients of the batch APIs, built and called through attempts.

Every batch API call runs inside :func:`chronominer.ext.attempt`: the
attempt's identity names the provider's key variable from the settings, the
client is built through the attempt with the constructors of
:mod:`chronominer.common.batch`, and a failure, on entry or of the
call, is passed to ``on_error``, whose decision drives the retry ladder of
:func:`~chronominer.common.retry.call_with_retry` with the retry policy
of the settings. Status, result, list and cancel calls are idempotent and
retried; a submission runs once, since a retried create can duplicate a
batch. Clients are kept per key variable, so one command builds each at most
once. A listing that names nothing raises
:class:`~chronominer.common.batch_reconcile.NotListed` inside its attempt, so
``on_error`` sees it like a not-found error.

Every client carries the response hooks of
:mod:`chronominer.common.capture`, and each attempt runs in its own
capture scope, so the response bodies of a call reach the caller verbatim.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from chronominer import ext
from chronominer.common.batch import (
    BatchBackend,
    BatchHandle,
    BatchResultItem,
    BatchStatusInfo,
    PartUpload,
    create_backend,
    create_client,
)
from chronominer.common.capture import (
    CaptureScope,
    capture_attempt,
    capture_hooks,
)
from chronominer.common.retry import RetryHooks, call_with_retry
from chronominer.common.usage import Usage
from chronominer.llm.retrying import AttemptDecisions
from chronominer.settings import Settings, SettingsError

__all__ = ["BatchClients", "MissingKeyError", "results_of", "status_of"]

T = TypeVar("T")

logger = logging.getLogger(__name__)


def status_of(handle: BatchHandle) -> Callable[[BatchBackend], BatchStatusInfo]:
    """Return the operation that asks a backend for the status of *handle*."""
    return lambda backend: backend.status(handle)


def results_of(handle: BatchHandle) -> Callable[[BatchBackend], list[BatchResultItem]]:
    """Return the operation that downloads the results of *handle*."""
    return lambda backend: backend.results(handle)


class MissingKeyError(OSError):
    """The key variable of a provider is unset."""


class BatchClients:
    """Batch backends of one command, one client per key variable.

    *role* is the attempt role of every call (``extract``). *sleep*
    replaces ``asyncio.sleep`` between retries. *args* holds the parsed
    command line, if any, for the extension's key variables.
    """

    def __init__(
        self,
        settings: Settings,
        role: str,
        *,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        args: argparse.Namespace | None = None,
    ) -> None:
        self.settings = settings
        self.role = role
        self.args = args
        self._policy = settings.retry.policy()
        self._sleep = sleep
        self._clients: dict[tuple[str, str], Any] = {}

    def close(self) -> None:
        """Close the SDK clients built so far; a later call builds them anew.

        A client without ``close`` is left alone and a failing close is
        logged, so closing never raises.
        """
        built, self._clients = list(self._clients.values()), {}
        for client in built:
            close = getattr(client, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - closing must not fail
                logger.warning("Could not close a batch client: %s", exc)

    def has_key(self, provider: str) -> bool:
        """Whether a key variable the extension names for *provider* is set.

        True when the extension names none; False when the settings name no
        key variable for the provider.
        """
        try:
            variables = ext.key_variables(
                provider, kind="batch", settings=self.settings, args=self.args
            )
        except SettingsError:
            return False
        return not variables or any(os.environ.get(name) for name in variables)

    def planner(self, provider: str) -> BatchBackend:
        """Return a backend of *provider* without a client, to plan parts.

        Planning wraps and splits the requests locally; it makes no call.
        """
        return create_backend(provider, None)

    def _client(self, identity: ext.Identity) -> Any:
        """Build the client of *identity*; its client options may set the key."""
        cache_key = (identity.provider, identity.api_key_env)
        client = self._clients.get(cache_key)
        if client is None:
            options = ext.client_options(identity)
            api_key = options.api_key or os.environ.get(identity.api_key_env)
            if not api_key:
                raise MissingKeyError(
                    f"The API key variable {identity.api_key_env} of provider "
                    f"{identity.provider} is not set."
                )
            timeouts = self.settings.timeouts
            client = create_client(
                identity.provider,
                api_key,
                timeout=timeouts.request,
                connect_timeout=timeouts.connect,
                event_hooks=capture_hooks(),
            )
            self._clients[cache_key] = client
        return client

    async def call(
        self,
        provider: str,
        operation: Callable[[BatchBackend], T],
        *,
        usage: Callable[[T], Usage | None] | None = None,
        model: str | None = None,
        retry: bool = True,
        label: str = "",
        on_capture: Callable[[CaptureScope], None] | None = None,
    ) -> T:
        """Run *operation* on the backend of *provider* inside an attempt.

        The operation runs in a worker thread; *usage* returns the usage to
        commit from its result. *model* enters the attempt identity. With
        *retry*, a failure is retried as the attempt's ``on_error`` decides
        within the settings' retry budget; pass ``retry=False`` for calls
        that are not idempotent. The last error propagates. *label* names
        the call in the retry log. *on_capture* receives each attempt's
        capture scope, with the response bodies it saw, after the attempt
        has succeeded or failed.
        """
        decisions = AttemptDecisions()

        async def one_attempt() -> T:
            attempt = ext.attempt(
                self.role,
                provider,
                None,
                settings=self.settings,
                build_client=self._client,
                model=model,
                kind="batch",
            )
            async with contextlib.AsyncExitStack() as stack:
                entered = await decisions.enter(stack, attempt)
                backend = create_backend(provider, entered.client)
                scope = stack.enter_context(capture_attempt())
                if on_capture is not None:
                    stack.callback(on_capture, scope)
                try:
                    result = await asyncio.to_thread(operation, backend)
                except Exception as exc:
                    decisions.failed(entered, exc)
                    raise
                if usage is not None:
                    entered.commit(usage(result))
                return result

        if not retry:
            return await one_attempt()

        return await call_with_retry(
            one_attempt,
            policy=self._policy,
            hooks=RetryHooks(on_error=decisions.decide, sleep=self._sleep),
            label=label or f"{provider} batch call",
        )

    async def delete_upload(
        self, upload: PartUpload, *, model: str | None = None
    ) -> bool:
        """Delete the provider file of *upload*; return whether it is gone.

        An upload sent inline has no file. A failure is logged, not raised.
        """
        if not upload.file_id:
            return True
        try:
            await self.call(
                upload.provider,
                lambda backend: backend.delete_upload(upload),
                model=model,
                label=f"deletion of upload {upload.file_id}",
            )
        except Exception as exc:  # noqa: BLE001 - a kept file is only reported
            logger.warning("Could not delete upload %s: %s", upload.file_id, exc)
            return False
        return True

    async def cleanup(self, handle: BatchHandle, *, model: str | None = None) -> None:
        """Delete the provider's input and result files of a finished batch.

        Call it only after the results are stored; a failure is logged.
        """
        try:
            await self.call(
                handle.provider,
                lambda backend: backend.cleanup(handle),
                model=model,
                label=f"file cleanup of batch {handle.batch_id}",
            )
        except Exception as exc:  # noqa: BLE001 - a kept file is only reported
            logger.warning(
                "Could not delete the files of batch %s: %s", handle.batch_id, exc
            )
