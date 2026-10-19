"""The client factory: LangChain chat models for each provider.

Supports OpenAI (Responses API), Anthropic, Google, OpenRouter and named
OpenAI-compatible custom endpoints. The run spec has already settled the
provider and stripped any ``provider:`` prefix from the model name; the
constructor kwargs come from :func:`~chronominer.llm.options.constructor_kwargs`.
Provider classes are imported inside the factory functions. SDK retries are
off (``max_retries=0``): the retry ladder of :mod:`chronominer.llm.caller`
sees every attempt.

Every chat model sends its requests through httpx clients that carry the
capture hooks of :mod:`chronominer.common.capture`, so an attempt's
request and response bodies reach its capture scope as they went over HTTP:
``ChatOpenAI`` takes them as ``http_client`` and ``http_async_client``;
``ChatAnthropic`` gets copies of its cached SDK clients on hooked httpx
clients; ``ChatGoogleGenerativeAI`` passes ``client_args`` to both httpx
clients of google-genai, with a transport that keeps async calls on httpx.
The httpx clients of one model share one TLS context.
"""

from __future__ import annotations

import functools
import logging
import os
import ssl
import warnings
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import httpx

from chronominer.common.capture import (
    acapture_response,
    async_capture_hooks,
    capture_hooks,
    capture_request,
    capture_response,
)
from chronominer.ext import ClientOptions
from chronominer.llm.options import constructor_kwargs
from chronominer.providers import PROVIDERS
from chronominer.settings import DEFAULT_API_KEYS, Settings
from chronominer.spec import ModelSpec

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "build_chat_model",
    "default_key_env",
    "get_api_key",
    "suppress_pydantic_serializer_warnings",
]

logger = logging.getLogger(__name__)


def default_key_env(provider: str) -> str | None:
    """Return the default key variable of *provider* (None for ``custom``)."""
    return DEFAULT_API_KEYS.get(provider)


def get_api_key(provider: str, api_key_env: str | None = None) -> str:
    """Return the API key from *api_key_env* or the provider's default variable.

    Raises:
        OSError: No variable is known, or it is unset.
    """
    env_key = api_key_env or default_key_env(provider)
    if env_key is None:
        raise OSError(
            "Custom endpoint API key not found: the endpoint names no "
            "api_key_env in the settings."
        )
    api_key = os.environ.get(env_key)
    if not api_key:
        raise OSError(
            f"API key not found for provider '{provider}'. "
            f"Please set the {env_key} environment variable."
        )
    return api_key


def suppress_pydantic_serializer_warnings() -> None:
    """Silence pydantic's "Pydantic serializer warnings" UserWarning.

    LangChain serializes OpenAI SDK response objects on every Responses API
    call, which raises this warning without signal. Only this message, only
    as a ``UserWarning`` from the pydantic package, is filtered.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"Pydantic serializer warnings",
        category=UserWarning,
        module=r"pydantic(\..*)?",
    )


def build_chat_model(
    model: ModelSpec,
    settings: Settings,
    *,
    api_key_env: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    options: ClientOptions | None = None,
) -> BaseChatModel:
    """Build the chat model of *model*, without SDK retries.

    *api_key_env* names the key variable (the attempt identity's); unset, the
    settings name it. *overrides* replace constructor kwargs. *options* are
    the client options of the attempt identity: their ``api_key`` replaces
    the key variable's value; ``base_url`` and ``payload_hook`` apply to
    OpenAI-compatible models only.

    Raises:
        OSError: The key variable is unset and *options* give no key.
        ImportError: The provider package is missing.
        ValueError: The provider is unknown or a custom model names no
            endpoint.
        SettingsError: The named endpoint is not defined.
    """
    provider = model.provider
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    options = options or ClientOptions()
    if options.api_key:
        api_key = options.api_key
    else:
        env_key = api_key_env or settings.api_key_env(provider, model.endpoint)
        api_key = get_api_key(provider, env_key)
    kwargs = {**constructor_kwargs(model, settings), **dict(overrides or {})}
    logger.debug("Creating %s model: %s", provider, model.model)
    return _FACTORIES[provider](api_key, kwargs, options)


def _missing(package: str, what: str) -> ImportError:
    return ImportError(
        f"{package} is required for {what}. Install with: pip install {package}"
    )


def _timeout_kwarg(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    timeout = kwargs.get("timeout")
    return {} if timeout is None else {"timeout": timeout}


def _ssl_context() -> ssl.SSLContext:
    """Return the TLS context the sync and async httpx clients of a model share.

    It is the context httpx builds for each client when given none. Loading
    its CA bundle is most of the time a client takes to build, so the
    clients of one model load it once.
    """
    return httpx.create_ssl_context()


@functools.cache
def _hooked_chat_openai() -> Callable[..., BaseChatModel]:
    """Return a ``ChatOpenAI`` subclass that passes payloads through a hook."""
    from langchain_core.language_models import LanguageModelInput
    from langchain_openai import ChatOpenAI
    from pydantic import Field

    class HookedChatOpenAI(ChatOpenAI):
        """``ChatOpenAI`` whose request payloads pass ``payload_hook``."""

        payload_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = Field(
            default=None, exclude=True
        )

        def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload: dict[str, Any] = super()._get_request_payload(
                input_, stop=stop, **kwargs
            )
            if self.payload_hook is None:
                return payload
            return self.payload_hook(payload)

    return HookedChatOpenAI


def _openai_compatible(
    api_key: str, kwargs: dict[str, Any], options: ClientOptions
) -> BaseChatModel:
    """Create a ``ChatOpenAI`` for OpenAI, OpenRouter or a custom endpoint.

    Its httpx clients carry the capture hooks and the kwargs' timeout.
    *options* may set the base URL and a request payload hook.
    """
    try:
        import openai
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise _missing("langchain-openai", "OpenAI-compatible models") from None
    suppress_pydantic_serializer_warnings()
    timeout = _timeout_kwarg(kwargs)
    verify = _ssl_context()
    clients = {
        "http_client": openai.DefaultHttpxClient(
            event_hooks=capture_hooks(), verify=verify, **timeout
        ),
        "http_async_client": openai.DefaultAsyncHttpxClient(
            event_hooks=async_capture_hooks(), verify=verify, **timeout
        ),
    }
    fields: dict[str, Any] = {**clients, **kwargs}
    if options.base_url:
        fields["base_url"] = options.base_url
    if options.payload_hook is not None:
        hooked = _hooked_chat_openai()
        return hooked(api_key=api_key, payload_hook=options.payload_hook, **fields)
    return ChatOpenAI(api_key=api_key, **fields)  # type: ignore[arg-type]


def _anthropic(
    api_key: str, kwargs: dict[str, Any], options: ClientOptions
) -> BaseChatModel:
    """Create a ``ChatAnthropic`` whose SDK clients carry the capture hooks."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise _missing("langchain-anthropic", "Anthropic models") from None
    model = ChatAnthropic(api_key=api_key, **kwargs)  # type: ignore[arg-type]
    _hook_anthropic(model)
    return model


def _hook_anthropic(model: Any) -> None:
    """Replace the SDK clients a ``ChatAnthropic`` caches with hooked copies.

    langchain-anthropic builds them in the cached properties ``_client`` and
    ``_async_client`` on httpx clients shared between instances; the copies
    keep every SDK option and take httpx clients of their own. A model
    without these properties (a test stand-in) is left as it is.
    """
    import anthropic

    sync_client = getattr(model, "_client", None)
    async_client = getattr(model, "_async_client", None)
    if not isinstance(sync_client, anthropic.Client) or not isinstance(
        async_client, anthropic.AsyncClient
    ):
        return
    verify = _ssl_context()
    model.__dict__["_client"] = sync_client.with_options(
        http_client=anthropic.DefaultHttpxClient(
            timeout=sync_client.timeout, event_hooks=capture_hooks(), verify=verify
        )
    )
    model.__dict__["_async_client"] = async_client.with_options(
        http_client=anthropic.DefaultAsyncHttpxClient(
            timeout=async_client.timeout,
            event_hooks=async_capture_hooks(),
            verify=verify,
        )
    )


def capture_either(response: httpx.Response) -> Awaitable[None] | None:
    """Capture hook for an ``httpx.Client`` and an ``httpx.AsyncClient`` alike.

    An async client awaits the returned coroutine; a sync client reads the
    body at once.
    """
    if isinstance(response.stream, httpx.AsyncByteStream):
        return acapture_response(response)
    capture_response(response)
    return None


class EitherTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """An HTTP transport that serves a sync and an async httpx client.

    google-genai builds its sync and async httpx clients from one set of
    ``client_args``; a transport among them also keeps its async calls on
    httpx where aiohttp is installed. It hands each request body to the
    capture scope, which one request hook cannot do for both clients.
    """

    def __init__(self) -> None:
        verify = _ssl_context()
        self._sync = httpx.HTTPTransport(verify=verify)
        self._async = httpx.AsyncHTTPTransport(verify=verify)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        capture_request(request)
        return self._sync.handle_request(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        capture_request(request)
        return await self._async.handle_async_request(request)

    def close(self) -> None:
        self._sync.close()

    async def aclose(self) -> None:
        await self._async.aclose()


def _google(
    api_key: str, kwargs: dict[str, Any], options: ClientOptions
) -> BaseChatModel:
    """Create a ``ChatGoogleGenerativeAI`` whose httpx clients capture bodies."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise _missing("langchain-google-genai", "Google models") from None
    client_args = {
        "transport": EitherTransport(),
        "event_hooks": {"response": [capture_either]},
        **dict(kwargs.get("client_args") or {}),
    }
    return ChatGoogleGenerativeAI(
        google_api_key=api_key, **{**kwargs, "client_args": client_args}
    )


type _Factory = Callable[[str, dict[str, Any], ClientOptions], BaseChatModel]

_FACTORIES: Mapping[str, _Factory] = MappingProxyType(
    {
        "openai": _openai_compatible,
        "anthropic": _anthropic,
        "google": _google,
        "openrouter": _openai_compatible,
        "custom": _openai_compatible,
    }
)
