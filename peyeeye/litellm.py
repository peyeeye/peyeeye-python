"""LiteLLM integration — redact prompts before the LLM, rehydrate after.

Two ways to plug peyeeye into a LiteLLM-based app:

**1. Wrap the completion function (direct usage):**

    import litellm
    from peyeeye import Peyeeye
    from peyeeye.litellm import with_peyeeye

    peyeeye = Peyeeye(api_key=os.environ["PEYEEYE_KEY"])
    completion = with_peyeeye(litellm.completion, client=peyeeye)

    resp = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi, I'm Ada"}],
    )
    print(resp.choices[0].message.content)  # already rehydrated

**2. Register a handler (LiteLLM proxy / callbacks):**

    import litellm
    from peyeeye.litellm import PeyeeyeHandler

    handler = PeyeeyeHandler(client=peyeeye)
    litellm.callbacks = [handler]

The module has **no hard dependency on LiteLLM** — the handler class falls
back to ``object`` when ``litellm`` isn't installed, so it can be imported in
dev/test environments without pulling the dep.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

from .client import Peyeeye, Shield

try:  # Optional: real CustomLogger base when litellm is installed.
    from litellm.integrations.custom_logger import CustomLogger  # type: ignore

    _LoggerBase: Any = CustomLogger
    _HAS_LITELLM = True
except Exception:  # pragma: no cover - exercised in CI without litellm
    _LoggerBase = object
    _HAS_LITELLM = False


__all__ = ["PeyeeyeHandler", "with_peyeeye", "redact_messages", "rehydrate_response"]


F = TypeVar("F", bound=Callable[..., Any])


def with_peyeeye(
    completion_fn: F,
    *,
    client: Peyeeye,
    stateless: bool = False,
) -> F:
    """Wrap ``litellm.completion`` / ``litellm.acompletion`` with peyeeye.

    Each call opens a fresh session, redacts the ``messages`` (both plain
    ``content`` strings and multimodal content lists), calls the inner
    function with the redacted payload, then rehydrates the response's
    ``choices[i].message.content`` before returning it.

    Works for both sync and async functions — the returned wrapper matches
    the inner function's sync-ness.
    """

    def _run(args: tuple, kwargs: dict) -> tuple[Shield, tuple, dict]:
        shield = _open_shield(client, stateless=stateless)
        new_kwargs = dict(kwargs)
        if "messages" in new_kwargs:
            new_kwargs["messages"] = redact_messages(new_kwargs["messages"], shield)
        return shield, args, new_kwargs

    if asyncio.iscoroutinefunction(completion_fn):

        @functools.wraps(completion_fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            shield, a, kw = _run(args, kwargs)
            try:
                resp = await completion_fn(*a, **kw)
                return rehydrate_response(resp, shield)
            finally:
                _safe_destroy(shield)

        return awrapper  # type: ignore[return-value]

    @functools.wraps(completion_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        shield, a, kw = _run(args, kwargs)
        try:
            resp = completion_fn(*a, **kw)
            return rehydrate_response(resp, shield)
        finally:
            _safe_destroy(shield)

    return wrapper  # type: ignore[return-value]


class PeyeeyeHandler(_LoggerBase):
    """LiteLLM ``CustomLogger``-compatible handler for PII redaction.

    Mutates ``data["messages"]`` in-place on the pre-call hook and rehydrates
    ``response.choices[].message.content`` on the post-call hook. Each request
    gets a fresh session keyed on the request's ``litellm_call_id`` so
    concurrent calls don't trample each other's token maps.

    Register with::

        import litellm
        litellm.callbacks = [PeyeeyeHandler(client=peyeeye)]
    """

    def __init__(self, *, client: Peyeeye, stateless: bool = False) -> None:
        if _HAS_LITELLM:
            try:
                super().__init__()  # type: ignore[misc]
            except TypeError:  # pragma: no cover
                pass
        self.client = client
        self.stateless = stateless
        self._shields: Dict[str, Shield] = {}

    # ---- pre-call ---------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        return self._pre_call(data)

    def pre_call_hook(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._pre_call(data)

    # ---- post-call --------------------------------------------------------

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        return self._post_call(data, response)

    def post_call_success_hook(self, data: Dict[str, Any], response: Any) -> Any:
        return self._post_call(data, response)

    # ---- internals --------------------------------------------------------

    def _pre_call(self, data: Dict[str, Any]) -> Dict[str, Any]:
        key = _call_key(data)
        shield = _open_shield(self.client, stateless=self.stateless)
        self._shields[key] = shield
        if "messages" in data:
            data["messages"] = redact_messages(data["messages"], shield)
        return data

    def _post_call(self, data: Dict[str, Any], response: Any) -> Any:
        key = _call_key(data)
        shield = self._shields.pop(key, None)
        if shield is None:
            return response
        try:
            return rehydrate_response(response, shield)
        finally:
            _safe_destroy(shield)


# ---------------------------------------------------------------------------
# Helpers exposed for users who want to wire things up themselves.
# ---------------------------------------------------------------------------


def redact_messages(
    messages: List[Dict[str, Any]],
    shield: Shield,
) -> List[Dict[str, Any]]:
    """Return a new message list with ``content`` redacted.

    Handles both plain-string content and the OpenAI-style multimodal content
    list (``[{"type": "text", "text": …}, {"type": "image_url", …}]``). Image
    and other non-text parts pass through untouched.
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, str):
            new_msg["content"] = shield.redact(content)
        elif isinstance(content, list):
            new_parts: List[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    new_parts.append(
                        {**part, "text": shield.redact(part.get("text", ""))}
                    )
                else:
                    new_parts.append(part)
            new_msg["content"] = new_parts
        out.append(new_msg)
    return out


def rehydrate_response(response: Any, shield: Shield) -> Any:
    """Rehydrate tokens in a LiteLLM ``ModelResponse``-shaped object.

    Handles ``response.choices[i].message.content`` (string or multimodal
    list) and dict-shaped responses. Returns the same object (mutated) or,
    for strings, the rehydrated string.
    """
    if isinstance(response, str):
        return shield.rehydrate(response)

    choices = _get(response, "choices")
    if choices is None:
        # Unknown shape — leave alone.
        return response

    for choice in choices:
        message = _get(choice, "message")
        if message is None:
            continue
        content = _get(message, "content")
        if isinstance(content, str):
            _set(message, "content", shield.rehydrate(content))
        elif isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    new_parts.append(
                        {**part, "text": shield.rehydrate(part.get("text", ""))}
                    )
                else:
                    new_parts.append(part)
            _set(message, "content", new_parts)
    return response


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _open_shield(client: Peyeeye, *, stateless: bool) -> Shield:
    opts: Dict[str, Any] = {"locale": "auto"}
    if stateless:
        opts["session"] = "stateless"
    return Shield(client, opts)


def _safe_destroy(shield: Shield) -> None:
    try:
        sid = shield.session_id
        if sid:
            shield._client.delete_session(sid)  # noqa: SLF001 - intentional
    except Exception:
        pass


def _call_key(data: Dict[str, Any]) -> str:
    return str(
        data.get("litellm_call_id")
        or data.get("call_id")
        or id(data)
    )


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        try:
            setattr(obj, name, value)
        except Exception:
            pass
