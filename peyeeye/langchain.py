"""LangChain integration — redact prompts before the LLM, rehydrate outputs.

Drop-in adapter that wraps any LangChain ``Runnable`` (chat model, LLM, or
chain) in a peyeeye session so PII never reaches the model and tokens never
leak back to the user::

    from langchain_openai import ChatOpenAI
    from peyeeye import Peyeeye
    from peyeeye.langchain import with_peyeeye

    peyeeye = Peyeeye(api_key=os.environ["PEYEEYE_KEY"])
    model = with_peyeeye(ChatOpenAI(model="gpt-4o-mini"), client=peyeeye)

    print(model.invoke("Hi, I'm Ada — email me at ada@a-e.com"))

Each ``invoke`` opens a fresh peyeeye session, so deterministic tokens hold
within one call and never bleed across requests.

The module has **no hard dependency on LangChain** — install ``langchain-core``
only if you want the returned object to also be a proper ``Runnable`` (and
therefore composable with ``|`` in LCEL pipelines). Without LangChain installed
you still get a callable wrapper that accepts strings or chat-message lists.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Union

from typing import Dict

from .client import Peyeeye, Shield

try:
    # Optional: enables LCEL composition (`prompt | model | parser`).
    from langchain_core.runnables import Runnable  # type: ignore

    _RunnableBase: Any = Runnable
    _HAS_LANGCHAIN = True
except Exception:  # pragma: no cover - exercised in CI without langchain
    _RunnableBase = object
    _HAS_LANGCHAIN = False


__all__ = ["PeyeeyeRunnable", "with_peyeeye"]


class PeyeeyeRunnable(_RunnableBase):
    """Wrap a LangChain Runnable (or any ``.invoke``-able) with peyeeye.

    - Redacts string prompt content and ``HumanMessage`` / ``SystemMessage``
      text using a fresh session per invocation.
    - Passes the redacted prompt to the inner runnable.
    - Rehydrates tokens in the response (strings, AIMessage ``.content``,
      or an object with a ``.content`` attribute).

    Attributes:
        inner: the wrapped model / chain / runnable.
        client: the :class:`~peyeeye.Peyeeye` client.
        stateless: if True, opens stateless sealed sessions — peyeeye stores
            no mapping server-side.
    """

    def __init__(
        self,
        inner: Any,
        *,
        client: Peyeeye,
        stateless: bool = False,
    ) -> None:
        self.inner = inner
        self.client = client
        self.stateless = stateless

    # ---- sync path --------------------------------------------------------

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        shield = self._open()
        try:
            redacted = _redact_input(input, shield)
            output = self.inner.invoke(redacted, config, **kwargs) if config is not None else self.inner.invoke(redacted, **kwargs)
            return _rehydrate_output(output, shield)
        finally:
            _safe_destroy(shield)

    def batch(
        self,
        inputs: Iterable[Any],
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Any]:
        # Each batch item gets its own session to avoid cross-prompt token
        # collisions that would confuse the LLM.
        return [self.invoke(item, config, **kwargs) for item in inputs]

    # ---- async path -------------------------------------------------------

    async def ainvoke(
        self, input: Any, config: Optional[Any] = None, **kwargs: Any
    ) -> Any:
        # Peyeeye's sync client is already non-blocking relative to the LLM
        # round-trip (one POST each). For LangChain's async codepath we still
        # call the inner runnable's ``ainvoke`` to avoid blocking the event
        # loop on the model call itself.
        shield = self._open()
        try:
            redacted = _redact_input(input, shield)
            if hasattr(self.inner, "ainvoke"):
                output = await self.inner.ainvoke(redacted, config, **kwargs) if config is not None else await self.inner.ainvoke(redacted, **kwargs)
            else:
                output = self.inner.invoke(redacted, config, **kwargs) if config is not None else self.inner.invoke(redacted, **kwargs)
            return _rehydrate_output(output, shield)
        finally:
            _safe_destroy(shield)

    # ---- helpers ----------------------------------------------------------

    def _open(self) -> Shield:
        opts: Dict[str, Any] = {"locale": "auto"}
        if self.stateless:
            opts["session"] = "stateless"
        return Shield(self.client, opts)


def with_peyeeye(
    inner: Any,
    *,
    client: Peyeeye,
    stateless: bool = False,
) -> PeyeeyeRunnable:
    """Convenience factory mirroring the rest of the SDK's ``with_*`` style."""
    return PeyeeyeRunnable(inner, client=client, stateless=stateless)


# ---------------------------------------------------------------------------
# Redact / rehydrate helpers — operate on the handful of shapes LangChain
# users actually pass around: plain strings, chat-message lists, or objects
# with a ``.content`` attribute (BaseMessage, AIMessage, etc.).
# ---------------------------------------------------------------------------


def _redact_input(value: Any, shield: Shield) -> Any:
    if isinstance(value, str):
        return shield.redact(value)
    if isinstance(value, list):
        return [_redact_message(m, shield) for m in value]
    if isinstance(value, tuple):
        # e.g. ("human", "Hi, I'm Ada") — LangChain's shorthand tuple message.
        return _redact_tuple_message(value, shield)
    if isinstance(value, dict):
        return _redact_dict_message(value, shield)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return _replace_content(value, shield.redact(content))
    return value


def _redact_message(msg: Any, shield: Shield) -> Any:
    if isinstance(msg, str):
        return shield.redact(msg)
    if isinstance(msg, tuple):
        return _redact_tuple_message(msg, shield)
    if isinstance(msg, dict):
        return _redact_dict_message(msg, shield)
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return _replace_content(msg, shield.redact(content))
    if isinstance(content, list):
        # Multi-modal content: only touch text parts.
        new_parts: List[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                new_parts.append({**part, "text": shield.redact(part.get("text", ""))})
            else:
                new_parts.append(part)
        return _replace_content(msg, new_parts)
    return msg


def _redact_tuple_message(value: tuple, shield: Shield) -> tuple:
    if len(value) == 2 and isinstance(value[1], str):
        return (value[0], shield.redact(value[1]))
    return value


def _redact_dict_message(value: dict, shield: Shield) -> dict:
    out = dict(value)
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = shield.redact(content)
    elif isinstance(content, list):
        new_parts: List[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                new_parts.append({**part, "text": shield.redact(part.get("text", ""))})
            else:
                new_parts.append(part)
        out["content"] = new_parts
    return out


def _replace_content(message: Any, new_content: Any) -> Any:
    """Return a message with ``content`` replaced, preserving the type."""
    if hasattr(message, "copy"):
        try:
            # Pydantic v1/v2 BaseMessage: .copy(update=...) stays in-type.
            return message.copy(update={"content": new_content})  # type: ignore[call-arg]
        except TypeError:
            pass
    if hasattr(message, "__class__") and hasattr(message, "content"):
        try:
            clone = message.__class__(content=new_content)
            for k, v in getattr(message, "__dict__", {}).items():
                if k != "content":
                    try:
                        setattr(clone, k, v)
                    except Exception:
                        pass
            return clone
        except Exception:
            message.content = new_content  # best-effort mutate
            return message
    return new_content


def _rehydrate_output(value: Any, shield: Shield) -> Any:
    if isinstance(value, str):
        return shield.rehydrate(value)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return _replace_content(value, shield.rehydrate(content))
    if isinstance(content, list):
        new_parts: List[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                new_parts.append(
                    {**part, "text": shield.rehydrate(part.get("text", ""))}
                )
            else:
                new_parts.append(part)
        return _replace_content(value, new_parts)
    return value


def _safe_destroy(shield: Shield) -> None:
    """Best-effort DELETE of the session; never raise through to the caller."""
    try:
        sid = shield.session_id
        if sid:
            shield._client.delete_session(sid)  # noqa: SLF001 - intentional
    except Exception:
        pass
