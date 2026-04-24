"""Tests for the LangChain integration (peyeeye.langchain).

Uses the same MockTransport-based router the rest of the SDK tests use, plus
hand-rolled stub runnables so we don't need ``langchain-core`` installed in
the SDK's dev env. The integration module itself falls back to ``object`` as
its base class when LangChain is unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

from peyeeye import Peyeeye
from peyeeye.langchain import PeyeeyeRunnable, with_peyeeye
from tests.conftest import Router, json_response


# ---------------------------------------------------------------------------
# Mock peyeeye backend
# ---------------------------------------------------------------------------


class _Backend:
    """Minimal redact/rehydrate backend keyed on session id."""

    def __init__(self) -> None:
        self.session_seq = 0
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _new_session(self) -> str:
        self.session_seq += 1
        sid = f"ses_{self.session_seq}"
        self.sessions[sid] = {"counter": 0, "tokens": {}, "reverse": {}}
        return sid

    def _tok(self, sid: str, raw: str, kind: str) -> str:
        s = self.sessions[sid]
        if raw in s["tokens"]:
            return s["tokens"][raw]
        s["counter"] += 1
        tok = f"[{kind}_{s['counter']}]"
        s["tokens"][raw] = tok
        s["reverse"][tok] = raw
        return tok

    def redact(self, sid: str, text: str) -> str:
        # Toy rules — enough for the tests.
        import re

        out = re.sub(r"\bAda\b", lambda _: self._tok(sid, "Ada", "PERSON"), text)
        out = re.sub(r"\bBen\b", lambda _: self._tok(sid, "Ben", "PERSON"), out)
        out = re.sub(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            lambda m: self._tok(sid, m.group(0), "EMAIL"),
            out,
        )
        return out


def _mount_backend(router: Router) -> _Backend:
    backend = _Backend()

    @router.route("POST", "/v1/redact")
    def _redact(req: httpx.Request) -> httpx.Response:
        body = req.read()
        import json

        payload = json.loads(body)
        requested = payload.get("session")
        stateless = requested == "stateless"
        if not requested or stateless:
            sid = backend._new_session()
        else:
            sid = requested
        text = payload["text"]
        if isinstance(text, list):
            redacted = [backend.redact(sid, t) for t in text]
        else:
            redacted = backend.redact(sid, text)
        resp: Dict[str, Any] = {
            "redacted": redacted,
            "session": sid,
            "entities": [],
            "latency_ms": 1,
            "expires_at": "2099-01-01T00:00:00Z",
        }
        if stateless:
            # Pretend the session id is itself a sealed blob — good enough
            # for the tests because rehydrate's handler looks up by session.
            resp["rehydration_key"] = sid
        return json_response(resp)

    @router.route("POST", "/v1/rehydrate")
    def _rehydrate(req: httpx.Request) -> httpx.Response:
        import json
        import re

        payload = json.loads(req.read())
        sid = payload["session"]
        sess = backend.sessions.get(sid)
        if sess is None:
            return httpx.Response(404, json={"code": "session_not_found"})
        text = re.sub(
            r"\[[A-Z]+_\d+\]",
            lambda m: sess["reverse"].get(m.group(0), m.group(0)),
            payload["text"],
        )
        return json_response({"text": text, "replaced": 1, "latency_ms": 1})

    @router.route("DELETE", "/v1/sessions/ses_1")
    def _delete(req: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    # Accept deletes for any session id without 404s.
    # (Can't easily do regex matching in the Router — so route each expected id.)
    for i in range(1, 50):

        @router.route("DELETE", f"/v1/sessions/ses_{i}")
        def _d(req: httpx.Request, _i: int = i) -> httpx.Response:
            return httpx.Response(204)

    return backend


# ---------------------------------------------------------------------------
# Stub LangChain primitives (we don't want a real langchain dep in SDK tests)
# ---------------------------------------------------------------------------


class _FakeMessage:
    """Stand-in for LangChain's BaseMessage / AIMessage."""

    def __init__(self, content: Any, role: str = "ai") -> None:
        self.content = content
        self.role = role

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_FakeMessage(role={self.role!r}, content={self.content!r})"


class _EchoRunnable:
    """Fake LLM that echoes the last user message back, preserving tokens.

    Records what it received so tests can assert the model never saw raw PII.
    """

    def __init__(self) -> None:
        self.last_input: Any = None

    def invoke(self, input: Any, config: Any = None) -> _FakeMessage:  # noqa: A002
        self.last_input = input
        text = _extract_text(input)
        return _FakeMessage(content=f"You said: {text}")

    async def ainvoke(self, input: Any, config: Any = None) -> _FakeMessage:  # noqa: A002
        return self.invoke(input, config)


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, tuple) and len(last) == 2:
            return str(last[1])
        if isinstance(last, dict):
            c = last.get("content")
            if isinstance(c, str):
                return c
        c = getattr(last, "content", None)
        if isinstance(c, str):
            return c
    c = getattr(value, "content", None)
    if isinstance(c, str):
        return c
    return str(value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_string_roundtrip_model_never_sees_raw_pii(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    echo = _EchoRunnable()
    wrapped = with_peyeeye(echo, client=client)

    out = wrapped.invoke("Hi, I'm Ada — email ada@a-e.com")

    # The inner "LLM" only saw redacted text.
    assert isinstance(echo.last_input, str)
    assert "Ada" not in echo.last_input
    assert "ada@a-e.com" not in echo.last_input
    assert "[PERSON_" in echo.last_input
    assert "[EMAIL_" in echo.last_input

    # Output is rehydrated back to the raw values.
    assert isinstance(out, _FakeMessage)
    assert "Ada" in out.content
    assert "ada@a-e.com" in out.content
    assert "[PERSON_" not in out.content


def test_chat_message_list_input_redacts_content_of_each_message(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    echo = _EchoRunnable()
    wrapped = with_peyeeye(echo, client=client)

    messages = [
        _FakeMessage(role="system", content="You are helpful."),
        _FakeMessage(role="human", content="Ben asked me to reach Ada"),
    ]
    out = wrapped.invoke(messages)

    # Each message's content was redacted before the model saw it.
    sent = echo.last_input
    assert isinstance(sent, list)
    assert sent[0].content == "You are helpful."
    assert "Ada" not in sent[1].content and "Ben" not in sent[1].content
    assert "[PERSON_1]" in sent[1].content and "[PERSON_2]" in sent[1].content

    # Rehydrated output has originals back.
    assert "Ada" in out.content and "Ben" in out.content


def test_tuple_shorthand_messages_are_redacted(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    echo = _EchoRunnable()
    wrapped = with_peyeeye(echo, client=client)

    out = wrapped.invoke([("system", "Be kind."), ("human", "Hi Ada")])
    sent = echo.last_input
    assert sent[0] == ("system", "Be kind.")
    assert sent[1][0] == "human"
    assert "Ada" not in sent[1][1] and "[PERSON_1]" in sent[1][1]
    assert "Ada" in out.content


def test_dict_message_input_redacts_content(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    echo = _EchoRunnable()
    wrapped = with_peyeeye(echo, client=client)

    out = wrapped.invoke(
        [{"role": "user", "content": "Ada sends regards to ben@e.com"}]
    )
    sent_content = echo.last_input[0]["content"]
    assert "Ada" not in sent_content
    assert "ben@e.com" not in sent_content
    assert out.content.count("Ada") == 1
    assert "ben@e.com" in out.content


def test_string_output_is_rehydrated(router: Router, client: Peyeeye) -> None:
    _mount_backend(router)

    class _PlainStringModel:
        def invoke(self, input: Any, config: Any = None) -> str:  # noqa: A002
            # Returns the same text back (already redacted).
            return str(input)

    wrapped = with_peyeeye(_PlainStringModel(), client=client)
    out = wrapped.invoke("Hello Ada")
    assert out == "Hello Ada"  # rehydrated string output


def test_multimodal_text_parts_are_redacted_images_passthrough(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    echo = _EchoRunnable()
    wrapped = with_peyeeye(echo, client=client)

    msg = _FakeMessage(
        role="human",
        content=[
            {"type": "text", "text": "Ada is in this photo:"},
            {"type": "image_url", "image_url": "https://example.com/ada.png"},
        ],
    )
    wrapped.invoke([msg])

    sent = echo.last_input[0].content
    assert sent[0]["text"] == "[PERSON_1] is in this photo:"
    assert sent[1] == {
        "type": "image_url",
        "image_url": "https://example.com/ada.png",
    }


def test_each_invoke_uses_a_fresh_session(router: Router, client: Peyeeye) -> None:
    _mount_backend(router)
    echo = _EchoRunnable()
    wrapped = with_peyeeye(echo, client=client)

    wrapped.invoke("Hi Ada")
    wrapped.invoke("Hi Ben")

    # Two distinct redact calls, neither carrying a session in the request
    # body (each should have opened a brand-new session).
    redacts = [c for c in router.calls if c.method == "POST" and c.path == "/v1/redact"]
    assert len(redacts) == 2
    for c in redacts:
        assert c.json_body is not None and c.json_body.get("session") is None


def test_stateless_mode_opens_sealed_session(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    echo = _EchoRunnable()
    wrapped = with_peyeeye(echo, client=client, stateless=True)

    wrapped.invoke("Hi Ada")

    body = router.call_for("POST", "/v1/redact").json_body
    assert body["session"] == "stateless"


def test_construct_directly_via_class(router: Router, client: Peyeeye) -> None:
    _mount_backend(router)
    echo = _EchoRunnable()
    # Constructor path is equivalent to with_peyeeye.
    direct = PeyeeyeRunnable(echo, client=client)
    assert isinstance(direct, PeyeeyeRunnable)
    out = direct.invoke("Hi Ada")
    assert "Ada" in out.content


def test_async_ainvoke_threads_through_shield(
    router: Router, client: Peyeeye
) -> None:
    import asyncio

    _mount_backend(router)
    echo = _EchoRunnable()
    wrapped = with_peyeeye(echo, client=client)

    async def run() -> Any:
        return await wrapped.ainvoke("Hi Ada")

    out = asyncio.run(run())
    assert "[PERSON_1]" not in out.content
    assert "Ada" in out.content
