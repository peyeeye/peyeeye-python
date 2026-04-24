"""Tests for the LiteLLM integration (peyeeye.litellm).

Uses a fake completion function instead of hitting the real ``litellm`` package
so the SDK's dev env doesn't need to pull it in — ``peyeeye.litellm`` falls
back to ``object`` as the CustomLogger base when litellm isn't installed.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

from peyeeye import Peyeeye
from peyeeye.litellm import (
    PeyeeyeHandler,
    redact_messages,
    rehydrate_response,
    with_peyeeye,
)
from tests.conftest import Router, json_response


# ---------------------------------------------------------------------------
# Mock peyeeye backend (same toy regex tokenizer used by test_langchain.py)
# ---------------------------------------------------------------------------


class _Backend:
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
        import json

        payload = json.loads(req.read())
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
        body: Dict[str, Any] = {
            "redacted": redacted,
            "session": sid,
            "entities": [],
            "latency_ms": 1,
            "expires_at": "2099-01-01T00:00:00Z",
        }
        if stateless:
            body["rehydration_key"] = sid
        return json_response(body)

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

    for i in range(1, 50):

        @router.route("DELETE", f"/v1/sessions/ses_{i}")
        def _d(req: httpx.Request, _i: int = i) -> httpx.Response:
            return httpx.Response(204)

    return backend


# ---------------------------------------------------------------------------
# Fake LiteLLM surface — just what we need for these tests.
# ---------------------------------------------------------------------------


class _FakeMessage(dict):
    """Acts like both a LiteLLM Message and a dict (attribute + item access)."""

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - rarely hit
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(role="assistant", content=content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_with_peyeeye_redacts_messages_and_rehydrates_response(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    seen: Dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Any:
        seen["messages"] = kwargs["messages"]
        content = kwargs["messages"][-1]["content"]
        return _FakeResponse(content=f"Hello {content}, welcome!")

    wrapped = with_peyeeye(fake_completion, client=client)
    resp = wrapped(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi, I'm Ada — ada@a-e.com"}],
    )

    sent = seen["messages"][0]["content"]
    assert "Ada" not in sent and "ada@a-e.com" not in sent
    assert "[PERSON_" in sent and "[EMAIL_" in sent

    out = resp.choices[0].message["content"]
    assert "Ada" in out and "ada@a-e.com" in out
    assert "[PERSON_" not in out


def test_with_peyeeye_preserves_system_message_and_multiple_turns(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    captured: Dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Any:
        captured["messages"] = kwargs["messages"]
        return _FakeResponse(content="ok, got it for [PERSON_1]")

    wrapped = with_peyeeye(fake_completion, client=client)
    resp = wrapped(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell Ben that Ada says hi"},
        ],
    )

    sent = captured["messages"]
    assert sent[0] == {"role": "system", "content": "You are helpful."}
    assert "Ada" not in sent[1]["content"] and "Ben" not in sent[1]["content"]
    assert "[PERSON_1]" in sent[1]["content"] and "[PERSON_2]" in sent[1]["content"]
    # [PERSON_1] rehydrates to whichever name the backend tokenized first
    # ("Ada" here — its regex runs before the Ben regex in the test backend).
    assert "Ada" in resp.choices[0].message["content"]
    assert "[PERSON_" not in resp.choices[0].message["content"]


def test_with_peyeeye_multimodal_parts_redacted_images_passthrough(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    captured: Dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Any:
        captured["messages"] = kwargs["messages"]
        return _FakeResponse(content="done")

    wrapped = with_peyeeye(fake_completion, client=client)
    wrapped(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Ada is in this photo:"},
                    {"type": "image_url", "image_url": "https://x/ada.png"},
                ],
            }
        ],
    )

    parts = captured["messages"][0]["content"]
    assert parts[0]["text"] == "[PERSON_1] is in this photo:"
    assert parts[1] == {"type": "image_url", "image_url": "https://x/ada.png"}


def test_with_peyeeye_async(router: Router, client: Peyeeye) -> None:
    import asyncio

    _mount_backend(router)

    async def fake_acompletion(**kwargs: Any) -> Any:
        content = kwargs["messages"][-1]["content"]
        return _FakeResponse(content=f"echo: {content}")

    wrapped = with_peyeeye(fake_acompletion, client=client)

    async def run() -> Any:
        return await wrapped(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi Ada"}],
        )

    resp = asyncio.run(run())
    assert "Ada" in resp.choices[0].message["content"]
    assert "[PERSON_" not in resp.choices[0].message["content"]


def test_with_peyeeye_opens_fresh_session_per_call(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)

    def fake_completion(**kwargs: Any) -> Any:
        return _FakeResponse(content=kwargs["messages"][-1]["content"])

    wrapped = with_peyeeye(fake_completion, client=client)
    wrapped(model="m", messages=[{"role": "user", "content": "Hi Ada"}])
    wrapped(model="m", messages=[{"role": "user", "content": "Hi Ben"}])

    redacts = [c for c in router.calls if c.method == "POST" and c.path == "/v1/redact"]
    assert len(redacts) == 2
    for c in redacts:
        assert c.json_body is not None and c.json_body.get("session") is None


def test_with_peyeeye_stateless_opts_in(router: Router, client: Peyeeye) -> None:
    _mount_backend(router)

    def fake_completion(**kwargs: Any) -> Any:
        return _FakeResponse(content=kwargs["messages"][-1]["content"])

    wrapped = with_peyeeye(fake_completion, client=client, stateless=True)
    wrapped(model="m", messages=[{"role": "user", "content": "Hi Ada"}])

    body = router.call_for("POST", "/v1/redact").json_body
    assert body["session"] == "stateless"


def test_handler_pre_and_post_call_hooks(router: Router, client: Peyeeye) -> None:
    _mount_backend(router)
    handler = PeyeeyeHandler(client=client)

    data: Dict[str, Any] = {
        "litellm_call_id": "req-1",
        "messages": [{"role": "user", "content": "Hi Ada"}],
    }
    out_data = handler.pre_call_hook(data)
    assert "Ada" not in out_data["messages"][0]["content"]
    assert "[PERSON_1]" in out_data["messages"][0]["content"]

    response = _FakeResponse(content="Hello [PERSON_1]!")
    out_resp = handler.post_call_success_hook(out_data, response)
    assert out_resp.choices[0].message["content"] == "Hello Ada!"


def test_redact_messages_helper_leaves_non_dict_untouched(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    from peyeeye.client import Shield

    shield = Shield(client, {"locale": "auto"})
    msgs: List[Any] = [
        "raw string, pass through",
        {"role": "user", "content": "Hi Ada"},
    ]
    out = redact_messages(msgs, shield)  # type: ignore[arg-type]
    assert out[0] == "raw string, pass through"
    assert "Ada" not in out[1]["content"]


def test_rehydrate_response_handles_dict_response(
    router: Router, client: Peyeeye
) -> None:
    _mount_backend(router)
    from peyeeye.client import Shield

    shield = Shield(client, {"locale": "auto"})
    # Prime a session with a real redact so the rehydrate call has something
    # to resolve.
    shield.redact("Hi Ada")

    resp = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello [PERSON_1]"}},
        ]
    }
    out = rehydrate_response(resp, shield)
    assert out["choices"][0]["message"]["content"] == "Hello Ada"


def test_rehydrate_response_plain_string(router: Router, client: Peyeeye) -> None:
    _mount_backend(router)
    from peyeeye.client import Shield

    shield = Shield(client, {"locale": "auto"})
    shield.redact("Hi Ada")

    out = rehydrate_response("Hello [PERSON_1]", shield)
    assert out == "Hello Ada"
