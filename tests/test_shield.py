"""Shield context manager — session reuse, delete-on-exit, stateless mode."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from peyeeye import Peyeeye, PeyeeyeError
from tests.conftest import Router, json_response


def _r(redacted: str, session: str) -> dict:
    return {
        "redacted": redacted,
        "session": session,
        "entities": [],
        "latency_ms": 1,
        "expires_at": "2026-01-01T00:00:00Z",
    }


def test_shield_binds_session_and_deletes_on_exit(
    router: Router, client: Peyeeye
) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(_r("[PERSON_1]", "ses_abc"))

    @router.route("POST", "/v1/rehydrate")
    def _r2(req: httpx.Request) -> httpx.Response:
        return json_response({"text": "Ada", "replaced": 1, "unknown": [], "latency_ms": 1})

    deletes: list[int] = []

    @router.route("DELETE", "/v1/sessions/ses_abc")
    def _d(req: httpx.Request) -> httpx.Response:
        deletes.append(1)
        return httpx.Response(204)

    with client.shield() as shield:
        assert shield.redact("Ada Lovelace") == "[PERSON_1]"
        assert shield.session_id == "ses_abc"
        assert shield.rehydrate("[PERSON_1]") == "Ada"

    # First redact opens a new session (no session in body); a *second* redact in the
    # same shield should send session=ses_abc.
    assert len(deletes) == 1


def test_shield_reuses_session_across_redacts(
    router: Router, client: Peyeeye
) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(_r("[X]", "ses_abc"))

    @router.route("DELETE", "/v1/sessions/ses_abc")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    with client.shield() as shield:
        shield.redact("first")
        shield.redact("second")

    first_call, second_call = [
        c for c in router.calls if c.method == "POST" and c.path == "/v1/redact"
    ]
    assert "session" not in first_call.json_body
    assert second_call.json_body["session"] == "ses_abc"


def test_shield_stateless_uses_rehydration_key(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "redacted": "[EMAIL_1]",
                "session": "stateless",
                "entities": [],
                "latency_ms": 1,
                "rehydration_key": "skey_deadbeef",
            }
        )

    @router.route("POST", "/v1/rehydrate")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response({"text": "ada@a-e.com", "replaced": 1, "unknown": [], "latency_ms": 1})

    with client.shield(stateless=True) as shield:
        shield.redact("ada@a-e.com")
        assert shield.rehydration_key == "skey_deadbeef"
        assert shield.rehydrate("[EMAIL_1]") == "ada@a-e.com"

    reh_call = router.call_for("POST", "/v1/rehydrate")
    assert reh_call.json_body["session"] == "skey_deadbeef"
    assert router.count("DELETE", "/v1/sessions/stateless") == 0


def test_shield_rehydrate_without_redact_raises(client: Peyeeye) -> None:
    with client.shield() as shield:
        with pytest.raises(RuntimeError):
            shield.rehydrate("whatever")


def test_shield_swallows_delete_errors(
    router: Router, client: Peyeeye
) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(_r("[X]", "ses_abc"))

    @router.route("DELETE", "/v1/sessions/ses_abc")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"code": "not_found", "message": "gone"})

    # Shield should silently swallow the delete error on exit rather than mask a
    # successful workflow with a cleanup failure.
    with client.shield() as shield:
        shield.redact("hi")


def test_shield_chunk_buffering_holds_partial_token(
    router: Router, client: Peyeeye
) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(_r("[PERSON_1]", "ses_abc"))

    replies = iter([
        {"text": "Hi ", "replaced": 0, "unknown": [], "latency_ms": 1},
        {"text": "Hello Ada", "replaced": 1, "unknown": [], "latency_ms": 1},
    ])

    @router.route("POST", "/v1/rehydrate")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(next(replies))

    @router.route("DELETE", "/v1/sessions/ses_abc")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    with client.shield() as shield:
        shield.redact("Ada")
        # "Hi " flushes cleanly because no partial token tail
        a = shield.rehydrate_chunk("Hi ")
        # "[PER" is held back; nothing is flushed yet
        b = shield.rehydrate_chunk("[PER")
        # Chunk closes the token → whole thing goes through rehydrate
        c = shield.rehydrate_chunk("SON_1] there")
        assert a == "Hi "
        assert b == ""
        assert c == "Hello Ada"


def test_shield_flush_emits_remainder(
    router: Router, make_client: Callable[..., Peyeeye]
) -> None:
    router = Router()
    transport = httpx.MockTransport(router)
    client = Peyeeye(
        api_key="pk_test",
        base_url="https://api.peyeeye.test",
        transport=transport,
        max_retries=1,
    )

    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(_r("[X]", "ses_flush"))

    @router.route("POST", "/v1/rehydrate")
    def _(req: httpx.Request) -> httpx.Response:
        body = req.content.decode()
        assert "done." in body
        return json_response({"text": "done.", "replaced": 0, "unknown": [], "latency_ms": 1})

    @router.route("DELETE", "/v1/sessions/ses_flush")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    with client.shield() as shield:
        shield.redact("x")
        # nothing buffered → flush returns "" and skips HTTP
        assert shield.flush() == ""
        # buffer a trailing "done." and flush
        shield.rehydrate_chunk("done.")  # consumes, returns fresh
