"""Core /v1/redact and /v1/rehydrate behavior."""

from __future__ import annotations

from typing import Any

import httpx

from peyeeye import Peyeeye
from tests.conftest import Router, json_response


def _redact_reply(session: str = "ses_abc") -> dict:
    return {
        "redacted": "Hi, I'm [PERSON_1].",
        "session": session,
        "entities": [
            {
                "token": "[PERSON_1]",
                "type": "PERSON",
                "span": [8, 20],
                "confidence": 0.98,
            }
        ],
        "latency_ms": 42,
        "expires_at": "2026-05-01T14:27:03Z",
    }


def test_redact_sends_expected_body(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(_redact_reply())

    r = client.redact("Hi, I'm Ada Lovelace.", locale="en-US", policy="default")
    assert r.session == "ses_abc"
    assert r.redacted == "Hi, I'm [PERSON_1]."
    assert r.entities[0].token == "[PERSON_1]"
    assert r.entities[0].span == (8, 20)
    assert r.entities[0].confidence == 0.98
    assert r.expires_at == "2026-05-01T14:27:03Z"

    call = router.call_for("POST", "/v1/redact")
    assert call.json_body["text"] == "Hi, I'm Ada Lovelace."
    assert call.json_body["locale"] == "en-US"
    assert call.json_body["policy"] == "default"
    assert call.headers["authorization"] == "Bearer pk_test_abc"


def test_redact_passes_through_list_input(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response({**_redact_reply(), "redacted": ["[PERSON_1]", "[EMAIL_1]"]})

    r = client.redact(["Ada", "ada@a-e.com"])
    assert r.redacted == ["[PERSON_1]", "[EMAIL_1]"]
    call = router.call_for("POST", "/v1/redact")
    assert call.json_body["text"] == ["Ada", "ada@a-e.com"]


def test_redact_passes_entities_placeholder_session(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(_redact_reply())

    client.redact(
        "hi",
        entities=["PERSON", "EMAIL"],
        placeholder="<{TYPE}>",
        session="ses_prior",
    )
    body: Any = router.call_for("POST", "/v1/redact").json_body
    assert body["entities"] == ["PERSON", "EMAIL"]
    assert body["placeholder"] == "<{TYPE}>"
    assert body["session"] == "ses_prior"


def test_redact_idempotency_key_sent_as_header(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(_redact_reply())

    client.redact("hi", idempotency_key="req_a1b2c3")
    call = router.call_for("POST", "/v1/redact")
    assert call.headers.get("idempotency-key") == "req_a1b2c3"


def test_redact_stateless_returns_rehydration_key(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "redacted": "Email [EMAIL_1]",
                "session": "stateless",
                "entities": [],
                "latency_ms": 5,
                "rehydration_key": "skey_deadbeef",
            }
        )

    r = client.redact("Email ada@a-e.com", session="stateless")
    assert r.session == "stateless"
    assert r.rehydration_key == "skey_deadbeef"


def test_rehydrate_sends_session_and_strict(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/rehydrate")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "text": "Hi Ada.",
                "replaced": 1,
                "unknown": [],
                "latency_ms": 7,
            }
        )

    r = client.rehydrate("Hi [PERSON_1].", session="ses_abc", strict=True)
    assert r.text == "Hi Ada."
    assert r.replaced == 1
    body = router.call_for("POST", "/v1/rehydrate").json_body
    assert body == {"text": "Hi [PERSON_1].", "session": "ses_abc", "strict": True}


def test_rehydrate_empty_text_skips_http(router: Router, client: Peyeeye) -> None:
    # Short-circuit: empty input doesn't need a round trip.
    r = client.rehydrate("", session="ses_abc")
    assert r.text == ""
    assert r.replaced == 0
    assert router.count("POST", "/v1/rehydrate") == 0


def test_rehydrate_with_skey(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/rehydrate")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response({"text": "ok", "replaced": 0, "unknown": [], "latency_ms": 1})

    client.rehydrate("hi", session="skey_xxxxxxxx")
    body = router.call_for("POST", "/v1/rehydrate").json_body
    assert body["session"] == "skey_xxxxxxxx"
