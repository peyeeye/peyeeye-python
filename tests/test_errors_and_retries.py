"""Error envelope parsing + retry/backoff behavior."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

import peyeeye.client as client_mod
from peyeeye import Peyeeye, PeyeeyeError
from tests.conftest import Router


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)


def test_error_payload_is_decoded(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"code": "invalid_request", "message": "Missing field 'text'."},
            headers={"X-Request-Id": "req_fa11"},
        )

    with pytest.raises(PeyeeyeError) as exc:
        client.redact("hi")
    err = exc.value
    assert err.code == "invalid_request"
    assert err.status == 400
    assert err.message == "Missing field 'text'."
    assert err.request_id == "req_fa11"


def test_error_falls_back_to_status_text_when_body_empty(
    router: Router, client: Peyeeye
) -> None:
    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(402, content=b"")

    with pytest.raises(PeyeeyeError) as exc:
        client.redact("hi")
    assert exc.value.status == 402
    # code defaults to "error" when the server body is empty
    assert exc.value.code == "error"


def test_retry_on_429_then_success(
    router: Router, make_client: Callable[..., Peyeeye], monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    client = make_client(max_retries=3)
    calls: list[int] = []

    @router.route("POST", "/v1/redact")
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(
                429,
                json={"code": "rate_limited", "message": "slow down"},
                headers={"Retry-After": "0"},
            )
        return httpx.Response(
            200,
            json={
                "redacted": "ok",
                "session": "ses_1",
                "entities": [],
                "latency_ms": 1,
                "expires_at": "2026-01-01T00:00:00Z",
            },
        )

    r = client.redact("hi")
    assert r.session == "ses_1"
    assert len(calls) == 3


def test_retry_on_500(
    router: Router, make_client: Callable[..., Peyeeye], monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    client = make_client(max_retries=2)
    hits: list[int] = []

    @router.route("POST", "/v1/rehydrate")
    def _(req: httpx.Request) -> httpx.Response:
        hits.append(1)
        if len(hits) < 2:
            return httpx.Response(503, json={"code": "detector_degraded", "message": "ml offline"})
        return httpx.Response(
            200,
            json={"text": "out", "replaced": 0, "unknown": [], "latency_ms": 1},
        )

    r = client.rehydrate("hi", session="ses_x")
    assert r.text == "out"
    assert len(hits) == 2


def test_retry_exhausted_raises(
    router: Router, make_client: Callable[..., Peyeeye], monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    client = make_client(max_retries=2)

    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"code": "internal_error", "message": "boom"},
        )

    with pytest.raises(PeyeeyeError) as exc:
        client.redact("hi")
    assert exc.value.status == 500
    assert exc.value.code == "internal_error"


def test_4xx_not_retried(
    router: Router, make_client: Callable[..., Peyeeye], monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    client = make_client(max_retries=5)
    count: list[int] = []

    @router.route("POST", "/v1/redact")
    def _(req: httpx.Request) -> httpx.Response:
        count.append(1)
        return httpx.Response(401, json={"code": "unauthorized", "message": "nope"})

    with pytest.raises(PeyeeyeError):
        client.redact("hi")
    assert count == [1]


def test_network_error_wrapped_as_peyeeyeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)

    def flaky(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(flaky)
    client = Peyeeye(
        api_key="pk_test",
        base_url="https://api.peyeeye.test",
        transport=transport,
        max_retries=1,
    )
    with pytest.raises(PeyeeyeError) as exc:
        client.redact("hi")
    assert exc.value.code == "network_error"
    assert exc.value.status == 0
