"""SSE stream parsing for /v1/redact/stream."""

from __future__ import annotations

import httpx
import pytest

from peyeeye import Peyeeye, PeyeeyeError
from tests.conftest import Router


_SSE_BODY = (
    b"event: session\n"
    b"data: {\"session\":\"ses_7fA2kLw9MxPq\"}\n\n"
    b"event: redacted\n"
    b"data: {\"text\":\"Hi, I'm [PERSON_1]\",\"entities\":1}\n\n"
    b"event: redacted\n"
    b"data: {\"text\":\" - card [CARD_1]\",\"entities\":1}\n\n"
    b"event: done\n"
    b"data: {\"chars\":37}\n\n"
)


def test_redact_stream_emits_events(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/redact/stream")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_SSE_BODY,
            headers={"Content-Type": "text/event-stream"},
        )

    events = list(client.redact_stream(["Hi, I'm Ada", " - card 4242 4242 4242 4242"]))
    assert [e.event for e in events] == ["session", "redacted", "redacted", "done"]
    assert events[0].data["session"] == "ses_7fA2kLw9MxPq"
    assert events[1].data == {"text": "Hi, I'm [PERSON_1]", "entities": 1}
    assert events[-1].data == {"chars": 37}


def test_redact_stream_sends_chunks_and_locale(
    router: Router, client: Peyeeye
) -> None:
    @router.route("POST", "/v1/redact/stream")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"event: done\ndata: {\"chars\":0}\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    list(client.redact_stream(["a", "b"], locale="en-US", policy="default"))
    call = router.call_for("POST", "/v1/redact/stream")
    assert call.json_body == {"chunks": ["a", "b"], "locale": "en-US", "policy": "default"}


def test_redact_stream_raises_on_403_plan_gate(
    router: Router, client: Peyeeye
) -> None:
    @router.route("POST", "/v1/redact/stream")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"code": "forbidden", "message": "Streaming requires Build."},
        )

    with pytest.raises(PeyeeyeError) as exc:
        list(client.redact_stream(["a"]))
    assert exc.value.code == "forbidden"
    assert exc.value.status == 403


def test_redact_stream_handles_trailing_event_without_blank_line(
    router: Router, client: Peyeeye
) -> None:
    body = b"event: done\ndata: {\"chars\":0}"  # no trailing \n\n

    @router.route("POST", "/v1/redact/stream")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"Content-Type": "text/event-stream"}
        )

    events = list(client.redact_stream([""]))
    assert len(events) == 1
    assert events[0].event == "done"
    assert events[0].data["chars"] == 0
