"""GET / DELETE /v1/sessions/:id coverage."""

from __future__ import annotations

import httpx

from peyeeye import Peyeeye
from tests.conftest import Router, json_response


def test_get_session(router: Router, client: Peyeeye) -> None:
    @router.route("GET", "/v1/sessions/ses_abc")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "id": "ses_abc",
                "locale": "en-US",
                "policy": "default",
                "chars_processed": 128,
                "entities_detected": 3,
                "created_at": "2026-04-23T12:00:00Z",
                "expires_at": "2026-04-23T12:15:00Z",
                "expired": False,
            }
        )

    info = client.get_session("ses_abc")
    assert info.id == "ses_abc"
    assert info.locale == "en-US"
    assert info.chars_processed == 128
    assert info.expired is False


def test_delete_session_tolerates_204(router: Router, client: Peyeeye) -> None:
    @router.route("DELETE", "/v1/sessions/ses_abc")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    client.delete_session("ses_abc")  # should not raise
    assert router.count("DELETE", "/v1/sessions/ses_abc") == 1


def test_delete_session_tolerates_200_empty_body(router: Router, client: Peyeeye) -> None:
    @router.route("DELETE", "/v1/sessions/ses_abc")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    client.delete_session("ses_abc")
