"""Custom-detector CRUD + templates + pattern test endpoint."""

from __future__ import annotations

import httpx
import pytest

from peyeeye import Peyeeye, PeyeeyeError
from tests.conftest import Router, json_response


def test_list_entities(router: Router, client: Peyeeye) -> None:
    @router.route("GET", "/v1/entities")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "builtin": [
                    {"id": "EMAIL", "category": "Contact", "sample": "a@b.com", "locales": ["all"]}
                ],
                "custom": [
                    {"id": "ORDER_ID", "kind": "regex", "pattern": r"#A-\d+", "enabled": True}
                ],
            }
        )

    out = client.list_entities()
    assert out.builtin[0].id == "EMAIL"
    assert out.custom[0].id == "ORDER_ID"
    assert out.custom[0].kind == "regex"


def test_create_entity_sends_full_body(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/entities")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response({"id": "ORDER_ID", "kind": "regex", "enabled": True}, status=201)

    d = client.create_entity(
        id="ORDER_ID",
        kind="regex",
        pattern=r"#A-\d{6,}",
        examples=["#A-884217"],
        confidence_floor=0.92,
    )
    body = router.call_for("POST", "/v1/entities").json_body
    assert body == {
        "id": "ORDER_ID",
        "kind": "regex",
        "pattern": r"#A-\d{6,}",
        "examples": ["#A-884217"],
        "confidence_floor": 0.92,
    }
    assert d.id == "ORDER_ID"
    assert d.enabled is True


def test_create_entity_without_pattern_omits_field(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/entities")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response({"id": "X", "kind": "fewshot", "enabled": True}, status=201)

    client.create_entity(id="X", kind="fewshot", examples=["a", "b"])
    body = router.call_for("POST", "/v1/entities").json_body
    assert "pattern" not in body
    assert body["examples"] == ["a", "b"]


def test_create_entity_forbidden_raises(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/entities")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"code": "forbidden", "message": "Custom entities require Build."},
        )

    with pytest.raises(PeyeeyeError) as exc:
        client.create_entity(id="ORDER_ID", pattern=r"\d+")
    assert exc.value.code == "forbidden"
    assert exc.value.status == 403


def test_update_entity_sends_only_present_fields(router: Router, client: Peyeeye) -> None:
    @router.route("PATCH", "/v1/entities/ORDER_ID")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "id": "ORDER_ID",
                "kind": "regex",
                "pattern": r"#A-\d+",
                "enabled": False,
                "confidence_floor": 0.9,
            }
        )

    d = client.update_entity("ORDER_ID", enabled=False)
    assert d.enabled is False
    body = router.call_for("PATCH", "/v1/entities/ORDER_ID").json_body
    assert body == {"enabled": False}


def test_delete_entity(router: Router, client: Peyeeye) -> None:
    @router.route("DELETE", "/v1/entities/ORDER_ID")
    def _(req: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    client.delete_entity("ORDER_ID")
    assert router.count("DELETE", "/v1/entities/ORDER_ID") == 1


def test_test_pattern(router: Router, client: Peyeeye) -> None:
    @router.route("POST", "/v1/entities/test")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "matches": [
                    {"value": "#A-884217", "start": 4, "end": 13},
                ],
                "count": 1,
            }
        )

    r = client.test_pattern(pattern=r"#A-\d+", text="ref #A-884217")
    assert r.count == 1
    assert r.matches[0].value == "#A-884217"
    assert r.matches[0].start == 4


def test_entity_templates(router: Router, client: Peyeeye) -> None:
    @router.route("GET", "/v1/entities/templates")
    def _(req: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "templates": [
                    {
                        "id": "STRIPE_KEY",
                        "name": "Stripe API key",
                        "description": "sk/pk/rk prefix",
                        "kind": "regex",
                        "pattern": r"\bsk_(live|test)_[A-Za-z0-9]{16,}\b",
                        "example": "sk_live_abcDEF…",
                        "category": "Credential",
                    }
                ]
            }
        )

    tpls = client.entity_templates()
    assert len(tpls) == 1
    assert tpls[0].id == "STRIPE_KEY"
    assert tpls[0].category == "Credential"
