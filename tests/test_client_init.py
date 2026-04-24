"""Construction, context management, version surface."""

from __future__ import annotations

import pytest

import re

import peyeeye
from peyeeye import Peyeeye, PeyeeyeError


def test_version_is_semver() -> None:
    assert re.match(r"^\d+\.\d+\.\d+(?:[-.].+)?$", peyeeye.__version__)


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="api_key is required"):
        Peyeeye(api_key="")


def test_public_surface() -> None:
    # These names make up the stable public API exported from the top-level package.
    for name in (
        "Peyeeye",
        "PeyeeyeError",
        "Shield",
        "DetectedEntity",
        "RedactResponse",
        "RehydrateResponse",
        "SessionInfo",
        "CustomDetector",
        "EntitiesList",
        "EntityTemplate",
        "TestPatternResponse",
        "StreamEvent",
    ):
        assert hasattr(peyeeye, name), f"missing public export: {name}"


def test_client_is_context_manager(client: Peyeeye) -> None:
    with client as c:
        assert c is client
    # Closing twice is a no-op (httpx.Client tolerates it).
    client.close()


def test_base_url_is_trimmed() -> None:
    c = Peyeeye(api_key="pk_test_abc", base_url="https://api.peyeeye.test///")
    assert c.base_url == "https://api.peyeeye.test"


def test_peyeeye_error_repr_and_attrs() -> None:
    e = PeyeeyeError(code="invalid_request", status=400, message="bad", request_id="req_1")
    assert e.code == "invalid_request"
    assert e.status == 400
    assert str(e) == "bad"
    assert e.request_id == "req_1"
    assert "invalid_request" in repr(e)
