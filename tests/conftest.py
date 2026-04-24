"""Test helpers: MockTransport + a recording router.

The router matches on ``(method, path)``, dispatches to a handler, and records
the matched calls so individual tests can assert on payloads and headers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pytest

from peyeeye import Peyeeye


@dataclass
class RecordedCall:
    method: str
    path: str
    headers: Dict[str, str]
    json_body: Any


Handler = Callable[[httpx.Request], httpx.Response]


@dataclass
class Router:
    """Dispatch an httpx.Request to one of several registered handlers."""

    handlers: Dict[Tuple[str, str], Handler] = field(default_factory=dict)
    calls: List[RecordedCall] = field(default_factory=list)

    def route(self, method: str, path: str) -> Callable[[Handler], Handler]:
        def deco(fn: Handler) -> Handler:
            self.handlers[(method.upper(), path)] = fn
            return fn

        return deco

    def __call__(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content) if request.content else None
        except json.JSONDecodeError:
            body = request.content.decode("utf-8", errors="replace")
        self.calls.append(
            RecordedCall(
                method=request.method,
                path=request.url.path,
                headers=dict(request.headers),
                json_body=body,
            )
        )
        key = (request.method.upper(), request.url.path)
        handler = self.handlers.get(key)
        if handler is None:
            return httpx.Response(
                404,
                json={"code": "not_found", "message": f"no handler for {key}"},
            )
        return handler(request)

    def call_for(self, method: str, path: str) -> RecordedCall:
        for c in reversed(self.calls):
            if c.method == method.upper() and c.path == path:
                return c
        raise AssertionError(f"no recorded call for {method} {path}")

    def count(self, method: str, path: str) -> int:
        return sum(1 for c in self.calls if c.method == method.upper() and c.path == path)


@pytest.fixture
def router() -> Router:
    return Router()


@pytest.fixture
def make_client(router: Router) -> Callable[..., Peyeeye]:
    """Factory that builds a Peyeeye client wired to the MockTransport."""

    def _make(api_key: str = "pk_test_abc", **kwargs: Any) -> Peyeeye:
        transport = httpx.MockTransport(router)
        return Peyeeye(
            api_key=api_key,
            base_url="https://api.peyeeye.test",
            transport=transport,
            max_retries=kwargs.pop("max_retries", 1),
            **kwargs,
        )

    return _make


@pytest.fixture
def client(make_client: Callable[..., Peyeeye]) -> Peyeeye:
    return make_client()


def json_response(body: Dict[str, Any], status: int = 200, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers or {})
