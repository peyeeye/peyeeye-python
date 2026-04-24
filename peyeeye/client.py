"""Synchronous client for the peyeeye.ai PII-redaction API.

    from peyeeye import Peyeeye
    pe = Peyeeye(api_key="pk_live_...")
    with pe.shield() as shield:
        safe = shield.redact("Hi, I'm Ada, ada@a-e.com")
        out = your_llm(safe)
        print(shield.rehydrate(out))
"""

from __future__ import annotations

import json
import random
import re
import time
from contextlib import contextmanager
from types import TracebackType
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

import httpx

from .errors import PeyeeyeError
from .models import (
    CustomDetector,
    EntitiesList,
    EntityTemplate,
    RedactResponse,
    RehydrateResponse,
    SessionInfo,
    StreamEvent,
    TestPatternResponse,
)

DEFAULT_BASE_URL = "https://api.peyeeye.ai"
USER_AGENT = "peyeeye-python/1.0.0"

# A token that ends at a chunk boundary looks like ``...[PER`` or ``...[EMAIL_1``.
# We hold it back until the next chunk closes the bracket, otherwise we'd emit a
# partial placeholder that the user's terminal would render verbatim.
_PARTIAL_TOKEN_TAIL = re.compile(r"\[[A-Z_0-9]*$")

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class Peyeeye:
    """Synchronous peyeeye client.

    Args:
        api_key: Bearer key from the dashboard, typically ``pk_live_...``.
        base_url: Override for the API host. Defaults to ``https://api.peyeeye.ai``.
        timeout: Per-request timeout in seconds. Defaults to 30s.
        max_retries: Max retries for 429 / 5xx responses. Defaults to 3.
        transport: Optional ``httpx.BaseTransport`` for advanced use / tests.
        default_headers: Extra headers merged onto every request.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        transport: Optional[httpx.BaseTransport] = None,
        default_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("Peyeeye: api_key is required.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        if default_headers:
            headers.update(default_headers)
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers,
            transport=transport,
        )

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client. Safe to call multiple times."""
        self._http.close()

    def __enter__(self) -> "Peyeeye":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # -- core: redact / rehydrate ---------------------------------------

    def redact(
        self,
        text: Union[str, Sequence[str]],
        *,
        locale: str = "auto",
        policy: Optional[Union[str, Mapping[str, Any]]] = None,
        entities: Optional[Sequence[str]] = None,
        placeholder: Optional[str] = None,
        session: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> RedactResponse:
        """Redact PII from ``text``.

        Args:
            text: String or list of strings. Lists are processed in one session.
            locale: BCP-47 language tag. ``"auto"`` lets the server detect.
            policy: Name of a saved policy or an inline policy object.
            entities: Restrict detection to these entity IDs.
            placeholder: Token template, default ``"[{TYPE}_{N}]"``.
            session: Existing ``ses_...`` id, or ``"stateless"`` for sealed mode.
            idempotency_key: Passed as the ``Idempotency-Key`` request header.
        """
        body: Dict[str, Any] = {
            "text": list(text) if _is_text_list(text) else text,
            "locale": locale,
        }
        if policy is not None:
            body["policy"] = policy
        if entities is not None:
            body["entities"] = list(entities)
        if placeholder is not None:
            body["placeholder"] = placeholder
        if session is not None:
            body["session"] = session
        data = self._post("/v1/redact", body, idempotency_key=idempotency_key)
        return RedactResponse.from_dict(data)

    def rehydrate(
        self,
        text: str,
        *,
        session: str,
        strict: bool = False,
    ) -> RehydrateResponse:
        """Swap placeholder tokens back to their original values.

        ``session`` may be either a ``ses_...`` id or a stateless
        ``skey_...`` rehydration key returned in stateless redact.
        """
        if not text:
            return RehydrateResponse(text="", replaced=0, unknown=[], latency_ms=0)
        body = {"text": text, "session": session, "strict": bool(strict)}
        data = self._post("/v1/rehydrate", body)
        return RehydrateResponse.from_dict(data)

    # -- streaming ------------------------------------------------------

    def redact_stream(
        self,
        chunks: Iterable[str],
        *,
        locale: str = "auto",
        policy: Optional[Union[str, Mapping[str, Any]]] = None,
    ) -> Iterator[StreamEvent]:
        """Stream-redact an iterable of chunks over SSE.

        Yields :class:`StreamEvent` values as they arrive. A streaming call
        always creates a fresh session — the first yielded event is of type
        ``session`` and carries the new ``ses_...`` id.

        Requires the Build plan or higher; otherwise the server returns 403.
        """
        body: Dict[str, Any] = {
            "chunks": list(chunks),
            "locale": locale,
        }
        if policy is not None:
            body["policy"] = policy

        with self._http.stream(
            "POST",
            "/v1/redact/stream",
            json=body,
            headers={"Accept": "*/*", "Content-Type": "application/json"},
        ) as resp:
            if not resp.is_success:
                _raise_from_response(resp, _read_body(resp))
            event: Optional[str] = None
            data_lines: List[str] = []
            for line in resp.iter_lines():
                if line == "":
                    if event is not None and data_lines:
                        yield StreamEvent(
                            event=event,
                            data=_decode_sse_data(data_lines),
                        )
                    event, data_lines = None, []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].lstrip())
            if event is not None and data_lines:
                yield StreamEvent(event=event, data=_decode_sse_data(data_lines))

    # -- session management --------------------------------------------

    def get_session(self, session_id: str) -> SessionInfo:
        """``GET /v1/sessions/:id`` — inspect a stateful session."""
        data = self._request("GET", f"/v1/sessions/{session_id}")
        return SessionInfo.from_dict(data)

    def delete_session(self, session_id: str) -> None:
        """``DELETE /v1/sessions/:id`` — drop a session immediately."""
        self._request("DELETE", f"/v1/sessions/{session_id}", allow_empty=True)

    # -- custom detectors ----------------------------------------------

    def list_entities(self) -> EntitiesList:
        """``GET /v1/entities`` — builtin catalog + org custom detectors."""
        return EntitiesList.from_dict(self._request("GET", "/v1/entities"))

    def create_entity(
        self,
        *,
        id: str,  # noqa: A002 - matches the API field name
        kind: str = "regex",
        pattern: Optional[str] = None,
        examples: Optional[Sequence[str]] = None,
        confidence_floor: Optional[float] = None,
    ) -> CustomDetector:
        """``POST /v1/entities`` — create or upsert a custom detector.

        ``kind`` is ``"regex"`` or ``"fewshot"``. When ``pattern`` is omitted
        the server induces one from ``examples`` at create time.
        """
        body: Dict[str, Any] = {"id": id, "kind": kind}
        if pattern is not None:
            body["pattern"] = pattern
        if examples is not None:
            body["examples"] = list(examples)
        if confidence_floor is not None:
            body["confidence_floor"] = float(confidence_floor)
        data = self._post("/v1/entities", body)
        return CustomDetector.from_dict(data)

    def update_entity(
        self,
        entity_id: str,
        *,
        pattern: Optional[str] = None,
        enabled: Optional[bool] = None,
        confidence_floor: Optional[float] = None,
    ) -> CustomDetector:
        """``PATCH /v1/entities/:id`` — partial-update a detector."""
        body: Dict[str, Any] = {}
        if pattern is not None:
            body["pattern"] = pattern
        if enabled is not None:
            body["enabled"] = bool(enabled)
        if confidence_floor is not None:
            body["confidence_floor"] = float(confidence_floor)
        data = self._request("PATCH", f"/v1/entities/{entity_id}", json=body)
        return CustomDetector.from_dict(data)

    def delete_entity(self, entity_id: str) -> None:
        """``DELETE /v1/entities/:id`` — retire a custom detector."""
        self._request("DELETE", f"/v1/entities/{entity_id}", allow_empty=True)

    def test_pattern(self, *, pattern: str, text: str) -> TestPatternResponse:
        """``POST /v1/entities/test`` — dry-run a regex against sample text."""
        data = self._post("/v1/entities/test", {"pattern": pattern, "text": text})
        return TestPatternResponse.from_dict(data)

    def entity_templates(self) -> List[EntityTemplate]:
        """``GET /v1/entities/templates`` — starter detector templates."""
        data = self._request("GET", "/v1/entities/templates")
        return [EntityTemplate.from_dict(t) for t in data.get("templates", [])]

    # -- high-level helpers --------------------------------------------

    @contextmanager
    def shield(
        self,
        *,
        locale: str = "auto",
        policy: Optional[Union[str, Mapping[str, Any]]] = None,
        entities: Optional[Sequence[str]] = None,
        placeholder: Optional[str] = None,
        stateless: bool = False,
        delete_on_exit: bool = True,
    ) -> Iterator["Shield"]:
        """Context manager pairing redact + rehydrate inside one session.

        Inside the ``with`` block ``shield.redact(text)`` returns the redacted
        string and ``shield.rehydrate(text)`` swaps tokens back. The session
        is torn down on exit unless ``delete_on_exit=False`` or stateless
        mode is used.
        """
        opts: Dict[str, Any] = {"locale": locale}
        if policy is not None:
            opts["policy"] = policy
        if entities is not None:
            opts["entities"] = list(entities)
        if placeholder is not None:
            opts["placeholder"] = placeholder
        if stateless:
            opts["session"] = "stateless"
        shield = Shield(self, opts)
        try:
            yield shield
        finally:
            if delete_on_exit and shield.session_id and not stateless:
                try:
                    self.delete_session(shield.session_id)
                except PeyeeyeError:
                    pass

    # -- HTTP plumbing --------------------------------------------------

    def _post(
        self,
        path: str,
        body: Any,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        headers: Dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return self._request("POST", path, json=body, headers=headers)  # type: ignore[return-value]

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        allow_empty: bool = False,
    ) -> Any:
        attempt = 0
        last_error: Optional[PeyeeyeError] = None
        while True:
            try:
                resp = self._http.request(method, path, json=json, headers=headers)
            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt, None)
                    attempt += 1
                    last_error = PeyeeyeError(
                        "network_error", 0, f"{type(e).__name__}: {e}", None
                    )
                    continue
                raise PeyeeyeError(
                    "network_error", 0, f"{type(e).__name__}: {e}", None
                ) from e

            if resp.is_success:
                if resp.status_code == 204 or not resp.content:
                    return {} if not allow_empty else None
                return resp.json()

            if resp.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                self._sleep_backoff(attempt, resp.headers.get("retry-after"))
                attempt += 1
                last_error = _error_from_response(resp)
                continue

            _raise_from_response(resp)
            # unreachable, _raise_from_response always raises
            assert last_error is not None
            raise last_error

    @staticmethod
    def _sleep_backoff(attempt: int, retry_after: Optional[str]) -> None:
        delay: Optional[float] = None
        if retry_after:
            try:
                delay = max(0.0, float(retry_after))
            except ValueError:
                delay = None
        if delay is None:
            base = 0.25 * (2**attempt)
            delay = min(15.0, base + random.uniform(0, base * 0.1))
        time.sleep(min(15.0, delay))


class Shield:
    """Helper returned by :meth:`Peyeeye.shield`.

    Remembers the server-side session so repeated ``redact`` / ``rehydrate``
    calls share the same token mapping. Supports streaming rehydration via
    :meth:`rehydrate_chunk` + :meth:`flush` for piping LLM output straight to
    a user terminal.
    """

    def __init__(self, client: Peyeeye, opts: Dict[str, Any]) -> None:
        self._client = client
        self._opts = opts
        self._stateless = opts.get("session") == "stateless"
        self.session_id: str = ""
        self.rehydration_key: Optional[str] = None
        self.last_redacted: Union[str, List[str], None] = None
        self._buf: str = ""

    def redact(self, text: Union[str, Sequence[str]]) -> Union[str, List[str]]:
        """Redact ``text`` in the shield's session.

        The first call establishes the session; subsequent calls reuse it so
        ``Ada Lovelace`` will resolve to the same ``[PERSON_1]`` token.
        """
        opts = dict(self._opts)
        if self.session_id:
            opts["session"] = self.session_id
        r = self._client.redact(text, **opts)
        if not self.session_id and not self._stateless:
            self.session_id = r.session
        if r.rehydration_key:
            self.rehydration_key = r.rehydration_key
        self.last_redacted = r.redacted
        return r.redacted

    def rehydrate(self, text: str, *, strict: bool = False) -> str:
        """Swap tokens back to their original values."""
        session = self.rehydration_key or self.session_id
        if not session:
            raise RuntimeError(
                "Shield.rehydrate called before redact(); no session to use."
            )
        return self._client.rehydrate(text, session=session, strict=strict).text

    def rehydrate_chunk(self, chunk: str) -> str:
        """Streaming-safe rehydrate: buffers partial tokens across chunks.

        Call this for each LLM chunk, then :meth:`flush` once upstream closes.
        Calling ``flush`` mid-stream can emit a half-formed token.
        """
        self._buf += chunk
        m = _PARTIAL_TOKEN_TAIL.search(self._buf)
        if m:
            safe, self._buf = self._buf[: m.start()], self._buf[m.start() :]
        else:
            safe, self._buf = self._buf, ""
        if not safe:
            return ""
        return self.rehydrate(safe)

    def flush(self) -> str:
        """Emit anything still buffered. Call once upstream has closed."""
        remainder, self._buf = self._buf, ""
        if not remainder:
            return ""
        return self.rehydrate(remainder)


# ---------------------------------------------------------------------- helpers


def _is_text_list(text: Any) -> bool:
    return isinstance(text, (list, tuple)) and not isinstance(text, (str, bytes))


def _read_body(resp: httpx.Response) -> Dict[str, Any]:
    try:
        resp.read()
        return resp.json()
    except Exception:
        return {}


def _error_from_response(resp: httpx.Response) -> PeyeeyeError:
    try:
        body = resp.json()
    except Exception:
        body = {}
    return PeyeeyeError(
        code=str(body.get("code") or "error"),
        status=resp.status_code,
        message=str(body.get("message") or resp.reason_phrase or "Error"),
        request_id=resp.headers.get("x-request-id") or body.get("request_id"),
    )


def _raise_from_response(
    resp: httpx.Response, body: Optional[Dict[str, Any]] = None
) -> None:
    if body is None:
        try:
            body = resp.json()
        except Exception:
            body = {}
    raise PeyeeyeError(
        code=str(body.get("code") or "error"),
        status=resp.status_code,
        message=str(body.get("message") or resp.reason_phrase or "Error"),
        request_id=resp.headers.get("x-request-id") or body.get("request_id"),
    )


def _decode_sse_data(lines: Sequence[str]) -> Dict[str, Any]:
    raw = "\n".join(lines)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}
