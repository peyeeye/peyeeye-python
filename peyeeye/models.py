"""Typed request/response models for peyeeye.

Dataclasses are used (rather than a heavy validation library) so the SDK has a
single runtime dependency (``httpx``) and plays well with mypy / pyright users.

Every model carries a :py:meth:`from_dict` classmethod that tolerates unknown
keys — forward-compatible with new server-side fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


def _tuple_span(v: Any) -> Tuple[int, int]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise ValueError(f"expected [start, end] span, got {v!r}")


@dataclass
class DetectedEntity:
    """One entity span returned from ``/v1/redact``."""

    token: str
    type: str
    span: Tuple[int, int]
    confidence: float
    value: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DetectedEntity":
        return cls(
            token=str(d.get("token", "")),
            type=str(d.get("type", "")),
            span=_tuple_span(d.get("span", [0, 0])),
            confidence=float(d.get("confidence", 0.0)),
            value=d.get("value"),
        )


@dataclass
class RedactResponse:
    """Response from ``POST /v1/redact``.

    ``redacted`` is a string when the request text was a string, or a list of
    strings when the request text was a list. ``rehydration_key`` is populated
    only in stateless mode (``session="stateless"``); otherwise ``session`` is
    a ``ses_...`` identifier and ``expires_at`` is an ISO-8601 timestamp.
    """

    redacted: Union[str, List[str]]
    session: str
    entities: List[DetectedEntity] = field(default_factory=list)
    latency_ms: int = 0
    rehydration_key: Optional[str] = None
    expires_at: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RedactResponse":
        return cls(
            redacted=d.get("redacted", ""),
            session=str(d.get("session", "")),
            entities=[DetectedEntity.from_dict(e) for e in d.get("entities", [])],
            latency_ms=int(d.get("latency_ms", 0)),
            rehydration_key=d.get("rehydration_key"),
            expires_at=d.get("expires_at"),
        )


@dataclass
class RehydrateResponse:
    """Response from ``POST /v1/rehydrate``."""

    text: str
    replaced: int
    unknown: List[str] = field(default_factory=list)
    latency_ms: int = 0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RehydrateResponse":
        return cls(
            text=str(d.get("text", "")),
            replaced=int(d.get("replaced", 0)),
            unknown=list(d.get("unknown", [])),
            latency_ms=int(d.get("latency_ms", 0)),
        )


@dataclass
class SessionInfo:
    """Response from ``GET /v1/sessions/:id``."""

    id: str
    locale: str
    policy: str
    chars_processed: int
    entities_detected: int
    created_at: Optional[str]
    expires_at: Optional[str]
    expired: bool

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SessionInfo":
        return cls(
            id=str(d.get("id", "")),
            locale=str(d.get("locale", "")),
            policy=str(d.get("policy", "")),
            chars_processed=int(d.get("chars_processed", 0)),
            entities_detected=int(d.get("entities_detected", 0)),
            created_at=d.get("created_at"),
            expires_at=d.get("expires_at"),
            expired=bool(d.get("expired", False)),
        )


@dataclass
class BuiltinEntity:
    id: str
    category: str
    sample: str
    locales: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BuiltinEntity":
        return cls(
            id=str(d.get("id", "")),
            category=str(d.get("category", "")),
            sample=str(d.get("sample", "")),
            locales=list(d.get("locales", [])),
        )


@dataclass
class CustomDetector:
    """A custom detector registered on an org."""

    id: str
    kind: str
    pattern: str = ""
    enabled: bool = True
    confidence_floor: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CustomDetector":
        return cls(
            id=str(d.get("id", "")),
            kind=str(d.get("kind", "regex")),
            pattern=str(d.get("pattern", "")),
            enabled=bool(d.get("enabled", True)),
            confidence_floor=(
                float(d["confidence_floor"]) if "confidence_floor" in d else None
            ),
        )


@dataclass
class EntitiesList:
    """Response from ``GET /v1/entities``."""

    builtin: List[BuiltinEntity]
    custom: List[CustomDetector]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EntitiesList":
        return cls(
            builtin=[BuiltinEntity.from_dict(x) for x in d.get("builtin", [])],
            custom=[CustomDetector.from_dict(x) for x in d.get("custom", [])],
        )


@dataclass
class EntityTemplate:
    """A starter-template entry from ``GET /v1/entities/templates``."""

    id: str
    name: str
    description: str
    kind: str
    pattern: str
    example: str
    category: str

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EntityTemplate":
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            kind=str(d.get("kind", "regex")),
            pattern=str(d.get("pattern", "")),
            example=str(d.get("example", "")),
            category=str(d.get("category", "")),
        )


@dataclass
class PatternMatch:
    value: str
    start: int
    end: int

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PatternMatch":
        return cls(
            value=str(d.get("value", "")),
            start=int(d.get("start", 0)),
            end=int(d.get("end", 0)),
        )


@dataclass
class TestPatternResponse:
    """Response from ``POST /v1/entities/test``."""

    matches: List[PatternMatch]
    count: int

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TestPatternResponse":
        return cls(
            matches=[PatternMatch.from_dict(m) for m in d.get("matches", [])],
            count=int(d.get("count", 0)),
        )


@dataclass
class StreamEvent:
    """A single server-sent event from ``POST /v1/redact/stream``.

    The server emits three event types:

    * ``session`` — fires once with the new session id in ``data["session"]``.
    * ``redacted`` — fires per chunk with ``data["text"]`` and ``data["entities"]``.
    * ``done`` — closes the stream with ``data["chars"]``.
    """

    event: str
    data: Dict[str, Any]
