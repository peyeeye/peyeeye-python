"""Exception types raised by the peyeeye client."""

from __future__ import annotations

from typing import Optional


class PeyeeyeError(Exception):
    """Raised for any non-2xx response from the peyeeye API.

    Attributes mirror the API error envelope:
        code        Stable machine code, e.g. ``invalid_request``, ``rate_limited``.
        status      HTTP status code.
        message     Human-readable message.
        request_id  Value of the ``X-Request-Id`` response header when present.
    """

    def __init__(
        self,
        code: str,
        status: int,
        message: str,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message
        self.request_id = request_id

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"PeyeeyeError(code={self.code!r}, status={self.status}, "
            f"message={self.message!r}, request_id={self.request_id!r})"
        )
