"""peyeeye — PII redaction & rehydration client.

    from peyeeye import Peyeeye

    pe = Peyeeye(api_key="pk_live_...")
    with pe.shield() as shield:
        safe = shield.redact("Hi, I'm Ada, ada@a-e.com")
        reply = call_your_model(safe)
        print(shield.rehydrate(reply))
"""

from .client import Peyeeye, Shield
from .errors import PeyeeyeError
from .models import (
    CustomDetector,
    DetectedEntity,
    EntitiesList,
    EntityTemplate,
    RedactResponse,
    RehydrateResponse,
    SessionInfo,
    StreamEvent,
    TestPatternResponse,
)

__all__ = [
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
]

__version__ = "1.0.1"
