# peyeeye

[![PyPI version](https://img.shields.io/pypi/v/peyeeye.svg)](https://pypi.org/project/peyeeye/)
[![Python versions](https://img.shields.io/pypi/pyversions/peyeeye.svg)](https://pypi.org/project/peyeeye/)
[![Downloads](https://static.pepy.tech/badge/peyeeye/month)](https://pepy.tech/project/peyeeye)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Types: py.typed](https://img.shields.io/badge/types-py.typed-informational.svg)](https://peps.python.org/pep-0561/)
[![CI](https://github.com/peyeeye/peyeeye-python/actions/workflows/test.yml/badge.svg)](https://github.com/peyeeye/peyeeye-python/actions/workflows/test.yml)
[![Homepage](https://img.shields.io/badge/homepage-peyeeye.ai-E8B4FF.svg)](https://peyeeye.ai)

Official Python client for [**peyeeye.ai**](https://peyeeye.ai) — redact PII on
the way _into_ your LLM prompts and rehydrate it on the way out.

- **Homepage**: <https://peyeeye.ai>
- **API reference**: <https://peyeeye.ai/docs>
- **PyPI**: <https://pypi.org/project/peyeeye/>

```bash
pip install peyeeye
```

Python 3.9+. Single runtime dependency: `httpx`. Fully type-hinted (`py.typed`).

## Get an API key

1. Sign up at **<https://peyeeye.ai/signup>** (free plan, no card required —
   1 M characters / month, all 30+ built-in detectors).
2. Head to **<https://peyeeye.ai/dashboard/keys>** → **New key**.
3. Copy the full `pk_live_…` (or `pk_test_…`) token shown once — we store only
   a hash after you close the dialog. Export it where your app reads env vars:

```bash
export PEYEEYE_KEY=pk_live_...
```

Test keys bypass billing and are rate-limited for development; live keys count
against your plan. Paid tiers (Build / Pro / Scale) unlock streaming, custom
detectors, and higher throughput — see <https://peyeeye.ai/pricing>.

## Quickstart

```python
import os
from peyeeye import Peyeeye
from anthropic import Anthropic

peyeeye = Peyeeye(api_key=os.environ["PEYEEYE_KEY"])
claude = Anthropic()

with peyeeye.shield() as shield:
    safe = shield.redact("Hi, I'm Ada, ada@a-e.com")
    reply = claude.messages.create(
        model="claude-sonnet-*",
        max_tokens=256,
        messages=[{"role": "user", "content": safe}],
    )
    print(shield.rehydrate(reply.content[0].text))
```

`shield()` opens a session, redacts, and cleans up on exit. Inside the block,
the same real value always maps to the same token — `Ada Lovelace` is always
`[PERSON_1]` — and tokens never leak across sessions.

## Low-level calls

Skip the `shield` helper when you need more control:

```python
r = peyeeye.redact("Card: 4242 4242 4242 4242")
# r.redacted    → "Card: [CARD_1]"
# r.session     → "ses_…"
# r.entities    → [DetectedEntity(token="[CARD_1]", type="CARD", span=(6, 25), confidence=0.99)]

clean = peyeeye.rehydrate("Confirmation for [CARD_1].", session=r.session)
# clean.text → "Confirmation for 4242 4242 4242 4242."
```

## Stateless sealed mode

Pass `stateless=True` and peyeeye never stores the mapping — the redact
response carries a sealed `skey_…` blob you hand back to rehydrate. Nothing
lives on the server between calls.

```python
with peyeeye.shield(stateless=True) as shield:
    safe = shield.redact("Email ada@a-e.com")
    clean = shield.rehydrate("Reply: [EMAIL_1]")
    # shield.rehydration_key is the skey_... blob, if you need to persist it
```

Or with raw calls:

```python
r = peyeeye.redact("Email ada@a-e.com", session="stateless")
# r.rehydration_key → "skey_…"
clean = peyeeye.rehydrate("[EMAIL_1] received.", session=r.rehydration_key)
```

## Streaming rehydration

When piping an LLM token stream straight to a user, naive rehydration breaks
on mid-token boundaries. `rehydrate_chunk()` buffers partial tokens across
chunks; call `flush()` once upstream closes.

```python
with peyeeye.shield() as shield:
    safe = shield.redact(prompt)
    for chunk in your_llm_stream(safe):
        sys.stdout.write(shield.rehydrate_chunk(chunk))
    sys.stdout.write(shield.flush())
```

Never call `flush()` while the stream is still delivering chunks — you'll emit
a half-formed placeholder.

## Streaming redact (SSE)

For the `/v1/redact/stream` endpoint (Build plan and higher):

```python
for event in peyeeye.redact_stream(["Hi, I'm Ada", " — card 4242 4242 4242 4242"]):
    if event.event == "session":
        session_id = event.data["session"]
    elif event.event == "redacted":
        print(event.data["text"])
    elif event.event == "done":
        print("chars:", event.data["chars"])
```

## Custom detectors

```python
peyeeye.create_entity(
    id="ORDER_ID",
    kind="regex",
    pattern=r"#A-\d{6,}",
    examples=["#A-884217", "#A-007431"],
    confidence_floor=0.9,
)

# dry-run a pattern before saving
peyeeye.test_pattern(pattern=r"#A-\d{6,}", text="ref #A-884217 and #A-1")
#   → TestPatternResponse(count=1, matches=[PatternMatch(value="#A-884217", ...)])

# inspect / update / retire
peyeeye.list_entities()
peyeeye.update_entity("ORDER_ID", enabled=False)
peyeeye.delete_entity("ORDER_ID")

# starter templates (Twilio SIDs, Stripe keys, AWS access keys, etc.)
for tpl in peyeeye.entity_templates():
    print(tpl.id, tpl.pattern)
```

## Sessions

```python
peyeeye.get_session("ses_…")       # SessionInfo
peyeeye.delete_session("ses_…")    # drop immediately
```

## Framework integrations

### LangChain

Drop-in wrapper around any LangChain `Runnable` (chat model, LLM, or chain).
Redacts the prompt before the model sees it, rehydrates tokens in the response,
and opens a fresh session per `invoke` so tokens never leak across requests.

```python
from langchain_openai import ChatOpenAI
from peyeeye import Peyeeye
from peyeeye.langchain import with_peyeeye

peyeeye = Peyeeye(api_key=os.environ["PEYEEYE_KEY"])
model = with_peyeeye(ChatOpenAI(model="gpt-4o-mini"), client=peyeeye)

print(model.invoke("Hi, I'm Ada — email me at ada@a-e.com"))
```

`peyeeye.langchain` has no hard dependency on LangChain — if `langchain-core`
is installed the wrapper is a proper `Runnable` (composable with `|` in LCEL
pipelines); without it you still get a callable with the same `.invoke` /
`.ainvoke` / `.batch` surface.

Opt into stateless sealed mode (no server-side mapping) with
`with_peyeeye(model, client=peyeeye, stateless=True)`.

Accepted prompt shapes: plain strings, chat-message lists (`HumanMessage`,
`SystemMessage`, …), tuple shorthand (`("human", "Hi Ada")`), dict messages
(`{"role": "user", "content": "…"}`), and multimodal content lists — image
parts pass through untouched.

### LiteLLM

Two ways to bolt peyeeye onto a LiteLLM-based app:

```python
import litellm
from peyeeye import Peyeeye
from peyeeye.litellm import with_peyeeye

peyeeye = Peyeeye(api_key=os.environ["PEYEEYE_KEY"])
completion = with_peyeeye(litellm.completion, client=peyeeye)

resp = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi, I'm Ada"}],
)
print(resp.choices[0].message.content)  # already rehydrated
```

`with_peyeeye` wraps sync or async completion functions and opens a fresh
session per call. For LiteLLM's proxy/callback path, register a
`PeyeeyeHandler` instead:

```python
import litellm
from peyeeye.litellm import PeyeeyeHandler

litellm.callbacks = [PeyeeyeHandler(client=peyeeye)]
```

Multimodal content (text + image parts), async `acompletion`, and stateless
sealed sessions (`with_peyeeye(..., stateless=True)`) are all supported.

## Errors

Every non-2xx response raises `PeyeeyeError` with `.code`, `.status`,
`.message`, and `.request_id`. 429 and 5xx responses are retried with
exponential backoff (`Retry-After` honoured); terminal errors raise
immediately.

```python
from peyeeye import PeyeeyeError

try:
    peyeeye.redact("…")
except PeyeeyeError as e:
    if e.code == "rate_limited":
        ...
    elif e.code == "forbidden":
        ...
    else:
        raise
```

## Configuration

```python
Peyeeye(
    api_key="pk_live_…",
    base_url="https://api.peyeeye.ai",
    timeout=30.0,
    max_retries=3,
)
```

For CI / air-gapped use, `Peyeeye(transport=httpx.MockTransport(handler))`
lets you mount a mock transport without monkey-patching.

## Method reference

| Method | HTTP | Purpose |
| --- | --- | --- |
| `peyeeye.redact(text, ...)` | `POST /v1/redact` | Redact PII; returns token stream + session. |
| `peyeeye.rehydrate(text, session=...)` | `POST /v1/rehydrate` | Substitute tokens back. Accepts `ses_…` or `skey_…`. |
| `peyeeye.redact_stream(chunks, ...)` | `POST /v1/redact/stream` (SSE) | Stream-safe redact. |
| `peyeeye.get_session(id)` | `GET /v1/sessions/{id}` | Inspect mapping metadata. |
| `peyeeye.delete_session(id)` | `DELETE /v1/sessions/{id}` | Evict a session. |
| `peyeeye.list_entities()` | `GET /v1/entities` | Built-ins + your custom detectors. |
| `peyeeye.create_entity(...)` | `POST /v1/entities` | Custom detector. |
| `peyeeye.update_entity(id, ...)` | `PATCH /v1/entities/{id}` | Toggle / tweak. |
| `peyeeye.delete_entity(id)` | `DELETE /v1/entities/{id}` | Retire. |
| `peyeeye.test_pattern(pattern, text)` | `POST /v1/entities/test` | Dry-run a regex. |
| `peyeeye.entity_templates()` | `GET /v1/entities/templates` | Starter patterns. |

Full request / response schemas: <https://peyeeye.ai/docs>.

## Using this SDK from an AI coding assistant

Drop these into your agent's context. Each snippet is self-contained and
compiles as-is.

```python
# Install
# pip install peyeeye

from peyeeye import Peyeeye, PeyeeyeError
import os

client = Peyeeye(api_key=os.environ["PEYEEYE_KEY"])  # or explicit base_url

# Round-trip: redact → call LLM → rehydrate (session-scoped)
with client.shield() as shield:
    safe = shield.redact("Hi, I'm Ada, ada@a-e.com")
    # ... send `safe` to the LLM, get `reply` back ...
    out = shield.rehydrate(reply)

# Stateless (zero server-side state; key is yours to persist)
with client.shield(stateless=True) as shield:
    safe = shield.redact("...")
    key = shield.rehydration_key  # skey_...
    clean = shield.rehydrate("[EMAIL_1] confirmed.")

# Low-level one-shot
r = client.redact("Card 4242 4242 4242 4242")
clean = client.rehydrate("Receipt: [CARD_1].", session=r.session)

# Error handling
try:
    client.redact(text)
except PeyeeyeError as e:
    # e.code ∈ {"rate_limited","forbidden","invalid_request","server_error", ...}
    # e.status, e.message, e.request_id
    raise
```

**Endpoint envelope**: all requests use `Authorization: Bearer <api_key>` against
`https://api.peyeeye.ai/v1/*`. Errors follow `{code, message, request_id}`
and surface as `PeyeeyeError`. Responses are plain JSON (dataclasses via
`from_dict`).

**Do**: reuse one `Peyeeye(...)` per process; call `.close()` or use it as a
context manager at shutdown.
**Don't**: open a new client per request, call `flush()` mid-stream, or parse
`skey_` blobs yourself — the API opens them.

## License

MIT.
