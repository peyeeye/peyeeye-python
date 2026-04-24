# AGENTS.md — peyeeye-python

Orientation for AI coding agents working on the official Python SDK for peyeeye.ai. Humans: read `README.md` first.

## What this is

`peyeeye` — synchronous Python client for the peyeeye PII redaction / rehydration API. Mirrors the full `/v1/*` surface: redact, rehydrate, streaming, sessions, custom detectors, stateless sealed mode. Published to PyPI as `peyeeye`; current version `1.0.0`.

## Layout

```
peyeeye/
  __init__.py       Public re-exports
  client.py         Peyeeye client + Shield context manager (core logic)
  errors.py         PeyeeyeError + typed subclasses (rate limit, auth, etc.)
  models.py         Dataclasses: RedactResponse, RehydrateResponse, etc.
  py.typed          PEP 561 marker — keep it
tests/              pytest suites, one per feature area
pyproject.toml      Build + deps (single runtime dep: httpx>=0.27)
```

## Run the tests

```
python -m pip install -e ".[dev]"
python -m pytest
```

Tests mock `httpx` via `respx` — no network. `conftest.py` wires the fixtures.

## Load-bearing invariants

- **Single runtime dep.** `httpx` only. Do not add `requests`, `pydantic`, `typing_extensions`, etc. — users install this into LLM pipelines where fewer deps is strictly better.
- **Streaming partial-token buffer** (`_PARTIAL_TOKEN_TAIL = re.compile(r"\[[A-Z_0-9]*$")` in `client.py`). When an LLM streams back `...[EMAIL` and the `_1]` arrives in the next chunk, `Shield.rehydrate_chunk()` holds the tail until the bracket closes. Breaking this prints literal `[EMAIL_1]` to end users. `flush()` must only be called after the upstream stream closes — calling it mid-stream emits a partial token.
- **`shield()` is a `@contextmanager`.** On exit it auto-closes the session server-side. Don't replace it with a plain function that returns an object — users rely on `with` for cleanup.
- **Retry policy** (`_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}`). Honors `Retry-After` when present, otherwise exponential back-off with jitter. `max_retries` default is 3. Do not retry 4xx other than 429 — those are caller errors.
- **`Idempotency-Key` is caller-supplied**, never auto-generated. Mutating the body between retries with the same key is the caller's problem, not ours.

## Auth

Bearer API key in the `Authorization` header. `pk_live_…` = prod, `pk_test_…` = test. `api_key` is required at construction — `ValueError` if empty. Never log or echo the key.

## Stateless sealed mode

`client.redact(..., session="stateless")` → response includes `rehydration_key: "skey_…"`. Pass the `skey_…` back as `session=` on `rehydrate`. The blob is AES-GCM-sealed server-side; the SDK just pipes bytes. Don't try to parse or validate `skey_…` — treat it as opaque.

## Versioning

SemVer. Breaking changes → major bump. The `v1` in the URL is the HTTP API version, not the SDK version — they're independent. `USER_AGENT` in `client.py` must match `pyproject.toml [project].version`.

## What NOT to do

- **Never** add a runtime dep beyond `httpx`. Dev deps are fine in `[dev]`.
- **Never** log or include `api_key` in exception messages — `PeyeeyeError.__str__` strips it.
- **Never** copy `reverse_index`-equivalent state out of a `Shield` — the server owns session determinism; the SDK just holds the `ses_…` id.
- **Do not** async-ify without an explicit feature request. The sync client is the product; an `AsyncPeyeeye` would be additive, not a replacement.
- **Do not** remove `py.typed` — removing it silently downgrades the downstream typing experience.
- **Do not** invent endpoints. `https://peyeeye.ai/docs` and the backend's `api/urls.py` are the source of truth.

## Where to look

- Client + retry + streaming: `peyeeye/client.py`
- Typed errors: `peyeeye/errors.py`
- Response models: `peyeeye/models.py`
- Tests: `tests/test_*.py` — one file per feature (redact/rehydrate, streaming, shield, sessions, entities, errors)
- Public API docs: https://peyeeye.ai/docs
