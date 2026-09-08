# databubble/client.py
"""
DataBubble — root client.

Usage:
    from databubble import DataBubble
    db = DataBubble(api_key="dbk_...")
    result = db.journeys.driver(df, outcome_col="sales", candidate_cols=[...])
    print(result)            # regression table
"""

from __future__ import annotations

import json
import time
from typing import Optional

from databubble._version import __version__
from databubble.exceptions import (
    AuthError, ForbiddenError, RateLimitError,
    SkillError, ServerError, DataBubbleError,
)
from databubble.skills import SkillsClient
from databubble.memory import MemoryClient
from databubble.journeys import JourneysClient
from databubble.model import ModelClient
from databubble.scorecard import ScorecardClient
from databubble.segments import SegmentsClient
from databubble.analysis import AnalysisClient
from databubble.qa_audit import QaAuditClient


DEFAULT_BASE_URL = "https://api.databubble.ai"

# Skills and memory are quick. Journeys run a full multi-step analysis on a
# single-worker server-side compute pool with no server-side timeout, so the
# client timeout is the only ceiling — 60s was routinely too short.
DEFAULT_TIMEOUT = 60.0
DEFAULT_JOURNEY_TIMEOUT = 300.0

# The compute pool returns 503 + Retry-After when it is saturated. That is a
# queue signal, not an error, so retry it rather than surfacing it.
DEFAULT_MAX_RETRIES = 3
RETRY_STATUS = (503,)
MAX_RETRY_SLEEP = 30.0


class _HTTPClient:
    """
    Thin HTTP client. Uses httpx if available, falls back to urllib.
    Handles auth header injection, retry-on-503, and error mapping.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max(0, int(max_retries))
        self._session = None
        self._init_session()

    @property
    def _headers(self) -> dict:
        return {
            "X-API-Key": self._api_key,
            "User-Agent": f"databubble-python/{__version__}",
        }

    def _init_session(self):
        try:
            import httpx
            self._session = httpx.Client(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
            )
            self._backend = "httpx"
        except ImportError:
            # Fall back to urllib — no session, headers injected per-request
            self._backend = "urllib"

    # -- errors -----------------------------------------------------------
    @staticmethod
    def _parse_body(raw_text: str, status_code: int) -> dict:
        """Never let a non-JSON error body (proxy HTML, empty 502) mask the status."""
        try:
            parsed = json.loads(raw_text) if raw_text else {}
            return parsed if isinstance(parsed, dict) else {"error": parsed}
        except (ValueError, TypeError):
            snippet = (raw_text or "").strip()[:200]
            return {"error": snippet or f"Non-JSON response (HTTP {status_code})"}

    def _raise_for_status(self, status_code: int, body: dict):
        # Route-level errors wrap in {"detail": {"error": ...}}; FastAPI's own
        # HTTPException uses {"detail": "<string>"}; middleware errors use a flat
        # {"error": ...}; the model/scorecard/segment-scorer score-and-predict
        # endpoints return their structured validation failures as a flat
        # {"error": "input_validation", "message": "<the real text>", ...}
        # with no "detail" wrapper at all (they bypass FastAPI's HTTPException
        # to carry missing_columns/unseen_levels alongside it) — "message" is
        # the human text there, "error" is just a category tag, so it has to
        # be checked first or every one of those errors surfaces as the
        # useless literal string "input_validation". Extract from all of
        # these — the string form carries the most useful text (row ceilings,
        # "Server at capacity") and used to be thrown away in favour of a
        # bare "HTTP 503".
        detail = body.get("detail")
        msg = None
        if isinstance(detail, dict):
            msg = detail.get("error") or detail.get("message")
        elif isinstance(detail, str) and detail.strip():
            msg = detail
        msg = msg or body.get("message") or body.get("error") or f"HTTP {status_code}"

        if status_code == 401:
            raise AuthError(msg, status_code, body)
        if status_code == 403:
            raise ForbiddenError(msg, status_code, body)
        if status_code == 429:
            raise RateLimitError(msg, status_code, body)
        if status_code in (400, 422):
            raise SkillError(msg, status_code, body)
        if status_code >= 500:
            raise ServerError(f"Server error ({status_code}): {msg}", status_code, body)
        if status_code >= 400:
            raise DataBubbleError(msg, status_code, body)

    @staticmethod
    def _retry_delay(headers, attempt: int) -> float:
        """Honour Retry-After when the server sets it, else exponential backoff."""
        raw = None
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            raw = None
        if raw:
            try:
                return min(float(raw), MAX_RETRY_SLEEP)
            except (TypeError, ValueError):
                pass
        return min(2.0 ** attempt, MAX_RETRY_SLEEP)

    # -- requests ---------------------------------------------------------
    def post_json(self, path: str, payload: dict, timeout: Optional[float] = None) -> dict:
        """POST with JSON body. Retries 503 (compute pool at capacity)."""
        url = f"{self._base_url}{path}"
        effective_timeout = timeout or self._timeout

        for attempt in range(self._max_retries + 1):
            if self._backend == "httpx":
                response = self._session.post(path, json=payload, timeout=effective_timeout)
                body = self._parse_body(response.text, response.status_code)
                if response.status_code in RETRY_STATUS and attempt < self._max_retries:
                    time.sleep(self._retry_delay(response.headers, attempt))
                    continue
                self._raise_for_status(response.status_code, body)
                return body

            import urllib.request, urllib.error
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json", **self._headers},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body = self._parse_body(e.read().decode("utf-8", "replace"), e.code)
                if e.code in RETRY_STATUS and attempt < self._max_retries:
                    time.sleep(self._retry_delay(e.headers, attempt))
                    continue
                self._raise_for_status(e.code, body)
                raise ServerError(f"Unexpected response ({e.code})", e.code, body)

        raise ServerError(
            f"Server still at capacity after {self._max_retries} retries. "
            "The compute pool runs one job at a time — retry shortly.",
            503,
            {},
        )

    def get_text(self, path: str) -> Optional[str]:
        """GET returning text — used for chart SVG fetches."""
        if self._backend == "httpx":
            response = self._session.get(path)
            if response.status_code == 404:
                return None
            body = self._parse_body(response.text, response.status_code) if response.status_code >= 400 else {}
            self._raise_for_status(response.status_code, body)
            return response.text

        import urllib.request, urllib.error
        req = urllib.request.Request(f"{self._base_url}{path}", headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            self._raise_for_status(e.code, self._parse_body(e.read().decode("utf-8", "replace"), e.code))
            return None

    def get_bytes(self, path: str) -> Optional[bytes]:
        if self._backend == "httpx":
            response = self._session.get(path)
            if response.status_code == 404:
                return None
            self._raise_for_status(response.status_code, {})
            return response.content

        import urllib.request, urllib.error
        req = urllib.request.Request(f"{self._base_url}{path}", headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            self._raise_for_status(e.code, {})
            return None

    def post_bytes(self, path: str, payload: dict) -> bytes:
        """
        POST with a JSON body, return the raw response bytes rather than
        parsing JSON — for routes that hand back a PDF, CSV, or ZIP
        (POST /v1/export/*, /v1/qa-audit/export, /v1/correlation/export and
        /v1/forecast/export with format="csv"/"zip"). The GET-only
        get_bytes() above doesn't cover these — they need a request body.
        """
        url = f"{self._base_url}{path}"

        if self._backend == "httpx":
            response = self._session.post(path, json=payload, timeout=self._timeout)
            if response.status_code >= 400:
                self._raise_for_status(
                    response.status_code,
                    self._parse_body(response.text, response.status_code),
                )
            return response.content

        import urllib.request, urllib.error
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", **self._headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = self._parse_body(e.read().decode("utf-8", "replace"), e.code)
            self._raise_for_status(e.code, body)
            raise ServerError(f"Unexpected response ({e.code})", e.code, body)

    def post_multipart(self, path: str, fields: dict, files) -> dict:
        """
        POST with multipart form data. Returns parsed response dict.

        files may be:
          - dict: {field_name: (filename, data, content_type)} — single file per field
          - list of (field_name, (filename, data, content_type)) — supports repeated field names
            for list[UploadFile] parameters (M-5: memory_files needs repeated "memory_files" key)
        """
        if self._backend != "httpx":
            raise DataBubbleError(
                "Multipart upload requires httpx. Install with: pip install httpx"
            )

        form_data = {k: v for k, v in fields.items() if v is not None}
        if isinstance(files, dict):
            file_list = [(k, v) for k, v in files.items()]
        else:
            file_list = list(files)
        response = self._session.post(path, data=form_data, files=file_list)
        body = self._parse_body(response.text, response.status_code)
        self._raise_for_status(response.status_code, body)
        return body

    def close(self):
        if self._backend == "httpx" and self._session:
            self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DataBubble:
    """
    DataBubble API client.

    Args:
        api_key:  Your API key (starts with dbk_). Get one at databubble.ai.
        base_url: API base URL. Defaults to https://api.databubble.ai.
                  Override for local development: http://localhost:8000
        timeout:  Request timeout in seconds for skills and memory. Default 60.
        journey_timeout: Request timeout for db.journeys.* calls. Default 300 —
                  journeys have no server-side timeout and queue behind a
                  single compute worker, so a short client timeout loses work
                  the server is still doing.
        max_retries: How many times to retry a 503 from the compute pool,
                  honouring Retry-After. Default 3. Set 0 to disable.

    Example:
        db = DataBubble(api_key="dbk_...")

        r = db.journeys.driver(df, outcome_col="sales",
                               candidate_cols=["price", "promotion"])
        print(r)                 # regression table
        r.estimates              # DataFrame
        r.explain()              # business narrative, on request
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        journey_timeout: float = DEFAULT_JOURNEY_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        import os
        resolved_key = api_key or os.environ.get("DATABUBBLE_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "API key required. Pass api_key= or set DATABUBBLE_API_KEY env var. "
                "Get a key at databubble.ai."
            )
        if not resolved_key.startswith("dbk_"):
            raise ValueError(
                f"Invalid API key format. Keys start with 'dbk_'. Got: {resolved_key[:8]}..."
            )

        self._http = _HTTPClient(resolved_key, base_url, timeout, max_retries)
        self.skills = SkillsClient(self._http)
        self.memory = MemoryClient(self._http)
        self.journeys = JourneysClient(self._http, timeout=journey_timeout)
        self.model = ModelClient(self._http)
        self.scorecard = ScorecardClient(self._http)
        self.segments = SegmentsClient(self._http)
        self.analysis = AnalysisClient(self._http)
        self.qa_audit = QaAuditClient(self._http)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"DataBubble(base_url='{self._http._base_url}')"
