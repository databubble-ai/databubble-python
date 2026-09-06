"""
SDK tests — no running server required.
All HTTP calls are intercepted with a mock _HTTPClient.
"""

import json
import pytest
import sys
import os

# Make SDK importable from sdk/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from unittest.mock import MagicMock, patch

from databubble import DataBubble
from databubble.models import SkillResult, MemoryResult, ReconciliationResult
from databubble.exceptions import (
    AuthError, ForbiddenError, RateLimitError,
    SkillError, SDKUsageError,
)
from databubble.skills import _resolve_single_column, _resolve_two_columns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_SKILL_RESPONSE = {
    "status": "ok",
    "skill_name": "univariate",
    "result": {
        "skill_name": "univariate",
        "column": "price",
        "summary": "Price shows right skew (skewness=2.31). Log transform recommended.",
        "findings": {
            "skewness": 2.31,
            "mean": 11.2,
            "median": 10.0,
            "std": 3.4,
            "is_bounded_ordinal": False,
        },
        "warnings": ["Right skew detected — consider log transform before modelling."],
        "recommendations": ["Apply log transform to price before regression."],
        "chapter_ref": "Chapter 04",
    },
    "n_rows": 30,
    "_meta": {"skill": "univariate", "tier": "developer", "key_prefix": "dbk_test12"},
}

MOCK_MEMORY_RESPONSE = {
    "memory_id": "abc-123",
    "memory_json": {"label": "POS data", "column_memories": []},
    "memory_markdown": "# Analytical Memory — POS data\n",
    "open_count": 1,
    "blocking_count": 0,
    "columns_with_univariate": 3,
}

@pytest.fixture
def mock_http():
    http = MagicMock()
    http.post_json.return_value = MOCK_SKILL_RESPONSE
    http.post_multipart.return_value = MOCK_MEMORY_RESPONSE
    return http

@pytest.fixture
def db(mock_http):
    client = DataBubble.__new__(DataBubble)
    client._http = mock_http
    from databubble.skills import SkillsClient
    from databubble.memory import MemoryClient
    client.skills = SkillsClient(mock_http)
    client.memory = MemoryClient(mock_http)
    return client

@pytest.fixture
def sample_df():
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "price": rng.exponential(scale=10, size=30),
        "sales": rng.normal(loc=100, scale=20, size=30),
        "region": rng.choice(["North", "South", "East"], size=30),
    })


# ---------------------------------------------------------------------------
# Case 1: univariate with Series
# ---------------------------------------------------------------------------

def test_univariate_series(db, sample_df):
    result = db.skills.univariate(sample_df["price"])
    assert isinstance(result, SkillResult)
    assert result.skill_name == "univariate"
    assert result.column == "price"
    assert result.findings["skewness"] == 2.31
    assert result.tier == "developer"
    assert result.has_warnings() is True

    call_args = db._http.post_json.call_args
    assert call_args[0][0] == "/v1/skills/univariate"
    payload = call_args[0][1]
    assert payload["column"] == "price"
    assert len(payload["data"]) == 30


# ---------------------------------------------------------------------------
# Case 2: univariate with DataFrame + column=
# ---------------------------------------------------------------------------

def test_univariate_dataframe_with_column(db, sample_df):
    result = db.skills.univariate(sample_df, column="price")
    assert result.column == "price"

    payload = db._http.post_json.call_args[0][1]
    assert payload["column"] == "price"
    assert len(payload["data"]) == 30


# ---------------------------------------------------------------------------
# Case 3: univariate with DataFrame without column= raises SDKUsageError
# ---------------------------------------------------------------------------

def test_univariate_dataframe_no_column_raises(db, sample_df):
    with pytest.raises(SDKUsageError) as exc_info:
        db.skills.univariate(sample_df)
    assert "column" in str(exc_info.value).lower()
    assert "univariate" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Case 4: univariate with unnamed Series raises SDKUsageError
# ---------------------------------------------------------------------------

def test_univariate_unnamed_series_raises(db):
    s = pd.Series([1.0, 2.0, 3.0])   # no .name
    with pytest.raises(SDKUsageError) as exc_info:
        db.skills.univariate(s)
    assert "unnamed" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Case 5: missing_values requires DataFrame
# ---------------------------------------------------------------------------

def test_missing_values_requires_dataframe(db, sample_df):
    db._http.post_json.return_value = {
        **MOCK_SKILL_RESPONSE,
        "result": {**MOCK_SKILL_RESPONSE["result"], "skill_name": "missing_values"},
    }
    result = db.skills.missing_values(sample_df)
    assert isinstance(result, SkillResult)

    payload = db._http.post_json.call_args[0][1]
    assert "columns" in payload
    assert set(payload["columns"]) == {"price", "sales", "region"}


def test_missing_values_rejects_series(db, sample_df):
    with pytest.raises(SDKUsageError) as exc_info:
        db.skills.missing_values(sample_df["price"])
    assert "DataFrame" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Case 6: leakage — outcome column required
# ---------------------------------------------------------------------------

def test_leakage_missing_outcome_raises(db, sample_df):
    with pytest.raises(SDKUsageError) as exc_info:
        db.skills.leakage(sample_df, outcome="revenue")  # not in df
    assert "revenue" in str(exc_info.value)
    assert "not found" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Case 7: bivariate with DataFrame + x= y=
# ---------------------------------------------------------------------------

def test_bivariate_dataframe(db, sample_df):
    db._http.post_json.return_value = {
        **MOCK_SKILL_RESPONSE,
        "result": {**MOCK_SKILL_RESPONSE["result"], "skill_name": "bivariate"},
    }
    result = db.skills.bivariate(sample_df, x="price", y="sales")
    assert isinstance(result, SkillResult)

    path, payload = db._http.post_json.call_args[0]
    # There is no "bivariate" skill slug server-side — only "bivariate_ts",
    # dispatched by mode=. Found 2026-09-06: this previously posted to
    # /v1/skills/bivariate, which 404'd every real call.
    assert path == "/v1/skills/bivariate_ts"
    assert "price" in payload["columns"]
    assert "sales" in payload["columns"]
    # Flat, top-level keys — api/routes/skill.py passes the request body
    # straight through as params, so a nested "params": {...} sub-object
    # (the previous shape here) is invisible server-side.
    assert payload["column"] == "price"
    assert payload["columns_2"] == ["sales"]
    assert payload["mode"] == "bivariate"


# ---------------------------------------------------------------------------
# Case 8: NaN serialised as None
# ---------------------------------------------------------------------------

def test_nan_serialised_as_none(db):
    import numpy as np
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], name="price")
    db.skills.univariate(s)
    payload = db._http.post_json.call_args[0][1]
    assert None in payload["data"]
    assert not any(v != v for v in payload["data"] if v is not None)  # no NaN in output


# ---------------------------------------------------------------------------
# Case 9: SkillResult convenience methods
# ---------------------------------------------------------------------------

def test_skill_result_convenience(db, sample_df):
    result = db.skills.univariate(sample_df["price"])
    assert result.has_warnings() is True
    assert result.has_blocking_issues() is False
    assert "price" in repr(result)
    assert "univariate" in repr(result)


# ---------------------------------------------------------------------------
# Case 10: memory export — MemoryResult shape
# ---------------------------------------------------------------------------

def test_memory_export(db, sample_df):
    result = db.memory.export(sample_df, label="POS data June 2026")
    assert isinstance(result, MemoryResult)
    assert result.memory_id == "abc-123"
    assert result.label == "POS data June 2026"
    assert result.open_count == 1
    assert result.columns_covered == 3
    assert "POS data" in repr(result)

    # Verify multipart was called with correct label
    call_kwargs = db._http.post_multipart.call_args
    assert call_kwargs is not None


# ---------------------------------------------------------------------------
# Case 11: DataBubble client — API key validation
# ---------------------------------------------------------------------------

def test_client_rejects_missing_key():
    with pytest.raises(ValueError) as exc_info:
        DataBubble(api_key="")
    assert "API key required" in str(exc_info.value)


def test_client_rejects_wrong_format():
    with pytest.raises(ValueError) as exc_info:
        DataBubble(api_key="sk-not-a-databubble-key")
    assert "dbk_" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Case 12: context manager
# ---------------------------------------------------------------------------

def test_context_manager(mock_http):
    client = DataBubble.__new__(DataBubble)
    client._http = mock_http
    from databubble.skills import SkillsClient
    from databubble.memory import MemoryClient
    client.skills = SkillsClient(mock_http)
    client.memory = MemoryClient(mock_http)

    with client as db:
        assert db is client

    mock_http.close.assert_called_once()
