"""
SDK 0.8.0 db.analysis.* and db.model.interval_calibration() tests.
No running server required — HTTP is mocked.

Response shapes below are trimmed from real payloads captured by replaying
each method's real request against a live platform TestClient during
implementation (DEF-0030 — docs/context/deferred_items.yaml in the platform
repo) — not invented shapes.
"""

import sys
import os

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock
from databubble.analysis import AnalysisClient
from databubble.model import ModelClient
from databubble.models import JourneyResult, SkillResult
from databubble.exceptions import SDKUsageError

RNG = np.random.default_rng(7)


@pytest.fixture
def df():
    n = 40
    return pd.DataFrame({
        "price": RNG.uniform(1, 10, n),
        "sales": RNG.uniform(50, 200, n),
    })


@pytest.fixture
def analysis_client():
    http = MagicMock()
    return AnalysisClient(http), http


def _journey_result(result_dict: dict) -> JourneyResult:
    return JourneyResult(
        journey_type="elasticity", halted=False, halt_reason=None, primary_estimate=None,
        plain_english_summary="", warnings=[], assumptions_met=True,
        raw={"result": result_dict},
    )


# ---------------------------------------------------------------------------
# db.analysis.power_plan
# ---------------------------------------------------------------------------

MOCK_SAMPLE_SIZE_RESPONSE = {
    "metric_type": "continuous", "baseline": {"mean": 100, "std": 20}, "mde": 5,
    "mde_type": "absolute", "effect_size": 0.25, "effect_size_metric": "cohens_d",
    "alpha": 0.05, "power_target": 0.8, "alternative": "two-sided", "ratio": 1.0,
    "n_per_group": {"group_a": 253, "group_b": 253}, "n_total": 506,
    "interpretation": "needs ~253 per group",
}


def test_power_plan_sample_size(analysis_client):
    client, http = analysis_client
    http.post_json.return_value = MOCK_SAMPLE_SIZE_RESPONSE

    result = client.power_plan(mode="sample_size", metric_type="continuous",
                                baseline_mean=100, baseline_std=20, mde=5)

    http.post_json.assert_called_once()
    assert http.post_json.call_args[0][0] == "/v1/power/plan"
    assert result.n_total == 506
    assert result.mde_absolute is None


def test_power_plan_requires_mde_for_sample_size(analysis_client):
    client, _ = analysis_client
    with pytest.raises(SDKUsageError, match="mde"):
        client.power_plan(mode="sample_size", metric_type="continuous")


def test_power_plan_requires_n_per_group_for_detectable_effect(analysis_client):
    client, _ = analysis_client
    with pytest.raises(SDKUsageError, match="n_per_group"):
        client.power_plan(mode="detectable_effect", metric_type="continuous")


def test_power_plan_bad_mode_raises(analysis_client):
    client, _ = analysis_client
    with pytest.raises(SDKUsageError, match="mode"):
        client.power_plan(mode="bogus", metric_type="continuous")


# ---------------------------------------------------------------------------
# db.analysis.eda / pulse / scope
# ---------------------------------------------------------------------------

MOCK_EDA_RESPONSE = {
    "status": "ok",
    "report": {
        "n_rows": 40, "n_cols": 2, "n_flagged_columns": 1, "n_critical_flags": 0,
        "plain_english_summary": "2 columns profiled.", "column_profiles": [],
        "critical_flags": [], "top_relationships": [], "journey_recommendations": [],
        "exportable": True,
    },
    "summary": "40 rows, 2 cols, 1 flagged.",
    "warnings": ["col X has 5% missing"],
    "chapter_ref": "Chapter 01",
}


def test_analysis_eda(analysis_client, df):
    client, http = analysis_client
    http.post_multipart.return_value = MOCK_EDA_RESPONSE

    result = client.eda(df)

    http.post_multipart.assert_called_once()
    assert http.post_multipart.call_args[0][0] == "/v1/eda"
    assert result.n_rows == 40
    assert result.n_flagged_columns == 1
    assert result.plain_english_summary == "2 columns profiled."


def test_analysis_eda_requires_two_columns(analysis_client):
    client, _ = analysis_client
    with pytest.raises(SDKUsageError, match="at least 2 columns"):
        client.eda(pd.DataFrame({"only_col": [1, 2, 3]}))


def test_analysis_eda_empty_df_raises(analysis_client):
    client, _ = analysis_client
    with pytest.raises(SDKUsageError, match="empty"):
        client.eda(pd.DataFrame({"a": [], "b": []}))


def test_analysis_eda_export_pdf(analysis_client, df):
    client, http = analysis_client
    http.post_multipart.return_value = MOCK_EDA_RESPONSE
    http.post_bytes.return_value = b"%PDF-1.4 fake"

    result = client.eda(df)
    pdf = result.export_pdf()

    http.post_bytes.assert_called_once_with("/v1/export/eda", {"eda": result.report})
    assert pdf == b"%PDF-1.4 fake"


MOCK_PULSE_RESPONSE = {
    "status": "ok",
    "report": {
        "n_rollups": 1, "raw_eda": {}, "rollups": [{"grain": "week", "n_rows": 12}],
        "deferred": [], "notes": [], "caveats": ["Ecological fallacy applies to rolled-up rates."],
    },
}


def test_analysis_pulse(analysis_client, df):
    client, http = analysis_client
    http.post_multipart.return_value = MOCK_PULSE_RESPONSE

    result = client.pulse(df)

    assert result.n_rollups == 1
    assert result.caveats == ["Ecological fallacy applies to rolled-up rates."]
    frame = result.rollups
    assert len(frame) == 1


def test_analysis_pulse_export_pdf(analysis_client, df):
    client, http = analysis_client
    http.post_multipart.return_value = MOCK_PULSE_RESPONSE
    http.post_bytes.return_value = b"%PDF-1.4 fake"

    result = client.pulse(df)
    pdf = result.export_pdf()

    http.post_bytes.assert_called_once_with("/v1/export/pulse", {"pulse": result.report})
    assert pdf == b"%PDF-1.4 fake"


MOCK_SCOPE_RESPONSE = {
    "status": "ok",
    "recommendation": {
        "regime": "sampling_safe", "journey_candidates": ["elasticity", "driver"],
        "recommended_grain": None, "recommended_sample_rows": 5000,
        "sampling_method": "random", "power_warning": None,
        "rationale": "Data looks i.i.d. — a random sample is representative.",
        "blocking": False,
    },
    "_meta": {"tier": "pro", "key_prefix": "dbk_test"},
}


def test_analysis_scope(analysis_client, df):
    client, http = analysis_client
    http.post_multipart.return_value = MOCK_SCOPE_RESPONSE

    result = client.scope(df.head(10), "Does price affect sales?")

    assert result.regime == "sampling_safe"
    assert result.blocking is False
    assert result.journey_candidates == ["elasticity", "driver"]


def test_analysis_scope_requires_problem_description(analysis_client, df):
    client, _ = analysis_client
    with pytest.raises(SDKUsageError, match="problem_description"):
        client.scope(df.head(10), "")


# ---------------------------------------------------------------------------
# db.analysis.export_skill_pack
# ---------------------------------------------------------------------------

MOCK_SKILL_PACK_RESPONSE = {
    "skill_name": "store-weekly-sales", "skill_md": "# SKILL.md contents",
    "profile_json": "{}", "updating_md": "# UPDATING.md contents",
    "n_rows": 40, "n_columns": 2, "blocking_count": 0,
}


def test_export_skill_pack(analysis_client, df):
    client, http = analysis_client
    http.post_multipart.return_value = MOCK_SKILL_PACK_RESPONSE

    result = client.export_skill_pack(df, table_label="Store weekly sales")

    assert result.skill_name == "store-weekly-sales"
    assert result.n_rows == 40


def test_export_skill_pack_save(analysis_client, df, tmp_path):
    client, http = analysis_client
    http.post_multipart.return_value = MOCK_SKILL_PACK_RESPONSE

    result = client.export_skill_pack(df, table_label="Store weekly sales")
    result.save(str(tmp_path))

    assert (tmp_path / "SKILL.md").read_text() == "# SKILL.md contents"
    assert (tmp_path / "profile.json").read_text() == "{}"
    assert (tmp_path / "UPDATING.md").read_text() == "# UPDATING.md contents"


def test_export_skill_pack_bad_outcome_raises(analysis_client, df):
    client, _ = analysis_client
    with pytest.raises(SDKUsageError, match="outcome"):
        client.export_skill_pack(df, table_label="x", outcome="not_a_column")


# ---------------------------------------------------------------------------
# db.analysis.correlation_export / forecast_export
# ---------------------------------------------------------------------------

MOCK_CORRELATION_EXPORT_RESPONSE = {
    "schema_version": "1.0", "kind": "correlation_diagnostic",
    "result": {"outcome": "sales", "n": 40, "predictors": ["price"]},
    "provenance": {}, "notes": [],
}


def test_correlation_export_json(analysis_client):
    client, http = analysis_client
    http.post_json.return_value = MOCK_CORRELATION_EXPORT_RESPONSE
    result_wrapper = _journey_result({"correlation_export_ref": {"predictors": ["price"]}})

    result = client.correlation_export(result_wrapper, format="json")

    http.post_json.assert_called_once_with(
        "/v1/correlation/export",
        {"result_ref": {"predictors": ["price"]}, "format": "json"},
    )
    assert result.outcome == "sales"


def test_correlation_export_csv_returns_bytes(analysis_client):
    client, http = analysis_client
    http.post_bytes.return_value = b"PK\x03\x04 fake zip"
    result_wrapper = _journey_result({"correlation_export_ref": {"predictors": ["price"]}})

    result = client.correlation_export(result_wrapper, format="csv")

    http.post_bytes.assert_called_once_with(
        "/v1/correlation/export",
        {"result_ref": {"predictors": ["price"]}, "format": "csv"},
    )
    assert result == b"PK\x03\x04 fake zip"


def test_correlation_export_missing_ref_raises(analysis_client):
    client, _ = analysis_client
    result_wrapper = _journey_result({})
    with pytest.raises(SDKUsageError, match="correlation_export_ref"):
        client.correlation_export(result_wrapper)


MOCK_FORECAST_EXPORT_RESPONSE = {
    "schema_version": "1.0", "kind": "forecast_card", "target": "sales",
    "model": {}, "forecast": [], "diagnostics": {}, "provenance": {}, "notes": [],
}


def test_forecast_export_json(analysis_client):
    client, http = analysis_client
    http.post_json.return_value = MOCK_FORECAST_EXPORT_RESPONSE
    result_wrapper = _journey_result({"forecast_export_ref": {"findings": {}, "family": "arima"}})

    result = client.forecast_export(result_wrapper, format="json")

    assert result.target == "sales"


def test_forecast_export_csv_returns_bytes(analysis_client):
    client, http = analysis_client
    http.post_bytes.return_value = b"date,forecast\n2026-01-01,10.0\n"
    result_wrapper = _journey_result({"forecast_export_ref": {"findings": {}, "family": "ets"}})

    result = client.forecast_export(result_wrapper, format="csv")

    assert result == b"date,forecast\n2026-01-01,10.0\n"


# ---------------------------------------------------------------------------
# db.model.interval_calibration
# ---------------------------------------------------------------------------

MOCK_INTERVAL_CALIBRATION_RESPONSE = {
    "result": {
        "summary": "Coverage: 94% (target 95%)", "findings": {"empirical_coverage": 0.94},
        "warnings": [], "recommendations": [], "chapter_ref": "Chapter 09",
        "skill_name": "interval_calibration", "column": "sales", "halted": False,
    },
    "_meta": {"tier": "pro"},
}


@pytest.fixture
def model_client():
    http = MagicMock()
    return ModelClient(http), http


def test_model_interval_calibration(model_client):
    client, http = model_client
    http.post_json.return_value = MOCK_INTERVAL_CALIBRATION_RESPONSE
    card = {"outcome": "sales", "terms": []}
    rows = pd.DataFrame({"price": [1.0, 2.0]})

    result = client.interval_calibration(card, rows, actuals=[10.0, 20.0])

    assert isinstance(result, SkillResult)
    payload = http.post_json.call_args[0][1]
    assert payload["actuals"] == [10.0, 20.0]
    assert payload["interval_kind"] == "prediction"
    assert result.summary == "Coverage: 94% (target 95%)"


def test_model_interval_calibration_mismatched_lengths_raises(model_client):
    client, _ = model_client
    card = {"outcome": "sales", "terms": []}
    rows = pd.DataFrame({"price": [1.0, 2.0]})

    with pytest.raises(SDKUsageError, match="actuals"):
        client.interval_calibration(card, rows, actuals=[10.0])


def test_model_interval_calibration_accepts_result_wrapper(model_client):
    """Card can be a ModelCardResult (has .raw), not just a dict."""
    from databubble.models import ModelCardResult

    client, http = model_client
    http.post_json.return_value = MOCK_INTERVAL_CALIBRATION_RESPONSE
    card = ModelCardResult(outcome="sales", kind="card", raw={"outcome": "sales", "terms": []})
    rows = pd.DataFrame({"price": [1.0]})

    client.interval_calibration(card, rows, actuals=[10.0])

    payload = http.post_json.call_args[0][1]
    assert payload["card"] == {"outcome": "sales", "terms": []}


# ---------------------------------------------------------------------------
# Client wiring
# ---------------------------------------------------------------------------

def test_client_exposes_analysis_namespace():
    from databubble import DataBubble

    db = DataBubble(api_key="dbk_test1234567890")
    assert isinstance(db.analysis, AnalysisClient)
