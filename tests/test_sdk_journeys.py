"""
SDK v0.3.0 journey method tests.
No running server required — HTTP is mocked.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock
from databubble import DataBubble, JourneyResult
from databubble.exceptions import SDKUsageError
from databubble.journeys import _df_to_rows_payload, _parse_journey_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

@pytest.fixture
def price_sales_df():
    n = 35
    price = RNG.uniform(8, 15, n)
    sales = 200 * price ** -0.85 + RNG.normal(0, 5, n)
    return pd.DataFrame({"price": price, "sales": sales})


@pytest.fixture
def driver_df():
    n = 40
    return pd.DataFrame({
        "sales":       RNG.normal(100, 20, n),
        "price":       RNG.uniform(8, 15, n),
        "promotion":   RNG.choice([0, 1], n),
        "region_code": RNG.choice([1, 2, 3, 4], n),
    })


@pytest.fixture
def segment_df():
    n = 60
    return pd.DataFrame({
        "recency":   RNG.integers(1, 365, n),
        "frequency": RNG.integers(1, 20, n),
        "spend":     RNG.uniform(10, 500, n),
        "segment":   RNG.choice(["High", "Mid", "Low"], n),
    })


@pytest.fixture
def ts_df():
    dates = pd.date_range("2022-01-01", periods=52, freq="W")
    values = 100 + np.arange(52) * 0.5 + RNG.normal(0, 5, 52)
    return pd.DataFrame({"week": dates.astype(str), "sales": values})


MOCK_ELASTICITY_RESPONSE = {
    "status": "ok",
    "journey_type": "elasticity",
    "result": {
        "primary_estimate": -0.847,
        "adj_r_squared": 0.612,
        "assumptions_met": True,
        "revenue_implication": "Demand is elastic. Price increases are likely to reduce total revenue.",
        "plain_english_summary": "A 1% increase in price is associated with a 0.85% decrease in demand.",
        "warnings": [],
        "halted": False,
        "halt_reason": None,
    },
    "_meta": {"journey": "elasticity", "tier": "business", "key_prefix": "dbk_test12"},
}

MOCK_DRIVER_RESPONSE = {
    "status": "ok",
    "journey_type": "driver",
    "result": {
        "primary_estimate": -8.2,
        "plain_english_summary": "Price is the strongest driver of sales.",
        "warnings": [],
        "halted": False,
        "halt_reason": None,
        "assumptions_met": True,
        "selection_output": {
            "recommended": ["price"],
            "caution": ["promotion"],
            "excluded": ["region_code"],
        },
    },
    "_meta": {"journey": "driver", "tier": "business", "key_prefix": "dbk_test12"},
}

MOCK_HALT_RESPONSE = {
    "status": "ok",
    "journey_type": "elasticity",
    "result": {
        "primary_estimate": None,
        "plain_english_summary": "",
        "warnings": ["post_campaign_spend flagged as leakage."],
        "halted": True,
        "halt_reason": "post_campaign_spend flagged as post-outcome leakage (r=0.71).",
        "assumptions_met": None,
    },
    "_meta": {"journey": "elasticity", "tier": "business", "key_prefix": "dbk_test12"},
}


@pytest.fixture
def mock_http_elasticity():
    http = MagicMock()
    http.post_json.return_value = MOCK_ELASTICITY_RESPONSE
    return http


@pytest.fixture
def db_elasticity(mock_http_elasticity):
    client = DataBubble.__new__(DataBubble)
    client._http = mock_http_elasticity
    from databubble.skills import SkillsClient
    from databubble.memory import MemoryClient
    from databubble.journeys import JourneysClient
    client.skills = SkillsClient(mock_http_elasticity)
    client.memory = MemoryClient(mock_http_elasticity)
    client.journeys = JourneysClient(mock_http_elasticity)
    return client


# ---------------------------------------------------------------------------
# Case 1: _df_to_rows_payload — correct shape
# ---------------------------------------------------------------------------

def test_df_to_rows_payload_shape(price_sales_df):
    payload = _df_to_rows_payload(price_sales_df, ["price", "sales"])
    assert payload["columns"] == ["price", "sales"]
    assert len(payload["rows"]) == 35
    assert len(payload["rows"][0]) == 2


# ---------------------------------------------------------------------------
# Case 2: NaN serialised as None in rows
# ---------------------------------------------------------------------------

def test_nan_in_rows_becomes_none():
    df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
    payload = _df_to_rows_payload(df, ["a"])
    assert payload["rows"][1][0] is None
    assert payload["rows"][0][0] == 1.0


# ---------------------------------------------------------------------------
# Case 3: elasticity — correct payload sent to API
# ---------------------------------------------------------------------------

def test_elasticity_payload(db_elasticity, price_sales_df):
    result = db_elasticity.journeys.elasticity(
        price_sales_df,
        price_col="price",
        sales_col="sales",
    )
    call_args = db_elasticity._http.post_json.call_args
    assert call_args[0][0] == "/v1/journeys/elasticity"
    payload = call_args[0][1]
    assert payload["column_map"]["price_col"] == "price"
    assert payload["column_map"]["sales_col"] == "sales"
    assert "rows" in payload["data"]
    assert len(payload["data"]["rows"]) == 35


# ---------------------------------------------------------------------------
# Case 4: elasticity — JourneyResult fields correct
# ---------------------------------------------------------------------------

def test_elasticity_result_fields(db_elasticity, price_sales_df):
    result = db_elasticity.journeys.elasticity(
        price_sales_df, price_col="price", sales_col="sales"
    )
    assert isinstance(result, JourneyResult)
    assert result.journey_type == "elasticity"
    assert result.primary_estimate == -0.847
    assert result.adj_r_squared == 0.612
    assert result.assumptions_met is True
    assert result.halted is False
    assert result.is_reliable() is True
    assert "elastic" in result.revenue_implication.lower()
    assert result.tier == "business"


# ---------------------------------------------------------------------------
# Case 5: halted journey — is_reliable() returns False
# ---------------------------------------------------------------------------

def test_halted_journey_not_reliable(price_sales_df):
    http = MagicMock()
    http.post_json.return_value = MOCK_HALT_RESPONSE
    client = DataBubble.__new__(DataBubble)
    client._http = http
    from databubble.journeys import JourneysClient
    from databubble.skills import SkillsClient
    from databubble.memory import MemoryClient
    client.skills = SkillsClient(http)
    client.memory = MemoryClient(http)
    client.journeys = JourneysClient(http)

    result = client.journeys.elasticity(
        price_sales_df, price_col="price", sales_col="sales"
    )
    assert result.halted is True
    assert result.is_reliable() is False
    assert result.primary_estimate is None
    assert "leakage" in result.halt_reason.lower()


# ---------------------------------------------------------------------------
# Case 6: driver — recommended/caution/excluded parsed
# ---------------------------------------------------------------------------

def test_driver_result_lists(driver_df):
    http = MagicMock()
    http.post_json.return_value = MOCK_DRIVER_RESPONSE
    client = DataBubble.__new__(DataBubble)
    client._http = http
    from databubble.journeys import JourneysClient
    from databubble.skills import SkillsClient
    from databubble.memory import MemoryClient
    client.skills = SkillsClient(http)
    client.memory = MemoryClient(http)
    client.journeys = JourneysClient(http)

    result = client.journeys.driver(
        driver_df,
        outcome_col="sales",
        candidate_cols=["price", "promotion", "region_code"],
    )
    assert result.recommended == ["price"]
    assert result.caution == ["promotion"]
    assert result.excluded == ["region_code"]
    assert result.primary_estimate == -8.2


# ---------------------------------------------------------------------------
# Case 7: SDKUsageError on missing column
# ---------------------------------------------------------------------------

def test_elasticity_missing_column_raises(db_elasticity, price_sales_df):
    with pytest.raises(SDKUsageError) as exc_info:
        db_elasticity.journeys.elasticity(
            price_sales_df,
            price_col="price",
            sales_col="revenue",   # not in df
        )
    assert "revenue" in str(exc_info.value)
    assert "not found" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Case 8: time_series — invalid objective raises SDKUsageError
# ---------------------------------------------------------------------------

def test_time_series_invalid_objective(ts_df):
    http = MagicMock()
    client = DataBubble.__new__(DataBubble)
    client._http = http
    from databubble.journeys import JourneysClient
    from databubble.skills import SkillsClient
    from databubble.memory import MemoryClient
    client.skills = SkillsClient(http)
    client.memory = MemoryClient(http)
    client.journeys = JourneysClient(http)

    with pytest.raises(SDKUsageError) as exc_info:
        client.journeys.time_series(
            ts_df,
            date_col="week",
            value_col="sales",
            objective="predict",   # invalid
        )
    assert "forecast" in str(exc_info.value)
    assert "decompose" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Case 9+: the 6 journeys added alongside item 5 (routes+SDK for
# classification/ab_test/clv/predictive_model/latent_factors/causal_inference)
# ---------------------------------------------------------------------------

def _make_client(http):
    client = DataBubble.__new__(DataBubble)
    client._http = http
    from databubble.journeys import JourneysClient
    from databubble.skills import SkillsClient
    from databubble.memory import MemoryClient
    client.skills = SkillsClient(http)
    client.memory = MemoryClient(http)
    client.journeys = JourneysClient(http)
    return client


@pytest.fixture
def classification_df():
    n = 40
    return pd.DataFrame({
        "churned": RNG.choice([0, 1], n),
        "tenure":  RNG.uniform(1, 60, n),
        "usage":   RNG.uniform(0, 100, n),
    })


@pytest.fixture
def ab_test_df():
    n = 40
    return pd.DataFrame({
        "variant":   RNG.choice(["A", "B"], n),
        "converted": RNG.choice([0, 1], n),
    })


@pytest.fixture
def clv_df():
    n = 40
    return pd.DataFrame({
        "tenure_months": RNG.uniform(1, 60, n),
        "churned":       RNG.choice([0, 1], n),
        "plan_tier":     RNG.uniform(1, 3, n),
    })


@pytest.fixture
def causal_df():
    n = 40
    return pd.DataFrame({
        "treated": RNG.choice([0, 1], n),
        "revenue": RNG.normal(100, 20, n),
        "store_id": RNG.integers(1, 10, n),
        "week": RNG.integers(1, 20, n),
    })


@pytest.fixture
def latent_df():
    n = 40
    return pd.DataFrame({f"q{i}": RNG.normal(size=n) for i in range(1, 7)})


def test_classification_payload(classification_df):
    http = MagicMock()
    http.post_json.return_value = {
        "status": "ok", "journey_type": "classification",
        "result": {"halted": False, "halt_reason": None, "plain_english_summary": "",
                   "warnings": [], "assumptions_met": None},
        "_meta": {"tier": "business", "key_prefix": "dbk_test12"},
    }
    client = _make_client(http)
    client.journeys.classification(
        classification_df, outcome_col="churned", candidate_cols=["tenure", "usage"],
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/classification"
    assert payload["column_map"] == {
        "outcome_col": "churned", "candidate_cols": ["tenure", "usage"], "confounder_cols": [],
    }


def test_ab_test_payload(ab_test_df):
    http = MagicMock()
    http.post_json.return_value = {
        "status": "ok", "journey_type": "ab_test",
        "result": {"halted": False, "halt_reason": None, "plain_english_summary": "",
                   "warnings": [], "assumptions_met": None},
        "_meta": {"tier": "business", "key_prefix": "dbk_test12"},
    }
    client = _make_client(http)
    client.journeys.ab_test(ab_test_df, group_col="variant", metric_col="converted")
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/ab_test"
    assert payload["column_map"] == {"group_col": "variant", "metric_col": "converted"}
    assert payload["options"]["alternative"] == "two-sided"


def test_ab_test_invalid_alternative_raises(ab_test_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.ab_test(
            ab_test_df, group_col="variant", metric_col="converted", alternative="both",
        )


def test_clv_payload(clv_df):
    http = MagicMock()
    http.post_json.return_value = {
        "status": "ok", "journey_type": "clv",
        "result": {"halted": False, "halt_reason": None, "plain_english_summary": "",
                   "warnings": [], "assumptions_met": None},
        "_meta": {"tier": "business", "key_prefix": "dbk_test12"},
    }
    client = _make_client(http)
    client.journeys.clv(
        clv_df, duration_col="tenure_months", event_col="churned",
        covariate_cols=["plan_tier"], margin_per_period=42.0,
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/clv"
    assert payload["column_map"] == {
        "duration_col": "tenure_months", "event_col": "churned", "covariate_cols": ["plan_tier"],
    }
    assert payload["options"]["margin_per_period"] == 42.0


def test_predictive_model_payload_uses_outcome_col_key(classification_df):
    # server key is outcome_col/candidate_cols even though the SDK's own
    # kwarg is target_col — this asserts the translation, not just pass-through.
    http = MagicMock()
    http.post_json.return_value = {
        "status": "ok", "journey_type": "predictive_model",
        "result": {"halted": False, "halt_reason": None, "plain_english_summary": "",
                   "warnings": [], "assumptions_met": None},
        "_meta": {"tier": "business", "key_prefix": "dbk_test12"},
    }
    client = _make_client(http)
    client.journeys.predictive_model(
        classification_df, target_col="churned", feature_cols=["tenure", "usage"],
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/predictive_model"
    assert payload["column_map"] == {"outcome_col": "churned", "candidate_cols": ["tenure", "usage"]}


def test_predictive_model_invalid_error_cost_preference_raises(classification_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.predictive_model(
            classification_df, target_col="churned", feature_cols=["tenure"],
            error_cost_preference="nonsense",
        )


def test_latent_factors_payload(latent_df):
    http = MagicMock()
    http.post_json.return_value = {
        "status": "ok", "journey_type": "latent_factors",
        "result": {"halted": False, "halt_reason": None, "plain_english_summary": "",
                   "warnings": [], "assumptions_met": None,
                   "handoffs": {"factorable": True, "n_factors": 2}},
        "_meta": {"tier": "business", "key_prefix": "dbk_test12"},
    }
    client = _make_client(http)
    result = client.journeys.latent_factors(
        latent_df, indicator_cols=[f"q{i}" for i in range(1, 7)], intent="efa",
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/latent_factors"
    assert payload["column_map"] == {"indicator_cols": [f"q{i}" for i in range(1, 7)]}
    assert payload["options"]["intent"] == "efa"
    assert result.raw["result"]["handoffs"]["n_factors"] == 2


def test_latent_factors_invalid_intent_raises(latent_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.latent_factors(
            latent_df, indicator_cols=[f"q{i}" for i in range(1, 7)], intent="ica",
        )


def test_causal_inference_payload(causal_df):
    http = MagicMock()
    http.post_json.return_value = {
        "status": "ok", "journey_type": "causal_inference",
        "result": {"halted": False, "halt_reason": None, "plain_english_summary": "",
                   "warnings": [], "assumptions_met": None,
                   "handoffs": {"design": "did", "identified": True, "did_estimate": 2.1}},
        "_meta": {"tier": "business", "key_prefix": "dbk_test12"},
    }
    client = _make_client(http)
    result = client.journeys.causal_inference(
        causal_df, treatment_col="treated", outcome_col="revenue",
        design="did", unit_col="store_id", time_col="week", treat_period=12,
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/causal_inference"
    assert payload["column_map"] == {
        "treatment_col": "treated", "outcome_col": "revenue",
        "covariate_cols": [], "unit_col": "store_id", "time_col": "week",
    }
    assert payload["options"]["design"] == "did"
    assert payload["options"]["treat_period"] == 12
    assert result.raw["result"]["handoffs"]["did_estimate"] == 2.1


def test_causal_inference_missing_treatment_col_raises(causal_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.causal_inference(
            causal_df, treatment_col="nonexistent", outcome_col="revenue",
        )


# ---------------------------------------------------------------------------
# Case 20+: the 7 journeys added 2026-08-16 to close the gap between the API
# (17 /v1/journeys/* routes) and the SDK, which had only wrapped 10.
# ---------------------------------------------------------------------------

def _ok_response(journey_type, handoffs=None):
    result = {
        "halted": False, "halt_reason": None, "plain_english_summary": "",
        "warnings": [], "assumptions_met": None,
    }
    if handoffs is not None:
        result["handoffs"] = handoffs
    return {
        "status": "ok", "journey_type": journey_type,
        "result": result,
        "_meta": {"tier": "business", "key_prefix": "dbk_test12"},
    }


@pytest.fixture
def spc_df():
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"day": dates.astype(str), "defect_rate": RNG.normal(2.0, 0.3, n)})


@pytest.fixture
def inventory_df():
    n = 52
    dates = pd.date_range("2025-01-01", periods=n, freq="W")
    return pd.DataFrame({"week": dates.astype(str), "units_sold": RNG.uniform(80, 150, n)})


@pytest.fixture
def churn_risk_df():
    n = 50
    return pd.DataFrame({
        "tenure_months": RNG.uniform(1, 60, n),
        "churned":       RNG.choice([0, 1], n),
        "plan_tier":     RNG.uniform(1, 3, n),
    })


@pytest.fixture
def mmm_df():
    n = 60
    dates = pd.date_range("2025-01-01", periods=n, freq="W")
    return pd.DataFrame({
        "week":          dates.astype(str),
        "revenue":       RNG.uniform(1000, 5000, n),
        "tv_spend":      RNG.uniform(0, 500, n),
        "search_spend":  RNG.uniform(0, 300, n),
    })


@pytest.fixture
def pay_equity_df():
    n = 80
    return pd.DataFrame({
        "salary": RNG.uniform(50000, 150000, n),
        "gender": RNG.choice(["M", "F"], n),
        "level":  RNG.integers(1, 6, n),
    })


@pytest.fixture
def cross_price_df():
    n = 40
    return pd.DataFrame({
        "units":   RNG.uniform(50, 200, n),
        "price_a": RNG.uniform(8, 15, n),
        "price_b": RNG.uniform(8, 15, n),
    })


@pytest.fixture
def intervention_df():
    n = 40
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"day": dates.astype(str), "conversions": RNG.uniform(10, 50, n)})


def test_spc_monitoring_payload(spc_df):
    http = MagicMock()
    http.post_json.return_value = _ok_response("spc_monitoring")
    client = _make_client(http)
    client.journeys.spc_monitoring(spc_df, date_col="day", value_col="defect_rate", baseline_n=30)
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/spc_monitoring"
    assert payload["column_map"] == {"date_col": "day", "value_col": "defect_rate"}
    assert payload["options"]["baseline_n"] == 30
    assert payload["options"]["min_baseline"] == 20


def test_spc_monitoring_missing_column_raises(spc_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.spc_monitoring(spc_df, date_col="day", value_col="nonexistent")


def test_forecast_inventory_payload(inventory_df):
    http = MagicMock()
    http.post_json.return_value = _ok_response("forecast_inventory")
    client = _make_client(http)
    client.journeys.forecast_inventory(
        inventory_df, date_col="week", value_col="units_sold", lead_time_periods=3,
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/forecast_inventory"
    assert payload["column_map"] == {"date_col": "week", "value_col": "units_sold"}
    assert payload["options"]["lead_time_periods"] == 3
    assert payload["options"]["service_level"] == 0.95


def test_churn_clv_at_risk_payload(churn_risk_df):
    http = MagicMock()
    http.post_json.return_value = _ok_response(
        "churn_clv_at_risk", handoffs={"revenue_at_risk_per_customer": 120.5},
    )
    client = _make_client(http)
    result = client.journeys.churn_clv_at_risk(
        churn_risk_df, duration_col="tenure_months", event_col="churned",
        covariate_cols=["plan_tier"], margin_per_period=42.0, cohort_size=5000,
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/churn_clv_at_risk"
    assert payload["column_map"] == {
        "duration_col": "tenure_months", "event_col": "churned", "covariate_cols": ["plan_tier"],
    }
    assert payload["options"]["margin_per_period"] == 42.0
    assert payload["options"]["cohort_size"] == 5000
    assert result.raw["result"]["handoffs"]["revenue_at_risk_per_customer"] == 120.5


def test_mmm_payload(mmm_df):
    http = MagicMock()
    http.post_json.return_value = _ok_response("mmm")
    client = _make_client(http)
    client.journeys.mmm(
        mmm_df, date_col="week", outcome_col="revenue",
        channel_cols=["tv_spend", "search_spend"],
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/mmm"
    assert payload["column_map"]["channel_cols"] == ["tv_spend", "search_spend"]
    assert payload["column_map"]["group_col"] is None
    assert payload["options"]["group_mode"] == "pooled"


def test_mmm_missing_channel_col_raises(mmm_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.mmm(
            mmm_df, date_col="week", outcome_col="revenue",
            channel_cols=["tv_spend", "nonexistent"],
        )


def test_pay_equity_payload(pay_equity_df):
    http = MagicMock()
    http.post_json.return_value = _ok_response(
        "pay_equity", handoffs={"mode": "binary", "adjusted_gap": -0.04},
    )
    client = _make_client(http)
    result = client.journeys.pay_equity(
        pay_equity_df, compensation_col="salary", protected_col="gender",
        factor_cols=["level"],
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/pay_equity"
    assert payload["column_map"] == {
        "compensation_col": "salary", "protected_col": "gender", "factor_cols": ["level"],
    }
    assert payload["options"]["log_compensation"] is True
    assert result.raw["result"]["handoffs"]["mode"] == "binary"


def test_cross_price_payload(cross_price_df):
    http = MagicMock()
    http.post_json.return_value = _ok_response("cross_price")
    client = _make_client(http)
    client.journeys.cross_price(
        cross_price_df, quantity_col="units", own_price_col="price_a",
        other_price_cols=["price_b"], own_portfolio=["price_b"],
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/cross_price"
    assert payload["column_map"]["other_price_cols"] == ["price_b"]
    assert payload["options"]["own_portfolio"] == ["price_b"]


def test_cross_price_invalid_own_portfolio_raises(cross_price_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.cross_price(
            cross_price_df, quantity_col="units", own_price_col="price_a",
            other_price_cols=["price_b"], own_portfolio=["price_c"],  # not in other_price_cols
        )


def test_intervention_lift_single_design_payload(intervention_df):
    http = MagicMock()
    http.post_json.return_value = _ok_response("intervention_lift")
    client = _make_client(http)
    client.journeys.intervention_lift(
        intervention_df, date_col="day", value_col="conversions",
        intervention_date="2026-01-15",
    )
    payload = http.post_json.call_args[0][1]
    assert http.post_json.call_args[0][0] == "/v1/journeys/intervention_lift"
    assert payload["column_map"] == {"date_col": "day", "value_col": "conversions"}
    assert payload["options"]["design"] == "single"
    assert payload["options"]["intervention_date"] == "2026-01-15"


def test_intervention_lift_control_design_requires_treatment_cols(intervention_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.intervention_lift(
            intervention_df, date_col="day", value_col="conversions",
            intervention_date="2026-01-15", design="control",
        )  # missing treatment_col/unit_col/time_col


def test_intervention_lift_missing_intervention_date_raises(intervention_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.intervention_lift(
            intervention_df, date_col="day", value_col="conversions",
            intervention_date=None,
        )


def test_intervention_lift_invalid_design_raises(intervention_df):
    client = _make_client(MagicMock())
    with pytest.raises(SDKUsageError):
        client.journeys.intervention_lift(
            intervention_df, date_col="day", value_col="conversions",
            intervention_date="2026-01-15", design="nonsense",
        )


# ---------------------------------------------------------------------------
# JourneyResult.export_pdf() / .export_exec_summary_pdf() (0.8.0 — DEF-0030)
#
# POST /v1/export/journey and .../exec-summary render a Mode 2 guided-session
# dict (journey_type/outcome/steps/final_output/brief). This SDK is Mode 1
# only, so JourneyResult._session_view() reshapes the envelope's already-flat
# fields into that shape — see models.py for the field-by-field mapping.
# ---------------------------------------------------------------------------

def _elasticity_response_for_pdf():
    return {
        "status": "ok", "journey_type": "elasticity",
        "result": {
            "halted": False, "halt_reason": None,
            "plain_english_summary": "A 1% price increase is associated with a 0.85% drop in sales.",
            "warnings": ["Sample size is modest (n=35)."],
            "assumptions_met": True,
            "steps": [{"display_name": "Univariate: price", "node": "univariate_price", "halted": False}],
            "primary_estimate": -0.85, "primary_estimate_caveat": None,
            "primary_label": "Price elasticity of demand", "ci_lower": -1.1, "ci_upper": -0.6,
            "causal_limitation": "Observational data — not a randomized experiment.",
            "outcome": "sales",
        },
        "_meta": {"tier": "business", "key_prefix": "dbk_test12"},
    }


def test_journey_export_pdf_reshapes_envelope_into_session(price_sales_df):
    http = MagicMock()
    http.post_json.return_value = _elasticity_response_for_pdf()
    http.post_bytes.return_value = b"%PDF-1.4 fake"
    client = _make_client(http)

    result = client.journeys.elasticity(price_sales_df, price_col="price", sales_col="sales")
    pdf = result.export_pdf()

    assert pdf == b"%PDF-1.4 fake"
    http.post_bytes.assert_called_once()
    called_path, called_payload = http.post_bytes.call_args[0]
    assert called_path == "/v1/export/journey"
    session = called_payload["session"]
    assert session["journey_type"] == "elasticity"
    assert session["outcome"] == "sales"
    assert session["steps"] == [{"display_name": "Univariate: price", "node": "univariate_price", "halted": False}]
    assert session["final_output"]["primary_estimate"] == -0.85
    assert session["final_output"]["plain_english_summary"] == "A 1% price increase is associated with a 0.85% drop in sales."
    assert session["brief"] == {}  # Mode 1 never collects business_problem/decision_at_stake


def test_journey_export_exec_summary_pdf(price_sales_df):
    http = MagicMock()
    http.post_json.return_value = _elasticity_response_for_pdf()
    http.post_bytes.return_value = b"%PDF-1.4 exec summary"
    client = _make_client(http)

    result = client.journeys.elasticity(price_sales_df, price_col="price", sales_col="sales")
    pdf = result.export_exec_summary_pdf()

    assert pdf == b"%PDF-1.4 exec summary"
    called_path = http.post_bytes.call_args[0][0]
    assert called_path == "/v1/export/journey/exec-summary"


def test_journey_export_pdf_without_http_raises():
    """A JourneyResult built by hand (no _http, e.g. reloaded from a saved
    dict) can't export a PDF — needs a live client to make the request."""
    result = JourneyResult(
        journey_type="elasticity", halted=False, halt_reason=None, primary_estimate=None,
        plain_english_summary="", warnings=[], assumptions_met=True,
        raw={"result": {}},
    )
    with pytest.raises(SDKUsageError, match="export_pdf"):
        result.export_pdf()
