# DataBubble SDK

Statistical Intelligence as a Service. Current version: **0.7.0**.

A thin, typed client over the DataBubble HTTP API (`/v1/*`) — handles
authentication (`X-API-Key`), request/response shaping, and session memory.
It does **not** expose the knowledge-base / `/ask` surface (no `knowledge` or
`ask` module) — that's a REST-only surface for now.

> **Status:** published on PyPI. MIT licensed — `LICENSE` file added
> 2026-08-01, matching the `license = {text = "MIT"}` declared in
> `pyproject.toml`.

```bash
pip install databubble          # httpx + pandas come with it as of 0.5.0+
pip install databubble[notebook]  # adds ipython, for inline chart rendering
```

(Installing straight from GitHub also still works, if you want an
unreleased commit: `pip install git+https://github.com/databubble-ai/databubble-python.git`,
or `pip install -e .` for local development.)

## Quick start

```python
from databubble import DataBubble
import pandas as pd

db = DataBubble(api_key="dbk_...")

# Regression — the output is a regression table, not a sentence
r = db.journeys.driver(df, outcome_col="sales", candidate_cols=["price", "promotion"])
print(r)                 # coef, std err, t, P>|t|, 95% CI, VIF, significance
r.estimates              # the same thing as a pandas DataFrame
r.coefficients["price"]  # -8.2
r.diagnostics            # n, adj R², assumptions_met, transformation applied
r.explain()              # the plain-English narrative, when you want it

# Univariate analysis
result = db.skills.univariate(df["price"])
result.to_frame()        # metric/value DataFrame
result.charts.show()     # renders inline in a notebook

# Missing value profiling
result = db.skills.missing_values(df)

# Export session memory for next session
mem = db.memory.export(df, label="POS data June 2026")
mem.save("pos_memory.json")
```

## Available skills

| Skill | Input | What it does |
|---|---|---|
| `univariate` | Series or DataFrame + column= | Distribution analysis, skewness, bounded ordinal detection, MNAR flags |
| `outliers` | Series or DataFrame + column= | IQR + Z-score outlier detection |
| `missing_values` | DataFrame | MCAR/MAR/MNAR profiling, treatment recommendations |
| `leakage` | DataFrame + outcome= | Post-outcome timing detection, correlation proxy check |
| `bivariate` | DataFrame + x= + y= | Relationship analysis, linearity check |
| `correlation` | DataFrame + x= + y= | Pearson + Spearman, non-linearity flag |
| `transformations` | Series or DataFrame + column= + transform= | Apply and assess log / sqrt / Box-Cox / reflect transforms |

## Journeys

Single-call, end-to-end analyses — `db.journeys.*` — available on **Pro,
Business and Enterprise** tiers, plus four of them (`elasticity`, `driver`,
`segmentation`, `time_series`) on a demo key. Not available on the free
Developer tier. (Pro was granted full journey access on 2026-08-05; the docs
said Business-and-above until 0.5.0.)

Each returns a `JourneyResult`. From 0.5.0 its primary surface is quantitative
— `.estimates` (DataFrame), `.coefficients`, `.effects`, `.diagnostics`,
`.confidence_interval`, `.significant`, `.selected_predictors` — and `print(r)`
renders a statsmodels-style regression table. The business narrative is still
there, at `.explain()`. Full reference: `docs/api/sdk/python.md`; upgrade notes:
`MIGRATION-0.5.md`.

```python
# Price elasticity of demand
result = db.journeys.elasticity(
    df, price_col="price", sales_col="revenue", confounder_cols=["region"],
)
if result.is_reliable():
    print(result.revenue_implication)

# Driver analysis — which variables drive the outcome?
r = db.journeys.driver(
    df, outcome_col="sales", candidate_cols=["price", "promotion", "region"],
)
print(r.selected_predictors)   # .recommended is deprecated — see MIGRATION-0.5.md

# Segmentation — discovery (no label_col) or classification (with label_col)
result = db.journeys.segmentation(df, feature_cols=["recency", "frequency", "spend"])

# Time series — forecast or decompose
result = db.journeys.time_series(df, date_col="week", value_col="sales", objective="forecast")

# Binary classification — churn, fraud, default, response
result = db.journeys.classification(df, outcome_col="churned", candidate_cols=["tenure", "usage"])

# A/B test — randomised experiment, two groups
result = db.journeys.ab_test(df, group_col="variant", metric_col="converted")

# Customer lifetime value / survival analysis
result = db.journeys.clv(df, duration_col="tenure_months", event_col="churned", margin_per_period=42.0)

# Predictive model — classifier built for prediction accuracy
result = db.journeys.predictive_model(df, target_col="default", feature_cols=["income", "credit_score"])

# Latent factors — PCA (compression) or EFA (latent constructs)
result = db.journeys.latent_factors(df, indicator_cols=["q1", "q2", "q3", "q4", "q5"], intent="efa")

# Causal inference — design-based effect from observational data (not a randomised test)
result = db.journeys.causal_inference(
    df, treatment_col="treated", outcome_col="revenue",
    design="did", unit_col="store_id", time_col="week", treat_period=12,
)

# SPC monitoring — is this process stable, or has it shifted?
result = db.journeys.spc_monitoring(df, date_col="day", value_col="defect_rate", baseline_n=60)

# Forecast to inventory — reorder point + safety stock from a demand forecast
result = db.journeys.forecast_inventory(
    df, date_col="week", value_col="units_sold", lead_time_periods=3,
)

# Churn -> CLV at risk — expected revenue at risk from churn probability x CLV
result = db.journeys.churn_clv_at_risk(
    df, duration_col="tenure_months", event_col="churned", margin_per_period=42.0,
)

# Marketing mix model — per-channel response curves, ROI, contribution share
result = db.journeys.mmm(
    df, date_col="week", outcome_col="revenue",
    channel_cols=["tv_spend", "search_spend", "social_spend"],
)

# Pay equity audit — adjusted pay-gap with raw/adjusted/explained breakdown
result = db.journeys.pay_equity(
    df, compensation_col="salary", protected_col="gender", factor_cols=["level", "tenure_years"],
)

# Cross-price elasticity & cannibalization
result = db.journeys.cross_price(
    df, quantity_col="units", own_price_col="price_a", other_price_cols=["price_b", "price_c"],
)

# Intervention lift — did this promo/price change/policy actually move the metric?
result = db.journeys.intervention_lift(
    df, date_col="day", value_col="conversions", intervention_date="2026-06-01",
)
```

## Tiers

| Tier | Price | Analysis calls/month | Knowledge Q&A calls/month | Skills |
|---|---|---|---|---|
| Developer | Free | 100 | 50 | Core analysis skills. No journeys. |
| Pro | $49/month | 2,000 | 1,000 | All skills + all journeys |
| Business | $299/month | 15,000 | Unlimited (fair use) | All skills + all journeys |
| Enterprise | Custom | Unlimited | Unlimited | Everything |

Get a key at [databubble.ai](https://databubble.ai).

## Related repositories

- **databubble** — the main platform (API + skills + journeys + front-end).
- **databubble-knowledge** — the knowledge / Obsidian vault.

## Releasing a new version

See `RELEASING.md` for the full flow. Short version:
`scripts/check_local.sh` (build + test + drift check) -> `scripts/publish_testpypi.sh`
(dry run) -> `scripts/publish_release.sh` (real PyPI, manually gated).

CI (`.github/workflows/ci.yml`) runs the test suite and `scripts/check_drift.py`
on every push and PR. The drift check compares the SDK's wrappers against
`api_manifest.json` — regenerate that file from the app repo whenever a journey
or skill is added, or the check will pass while the SDK falls behind.
