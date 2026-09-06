# Changelog

Tracks what each merged PR/branch did — not a commit-by-commit log (`git log`
is authoritative for that), but a short, human-readable note per change so
anyone (including future you) can see what shipped and why without reading
every diff. Add an entry here as part of each PR, in the same PR.

Format: date, branch/PR, one or two lines on what changed and why.

## 2026-09-06 — 0.7.0: fix 4 broken skill methods, add 7 missing ones

`scripts/check_drift.py`'s manifest mode was comparing against `api_manifest.json`
from 2026-08-18 — three weeks and 129 app commits stale — so it kept reporting
clean while real drift accumulated silently. Running it in sibling-checkout
mode against a live app checkout instead found two separate problems:

1. **`leakage()`, `bivariate()`, `correlation()`, and `transformations()` were
   broken against the real API.** `api/routes/skill.py` passes the whole
   request body straight through as the dispatch layer's params dict — there
   is no nested `"params": {...}` sub-object it unwraps — but all four
   methods sent their skill-specific args nested that way. Confirmed by
   calling the server's own dispatch functions directly: the nested shape
   raises `'outcome' is required` (etc.) every time; the flat shape works.
   `bivariate()` was also posting to `/v1/skills/bivariate`, which doesn't
   exist server-side (only `bivariate_ts`, mode-dispatched) — a 404 on every
   call — and using `x`/`y` params the dispatch layer never reads at all
   (it wants `column`/`columns_2`). Fixed all four; verified end-to-end by
   running each SDK method's built payload straight through the actual
   `skills.dispatch` functions (not just re-checking the drift manifest).
2. **7 registered skills had no SDK method at all**: `data_quality`,
   `nonlinearity`, `linear_regression`, `confounding_remedy`, `oaxaca`,
   `oaxaca_detailed`, `oaxaca_yun_normalized`. Added all seven, matching
   `skills/dispatch.py`'s actual required/optional params exactly.

Also fixed `check_drift.py` itself: its journey-route regex assumed
`@journeys_router.post("/journeys/x")` with nothing else inside the parens,
so it stopped matching anything once the Phase 3 typed-`response_model`
work added a second argument to every route decorator — a false "17 missing
journeys" alarm with zero real drift behind it.

## 2026-08-29 — 0.6.1: pin the mlflow extra to a real dependency

`[mlflow]`'s dependency on `databubble-scoring[mlflow]` was unpinned (no version constraint at
all) in 0.6.0's published metadata, which had shipped before `databubble-scoring` had any real
PyPI release to point at. Now pinned to `>=0.1.0`, `databubble-scoring`'s actual first real PyPI
release (2026-08-29). `pip install databubble[mlflow]` now resolves and imports end-to-end from
PyPI alone for the first time — no editable sibling checkout required.

## 2026-08-24 — 0.6.0: portable model artifacts + offline scoring

Phase 3+5 of the SDK train→predict arc. Adds `db.model` / `db.scorecard` /
`db.segments` — typed clients over the platform's existing model-card/
scorecard/segment-scorer export+score endpoints. `db.model.export(result)`
turns a fitted `JourneyResult` into a portable artifact (coefficients, or a
`FittedPipeline` of primitives, plus a replayable recipe — never a pickled
object), which `.save()`/reload and `db.model.predict(card, new_df)` can
score independent of the session. Also fixes `client.py`'s error-message
extraction: the model/scorecard/segment-scorer score-and-predict endpoints'
structured 422s carry the real message under `"message"`, not `"detail"` —
was silently discarded in favour of the literal string "input_validation".

Each of the three artifact types also gets `.to_mlflow()`: writes a real
MLflow pyfunc model directory, scored fully offline
(`mlflow.pyfunc.load_model(path).predict(df)`) with zero DataBubble network
call at inference. Lazily imports `databubble_scoring.mlflow_pyfunc` so
plain `import databubble` never requires it; new `mlflow` optional-dependency
group (currently resolves only via an editable sibling checkout of
`databubble-scoring` — not yet on PyPI).

Also folds in an unreleased fix from the same period: `.effects` misread
`partial_r_squared_dominance` (a `{"shares": [...], "sum",
"reconciles_to_r2"}` dict, not a flat `{predictor: value}` mapping) as
predictor rows, and merging 3+ effect-size sources that each carry their own
`rank` column raised a pandas `MergeError` on every real
`driver_analysis`/`elasticity` response, not just an edge case.

35 new tests across `test_sdk_model_surface.py`, `test_sdk_mlflow.py`, and
`test_ds_surface.py`. Version bump was written into the model-artifact
commit's own message (0818184/39dbb72/e7573b4 all landed on `ds-first-output`
without ever touching `pyproject.toml`) — applied here as part of merging the
branch, bumping straight from the last PyPI release (0.4.0) since 0.5.0 was
also never actually published.

## 2026-08-18 — 0.5.0: data-scientist-first output

Inverts the result surface. `JourneyResult` now leads with the quantitative
output — `.estimates` (DataFrame of coef / std err / t / p / CI / VIF),
`.coefficients`, `.effects`, `.diagnostics`, `.confidence_interval`,
`.significant`, `.n_observations`, `.selected_predictors`, `.handoffs` — and
`repr()` renders a statsmodels-style table instead of a one-line summary.
`_repr_html_` renders the same thing as a table in notebooks. The business
narrative moved to `.explain()`; `plain_english_summary` is unchanged.

Reads everything lazily off `.raw` via properties rather than lifting fields at
parse time, so an envelope change can no longer silently produce an empty list.

Fixes: `.recommended` / `.caution` / `.excluded` never worked against the live
API — they read `result["selection_output"]`, which the platform does not
return (the real names are `selected_predictors` / `excluded_predictors`, and
the selection block is a sibling of `result`). They now resolve correctly and
emit DeprecationWarning; removal in 1.0. The unit test that "covered" this used
a hand-written fixture with a `selection_output` key that no live response has.

Also: `transformations` wrapper added (7th registered skill, previously
unwrapped); `SkillResult.charts` / `JourneyResult.charts` with lazy fetch from
`GET /v1/charts/{name}`; `journey_timeout` default 300s (journeys have no
server-side timeout); automatic retry of 503 honouring `Retry-After`;
`User-Agent: databubble-python/<version>`; string `detail` and non-JSON error
bodies no longer collapse to "HTTP 503"; vectorised payload serialisation
(replaces a per-row `iterrows()` loop); client-side 500k row pre-check;
`httpx` + `pandas` are now real dependencies; CI workflow runs the tests and
the drift check; `check_journey_drift.py` superseded by `check_drift.py`, which
also covers skills and can run in CI against a committed `api_manifest.json`.
Tier docs corrected: journeys are Pro and above, not Business and above.
Packaging: `license` moved to an SPDX string with `license-files` (the TOML
table form is deprecated by setuptools and stops building in Feb 2027).

## 2026-07-18 — `docs/platform-docs-2026-07-18`

Corrected a code-sample bug in `README.md` (draft had `from databubble import
Client`; the real class is `DataBubble`), and added a pre-PyPI/license-status
note, a "does not expose the knowledge base" clarification, and a related-
repositories section. Added `CONTRIBUTING.md`, `SECURITY.md`, and
`LICENSE_DECISION.md` (license choice is still an open owner decision — MIT
recommended, not yet finalized). Merged directly to `main` (fast-forward, no
conflicts).

## 2026-08-01 — MIT LICENSE file added

`pyproject.toml` already declared `license = {text = "MIT"}` but no `LICENSE`
file existed — the last blocker before a PyPI publish. Owner picked MIT
(recommended in `LICENSE_DECISION.md`) over Apache-2.0/copyleft alternatives;
README release checklist updated to match.

## 2026-08-05 — item 5: 6 new journey wrappers (classification, ab_test, clv,
predictive_model, latent_factors, causal_inference)

`JourneysClient` goes from 4 to 10 methods, matching the 2 new `/v1/journeys/*`
routes added the same day (`api/routes/journeys.py`) for the two graph-only
journeys (latent_factors, causal_inference — neither has a classic
`journey_*.py` driver; both run via `graph_runner.run_graph`). `column_map`
keys verified directly against each route body, not assumed from the review
docs that specced this — `predictive_model`'s server-side key is
`outcome_col`/`candidate_cols` even though the SDK kwarg is `target_col`
(matches `classification`'s shape; the route reuses it). `latent_factors` and
`causal_inference` results carry their domain payload under
`result.raw["result"]["handoffs"]` rather than dedicated `JourneyResult`
fields — deliberately not extending `models.py`/`_parse_journey_result` for
these two to match the SDK's existing scope (only `driver`'s
recommended/caution/excluded gets bespoke parsing; everything else already
leans on `.raw`). Added `README.md` examples for all 6. Version bump to
0.3.0 and PyPI publish are deferred to the (owner-run) publish step, per the
review's own item 5/6 split.

## 2026-08-16 — 7 more journey wrappers, closing the API/SDK gap

`JourneysClient` goes from 10 to 17 methods, catching up to every
`/v1/journeys/*` route the API has: `spc_monitoring`, `forecast_inventory`,
`churn_clv_at_risk`, `mmm`, `pay_equity`, `cross_price`, `intervention_lift`.
These 7 routes shipped in the main app on 2026-08-12/13 (`api/routes/
journeys.py`, all graph-only journeys run via `_run_graph_journey_sync`) but
were never wrapped here — found by diffing the API's route list against
`JourneysClient`'s method list while investigating an unrelated support
question, not by a planned audit. `column_map`/`options` keys verified
directly against each route's docstring and body, not assumed. All 7 return
their domain payload under `result.raw["result"]["handoffs"]`, matching
`latent_factors`/`causal_inference`'s existing convention rather than adding
bespoke `JourneyResult` fields. 21 new tests in `tests/test_sdk_journeys.py`
(payload-shape + validation-error cases per journey, mocked HTTP, no live
server). Version bump to 0.4.0.

Also added `scripts/check_local.sh` (build+test+twine-check+the drift check
that would have caught this gap earlier), `scripts/publish_testpypi.sh`, and
`scripts/publish_release.sh` — a manual-trigger, staged release flow mirroring
`code/scripts/deploy_test.sh`/`promote_demo.sh`'s local-dev -> test -> demo
pattern. See `RELEASING.md` for the updated flow.
