# databubble/skills.py
"""
SkillsClient — typed methods for every DataBubble skill.

Column rules:
  Single-column skills (univariate, outliers):
    - Pass a pd.Series → SDK serialises it
    - Pass a pd.DataFrame + column="price" → SDK extracts and serialises
    - Pass a pd.DataFrame without column= → SDKUsageError (never auto-pick)

  Whole-dataset skills (missing_values, leakage):
    - Pass a pd.DataFrame → SDK serialises all columns

  Two-column skills (bivariate, correlation):
    - Pass a pd.DataFrame + x="price", y="sales"
    - Or pass two pd.Series as positional args
"""

from __future__ import annotations

from typing import Optional, Union
import json

from databubble.models import SkillResult
from databubble.exceptions import SDKUsageError


def _sanitise_value(v, col_name: str):
    """
    Convert a single value for JSON transport.
    NaN / pd.NA / None → None (documented: treated as missing by the API).
    Inf/-Inf → SDKUsageError (JSON cannot represent infinity; caller must handle it).

    M-4: pd.NA (nullable Int64/boolean/string dtypes) breaks `v != v` because
    `pd.NA != pd.NA` returns pd.NA, which raises TypeError in a boolean context.
    Use pd.isna() which handles float NaN, pd.NA, numpy.nan, and None uniformly.
    """
    import math
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except (ImportError, TypeError, ValueError):
        # pd.isna raises TypeError for unhashable types (dicts, lists) — not missing
        pass
    if isinstance(v, float) and math.isinf(v):
        raise SDKUsageError(
            f"Column '{col_name}' contains infinite values (inf or -inf). "
            "Remove or replace them before calling skills. "
            "Note: NaN values are accepted and treated as missing."
        )
    return v


def _series_to_payload(series, col_name: str) -> dict:
    """
    Serialise a pd.Series to the JSON body format.
    NaN → null (documented API contract: treated as missing value).
    Inf raises SDKUsageError — JSON has no Infinity representation.
    """
    return {
        "column": col_name,
        "data": [_sanitise_value(v, col_name) for v in series.tolist()],
    }


def _df_to_payload(df, columns: list[str]) -> dict:
    """Serialise selected DataFrame columns to JSON body format."""
    return {
        "columns": columns,
        "data": {col: [_sanitise_value(v, col) for v in df[col].tolist()] for col in columns},
    }


def _parse_skill_result(response: dict, http=None) -> SkillResult:
    """Build a SkillResult from the API response dict."""
    result = response.get("result", response)
    meta = response.get("_meta", {})
    return SkillResult(
        summary=result.get("summary", ""),
        findings=result.get("findings", {}),
        warnings=result.get("warnings", []),
        recommendations=result.get("recommendations", []),
        chapter_ref=result.get("chapter_ref", ""),
        skill_name=result.get("skill_name", meta.get("skill", "")),
        column=result.get("column"),
        n_rows=response.get("n_rows"),
        tier=meta.get("tier"),
        key_prefix=meta.get("key_prefix"),
        halted=result.get("halted", False),
        halt_reason=result.get("halt_reason"),
        raw=response,
        _http=http,
    )


def _parse_skill_result_first_output(response: dict, http=None) -> SkillResult:
    """Like _parse_skill_result, but for the `outputs[]` envelope shape.

    api/routes/skill.py returns `outputs: [...]` instead of `result: {...}`
    whenever the underlying skill function returns a list — which
    bivariate_ts always does server-side (one entry per Column-2 selection),
    even when the SDK only ever asks for one. This unwraps that single
    entry so callers still get one SkillResult back, matching bivariate()'s
    existing return type.
    """
    outputs = response.get("outputs")
    if outputs:
        return _parse_skill_result(
            {"result": outputs[0], "_meta": response.get("_meta", {}), "n_rows": response.get("n_rows")},
            http=http,
        )
    return _parse_skill_result(response, http=http)


class SkillsClient:
    def __init__(self, http_client):
        self._http = http_client   # injected from DataBubble root client

    def _call(self, skill_name: str, payload: dict) -> SkillResult:
        response = self._http.post_json(f"/v1/skills/{skill_name}", payload)
        return _parse_skill_result(response, http=self._http)

    def _call_first_output(self, skill_name: str, payload: dict) -> SkillResult:
        response = self._http.post_json(f"/v1/skills/{skill_name}", payload)
        return _parse_skill_result_first_output(response, http=self._http)

    # -----------------------------------------------------------------------
    # Single-column skills
    # -----------------------------------------------------------------------

    def univariate(
        self,
        data,
        column: Optional[str] = None,
    ) -> SkillResult:
        """
        Univariate distribution analysis on one column.

        Args:
            data:   pd.Series, or pd.DataFrame (requires column= arg)
            column: Column name — required when data is a DataFrame.

        Returns:
            SkillResult with skewness, kurtosis, mean, median, std,
            bounded_ordinal detection, MNAR flags, and recommendations.
        """
        payload = _resolve_single_column(data, column, skill="univariate")
        return self._call("univariate", payload)

    def outliers(
        self,
        data,
        column: Optional[str] = None,
    ) -> SkillResult:
        """
        Univariate outlier detection (IQR + Z-score).
        For relational/Type 2 outliers, use a bivariate or regression skill.

        Args:
            data:   pd.Series, or pd.DataFrame (requires column= arg)
            column: Column name — required when data is a DataFrame.
        """
        payload = _resolve_single_column(data, column, skill="outliers")
        return self._call("outliers", payload)

    #: Valid `transform` values for db.skills.transformations()
    #: (skills/analysis/transformations.py:UNIVARIATE_TRANSFORMS).
    TRANSFORMS = ("log", "sqrt", "box-cox", "reflect-log", "reflect-sqrt")

    def transformations(
        self,
        data,
        transform: str,
        column: Optional[str] = None,
    ) -> SkillResult:
        """
        Apply and assess a univariate transformation.

        Wraps the 7th registered skill, which had no SDK method before 0.5.0.

        Args:
            data:      pd.Series, or pd.DataFrame (requires column= arg)
            transform: one of "log", "sqrt", "box-cox", "reflect-log",
                       "reflect-sqrt". Validated server-side; the list is
                       mirrored on SkillsClient.TRANSFORMS.
            column:    Column name — required when data is a DataFrame.

        Returns:
            SkillResult with before/after skewness, whether the transform
            improved the distribution, and chart references in .charts.
        """
        if not transform:
            raise SDKUsageError(
                "db.skills.transformations() requires transform=. "
                f"One of: {', '.join(SkillsClient.TRANSFORMS)}"
            )
        payload = _resolve_single_column(data, column, skill="transformations")
        # Flat, top-level key — api/routes/skill.py passes the whole request
        # body straight through as params (`params_dict = body`), so a
        # nested "params": {...} sub-object is invisible server-side and
        # this call would 422 with "'transform' is required" every time.
        # Found 2026-09-06 auditing the SDK against the live dispatch layer.
        payload["transform"] = transform
        return self._call("transformations", payload)

    # -----------------------------------------------------------------------
    # Whole-dataset skills
    # -----------------------------------------------------------------------

    def missing_values(self, df) -> SkillResult:
        """
        Missing value profiling across all columns.
        Detects MCAR / MAR / POSSIBLE_MNAR per column.
        Recommends treatment per column.

        Args:
            df: pd.DataFrame — all columns profiled.
        """
        _require_dataframe(df, skill="missing_values")
        payload = _df_to_payload(df, list(df.columns))
        return self._call("missing_values", payload)

    def leakage(self, df, outcome: str) -> SkillResult:
        """
        Data leakage detection.
        Checks for post-outcome timing patterns and high-correlation proxies.

        Args:
            df:      pd.DataFrame
            outcome: Name of the outcome column.
        """
        _require_dataframe(df, skill="leakage")
        if outcome not in df.columns:
            raise SDKUsageError(
                f"outcome='{outcome}' not found in DataFrame columns: {list(df.columns)}"
            )
        predictor_cols = [c for c in df.columns if c != outcome]
        # Flat, top-level keys — see the note in transformations() above.
        payload = {
            **_df_to_payload(df, list(df.columns)),
            "outcome": outcome,
            "predictor_cols": predictor_cols,
        }
        return self._call("leakage", payload)

    # -----------------------------------------------------------------------
    # Two-column skills
    # -----------------------------------------------------------------------

    def bivariate(
        self,
        data,
        x: Optional[str] = None,
        y: Optional[str] = None,
        y_series=None,
    ) -> SkillResult:
        """
        Bivariate relationship analysis.
        Detects linearity, non-linearity, and correlation strength.

        Args:
            data: pd.DataFrame (requires x= and y=) or pd.Series for x
            x:    Column name for x variable (predictor)
            y:    Column name for y variable (outcome)
            y_series: pd.Series for y when data is a Series for x
        """
        payload, x_name, y_name = _resolve_two_columns(data, x, y, y_series, skill="bivariate")
        # Server-side there is no "bivariate" skill slug — only "bivariate_ts",
        # which dispatches on mode= to either the bivariate or timeseries path
        # (skills/dispatch.py:_run_bivariate_or_ts). It also always returns a
        # list (one entry per Column-2 selection) even for the single column
        # this method sends, hence _call_first_output instead of _call.
        # Found 2026-09-06: this method previously posted to /v1/skills/bivariate
        # directly, which 404'd every time — "Unknown skill: 'bivariate'".
        payload["column"] = x_name
        payload["columns_2"] = [y_name]
        payload["mode"] = "bivariate"
        return self._call_first_output("bivariate_ts", payload)

    def correlation(
        self,
        data,
        x: Optional[str] = None,
        y: Optional[str] = None,
        y_series=None,
    ) -> SkillResult:
        """
        Correlation analysis between two columns.
        Reports Pearson + Spearman, flags non-linearity risk.

        Args:
            data: pd.DataFrame (requires x= and y=) or pd.Series for x
            x:    Column name for x variable
            y:    Column name for y variable
            y_series: pd.Series for y when data is a Series for x
        """
        payload, x_name, y_name = _resolve_two_columns(data, x, y, y_series, skill="correlation")
        # skills/dispatch.py:_run_correlation reads outcome/predictor_cols,
        # not x/y — it computes y's correlation against each predictor.
        payload["outcome"] = y_name
        payload["predictor_cols"] = [x_name]
        return self._call("correlation", payload)

    # -----------------------------------------------------------------------
    # Whole-dataset / multi-column skills, added 2026-09-06 — these had no
    # SDK wrapper at all (scripts/check_drift.py against a live app
    # checkout found 7 registered skills with no corresponding method here).
    # -----------------------------------------------------------------------

    def data_quality(
        self,
        df,
        data_grain: Optional[str] = None,
        data_grain_column: Optional[str] = None,
    ) -> SkillResult:
        """
        Duplication and grain check — exact whole-row duplicates, repeats of
        a stated grain column, and candidate-key near-misses. Run this
        first: duplicated rows silently contaminate every statistic that
        follows.

        Args:
            df:                pd.DataFrame — all columns profiled.
            data_grain:        Optional. What one row is supposed to represent.
            data_grain_column: Optional. Column(s) that should be unique per data_grain.
        """
        _require_dataframe(df, skill="data_quality")
        payload = _multicol_payload(
            df, list(df.columns),
            data_grain=data_grain, data_grain_column=data_grain_column,
        )
        return self._call("data_quality", payload)

    def nonlinearity(
        self,
        data,
        column: Optional[str] = None,
        outcome: Optional[str] = None,
        y_series=None,
    ) -> SkillResult:
        """
        Functional-form check between a predictor and an outcome — is the
        relationship linear, or does it need a transform first (see
        transformations())?

        Args:
            data:     pd.DataFrame (requires column= and outcome=) or pd.Series for column
            column:   Predictor column name.
            outcome:  Outcome column name.
            y_series: pd.Series for outcome when data is a Series for column.
        """
        payload, x_name, y_name = _resolve_two_columns(data, column, outcome, y_series, skill="nonlinearity")
        payload["column"] = x_name
        payload["outcome"] = y_name
        return self._call("nonlinearity", payload)

    def linear_regression(
        self,
        df,
        outcome: str,
        predictor_cols: list[str],
        log_transformed: bool = False,
        log_x_transformed: bool = True,
    ) -> SkillResult:
        """
        OLS regression via statsmodels — canonical prerequisite for oaxaca()
        and confounding_remedy().

        Args:
            df:                 pd.DataFrame
            outcome:            Outcome column name.
            predictor_cols:     One or more predictor column names.
            log_transformed:    Whether `outcome` is already log-transformed
                                 (changes coefficient wording, e.g. elasticity).
            log_x_transformed:  Whether predictors are already log-transformed.
        """
        _require_dataframe(df, skill="linear_regression")
        if outcome not in df.columns:
            raise SDKUsageError(f"outcome='{outcome}' not found in DataFrame columns: {list(df.columns)}")
        missing = [c for c in predictor_cols if c not in df.columns]
        if missing:
            raise SDKUsageError(f"predictor_cols not found in DataFrame: {missing}")
        payload = _multicol_payload(
            df, [outcome, *predictor_cols],
            outcome=outcome, predictor_cols=predictor_cols,
            log_transformed=log_transformed, log_x_transformed=log_x_transformed,
        )
        return self._call("linear_regression", payload)

    def confounding_remedy(
        self,
        df,
        outcome: str,
        focal_col: str,
        remedy: str,
        period_col: str,
        other_predictors: Optional[list[str]] = None,
        drop_period_values: Optional[list] = None,
        log_predictors: Optional[list[str]] = None,
        log_outcome: bool = False,
        expected_sign: str = "negative",
        min_rows_after: int = 30,
    ) -> SkillResult:
        """
        Adjusts a focal predictor's effect for a confound tied to a period
        (e.g. a policy or seasonal change), via restriction or control.

        Args:
            df:                 pd.DataFrame
            outcome:            Outcome column name.
            focal_col:          The predictor whose effect you want, net of the confound.
            remedy:             "period_restriction" or "period_control".
            period_col:         Column identifying the confounded period.
            other_predictors:   Additional predictors to include.
            drop_period_values: Period values to exclude (period_restriction).
            log_predictors:     Which predictor columns to log-transform.
            log_outcome:        Whether to log-transform the outcome.
            expected_sign:      "negative" or "positive" — the focal effect's expected direction.
            min_rows_after:     Minimum rows required after any restriction.
        """
        _require_dataframe(df, skill="confounding_remedy")
        other_predictors = other_predictors or []
        needed = {outcome, focal_col, period_col, *other_predictors}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise SDKUsageError(f"Columns not found in DataFrame: {sorted(missing)}")
        payload = _multicol_payload(
            df, sorted(needed),
            outcome=outcome, focal_col=focal_col, remedy=remedy, period_col=period_col,
            other_predictors=other_predictors, drop_period_values=drop_period_values,
            log_predictors=log_predictors, log_outcome=log_outcome,
            expected_sign=expected_sign, min_rows_after=min_rows_after,
        )
        return self._call("confounding_remedy", payload)

    def _oaxaca_payload(
        self,
        skill_name: str,
        df,
        compensation_col: str,
        protected_col: str,
        factor_cols: list[str],
        reference_group: Optional[str],
        log_compensation: bool,
        proxy_max_assoc: float,
    ) -> dict:
        _require_dataframe(df, skill=skill_name)
        needed = {compensation_col, protected_col, *factor_cols}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise SDKUsageError(f"Columns not found in DataFrame: {sorted(missing)}")
        return _multicol_payload(
            df, sorted(needed),
            compensation_col=compensation_col, protected_col=protected_col, factor_cols=factor_cols,
            reference_group=reference_group, log_compensation=log_compensation,
            proxy_max_assoc=proxy_max_assoc,
        )

    def oaxaca(
        self,
        df,
        compensation_col: str,
        protected_col: str,
        factor_cols: list[str],
        reference_group: Optional[str] = None,
        log_compensation: bool = True,
        proxy_max_assoc: float = 0.90,
    ) -> SkillResult:
        """
        Oaxaca-Blinder pay-gap decomposition: splits a compensation gap
        between `protected_col` groups into the part "explained" by
        `factor_cols` and the "unexplained" residual.

        Args:
            df:               pd.DataFrame
            compensation_col: Outcome column (e.g. salary).
            protected_col:    Group column (e.g. gender) — binary or multi-group.
            factor_cols:      Legitimate explanatory factors (e.g. tenure, level).
            reference_group:  Which group value is the reference. Defaults server-side.
            log_compensation: Whether to log-transform compensation before decomposing.
            proxy_max_assoc:  Max allowed association between a factor and protected_col
                               before it's flagged as a likely proxy.
        """
        payload = self._oaxaca_payload(
            "oaxaca", df, compensation_col, protected_col, factor_cols,
            reference_group, log_compensation, proxy_max_assoc,
        )
        return self._call("oaxaca", payload)

    def oaxaca_detailed(
        self,
        df,
        compensation_col: str,
        protected_col: str,
        factor_cols: list[str],
        reference_group: Optional[str] = None,
        log_compensation: bool = True,
        proxy_max_assoc: float = 0.90,
    ) -> SkillResult:
        """Like oaxaca(), with the explained component broken out per factor."""
        payload = self._oaxaca_payload(
            "oaxaca_detailed", df, compensation_col, protected_col, factor_cols,
            reference_group, log_compensation, proxy_max_assoc,
        )
        return self._call("oaxaca_detailed", payload)

    def oaxaca_yun_normalized(
        self,
        df,
        compensation_col: str,
        protected_col: str,
        factor_cols: list[str],
        reference_group: Optional[str] = None,
        log_compensation: bool = True,
        proxy_max_assoc: float = 0.90,
    ) -> SkillResult:
        """Like oaxaca_detailed(), with Yun's normalization so per-factor
        contributions don't depend on category-reference-level choice."""
        payload = self._oaxaca_payload(
            "oaxaca_yun_normalized", df, compensation_col, protected_col, factor_cols,
            reference_group, log_compensation, proxy_max_assoc,
        )
        return self._call("oaxaca_yun_normalized", payload)


# ---------------------------------------------------------------------------
# Input resolution helpers — used by SkillsClient methods
# ---------------------------------------------------------------------------

def _resolve_single_column(data, column: Optional[str], skill: str) -> dict:
    """
    Resolve single-column input to a JSON payload dict.

    Accepts:
      - pd.Series → use Series.name or raise if unnamed
      - pd.DataFrame + column= → extract that column
      - pd.DataFrame without column= → SDKUsageError
    """
    try:
        import pandas as pd
    except ImportError:
        raise SDKUsageError("pandas is required. Install with: pip install pandas")

    if isinstance(data, pd.Series):
        col_name = column or data.name
        if not col_name:
            raise SDKUsageError(
                f"{skill}() received an unnamed Series. "
                f"Either name the Series (series.name = 'price') "
                f"or pass column='price' as an argument."
            )
        return _series_to_payload(data, col_name)

    if isinstance(data, pd.DataFrame):
        if column is None:
            raise SDKUsageError(
                f"{skill}() received a DataFrame but no column was specified. "
                f"Pass column='price' or pass a Series directly: df['price']"
            )
        if column not in data.columns:
            raise SDKUsageError(
                f"column='{column}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}"
            )
        return _series_to_payload(data[column], column)

    raise SDKUsageError(
        f"{skill}() expects a pd.Series or pd.DataFrame. Got {type(data).__name__}."
    )


def _multicol_payload(df, columns: list[str], **extra_params) -> dict:
    """
    Build a multi-column payload with flat, top-level extra params.

    api/routes/skill.py passes the whole request body straight through as
    the dispatch layer's `params` dict — there is no nested "params"
    sub-object it unwraps. So confounding_remedy/oaxaca/data_quality/etc.
    all need their skill-specific args (outcome=, factor_cols=, ...) as
    plain top-level keys alongside "columns"/"data", not nested.
    """
    return {**_df_to_payload(df, columns), **extra_params}


def _resolve_two_columns(data, x, y, y_series, skill: str) -> tuple[dict, str, str]:
    """
    Resolve two-column input to a JSON payload dict.

    Accepts:
      - pd.DataFrame + x= + y=
      - pd.Series (as data) + pd.Series (as y_series)

    Returns (payload, x_name, y_name) — the resolved names are returned
    alongside the payload because each caller needs them under different
    flat, skill-specific keys (bivariate wants column/columns_2, correlation
    wants outcome/predictor_cols) rather than a generic "params": {x, y}
    that api/routes/skill.py's dispatch layer never reads.
    """
    try:
        import pandas as pd
    except ImportError:
        raise SDKUsageError("pandas is required. Install with: pip install pandas")

    if isinstance(data, pd.DataFrame):
        if x is None or y is None:
            raise SDKUsageError(
                f"{skill}() requires x= and y= column names when passing a DataFrame. "
                f"Example: db.skills.{skill}(df, x='price', y='sales')"
            )
        for col in (x, y):
            if col not in data.columns:
                raise SDKUsageError(
                    f"Column '{col}' not found in DataFrame. "
                    f"Available: {list(data.columns)}"
                )
        return _df_to_payload(data, [x, y]), x, y

    if isinstance(data, pd.Series):
        if y_series is None or not isinstance(y_series, pd.Series):
            raise SDKUsageError(
                f"{skill}() with two Series: pass x Series as first arg "
                f"and y Series as y_series=. "
                f"Example: db.skills.{skill}(df['price'], y_series=df['sales'])"
            )
        x_name = x or data.name or "x"
        y_name = y or y_series.name or "y"
        payload = {
            "columns": [x_name, y_name],
            "data": {
                x_name: _series_to_payload(data, x_name)["data"],
                y_name: _series_to_payload(y_series, y_name)["data"],
            },
        }
        return payload, x_name, y_name

    raise SDKUsageError(
        f"{skill}() expects a pd.DataFrame or pd.Series. Got {type(data).__name__}."
    )


def _require_dataframe(data, skill: str) -> None:
    try:
        import pandas as pd
    except ImportError:
        raise SDKUsageError("pandas is required. Install with: pip install pandas")
    if not isinstance(data, pd.DataFrame):
        raise SDKUsageError(
            f"{skill}() requires a pd.DataFrame. Got {type(data).__name__}. "
            f"Pass the full DataFrame — this skill analyses all columns."
        )
