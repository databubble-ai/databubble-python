# databubble/analysis.py
"""
AnalysisClient — db.analysis.* : standalone analytical operations that are
not skills (no SKILL_REGISTRY entry, no session), not journeys (no graph, no
tier-gated multi-step orchestration), and not a portable scoring artifact.

    report = db.analysis.eda(df)
    report.n_flagged_columns
    pdf = report.export_pdf()

Added in 0.8.0 to close a real gap: these routes existed on the platform with
zero SDK reach — see DEF-0030 in the platform repo's
docs/context/deferred_items.yaml.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Union

from databubble.exceptions import SDKUsageError
from databubble.models import (
    EDAResult, PulseResult, ScopeResult, PowerPlanResult, SkillPackResult,
    CorrelationExportResult, ForecastExportResult,
)
from databubble._scoring_common import extract_ref


def _require_dataframe(df, method: str):
    try:
        import pandas as pd
    except ImportError:
        raise SDKUsageError("pandas is required.")
    if not isinstance(df, pd.DataFrame):
        raise SDKUsageError(f"{method}() requires a pd.DataFrame. Got {type(df).__name__}.")
    if df.empty:
        raise SDKUsageError(f"{method}(): DataFrame is empty.")
    return df


class AnalysisClient:
    def __init__(self, http_client):
        self._http = http_client

    def eda(
        self,
        df,
        data_grain: Optional[str] = None,
        data_grain_column: Optional[str] = None,
        column_metadata: Optional[list] = None,
        inherited_memories: Optional[dict] = None,
    ) -> EDAResult:
        """
        Full EDA health analysis on a DataFrame — POST /v1/eda.

        Args:
            df:                  pd.DataFrame, at least 2 columns.
            data_grain:          What one row represents, e.g. "customer" —
                                  enables the grain-anchored duplication check.
            data_grain_column:   Column(s) that should be unique per data_grain.
            column_metadata:     List of column-metadata dicts (unit/data_type/
                                  role) from a prior db.analysis.scope() call or
                                  your own data dictionary — an explicit mapping
                                  is trusted over name-matching heuristics.
            inherited_memories:  Dict of col_name -> ColumnMemory dict, to skip
                                  univariate for columns already covered.

        Returns:
            EDAResult — .n_flagged_columns, .report (full dict), .export_pdf().

        Tier: same as db.skills.* — developer key and up.
        """
        df = _require_dataframe(df, "db.analysis.eda")
        if len(df.columns) < 2:
            raise SDKUsageError("db.analysis.eda(): DataFrame must have at least 2 columns.")

        fields = {
            "data_grain": data_grain,
            "data_grain_column": data_grain_column,
            "column_metadata": json.dumps(column_metadata) if column_metadata is not None else None,
            "inherited_memories": json.dumps(inherited_memories) if inherited_memories else None,
        }
        response = self._http.post_multipart(
            "/v1/eda", fields=fields,
            files={"file": ("data.csv", df.to_csv(index=False), "text/csv")},
        )
        report = response.get("report", {})
        return EDAResult(
            n_rows=report.get("n_rows", 0),
            n_cols=report.get("n_cols", 0),
            n_flagged_columns=report.get("n_flagged_columns", 0),
            n_critical_flags=report.get("n_critical_flags", 0),
            summary=response.get("summary", ""),
            warnings=response.get("warnings", []),
            chapter_ref=response.get("chapter_ref", ""),
            raw=response,
            _http=self._http,
        )

    def pulse(
        self,
        df,
        data_grain: Optional[str] = None,
        data_grain_column: Optional[str] = None,
        column_metadata: Optional[list] = None,
    ) -> PulseResult:
        """
        Aggregate-profiling ("Pulse of Data") diagnostic — POST /v1/pulse.
        Auto-resolves logical roll-up grains, picks a defensible aggregation
        per measure, and surfaces anomaly callouts. Detect-and-report only —
        never repairs, never blocks.

        Args:
            df:                pd.DataFrame, at least 2 columns.
            data_grain:        Suggested grain, tried first ahead of inference.
            data_grain_column: Column(s) identifying that grain.
            column_metadata:   Column-metadata dicts — see eda()'s docstring.

        Returns:
            PulseResult — .n_rollups, .rollups (DataFrame), .caveats, .export_pdf().
            Read .caveats wherever you render .rollups — they qualify results
            Pulse DID produce (ecological fallacy, size-driven relationships,
            unweighted rate means), distinct from .deferred (what it could not do).
        """
        df = _require_dataframe(df, "db.analysis.pulse")
        if len(df.columns) < 2:
            raise SDKUsageError("db.analysis.pulse(): DataFrame must have at least 2 columns.")

        fields = {
            "data_grain": data_grain,
            "data_grain_column": data_grain_column,
            "column_metadata": json.dumps(column_metadata) if column_metadata is not None else None,
        }
        response = self._http.post_multipart(
            "/v1/pulse", fields=fields,
            files={"file": ("data.csv", df.to_csv(index=False), "text/csv")},
        )
        report = response.get("report", {})
        return PulseResult(
            n_rollups=report.get("n_rollups", 0),
            deferred=report.get("deferred", []),
            notes=report.get("notes", []),
            caveats=report.get("caveats", []),
            raw=response,
            _http=self._http,
        )

    def scope(self, df_sample, problem_description: str) -> ScopeResult:
        """
        Scoping pre-flight — POST /v1/scope. A SMALL SAMPLE plus a free-text
        question about what you're trying to do, before committing to full
        ingestion. Not a skill or journey: an advisory recommendation for
        regime, sample size, and grain.

        Args:
            df_sample:            A small pd.DataFrame sample (not the full
                                   dataset — this is a pre-flight check).
            problem_description:  What you're trying to figure out, in plain
                                   English. Required, cannot be empty.

        Returns:
            ScopeResult — .regime, .journey_candidates, .recommended_grain,
            .rationale, .blocking (True means out-of-scope, still a 200 —
            read .rationale for why rather than treating this as an error).
        """
        df_sample = _require_dataframe(df_sample, "db.analysis.scope")
        if not problem_description or not problem_description.strip():
            raise SDKUsageError("db.analysis.scope(): problem_description is required and cannot be empty.")

        response = self._http.post_multipart(
            "/v1/scope",
            fields={"problem_description": problem_description},
            files={"file": ("sample.csv", df_sample.to_csv(index=False), "text/csv")},
        )
        rec = response.get("recommendation", {})
        return ScopeResult(
            regime=rec.get("regime", ""),
            journey_candidates=rec.get("journey_candidates", []),
            rationale=rec.get("rationale", ""),
            blocking=rec.get("blocking", False),
            recommended_grain=rec.get("recommended_grain"),
            recommended_sample_rows=rec.get("recommended_sample_rows"),
            raw=response,
        )

    def power_plan(
        self,
        mode: str,
        metric_type: str,
        baseline_mean: Optional[float] = None,
        baseline_std: Optional[float] = None,
        baseline_rate: Optional[float] = None,
        mde: Optional[float] = None,
        mde_type: str = "absolute",
        n_per_group: Optional[int] = None,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0,
        alternative: str = "two-sided",
    ) -> PowerPlanResult:
        """
        Sample-size / detectable-effect planning — POST /v1/power/plan.
        Stateless, no dataset involved — same shape as db.model.compare().

        Args:
            mode:          "sample_size" (given an MDE, how many rows do I need)
                           or "detectable_effect" (given n_per_group, what's the
                           smallest effect I could detect).
            metric_type:   "continuous" or "binary".
            baseline_mean, baseline_std:  Required for metric_type="continuous".
            baseline_rate: Required for metric_type="binary".
            mde:           Minimum detectable effect. Required when mode="sample_size".
            mde_type:      "absolute" or "relative".
            n_per_group:   Required when mode="detectable_effect".
            alpha, power, ratio, alternative:  Standard test-design parameters.

        Returns:
            PowerPlanResult — .interpretation, plus .n_total (sample_size mode)
            or .mde_absolute (detectable_effect mode) on .raw.
        """
        if mode not in ("sample_size", "detectable_effect"):
            raise SDKUsageError("db.analysis.power_plan(): mode must be 'sample_size' or 'detectable_effect'.")
        if mode == "sample_size" and mde is None:
            raise SDKUsageError("db.analysis.power_plan(): mde is required when mode='sample_size'.")
        if mode == "detectable_effect" and n_per_group is None:
            raise SDKUsageError("db.analysis.power_plan(): n_per_group is required when mode='detectable_effect'.")

        payload = {
            "mode": mode, "metric_type": metric_type,
            "baseline_mean": baseline_mean, "baseline_std": baseline_std,
            "baseline_rate": baseline_rate, "mde": mde, "mde_type": mde_type,
            "n_per_group": n_per_group, "alpha": alpha, "power": power,
            "ratio": ratio, "alternative": alternative,
        }
        response = self._http.post_json("/v1/power/plan", payload)
        return PowerPlanResult(interpretation=response.get("interpretation", ""), raw=response)

    def export_skill_pack(
        self,
        df,
        table_label: str,
        description: Optional[str] = None,
        source_description: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> SkillPackResult:
        """
        Profile a DataFrame in isolation and render it as a downloadable
        Anthropic Agent Skill bundle (SKILL.md, profile.json, UPDATING.md) —
        POST /v1/export/skill. Single-table export.

        Args:
            df:                   pd.DataFrame to profile.
            table_label:          Human-readable label for this table.
            description:          Optional override for the skill's description.
            source_description:   Optional override for where this data came
                                   from (defaults to a filename/shape summary).
            outcome:               Optional outcome column name, if this table
                                    has one — must be a column in df.

        Returns:
            SkillPackResult — .save(directory) writes the 3-file bundle.
        """
        df = _require_dataframe(df, "db.analysis.export_skill_pack")
        if not table_label or not table_label.strip():
            raise SDKUsageError("db.analysis.export_skill_pack(): table_label is required and cannot be empty.")
        if outcome is not None and outcome not in df.columns:
            raise SDKUsageError(f"db.analysis.export_skill_pack(): outcome='{outcome}' not found in DataFrame columns.")

        fields = {
            "table_label": table_label,
            "description": description,
            "source_description": source_description,
            "outcome": outcome,
        }
        response = self._http.post_multipart(
            "/v1/export/skill", fields=fields,
            files={"file": ("data.csv", df.to_csv(index=False), "text/csv")},
        )
        return SkillPackResult(
            skill_name=response.get("skill_name", ""),
            n_rows=response.get("n_rows", 0),
            n_columns=response.get("n_columns", 0),
            blocking_count=response.get("blocking_count", 0),
            raw=response,
        )

    def correlation_export(
        self, source: Any, format: str = "json",
    ) -> Union[CorrelationExportResult, bytes]:
        """
        Export the correlation / variable-selection diagnostic from an
        elasticity/driver JourneyResult — POST /v1/correlation/export.

        Args:
            source: a JourneyResult from db.journeys.elasticity()/driver(), or
                    the raw correlation_export_ref dict
                    (result.raw["result"]["correlation_export_ref"]).
            format: "json" -> CorrelationExportResult. "csv" -> raw bytes (a
                    ZIP of correlation_matrix.csv + predictor_assessments.csv
                    + README.txt) — write it to a .zip file yourself.

        Returns:
            CorrelationExportResult (format="json") or bytes (format="csv").

        Example:
            result = db.journeys.elasticity(df, price_col="price", sales_col="sales")
            diag = db.analysis.correlation_export(result)
            diag.save("correlation_diagnostic.json")
        """
        ref = extract_ref(source, "correlation_export_ref", "db.analysis.correlation_export")
        payload = {"result_ref": ref, "format": "json" if format == "json" else "csv"}
        if format == "json":
            response = self._http.post_json("/v1/correlation/export", payload)
            return CorrelationExportResult(raw=response)
        return self._http.post_bytes("/v1/correlation/export", payload)

    def forecast_export(
        self, source: Any, format: str = "json",
    ) -> Union[ForecastExportResult, bytes]:
        """
        Export the fitted forecast card from a time_series JourneyResult —
        POST /v1/forecast/export.

        Args:
            source: a JourneyResult from db.journeys.time_series(), or the raw
                    forecast_export_ref dict
                    (result.raw["result"]["forecast_export_ref"]).
            format: "json" -> ForecastExportResult. "csv" -> raw bytes (a
                    forecast.csv table) — write it to a .csv file yourself.

        Returns:
            ForecastExportResult (format="json") or bytes (format="csv").
        """
        ref = extract_ref(source, "forecast_export_ref", "db.analysis.forecast_export")
        payload = {"result_ref": ref, "format": "json" if format == "json" else "csv"}
        if format == "json":
            response = self._http.post_json("/v1/forecast/export", payload)
            return ForecastExportResult(raw=response)
        return self._http.post_bytes("/v1/forecast/export", payload)
