# databubble/models.py
"""
Typed return objects for the DataBubble SDK.

Deliberately lightweight — plain dataclasses, no Pydantic dependency.
Every object exposes `.raw` for full API response access.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from databubble.exceptions import SDKUsageError

_MLFLOW_EXTRA_MSG = (
    "to_mlflow() requires databubble-scoring with the mlflow extra installed — "
    "install with `pip install databubble-scoring[mlflow]`."
)


def _save_as_mlflow_model(raw: dict, path: str) -> None:
    """
    Shared by ModelCardResult/ScorecardResult/SegmentScorerResult.to_mlflow().
    Lazily imported so plain `import databubble` never requires
    databubble-scoring or mlflow to be installed — only calling .to_mlflow()
    does.
    """
    try:
        from databubble_scoring.mlflow_pyfunc import save_databubble_model
    except ImportError as e:
        raise SDKUsageError(_MLFLOW_EXTRA_MSG) from e
    save_databubble_model(raw, path)


@dataclass
class SkillResult:
    """
    Return type for all skill calls.

    Attributes:
        summary         Plain-English summary of the finding (token-efficient).
        findings        Dict of structured findings — skewness, mean, mechanism etc.
        warnings        List of warning strings — assumption violations, data issues.
        recommendations List of recommendation strings — what to do next.
        chapter_ref     Knowledge chapter reference e.g. "Chapter 04".
        skill_name      Which skill was called.
        column          Column analysed (None for whole-dataset skills).
        n_rows          Row count of the input data.
        tier            API tier used for this call.
        key_prefix      Key prefix for audit trail.
        raw             Full API response dict — access anything not surfaced above.
    """
    summary: str
    findings: dict[str, Any]
    warnings: list[str]
    recommendations: list[str]
    chapter_ref: str
    skill_name: str
    column: Optional[str] = None
    n_rows: Optional[int] = None
    tier: Optional[str] = None
    key_prefix: Optional[str] = None
    halted: bool = False
    halt_reason: Optional[str] = None
    raw: dict = field(default_factory=dict)
    # Injected by SkillsClient so .charts can lazily fetch by filename.
    _http: Any = field(default=None, repr=False, compare=False)

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def has_blocking_issues(self) -> bool:
        """True if any warning uses the word 'blocking' or 'halt'."""
        lower = [w.lower() for w in self.warnings]
        return any("blocking" in w or "halt" in w for w in lower)

    # -- data-scientist surface (0.5.0) -----------------------------------
    @property
    def charts(self):
        """
        ChartSet built from the chart_* keys the skill returned inside
        `findings` (univariate, bivariate, outliers, correlation, transformations).
        Charts are fetched lazily from GET /v1/charts/{filename}.
        """
        from databubble.charts import from_response

        payload = self.findings if isinstance(self.findings, dict) else {}
        return from_response(payload, http=self._http)

    def to_frame(self):
        """
        Findings as a two-column DataFrame (metric, value). Nested/None values
        and chart references are dropped — use .findings for those.
        """
        import pandas as pd

        if not isinstance(self.findings, dict):
            return pd.DataFrame({"value": self.findings or []})
        rows = [
            (key, value)
            for key, value in self.findings.items()
            if not key.startswith("chart_") and not isinstance(value, (dict, list))
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def _repr_html_(self) -> str:
        from databubble.tables import escape, frame_to_html

        title = f"{self.skill_name}" + (f" — {self.column}" if self.column else "")
        head = (
            "<div style='font-family:system-ui'>"
            f"<p style='margin:0 0 6px 0'><strong>{escape(title)}</strong></p>"
        )
        body = frame_to_html(self.to_frame(), "")
        warn = ""
        if self.warnings:
            warn = "<ul style='margin:6px 0'>" + "".join(
                f"<li>{escape(w)}</li>" for w in self.warnings[:10]
            ) + "</ul>"
        chart_note = ""
        if len(self.charts):
            chart_note = (
                "<p style='color:#666;margin:4px 0 0 0'>"
                f"{len(self.charts)} chart(s) available — <code>.charts.show()</code></p>"
            )
        return head + body + warn + chart_note + "</div>"

    def __repr__(self) -> str:
        warn_str = f", {len(self.warnings)} warnings" if self.warnings else ""
        col_str = f" on '{self.column}'" if self.column else ""
        return f"SkillResult({self.skill_name}{col_str}{warn_str})"


@dataclass
class TreatmentRecord:
    """One platform recommendation on one column."""
    column: str
    issue: str
    recommendation: str
    severity: str          # "blocking" / "warning" / "informational"
    status: str            # "open" / "applied" / "deferred" / "overridden"


@dataclass
class ColumnMemory:
    """Analytical record for one column from a memory file."""
    name: str
    skewness: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    missing_pct: float
    missing_mechanism: Optional[str]
    variable_type: Optional[str]
    is_bounded_ordinal: bool
    blocking_issues: list[str]
    treatments: list[TreatmentRecord]

    @property
    def has_blocking_issues(self) -> bool:
        return len(self.blocking_issues) > 0

    @property
    def open_treatments(self) -> list[TreatmentRecord]:
        return [t for t in self.treatments if t.status == "open"]


@dataclass
class MemoryResult:
    """
    Return type for db.memory.export().

    Attributes:
        memory_id       UUID for this memory file.
        label           User-given label.
        memory_json     Full memory dict — pass to save() or load back later.
        memory_markdown Human-readable narrative string.
        open_count      Number of unresolved recommendations.
        blocking_count  Number of columns with blocking issues.
        columns_covered Number of columns with univariate coverage.
        raw             Full API response.
    """
    memory_id: str
    label: str
    memory_json: dict
    memory_markdown: str
    open_count: int
    blocking_count: int
    columns_covered: int
    raw: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Write memory_json to disk as a .json file."""
        import json
        with open(path, "w") as f:
            json.dump(self.memory_json, f, indent=2)
        print(f"Memory saved to {path}")

    def save_markdown(self, path: str) -> None:
        """Write human-readable narrative to disk as a .md file."""
        with open(path, "w") as f:
            f.write(self.memory_markdown)
        print(f"Markdown summary saved to {path}")

    def __repr__(self) -> str:
        return (
            f"MemoryResult('{self.label}', "
            f"{self.columns_covered} columns covered, "
            f"{self.open_count} open items)"
        )


@dataclass
class ColumnReconciliation:
    name: str
    status: str            # matched / missing_in_new / new_column / stats_shifted
    memory_source: Optional[str]
    shift_note: Optional[str]


@dataclass
class ReconciliationResult:
    """
    Return type for db.memory.reconcile().

    Attributes:
        memories_loaded         Labels of loaded memory files.
        columns_matched         Per-column reconciliation status.
        open_items              Unresolved recommendations from all memories.
        verified_treatments     Treatments user claimed that platform could verify.
        unverifiable_treatments Treatments platform could not confirm from data.
        ready_to_advance        True when all columns have univariate coverage (Option A).
        suggested_next          Plain-English suggestion for next analysis step.
        inherited_column_memories  Dict of col_name → column facts for EDA orchestrator.
        raw                     Full API response.
    """
    memories_loaded: list[str]
    columns_matched: list[ColumnReconciliation]
    open_items: list[str]
    verified_treatments: list[str]
    unverifiable_treatments: list[str]
    ready_to_advance: bool
    suggested_next: str
    inherited_column_memories: dict
    raw: dict = field(default_factory=dict)

    @property
    def new_columns(self) -> list[str]:
        return [c.name for c in self.columns_matched if c.status == "new_column"]

    @property
    def shifted_columns(self) -> list[str]:
        return [c.name for c in self.columns_matched if c.status == "stats_shifted"]

    def to_frame(self):
        """Per-column reconciliation status as a DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "column": c.name,
                    "status": c.status,
                    "memory_source": c.memory_source,
                    "shift_note": c.shift_note,
                }
                for c in self.columns_matched
            ]
        )

    def __repr__(self) -> str:
        return (
            f"ReconciliationResult("
            f"{len(self.memories_loaded)} memories, "
            f"ready={self.ready_to_advance}, "
            f"open={len(self.open_items)})"
        )




# ---------------------------------------------------------------------------
# JourneyResult — data-scientist-first surface (0.5.0)
# ---------------------------------------------------------------------------

_DEPRECATED_SELECTION = (
    "JourneyResult.{old} is deprecated and will be removed in 1.0. "
    "It never worked against the live API: the envelope names these fields "
    "`selected_predictors` / `excluded_predictors`, and the selection reasoning "
    "is a sibling of `result`, not inside it. Use .{new} instead."
)


@dataclass
class JourneyResult:
    """
    Return type for all db.journeys.* calls.

    The primary surface is quantitative:

        r = db.journeys.driver(df, outcome_col="sales", candidate_cols=[...])
        print(r)                # statsmodels-style regression table
        r.estimates             # DataFrame: coef, std err, t, p, CI, VIF
        r.coefficients          # Series: predictor -> coefficient
        r.effects               # DataFrame: standardised coef, effect size, partial R2
        r.diagnostics           # Series: n, adj R2, assumptions_met, transformation...
        r.charts                # ChartSet (empty until the API returns charts)

    The business narrative is still there, it is just no longer the default view:

        r.explain()             # plain-English summary + revenue implication
        r.plain_english_summary # the raw narrative string

    Everything the API returned is always available at r.raw.
    """

    journey_type: str
    halted: bool
    halt_reason: Optional[str]
    primary_estimate: Optional[float]
    plain_english_summary: str
    warnings: list[str]
    assumptions_met: Optional[bool]
    adj_r_squared: Optional[float] = None
    revenue_implication: Optional[str] = None
    tier: Optional[str] = None
    key_prefix: Optional[str] = None
    raw: dict = field(default_factory=dict)
    # Injected by JourneysClient so .charts can lazily fetch by filename.
    _http: Any = field(default=None, repr=False, compare=False)

    # -- raw navigation ---------------------------------------------------
    @property
    def result(self) -> dict:
        """The `result` envelope — every field the API returned for this journey."""
        value = self.raw.get("result")
        return value if isinstance(value, dict) else {}

    @property
    def handoffs(self) -> dict:
        """
        Domain payload for the graph-based journeys (mmm, pay_equity,
        cross_price, spc_monitoring, latent_factors, causal_inference,
        churn_clv_at_risk, forecast_inventory, intervention_lift).
        Empty dict for the direct-function journeys.
        """
        value = self.result.get("handoffs")
        return value if isinstance(value, dict) else {}

    # -- quantitative surface ---------------------------------------------
    @property
    def estimates(self):
        """Per-predictor coefficient table as a DataFrame (empty if absent)."""
        from databubble.tables import ESTIMATE_COLUMNS, rows_to_frame

        rows = self.result.get("estimates") or self.handoffs.get("estimates")
        return rows_to_frame(rows, ESTIMATE_COLUMNS)

    @property
    def coefficients(self):
        """Series mapping predictor name -> coefficient."""
        import pandas as pd

        frame = self.estimates
        if frame.empty or "name" not in frame.columns or "coefficient" not in frame.columns:
            return pd.Series(dtype="float64")
        return pd.Series(
            frame["coefficient"].values, index=frame["name"].values, name="coefficient"
        )

    @property
    def effects(self):
        """
        Effect-size table: standardised coefficients, effect sizes, partial R²
        and dominance, merged on predictor name where the API provides them.
        """
        import pandas as pd
        from databubble.tables import mapping_to_frame, rows_to_frame

        # Each (frame, source_label) pair — the label only ever gets used to
        # disambiguate a column name that turns out to collide across sources
        # (see below); it never appears in the output otherwise.
        frames: list[tuple[Any, str]] = []
        for key, label in (
            ("standardized_coefficients", "std"),
            ("effect_sizes", "effect_size"),
            ("partial_r_squared", "partial_r2"),
        ):
            value = self.result.get(key)
            if isinstance(value, dict):
                frames.append((mapping_to_frame(value, "predictor", label), label))
            elif isinstance(value, list):
                frame = rows_to_frame(value)
                if not frame.empty:
                    name_col = "predictor" if "predictor" in frame.columns else "name"
                    if name_col in frame.columns:
                        frame = frame.rename(columns={name_col: "predictor"})
                    frames.append((frame, label))

        # partial_r_squared_dominance is NOT a flat {predictor: value} mapping —
        # it's {"shares": [{"name", "dominance_r2", "share_of_model_r2", "rank"}, ...],
        # "sum": float, "reconciles_to_r2": bool}. Feeding the dict straight into
        # mapping_to_frame treated "shares"/"sum"/"reconciles_to_r2" themselves as
        # predictor names, producing three garbage rows alongside the real ones —
        # caught by testing against a live-captured driver_analysis response, not
        # by the fixtures (which never exercised dominance's actual shape). The
        # model-level "sum"/"reconciles_to_r2" scalars aren't per-predictor, so
        # they're intentionally left out of this table — still reachable via
        # .raw["partial_r_squared_dominance"].
        dominance = self.result.get("partial_r_squared_dominance")
        if isinstance(dominance, dict) and isinstance(dominance.get("shares"), list):
            dom_frame = rows_to_frame(dominance["shares"])
            if not dom_frame.empty:
                name_col = "predictor" if "predictor" in dom_frame.columns else "name"
                if name_col in dom_frame.columns:
                    dom_frame = dom_frame.rename(columns={name_col: "predictor"})
                frames.append((dom_frame, "dominance"))

        frames = [(f, label) for f, label in frames if not f.empty and "predictor" in f.columns]
        if not frames:
            return pd.DataFrame()

        # pandas' merge(..., suffixes=) only ever disambiguates ONE pairwise
        # collision — a third frame sharing a column name already suffixed
        # into e.g. "rank_x"/"rank_y" by an earlier merge raises MergeError
        # instead of silently overwriting it. standardized_coefficients,
        # partial_r_squared and the dominance shares all carry their own
        # "rank" column, so with 3+ sources this isn't a hypothetical — it's
        # every driver_analysis/elasticity result with dominance data
        # (caught the same way as the garbage-rows bug above: testing against
        # a live-captured response, not the fixtures). Disambiguate up front
        # instead: any non-key column name that appears in more than one
        # frame gets its source label appended, so merges never collide
        # regardless of how many sources are present.
        from collections import Counter
        col_counts = Counter()
        for frame, _ in frames:
            col_counts.update(c for c in frame.columns if c != "predictor")
        renamed = []
        for frame, label in frames:
            rename_map = {
                c: f"{c}_{label}" for c in frame.columns
                if c != "predictor" and col_counts[c] > 1
            }
            renamed.append(frame.rename(columns=rename_map) if rename_map else frame)

        merged = renamed[0]
        for frame in renamed[1:]:
            merged = merged.merge(frame, on="predictor", how="outer")
        return merged

    @property
    def diagnostics(self):
        """Model-level diagnostics as a Series (only fields the API returned)."""
        import pandas as pd
        from databubble.tables import DIAGNOSTIC_FIELDS

        data = {}
        for key in DIAGNOSTIC_FIELDS:
            if key in self.result and self.result[key] is not None:
                data[key] = self.result[key]
        for key in ("shapiro_pvalue", "bp_pvalue", "durbin_watson", "normality_ok",
                    "homoscedasticity_ok", "residual_std"):
            source = self.result.get("residual_diagnostics")
            if isinstance(source, dict) and key in source:
                data[key] = source[key]
        return pd.Series(data, dtype="object")

    @property
    def selected_predictors(self) -> list[str]:
        """Predictors that were actually fitted."""
        for source in (self.result, self.raw.get("selection") or {}):
            if isinstance(source, dict):
                value = source.get("selected_predictors") or source.get("recommended")
                if isinstance(value, list):
                    return value
        frame = self.estimates
        if not frame.empty and "name" in frame.columns:
            return [str(n) for n in frame["name"].tolist()]
        return []

    @property
    def excluded_predictors(self) -> list[str]:
        for source in (self.result, self.raw.get("selection") or {}):
            if isinstance(source, dict):
                value = source.get("excluded_predictors") or source.get("excluded")
                if isinstance(value, list):
                    return value
        return []

    @property
    def focus_predictor(self) -> Optional[str]:
        return self.result.get("focus_predictor")

    @property
    def n_observations(self) -> Optional[int]:
        value = self.result.get("n_observations") or self.result.get("n_clean")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def confidence_interval(self) -> Optional[tuple]:
        low = self.result.get("ci_lower", self.result.get("primary_ci_lower"))
        high = self.result.get("ci_upper", self.result.get("primary_ci_upper"))
        if low is None and high is None:
            return None
        return (low, high)

    @property
    def significant(self) -> Optional[bool]:
        value = self.result.get("significant", self.result.get("primary_significant"))
        return value if isinstance(value, bool) else None

    @property
    def charts(self):
        """
        ChartSet for this journey.

        Empty until the platform returns chart content on /v1/journeys/* — see
        the Track B spec. When it does, no SDK change is needed: inline SVG and
        filename shapes are both handled here already.
        """
        from databubble.charts import from_response

        merged = dict(self.result)
        merged.update({k: v for k, v in self.raw.items() if k.startswith("chart") or k == "charts"})
        return from_response(merged, http=self._http)

    # -- PDF export ---------------------------------------------------------
    def _session_view(self) -> dict:
        """
        POST /v1/export/journey and /v1/export/journey/exec-summary render a
        Mode 2 (guided-session) dict: journey_type/outcome/steps/final_output/
        brief (api/routes/export.py + api/export/journey_pdf.py in the
        platform repo). This SDK is Mode 1 (stateless) only and never has a
        live session object — but every `final_output` field the renderer
        reads is already flat on this JourneyResponseEnvelope (api/envelope.py),
        because Mode 1 (journeys/journey_run.py) populates the same fields
        Mode 2's build_final_output() does. `brief` (business_problem /
        decision_at_stake) is the one thing genuinely absent — Mode 1 never
        collects free-text framing from the caller — so it's sent empty;
        the renderer already treats it as optional. This is a faithful
        re-shaping of data this result already has, not invented content.
        """
        result = self.result
        final_output = {
            k: result.get(k) for k in (
                "plain_english_summary", "primary_estimate", "primary_estimate_caveat",
                "primary_label", "ci_lower", "ci_upper", "assumptions_met",
                "causal_limitation", "warnings", "halted", "halt_reason",
            )
        }
        return {
            "journey_type": self.journey_type,
            "outcome": result.get("outcome", ""),
            "steps": result.get("steps", []),
            "final_output": final_output,
            "brief": {},
        }

    def export_pdf(self) -> bytes:
        """
        Render this journey as a full walkthrough PDF — POST /v1/export/journey.

        Example:
            result = db.journeys.elasticity(df, price_col="price", sales_col="sales")
            with open("elasticity.pdf", "wb") as f:
                f.write(result.export_pdf())

        See JourneyResult._session_view() for how a Mode 1 (stateless) result
        is reshaped into the session dict this route expects.
        """
        if self._http is None:
            raise SDKUsageError("export_pdf() requires a JourneyResult returned by db.journeys.*().")
        return self._http.post_bytes("/v1/export/journey", {"session": self._session_view()})

    def export_exec_summary_pdf(self) -> bytes:
        """
        Render a one-page, decision-first summary PDF (no methodology, no
        step-by-step) — POST /v1/export/journey/exec-summary.
        """
        if self._http is None:
            raise SDKUsageError("export_exec_summary_pdf() requires a JourneyResult returned by db.journeys.*().")
        return self._http.post_bytes("/v1/export/journey/exec-summary", {"session": self._session_view()})

    # -- gates ------------------------------------------------------------
    def is_reliable(self) -> bool:
        """True when the journey completed and assumptions were not violated."""
        return not self.halted and self.assumptions_met is not False

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    # -- views ------------------------------------------------------------
    def to_frame(self):
        """Alias for .estimates — the tabular view of this result."""
        return self.estimates

    def summary(self, max_rows: int = 50) -> str:
        """statsmodels-style text summary. This is what repr() shows."""
        from databubble.tables import estimate_table, header_block, warnings_block

        if self.halted:
            return (
                f"{self.journey_type} — HALTED\n"
                + "=" * 78
                + f"\n{self.halt_reason or 'no reason given'}\n"
                + warnings_block(self.warnings)
            )

        pairs = [
            ("Observations", self.n_observations),
            ("Adj. R²", self.adj_r_squared),
            ("Assumptions met", self.assumptions_met),
            ("Focus predictor", self.focus_predictor),
            ("Estimate", self.primary_estimate),
            ("Significant", self.significant),
            ("Transformation", self.result.get("transformation_applied")),
            ("Tier", self.tier),
        ]
        interval = self.confidence_interval
        if interval and interval[0] is not None:
            pairs.append(("95% CI", f"[{interval[0]:.4g}, {interval[1]:.4g}]"))

        title = self.result.get("primary_label") or f"{self.journey_type} journey"
        blocks = [header_block(title, pairs) + "\n", estimate_table(self.estimates, max_rows)]

        if self.estimates.empty and self.handoffs:
            blocks.append(
                "  Domain output for this journey is in .handoffs — keys: "
                + ", ".join(sorted(self.handoffs)[:12])
            )

        blocks.append(warnings_block(self.warnings))
        blocks.append("\nNarrative: r.explain()   Full response: r.raw")
        return "\n".join(b for b in blocks if b != "")

    def explain(self) -> str:
        """The business-facing narrative — now opt-in rather than the default."""
        parts = [self.plain_english_summary or "(no narrative returned)"]
        if self.revenue_implication:
            parts.append(f"\nRevenue implication: {self.revenue_implication}")
        if self.result.get("causal_limitation"):
            parts.append(f"\nCausal limitation: {self.result['causal_limitation']}")
        if self.warnings:
            parts.append("\nWarnings:\n" + "\n".join(f"  - {w}" for w in self.warnings))
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        from databubble.tables import escape, frame_to_html

        if self.halted:
            return (
                f"<div><strong>{escape(self.journey_type)} — HALTED</strong>"
                f"<p>{escape(self.halt_reason)}</p></div>"
            )

        rows = "".join(
            f"<td style='padding:2px 12px 2px 0'><strong>{escape(label)}</strong> {escape(value)}</td>"
            for label, value in (
                ("n", self.n_observations),
                ("Adj. R²", self.adj_r_squared),
                ("Assumptions", self.assumptions_met),
                ("Estimate", self.primary_estimate),
            )
            if value is not None
        )
        head = (
            f"<div style='font-family:system-ui'><p style='margin:0 0 6px 0'>"
            f"<strong>{escape(self.result.get('primary_label') or self.journey_type)}</strong></p>"
            f"<table style='margin-bottom:8px'><tr>{rows}</tr></table>"
        )
        body = frame_to_html(self.estimates, "Estimates")
        if not body and self.handoffs:
            body = f"<em>Domain output in <code>.handoffs</code>: {escape(', '.join(sorted(self.handoffs)[:12]))}</em>"
        warn = ""
        if self.warnings:
            warn = "<ul style='margin:6px 0'>" + "".join(
                f"<li>{escape(w)}</li>" for w in self.warnings[:10]
            ) + "</ul>"
        foot = "<p style='color:#666;margin:6px 0 0 0'><code>.explain()</code> for the narrative, <code>.raw</code> for everything.</p></div>"
        return head + body + warn + foot

    def __repr__(self) -> str:
        return self.summary()

    # -- deprecated -------------------------------------------------------
    @property
    def recommended(self) -> list[str]:
        import warnings as _w

        _w.warn(
            _DEPRECATED_SELECTION.format(old="recommended", new="selected_predictors"),
            DeprecationWarning,
            stacklevel=2,
        )
        selection = self.raw.get("selection")
        if isinstance(selection, dict) and isinstance(selection.get("recommended"), list):
            return selection["recommended"]
        legacy = self.result.get("selection_output")
        if isinstance(legacy, dict) and isinstance(legacy.get("recommended"), list):
            return legacy["recommended"]
        return self.selected_predictors

    @property
    def caution(self) -> list[str]:
        import warnings as _w

        _w.warn(
            _DEPRECATED_SELECTION.format(old="caution", new="raw['selection']['caution']"),
            DeprecationWarning,
            stacklevel=2,
        )
        for source in (self.raw.get("selection"), self.result.get("selection_output")):
            if isinstance(source, dict) and isinstance(source.get("caution"), list):
                return source["caution"]
        return []

    @property
    def excluded(self) -> list[str]:
        import warnings as _w

        _w.warn(
            _DEPRECATED_SELECTION.format(old="excluded", new="excluded_predictors"),
            DeprecationWarning,
            stacklevel=2,
        )
        for source in (self.raw.get("selection"), self.result.get("selection_output")):
            if isinstance(source, dict) and isinstance(source.get("excluded"), list):
                return source["excluded"]
        return self.excluded_predictors


# ---------------------------------------------------------------------------
# Portable model artifacts — db.model / db.scorecard / db.segments (0.6.0)
#
# Cards/scorecards/scorers never pickle a fitted object — coefficients (or a
# FittedPipeline of primitives) plus a replayable recipe, so scoring never
# needs the platform again once you have the artifact. Export once from a
# fitted JourneyResult, then predict/score independently, any time, offline.
# ---------------------------------------------------------------------------

@dataclass
class ModelCardResult:
    """
    Return type for db.model.export() — a portable JSON model card (linear or
    fixed-effect regression) or bundle (one card per group, fixed_effect/
    by_group journeys).

        card = db.model.export(result)
        card.coefficients        # Series (card) or DataFrame (bundle)
        card.save("model.json")  # reload with json.load(), pass to .predict()
    """
    outcome: str
    kind: str  # "card" | "bundle"
    model_family: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def coefficients(self):
        """Series (single card, term -> coefficient) or DataFrame (bundle, one row per group x term)."""
        import pandas as pd

        if self.kind == "bundle":
            rows = [
                {"group": card.get("group_value"), "term": t["name"], "coefficient": t["coefficient"]}
                for card in self.raw.get("cards", [])
                for t in card.get("terms", [])
            ]
            return pd.DataFrame(rows)
        terms = self.raw.get("terms", [])
        return pd.Series({t["name"]: t["coefficient"] for t in terms}, name="coefficient")

    def save(self, path: str) -> None:
        """Write the portable card JSON to disk."""
        import json
        with open(path, "w") as f:
            json.dump(self.raw, f, indent=2)
        print(f"Model card saved to {path}")

    def to_mlflow(self, path: str) -> None:
        """
        Save this card as a local MLflow pyfunc model directory — load it
        with mlflow.pyfunc.load_model(path).predict(df) and score fully
        offline, zero DataBubble network call at inference. Requires
        `pip install databubble-scoring[mlflow]`.
        """
        if self.kind == "bundle":
            raise SDKUsageError(
                "to_mlflow() does not support bundles (fixed_effect/by_group cards "
                "with multiple groups) yet — export a single group's card instead."
            )
        _save_as_mlflow_model(self.raw, path)

    def __repr__(self) -> str:
        if self.kind == "bundle":
            return f"ModelCardResult(bundle, {len(self.raw.get('cards', []))} groups, outcome='{self.outcome}')"
        return f"ModelCardResult('{self.outcome}', {len(self.raw.get('terms', []))} terms)"


@dataclass
class PredictionResult:
    """Return type for db.model.predict()."""
    outcome: str
    n_scored: int
    log_back_transformed: bool
    warnings: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def predictions(self):
        """DataFrame: prediction, plus ci_lower/ci_upper/pi_lower/pi_upper/group_used when the card provides them."""
        import pandas as pd

        cols: dict[str, Any] = {"prediction": self.raw.get("predictions")}
        for key in ("group_used", "ci_lower", "ci_upper", "pi_lower", "pi_upper"):
            value = self.raw.get(key)
            if value is not None:
                cols[key] = value
        return pd.DataFrame(cols)

    def __repr__(self) -> str:
        return f"PredictionResult(n_scored={self.n_scored}, outcome='{self.outcome}')"


@dataclass
class ComparisonResult:
    """
    Return type for db.model.compare(). Shape depends on mode:
    "ic" (>=2 cards) -> ranked candidates by AIC/BIC; "nested" (exactly 2
    cards: [reduced, full]) -> a single F-test verdict, read from .raw.
    """
    mode: str
    outcome: str
    raw: dict = field(default_factory=dict)

    @property
    def candidates(self):
        """Ranked-candidate DataFrame — empty for mode='nested' (use .raw for the F-test fields)."""
        from databubble.tables import rows_to_frame

        return rows_to_frame(self.raw.get("candidates"))

    @property
    def best_label(self) -> Optional[str]:
        return self.raw.get("best_label")

    def __repr__(self) -> str:
        if self.mode == "nested":
            verdict = "full model preferred" if self.raw.get("full_model_preferred") else "reduced model preferred"
            p = self.raw.get("p_value")
            return f"ComparisonResult(nested, {verdict}, p={p:.4g})" if p is not None else f"ComparisonResult(nested, {verdict})"
        return f"ComparisonResult(ic, best='{self.best_label}', {len(self.raw.get('candidates', []))} candidates)"


@dataclass
class DriftResult:
    """Return type for db.model.drift() — feature drift vs. the card's training-time snapshot."""
    applicable: bool
    drift_detected: bool
    n_features_checked: int
    n_features_drifted: int
    interpretation: str
    raw: dict = field(default_factory=dict)

    @property
    def per_feature(self):
        """DataFrame: column, kind, drifted, detail, plus reference/observed stats."""
        from databubble.tables import rows_to_frame

        return rows_to_frame(self.raw.get("per_feature"))

    def __repr__(self) -> str:
        if not self.applicable:
            return f"DriftResult(not applicable: {self.raw.get('reason')})"
        return f"DriftResult(drift_detected={self.drift_detected}, {self.n_features_drifted}/{self.n_features_checked} features)"


@dataclass
class ScorecardResult:
    """
    Return type for db.scorecard.export() — a portable JSON scorecard for a
    classification-family journey (classification, predictive_model).
    """
    outcome: str
    raw: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Write the portable scorecard JSON to disk."""
        import json
        with open(path, "w") as f:
            json.dump(self.raw, f, indent=2)
        print(f"Scorecard saved to {path}")

    def to_mlflow(self, path: str) -> None:
        """
        Save this scorecard as a local MLflow pyfunc model directory — load
        it with mlflow.pyfunc.load_model(path).predict(df) and score fully
        offline, zero DataBubble network call at inference. Requires
        `pip install databubble-scoring[mlflow]`.
        """
        _save_as_mlflow_model(self.raw, path)

    @property
    def auc(self) -> Optional[float]:
        return (self.raw.get("provenance") or {}).get("auc")

    def __repr__(self) -> str:
        return f"ScorecardResult('{self.outcome}', auc={self.auc})"


@dataclass
class ScoreResult:
    """Return type for db.scorecard.score()."""
    n_scored: int
    low_confidence_count: int
    threshold_used: Optional[float] = None
    raw: dict = field(default_factory=dict)

    @property
    def predictions(self):
        """DataFrame: label, probability (top class), low_confidence — one row per scored observation."""
        import pandas as pd

        labels = self.raw.get("labels") or []
        probs = self.raw.get("probabilities") or []
        low_conf = self.raw.get("low_confidence") or [None] * len(labels)
        top_prob = [max(p.values()) if isinstance(p, dict) and p else None for p in probs]
        return pd.DataFrame({"label": labels, "probability": top_prob, "low_confidence": low_conf})

    def __repr__(self) -> str:
        return f"ScoreResult(n_scored={self.n_scored}, low_confidence={self.low_confidence_count})"


@dataclass
class SegmentScorerResult:
    """Return type for db.segments.export() — a portable JSON segment scorer."""
    raw: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Write the portable scorer JSON to disk."""
        import json
        with open(path, "w") as f:
            json.dump(self.raw, f, indent=2)
        print(f"Segment scorer saved to {path}")

    def to_mlflow(self, path: str) -> None:
        """
        Save this scorer as a local MLflow pyfunc model directory — load it
        with mlflow.pyfunc.load_model(path).predict(df) and score fully
        offline, zero DataBubble network call at inference. Requires
        `pip install databubble-scoring[mlflow]`.
        """
        _save_as_mlflow_model(self.raw, path)

    @property
    def segments(self) -> list[str]:
        return sorted(set((self.raw.get("cluster_label_map") or {}).values()))

    def __repr__(self) -> str:
        return f"SegmentScorerResult({len(self.segments)} segments)"


@dataclass
class SegmentScoreResult:
    """Return type for db.segments.score()."""
    n_scored: int
    raw: dict = field(default_factory=dict)

    @property
    def assignments(self):
        """DataFrame: segment, raw_label — one row per scored observation."""
        from databubble.tables import rows_to_frame

        return rows_to_frame(self.raw.get("assignments"))

    @property
    def segment_distribution(self) -> dict:
        return self.raw.get("segment_distribution") or {}

    def __repr__(self) -> str:
        return f"SegmentScoreResult(n_scored={self.n_scored}, distribution={self.segment_distribution})"


# ---------------------------------------------------------------------------
# db.analysis / db.qa_audit — standalone analytical operations (0.8.0)
#
# Not skills (no session, no SKILL_REGISTRY entry), not journeys (no graph,
# no tier-gated multi-step orchestration), not a portable scoring artifact.
# Each of these wraps one of the platform routes that had no SDK method at
# all before 0.8.0 — see DEF-0030 in the platform repo's
# docs/context/deferred_items.yaml for the inventory this closes.
# ---------------------------------------------------------------------------

@dataclass
class EDAResult:
    """
    Return type for db.analysis.eda() — POST /v1/eda.

    Attributes:
        n_rows, n_cols        Shape of the analysed frame.
        n_flagged_columns     Columns with at least one data-quality flag.
        n_critical_flags      Flags severe enough to block downstream analysis.
        summary                Token-efficient one-line summary (LLM-facing).
        plain_english_summary  Longer narrative from the report itself.
        warnings                List of warning strings.
        chapter_ref             Knowledge-base chapter reference.
        raw                      Full API response — report.column_profiles,
                                  .critical_flags, .top_relationships,
                                  .journey_recommendations, .data_quality, etc.
    """
    n_rows: int
    n_cols: int
    n_flagged_columns: int
    n_critical_flags: int
    summary: str
    warnings: list[str] = field(default_factory=list)
    chapter_ref: str = ""
    raw: dict = field(default_factory=dict)
    _http: Any = field(default=None, repr=False, compare=False)

    @property
    def report(self) -> dict:
        """The full EDAReport dict."""
        return self.raw.get("report") or {}

    @property
    def plain_english_summary(self) -> str:
        return self.report.get("plain_english_summary", "")

    def export_pdf(self) -> bytes:
        """Render this report as a PDF — POST /v1/export/eda."""
        if self._http is None:
            raise SDKUsageError("export_pdf() requires an EDAResult returned by db.analysis.eda().")
        return self._http.post_bytes("/v1/export/eda", {"eda": self.report})

    def __repr__(self) -> str:
        return (f"EDAResult({self.n_rows} rows x {self.n_cols} cols, "
                f"{self.n_flagged_columns} flagged, {self.n_critical_flags} critical)")


@dataclass
class PulseResult:
    """
    Return type for db.analysis.pulse() — POST /v1/pulse ("Pulse of Data").

    Attributes:
        n_rollups   Number of roll-up grains resolved and profiled.
        deferred    What Pulse could NOT do (distinct from caveats below).
        notes       Free-text notes from the profiling run.
        caveats     Standing interpretation warnings (ecological fallacy /
                    Simpson's paradox etc.) that apply to results Pulse DID
                    produce — render these wherever the roll-up results are.
        raw         Full API response — report.rollups, report.raw_eda, etc.
    """
    n_rollups: int
    deferred: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    _http: Any = field(default=None, repr=False, compare=False)

    @property
    def report(self) -> dict:
        """The full PulseReport dict."""
        return self.raw.get("report") or {}

    @property
    def rollups(self):
        """Roll-up profiles as a DataFrame — one row per resolved grain (empty if n_rollups == 0)."""
        from databubble.tables import rows_to_frame

        return rows_to_frame(self.report.get("rollups"))

    def export_pdf(self) -> bytes:
        """Render this report as a PDF — POST /v1/export/pulse."""
        if self._http is None:
            raise SDKUsageError("export_pdf() requires a PulseResult returned by db.analysis.pulse().")
        return self._http.post_bytes("/v1/export/pulse", {"pulse": self.report})

    def __repr__(self) -> str:
        return f"PulseResult({self.n_rollups} roll-ups, {len(self.deferred)} deferred)"


@dataclass
class ScopeResult:
    """
    Return type for db.analysis.scope() — POST /v1/scope. An advisory
    pre-flight, not a skill or journey: a small sample + a free-text question
    in, a recommendation for how to get an analysis-ready table out.

    Attributes:
        regime                    Classified data regime.
        journey_candidates         Journey types this data plausibly supports.
        recommended_grain          Suggested roll-up grain, if any.
        recommended_sample_rows    Suggested sample size for full ingestion.
        rationale                  Plain-English explanation of the recommendation.
        blocking                   True if this is an out-of-scope verdict, not
                                    just advice — still a 200, not an error.
        raw                        Full API response.
    """
    regime: str
    journey_candidates: list[str]
    rationale: str
    blocking: bool
    recommended_grain: Optional[str] = None
    recommended_sample_rows: Optional[int] = None
    raw: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ScopeResult(regime='{self.regime}', blocking={self.blocking})"


@dataclass
class PowerPlanResult:
    """
    Return type for db.analysis.power_plan() — POST /v1/power/plan.
    Stateless sample-size / detectable-effect planning, no dataset involved.
    Shape depends on mode: "sample_size" -> n_per_group/n_total; the same
    request answered inverted ("detectable_effect") -> mde_absolute instead.
    Both are on .raw regardless of mode; only interpretation is common to both.

    Attributes:
        interpretation   Plain-English statement of the result.
        raw               Full response — n_total/n_per_group (sample_size mode)
                          or mde_absolute/mde_relative (detectable_effect mode),
                          plus effect_size, alpha, power_target, alternative.
    """
    interpretation: str
    raw: dict = field(default_factory=dict)

    @property
    def n_total(self) -> Optional[int]:
        """sample_size mode only — None for detectable_effect mode."""
        return self.raw.get("n_total")

    @property
    def mde_absolute(self) -> Optional[float]:
        """detectable_effect mode only — None for sample_size mode."""
        return self.raw.get("mde_absolute")

    def __repr__(self) -> str:
        return f"PowerPlanResult({self.interpretation})"


@dataclass
class SkillPackResult:
    """
    Return type for db.analysis.export_skill_pack() — POST /v1/export/skill.
    Profiles one uploaded dataset in isolation and renders it as a
    downloadable Anthropic Agent Skill bundle (SKILL.md, profile.json,
    UPDATING.md) — distinct from db.skills.* (which runs one statistical
    check) and db.model/scorecard/segments (which export a fitted artifact).

    Attributes:
        skill_name       Slugified name for the bundle.
        n_rows, n_columns Shape of the profiled dataset.
        blocking_count    Count of blocking data-quality issues found.
        raw               Full response — skill_md/profile_json/updating_md
                          strings, read via .save() rather than by hand.
    """
    skill_name: str
    n_rows: int
    n_columns: int
    blocking_count: int
    raw: dict = field(default_factory=dict)

    def save(self, directory: str) -> None:
        """Write SKILL.md, profile.json and UPDATING.md into `directory`."""
        import os

        os.makedirs(directory, exist_ok=True)
        files = {
            "SKILL.md": self.raw.get("skill_md", ""),
            "profile.json": self.raw.get("profile_json", ""),
            "UPDATING.md": self.raw.get("updating_md", ""),
        }
        for name, content in files.items():
            with open(os.path.join(directory, name), "w") as f:
                f.write(content)
        print(f"Skill pack '{self.skill_name}' saved to {directory}/")

    def __repr__(self) -> str:
        return f"SkillPackResult('{self.skill_name}', {self.n_rows}x{self.n_columns}, blocking={self.blocking_count})"


@dataclass
class CorrelationExportResult:
    """
    Return type for db.analysis.correlation_export(..., format="json").
    format="csv"/"zip" return raw bytes instead (nothing further to wrap) —
    see AnalysisClient.correlation_export()'s docstring.
    """
    raw: dict = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        return (self.raw.get("result") or {}).get("outcome", "")

    def save(self, path: str) -> None:
        """Write the diagnostic JSON to disk."""
        import json
        with open(path, "w") as f:
            json.dump(self.raw, f, indent=2)
        print(f"Correlation diagnostic saved to {path}")

    def __repr__(self) -> str:
        return f"CorrelationExportResult(outcome='{self.outcome}')"


@dataclass
class ForecastExportResult:
    """
    Return type for db.analysis.forecast_export(..., format="json").
    format="csv" returns raw bytes instead — see
    AnalysisClient.forecast_export()'s docstring.
    """
    raw: dict = field(default_factory=dict)

    @property
    def target(self) -> Optional[str]:
        return self.raw.get("target")

    def save(self, path: str) -> None:
        """Write the forecast card JSON to disk."""
        import json
        with open(path, "w") as f:
            json.dump(self.raw, f, indent=2)
        print(f"Forecast card saved to {path}")

    def __repr__(self) -> str:
        return f"ForecastExportResult(target='{self.target}')"


@dataclass
class ExtractedClaimResult:
    """
    Return type for db.qa_audit.extract() — POST /v1/qa-audit/extract.
    Normalized, diffable extraction from a pasted analysis or an uploaded
    artifact (PDF/PPTX/DOCX/TXT/MD). Fails closed: a paste/file this
    couldn't parse comes back with extracted=False and abstain_reason set,
    never an exception.

    Attributes:
        extracted        Whether anything was successfully extracted.
        confidence        "high"/"medium"/"low", when extracted.
        abstain_reason    Why nothing was extracted, when extracted is False.
        raw               Full response — dependent_variable, independent_variables,
                          method claimed, coefficients, etc.
    """
    extracted: bool
    confidence: Optional[str] = None
    abstain_reason: Optional[str] = None
    raw: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        if not self.extracted:
            return f"ExtractedClaimResult(extracted=False, reason='{self.abstain_reason}')"
        return f"ExtractedClaimResult(extracted=True, confidence='{self.confidence}')"


@dataclass
class ClaimDiffResult:
    """
    Return type for db.qa_audit.diff() — POST /v1/qa-audit/diff. Pure,
    deterministic comparison of an extracted claim (db.qa_audit.extract())
    against a completed journey's envelope — no LLM call.

    Attributes:
        summary     Plain-English verdict.
        abstained   True if the comparison could not be made at all
                    (envelope incomplete, claim not extracted).
        raw         Full response — findings[] (per-dimension agree/disagree).
    """
    summary: str
    abstained: bool = False
    raw: dict = field(default_factory=dict)
    _http: Any = field(default=None, repr=False, compare=False)

    @property
    def findings(self):
        """Per-dimension comparison findings as a DataFrame."""
        from databubble.tables import rows_to_frame

        return rows_to_frame(self.raw.get("findings"))

    def __repr__(self) -> str:
        return f"ClaimDiffResult(abstained={self.abstained}, {len(self.raw.get('findings') or [])} findings)"
