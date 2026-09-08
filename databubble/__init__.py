# databubble/__init__.py
"""
DataBubble SDK — Statistical Intelligence as a Service.

Quick start:
    from databubble import DataBubble
    db = DataBubble(api_key="dbk_...")
    r = db.journeys.driver(df, outcome_col="sales", candidate_cols=["price"])
    print(r)              # regression table — coef, std err, t, p, CI
    r.estimates           # pandas DataFrame
    r.explain()           # business narrative, on request
"""

from databubble._version import __version__
from databubble.client import DataBubble
from databubble.models import (
    SkillResult, MemoryResult, ReconciliationResult, JourneyResult,
    ModelCardResult, PredictionResult, ComparisonResult, DriftResult,
    ScorecardResult, ScoreResult, SegmentScorerResult, SegmentScoreResult,
    EDAResult, PulseResult, ScopeResult, PowerPlanResult, SkillPackResult,
    CorrelationExportResult, ForecastExportResult,
    ExtractedClaimResult, ClaimDiffResult,
)
from databubble.charts import Chart, ChartSet
from databubble.exceptions import (
    DataBubbleError, AuthError, ForbiddenError,
    RateLimitError, SkillError, ServerError, SDKUsageError,
)

__all__ = [
    "DataBubble",
    "SkillResult",
    "MemoryResult",
    "ReconciliationResult",
    "JourneyResult",
    "ModelCardResult",
    "PredictionResult",
    "ComparisonResult",
    "DriftResult",
    "ScorecardResult",
    "ScoreResult",
    "SegmentScorerResult",
    "SegmentScoreResult",
    "EDAResult",
    "PulseResult",
    "ScopeResult",
    "PowerPlanResult",
    "SkillPackResult",
    "CorrelationExportResult",
    "ForecastExportResult",
    "ExtractedClaimResult",
    "ClaimDiffResult",
    "Chart",
    "ChartSet",
    "DataBubbleError",
    "AuthError",
    "ForbiddenError",
    "RateLimitError",
    "SkillError",
    "ServerError",
    "SDKUsageError",
]
