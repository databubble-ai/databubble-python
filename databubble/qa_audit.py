# databubble/qa_audit.py
"""
QaAuditClient — db.qa_audit.* : "Compare with my analysis" (Analysis QA/Audit).

Three steps, matching the platform's own confirm-before-diff UX — extract
first so a caller can see and confirm what was understood, before the
(cheap, deterministic) diff runs:

    claim = db.qa_audit.extract(text=pasted_summary)
    report = db.qa_audit.diff(claim, envelope=result.result)
    pdf = db.qa_audit.export(claim, envelope=result.result, report=report)

Requires business or enterprise tier (or a demo key) — see
api.keystore.TIER_QA_AUDIT_ACCESS on the platform. Added in 0.8.0; see
DEF-0030 in the platform repo's docs/context/deferred_items.yaml.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from databubble.exceptions import SDKUsageError
from databubble.models import ExtractedClaimResult, ClaimDiffResult


def _artifact_dict(value: Any, method: str, arg_name: str) -> dict:
    """claim/envelope/report each accept the SDK's own result wrapper (has
    .raw) or a plain dict — matches _scoring_common.artifact_dict's contract,
    duplicated here rather than imported since these aren't scoring artifacts."""
    if hasattr(value, "raw") and isinstance(value.raw, dict):
        return value.raw
    if isinstance(value, dict):
        return value
    raise SDKUsageError(
        f"{method}(): {arg_name} must be a dict or a result object with .raw. "
        f"Got {type(value).__name__}."
    )


class QaAuditClient:
    def __init__(self, http_client):
        self._http = http_client

    def extract(
        self,
        text: Optional[str] = None,
        file: Optional[Union[str, bytes]] = None,
        filename: Optional[str] = None,
        artifact_type: str = "auto",
        journey_type: Optional[str] = None,
    ) -> ExtractedClaimResult:
        """
        Extract a single statistical claim from pasted analysis text or an
        artifact file — POST /v1/qa-audit/extract. Exactly one of `text` or
        `file` must be given.

        Args:
            text:           Pasted analysis text (e.g. a copy-pasted regression
                             summary, or a written narrative).
            file:            A file path (str) or raw bytes of an uploaded
                             artifact (PDF/PPTX/DOCX/TXT/MD).
            filename:        Required when `file` is raw bytes, so the server
                             knows the file type. Inferred from the path when
                             `file` is a str path.
            artifact_type:   "auto" (default), or an explicit hint.
            journey_type:    Optional hint for the LLM fallback prompt —
                             "segmentation" or "time_series" pick a
                             journey-specific extractor; anything else falls
                             back to the regression-shaped prompts.

        Returns:
            ExtractedClaimResult — .extracted, .confidence, .abstain_reason.
            Never raises on an unrecognized artifact: extracted=False with
            abstain_reason set is the fail-closed result, not an error.
        """
        if bool(text) == bool(file):
            raise SDKUsageError(
                "db.qa_audit.extract(): provide exactly one of text= or file=."
            )

        fields = {"artifact_type": artifact_type, "journey_type": journey_type}
        if text is not None:
            fields["text"] = text
            response = self._http.post_multipart("/v1/qa-audit/extract", fields=fields, files={})
        else:
            if isinstance(file, str):
                with open(file, "rb") as f:
                    file_bytes = f.read()
                resolved_name = filename or file
            else:
                if filename is None:
                    raise SDKUsageError(
                        "db.qa_audit.extract(): filename= is required when file= is raw bytes."
                    )
                file_bytes = file
                resolved_name = filename
            response = self._http.post_multipart(
                "/v1/qa-audit/extract", fields=fields,
                files={"file": (resolved_name, file_bytes, "application/octet-stream")},
            )

        return ExtractedClaimResult(
            extracted=response.get("extracted", False),
            confidence=response.get("confidence"),
            abstain_reason=response.get("abstain_reason"),
            raw=response,
        )

    def diff(self, claim: Any, envelope: Any) -> ClaimDiffResult:
        """
        Diff an extracted claim against a completed journey's envelope —
        POST /v1/qa-audit/diff. Pure and deterministic, no LLM call.

        Args:
            claim:     an ExtractedClaimResult (from .extract()) or dict.
            envelope:  a JourneyResult's .result dict (the envelope), or the
                       raw envelope dict.

        Returns:
            ClaimDiffResult — .summary, .abstained, .findings (DataFrame).
        """
        claim_dict = _artifact_dict(claim, "db.qa_audit.diff", "claim")
        envelope_dict = _artifact_dict(envelope, "db.qa_audit.diff", "envelope")

        response = self._http.post_json(
            "/v1/qa-audit/diff", {"claim": claim_dict, "envelope": envelope_dict}
        )
        return ClaimDiffResult(
            summary=response.get("summary", ""),
            abstained=response.get("abstained", False),
            raw=response,
            _http=self._http,
        )

    def export(self, claim: Any, envelope: Any, report: Any) -> bytes:
        """
        Render a prior diff() result as a downloadable PDF comparison report —
        POST /v1/qa-audit/export.

        Args:
            claim:     an ExtractedClaimResult (from .extract()) or dict.
            envelope:  a JourneyResult's .result dict, or the raw dict.
            report:    a ClaimDiffResult (from .diff()) or dict.

        Returns:
            Raw PDF bytes — write them to a file yourself.

        Example:
            pdf = db.qa_audit.export(claim, result.result, report)
            with open("comparison_report.pdf", "wb") as f:
                f.write(pdf)
        """
        claim_dict = _artifact_dict(claim, "db.qa_audit.export", "claim")
        envelope_dict = _artifact_dict(envelope, "db.qa_audit.export", "envelope")
        report_dict = _artifact_dict(report, "db.qa_audit.export", "report")

        return self._http.post_bytes(
            "/v1/qa-audit/export",
            {"claim": claim_dict, "envelope": envelope_dict, "report": report_dict},
        )
