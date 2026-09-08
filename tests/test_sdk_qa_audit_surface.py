"""
SDK 0.8.0 db.qa_audit.* tests ("Compare with my analysis"). No running
server required — HTTP is mocked. Response shapes trimmed from real payloads
captured by replaying each method's request against a live platform
TestClient during implementation (DEF-0030).
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock
from databubble.qa_audit import QaAuditClient
from databubble.models import ExtractedClaimResult, ClaimDiffResult
from databubble.exceptions import SDKUsageError


@pytest.fixture
def qa_audit_client():
    http = MagicMock()
    return QaAuditClient(http), http


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

MOCK_EXTRACTED_CLAIM_RESPONSE = {
    "extracted": True, "source_tier": 1, "format_detected": "written_summary",
    "confidence": "medium", "abstain_reason": None,
    "dependent_variable": "sales", "independent_variables": ["price"],
}

MOCK_ABSTAIN_RESPONSE = {
    "extracted": False, "source_tier": None, "format_detected": None,
    "confidence": None, "abstain_reason": "no_recognizable_statistical_content",
}


def test_extract_from_text(qa_audit_client):
    client, http = qa_audit_client
    http.post_multipart.return_value = MOCK_EXTRACTED_CLAIM_RESPONSE

    result = client.extract(text="OLS Regression Results\nprice coef -1.2")

    http.post_multipart.assert_called_once()
    assert http.post_multipart.call_args[0][0] == "/v1/qa-audit/extract"
    fields = http.post_multipart.call_args[1]["fields"]
    assert fields["text"] == "OLS Regression Results\nprice coef -1.2"
    assert result.extracted is True
    assert result.confidence == "medium"


def test_extract_abstains_gracefully_not_an_exception(qa_audit_client):
    client, http = qa_audit_client
    http.post_multipart.return_value = MOCK_ABSTAIN_RESPONSE

    result = client.extract(text="asdf random text with no stats content")

    assert result.extracted is False
    assert result.abstain_reason == "no_recognizable_statistical_content"


def test_extract_from_file_path(qa_audit_client, tmp_path):
    client, http = qa_audit_client
    http.post_multipart.return_value = MOCK_EXTRACTED_CLAIM_RESPONSE
    path = tmp_path / "summary.txt"
    path.write_text("OLS Regression Results\nprice coef -1.2")

    result = client.extract(file=str(path))

    files = http.post_multipart.call_args[1]["files"]
    assert files["file"][0] == str(path)
    assert result.extracted is True


def test_extract_from_bytes_requires_filename(qa_audit_client):
    client, _ = qa_audit_client
    with pytest.raises(SDKUsageError, match="filename"):
        client.extract(file=b"raw bytes")


def test_extract_requires_exactly_one_of_text_or_file(qa_audit_client):
    client, _ = qa_audit_client
    with pytest.raises(SDKUsageError, match="exactly one"):
        client.extract()
    with pytest.raises(SDKUsageError, match="exactly one"):
        client.extract(text="a", file="b.txt")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

MOCK_DIFF_RESPONSE = {
    "journey_type": "elasticity", "mode": "auto", "envelope_incomplete": False,
    "findings": [{"dimension": "sign", "claimed": "negative", "observed": "negative", "agrees": True}],
    "summary": "Claim agrees with the elasticity result on sign and magnitude.",
    "abstained": False, "abstain_reason": None,
}


def test_diff_with_dicts(qa_audit_client):
    client, http = qa_audit_client
    http.post_json.return_value = MOCK_DIFF_RESPONSE
    claim = {"extracted": True, "dependent_variable": "sales"}
    envelope = {"journey_type": "elasticity", "primary_estimate": -0.85}

    result = client.diff(claim, envelope)

    http.post_json.assert_called_once_with(
        "/v1/qa-audit/diff", {"claim": claim, "envelope": envelope}
    )
    assert result.summary == "Claim agrees with the elasticity result on sign and magnitude."
    assert result.abstained is False
    assert len(result.findings) == 1


def test_diff_accepts_result_wrappers(qa_audit_client):
    """claim/envelope can be ExtractedClaimResult / a JourneyResult's .result — anything with .raw."""
    client, http = qa_audit_client
    http.post_json.return_value = MOCK_DIFF_RESPONSE
    claim = ExtractedClaimResult(extracted=True, raw={"extracted": True, "dependent_variable": "sales"})

    client.diff(claim, {"journey_type": "elasticity"})

    payload = http.post_json.call_args[0][1]
    assert payload["claim"] == {"extracted": True, "dependent_variable": "sales"}


def test_diff_wrong_type_raises(qa_audit_client):
    client, _ = qa_audit_client
    with pytest.raises(SDKUsageError, match="claim"):
        client.diff(42, {})


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def test_export_returns_pdf_bytes(qa_audit_client):
    client, http = qa_audit_client
    http.post_bytes.return_value = b"%PDF-1.4 comparison report"
    claim = {"extracted": True}
    envelope = {"journey_type": "elasticity"}
    report = {"journey_type": "elasticity", "summary": "ok", "findings": []}

    pdf = client.export(claim, envelope, report)

    http.post_bytes.assert_called_once_with(
        "/v1/qa-audit/export", {"claim": claim, "envelope": envelope, "report": report}
    )
    assert pdf == b"%PDF-1.4 comparison report"


def test_export_accepts_result_wrappers(qa_audit_client):
    client, http = qa_audit_client
    http.post_bytes.return_value = b"%PDF-1.4"
    claim = ExtractedClaimResult(extracted=True, raw={"extracted": True})
    report = ClaimDiffResult(summary="ok", raw={"summary": "ok", "findings": []})

    client.export(claim, {"journey_type": "elasticity"}, report)

    payload = http.post_bytes.call_args[0][1]
    assert payload["claim"] == {"extracted": True}
    assert payload["report"] == {"summary": "ok", "findings": []}


# ---------------------------------------------------------------------------
# Client wiring
# ---------------------------------------------------------------------------

def test_client_exposes_qa_audit_namespace():
    from databubble import DataBubble

    db = DataBubble(api_key="dbk_test1234567890")
    assert isinstance(db.qa_audit, QaAuditClient)
