"""tests/test_api.py — Integration tests for the FastAPI application endpoints.

Uses FastAPI's ``TestClient`` (backed by ``httpx``) with mocked OpenAI calls
so that no real network requests are made during testing.

Covers:
- GET  /health
- GET  /
- POST /api/audit/prompt        (success, validation error, engine unavailable, OpenAI error)
- POST /api/audit/conversation  (success, validation error, engine unavailable, OpenAI error)
- POST /api/audit/report        (success, download headers, OpenAI error)
- POST /htmx/audit/prompt       (success, empty prompt)
- POST /htmx/audit/conversation (success, invalid JSON, empty body)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from teen_safety_auditor.main import create_app
from teen_safety_auditor.models import (
    AuditResult,
    ConversationAuditResult,
    AuditReport,
    AuditReportSummary,
    TurnAuditResult,
    CategoryScore,
)
from teen_safety_auditor.policy import PolicyCategory, RiskLevel
from teen_safety_auditor.auditor import AuditEngine, OpenAICallError
from teen_safety_auditor import __version__


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

FAKE_API_KEY = "sk-test-key-for-integration-tests"


def _make_safe_audit_result() -> AuditResult:
    """Build a minimal safe AuditResult for mocking."""
    category_scores = [
        CategoryScore(
            category=cat,
            score=0.05,
            risk_level=RiskLevel.SAFE,
        )
        for cat in PolicyCategory
    ]
    return AuditResult(
        prompt_snippet="Hello, how do I stay safe online?",
        response_snippet=None,
        category_scores=category_scores,
        overall_score=0.05,
        overall_risk_level=RiskLevel.SAFE,
        flagged_categories=[],
        reasoning="Content is benign.",
    )


def _make_safe_conversation_result() -> ConversationAuditResult:
    """Build a minimal safe ConversationAuditResult for mocking."""
    category_scores = [
        CategoryScore(category=cat, score=0.05, risk_level=RiskLevel.SAFE)
        for cat in PolicyCategory
    ]
    turn = TurnAuditResult(
        turn_index=0,
        role="user",
        content_snippet="Hello!",
        category_scores=category_scores,
        overall_score=0.05,
        overall_risk_level=RiskLevel.SAFE,
        flagged_categories=[],
        reasoning="Content is safe.",
    )
    return ConversationAuditResult(
        turn_results=[turn],
        total_turns=1,
        flagged_turns=0,
        overall_score=0.05,
        overall_risk_level=RiskLevel.SAFE,
        most_flagged_categories=[],
    )


def _make_safe_audit_report(conv_result: ConversationAuditResult) -> AuditReport:
    """Build a minimal AuditReport for mocking."""
    import uuid
    from datetime import datetime, timezone

    summary = AuditReportSummary(
        total_turns=conv_result.total_turns,
        flagged_turns=conv_result.flagged_turns,
        safe_turns=conv_result.total_turns,
        violation_turns=0,
        caution_turns=0,
        overall_score=conv_result.overall_score,
        overall_risk_level=conv_result.overall_risk_level,  # type: ignore[arg-type]
        most_flagged_categories=[],
        compliance_rate=100.0,
    )
    return AuditReport(
        report_id=str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc),
        schema_version="1.0",
        summary=summary,
        flagged_turns=[],
        all_turn_results=conv_result.turn_results,
        tool_version=__version__,
        metadata={},
    )


def _make_test_client(
    engine_ready: bool = True,
    audit_prompt_result: AuditResult | None = None,
    audit_conversation_result: ConversationAuditResult | None = None,
    audit_report_result: AuditReport | None = None,
    audit_prompt_side_effect: Exception | None = None,
    audit_conversation_side_effect: Exception | None = None,
    audit_report_side_effect: Exception | None = None,
) -> TestClient:
    """Create a TestClient with a mocked AuditEngine attached to app state.

    Parameters
    ----------
    engine_ready:
        If False, ``app.state.engine`` is set to None (simulates missing API key).
    audit_prompt_result:
        Return value for ``engine.audit_prompt``.
    audit_conversation_result:
        Return value for ``engine.audit_conversation``.
    audit_report_result:
        Return value for ``engine.build_report``.
    audit_prompt_side_effect:
        Exception to raise from ``engine.audit_prompt``.
    audit_conversation_side_effect:
        Exception to raise from ``engine.audit_conversation``.
    audit_report_side_effect:
        Exception to raise from ``engine.build_report``.
    """
    app = create_app()

    if not engine_ready:
        # Override lifespan by directly patching state after creation
        # We use the TestClient context manager to handle lifespan
        with TestClient(app, raise_server_exceptions=True) as client:
            client.app.state.engine = None  # type: ignore[union-attr]
            return client

    # Build a mock engine
    mock_engine = MagicMock(spec=AuditEngine)

    if audit_prompt_side_effect is not None:
        mock_engine.audit_prompt = AsyncMock(side_effect=audit_prompt_side_effect)
    else:
        mock_engine.audit_prompt = AsyncMock(
            return_value=audit_prompt_result or _make_safe_audit_result()
        )

    if audit_conversation_side_effect is not None:
        mock_engine.audit_conversation = AsyncMock(
            side_effect=audit_conversation_side_effect
        )
    else:
        mock_engine.audit_conversation = AsyncMock(
            return_value=audit_conversation_result or _make_safe_conversation_result()
        )

    if audit_report_side_effect is not None:
        mock_engine.build_report = AsyncMock(side_effect=audit_report_side_effect)
    else:
        safe_conv = _make_safe_conversation_result()
        mock_engine.build_report = AsyncMock(
            return_value=audit_report_result or _make_safe_audit_report(safe_conv)
        )

    # Patch OPENAI_API_KEY so lifespan does not fail to build a real engine,
    # then override state.engine after startup.
    with patch.dict("os.environ", {"OPENAI_API_KEY": FAKE_API_KEY}):
        with patch("teen_safety_auditor.main.create_engine", return_value=mock_engine):
            client = TestClient(app, raise_server_exceptions=False)
            return client


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for GET /health."""

    def test_health_returns_200(self) -> None:
        client = _make_test_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_structure(self) -> None:
        client = _make_test_client()
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "engine_ready" in data
        assert "version" in data
        assert "timestamp" in data

    def test_health_status_is_ok(self) -> None:
        client = _make_test_client()
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_version_matches(self) -> None:
        client = _make_test_client()
        resp = client.get("/health")
        assert resp.json()["version"] == __version__

    def test_health_engine_ready_true_when_engine_set(self) -> None:
        client = _make_test_client(engine_ready=True)
        resp = client.get("/health")
        assert resp.json()["engine_ready"] is True


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


class TestMainUI:
    """Tests for GET /."""

    def test_index_returns_200(self) -> None:
        client = _make_test_client()
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_content_type_html(self) -> None:
        client = _make_test_client()
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_index_contains_html_tag(self) -> None:
        client = _make_test_client()
        resp = client.get("/")
        assert "<html" in resp.text.lower() or "<!doctype" in resp.text.lower()


# ---------------------------------------------------------------------------
# POST /api/audit/prompt
# ---------------------------------------------------------------------------


class TestAuditPromptEndpoint:
    """Integration tests for POST /api/audit/prompt."""

    def test_valid_prompt_returns_200(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "How do I make friends safely online?"},
        )
        assert resp.status_code == 200

    def test_response_is_audit_result_structure(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "Hello there!"},
        )
        data = resp.json()
        assert "prompt_snippet" in data
        assert "overall_score" in data
        assert "overall_risk_level" in data
        assert "category_scores" in data
        assert "flagged_categories" in data

    def test_safe_result_has_safe_risk_level(self) -> None:
        safe_result = _make_safe_audit_result()
        client = _make_test_client(audit_prompt_result=safe_result)
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "Good morning!"},
        )
        assert resp.status_code == 200
        assert resp.json()["overall_risk_level"] == "safe"

    def test_with_optional_response_field(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/api/audit/prompt",
            json={
                "prompt": "What is safe browsing?",
                "response": "Safe browsing means using HTTPS and avoiding suspicious links.",
            },
        )
        assert resp.status_code == 200

    def test_empty_prompt_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/prompt", json={"prompt": ""})
        assert resp.status_code == 422

    def test_missing_prompt_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/prompt", json={})
        assert resp.status_code == 422

    def test_whitespace_prompt_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/prompt", json={"prompt": "   "})
        assert resp.status_code == 422

    def test_openai_error_returns_500(self) -> None:
        client = _make_test_client(
            audit_prompt_side_effect=OpenAICallError("Rate limit exceeded")
        )
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "Test prompt"},
        )
        assert resp.status_code == 500

    def test_openai_error_detail_in_response(self) -> None:
        client = _make_test_client(
            audit_prompt_side_effect=OpenAICallError("Timeout error")
        )
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "Test"},
        )
        assert resp.status_code == 500
        assert "Timeout error" in resp.json().get("detail", "")

    def test_category_scores_all_present(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "Hello!"},
        )
        data = resp.json()
        assert len(data["category_scores"]) == len(list(PolicyCategory))

    def test_audit_timestamp_in_response(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "Hello!"},
        )
        data = resp.json()
        assert "audit_timestamp" in data
        assert data["audit_timestamp"] is not None

    def test_reasoning_in_response(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "Hello!"},
        )
        data = resp.json()
        assert "reasoning" in data

    def test_unexpected_error_returns_500(self) -> None:
        client = _make_test_client(
            audit_prompt_side_effect=RuntimeError("Unexpected internal error")
        )
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "Test"},
        )
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /api/audit/conversation
# ---------------------------------------------------------------------------


class TestAuditConversationEndpoint:
    """Integration tests for POST /api/audit/conversation."""

    _valid_body: dict[str, Any] = {
        "turns": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }

    def test_valid_conversation_returns_200(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/conversation", json=self._valid_body)
        assert resp.status_code == 200

    def test_response_is_conversation_audit_result_structure(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/conversation", json=self._valid_body)
        data = resp.json()
        assert "turn_results" in data
        assert "total_turns" in data
        assert "flagged_turns" in data
        assert "overall_score" in data
        assert "overall_risk_level" in data

    def test_empty_turns_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/conversation", json={"turns": []})
        assert resp.status_code == 422

    def test_missing_turns_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/conversation", json={})
        assert resp.status_code == 422

    def test_no_user_turn_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/api/audit/conversation",
            json={"turns": [{"role": "assistant", "content": "Hi"}]},
        )
        assert resp.status_code == 422

    def test_invalid_role_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/api/audit/conversation",
            json={"turns": [{"role": "bot", "content": "Hello"}]},
        )
        assert resp.status_code == 422

    def test_openai_error_returns_500(self) -> None:
        client = _make_test_client(
            audit_conversation_side_effect=OpenAICallError("API down")
        )
        resp = client.post("/api/audit/conversation", json=self._valid_body)
        assert resp.status_code == 500

    def test_safe_result_has_safe_risk_level(self) -> None:
        safe_result = _make_safe_conversation_result()
        client = _make_test_client(audit_conversation_result=safe_result)
        resp = client.post("/api/audit/conversation", json=self._valid_body)
        assert resp.status_code == 200
        assert resp.json()["overall_risk_level"] == "safe"

    def test_audit_timestamp_present(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/conversation", json=self._valid_body)
        data = resp.json()
        assert "audit_timestamp" in data

    def test_most_flagged_categories_list(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/conversation", json=self._valid_body)
        data = resp.json()
        assert isinstance(data["most_flagged_categories"], list)

    def test_unexpected_error_returns_500(self) -> None:
        client = _make_test_client(
            audit_conversation_side_effect=RuntimeError("Unexpected")
        )
        resp = client.post("/api/audit/conversation", json=self._valid_body)
        assert resp.status_code == 500

    def test_single_user_turn_accepted(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/api/audit/conversation",
            json={"turns": [{"role": "user", "content": "Just me talking."}]},
        )
        assert resp.status_code == 200

    def test_total_turns_matches_result(self) -> None:
        result = _make_safe_conversation_result()
        client = _make_test_client(audit_conversation_result=result)
        resp = client.post("/api/audit/conversation", json=self._valid_body)
        assert resp.json()["total_turns"] == result.total_turns


# ---------------------------------------------------------------------------
# POST /api/audit/report
# ---------------------------------------------------------------------------


class TestAuditReportEndpoint:
    """Integration tests for POST /api/audit/report."""

    _valid_body: dict[str, Any] = {
        "turns": [
            {"role": "user", "content": "How do I stay safe online?"},
            {"role": "assistant", "content": "Here are some tips for online safety..."},
        ]
    }

    def test_valid_request_returns_200(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        assert resp.status_code == 200

    def test_response_content_type_is_json(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        assert "application/json" in resp.headers["content-type"]

    def test_response_has_content_disposition_attachment(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        cd = resp.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert ".json" in cd

    def test_filename_starts_with_teen_safety_audit(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        cd = resp.headers.get("content-disposition", "")
        assert "teen_safety_audit_" in cd

    def test_response_body_is_valid_json(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        data = resp.json()
        assert isinstance(data, dict)

    def test_response_has_report_id(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        data = resp.json()
        assert "report_id" in data
        assert len(data["report_id"]) > 0

    def test_response_has_summary(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        data = resp.json()
        assert "summary" in data
        assert "total_turns" in data["summary"]

    def test_response_has_schema_version(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        data = resp.json()
        assert data.get("schema_version") == "1.0"

    def test_response_has_policy_reference(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        data = resp.json()
        assert "policy_reference" in data
        assert data["policy_reference"].startswith("http")

    def test_openai_error_returns_500(self) -> None:
        client = _make_test_client(
            audit_report_side_effect=OpenAICallError("OpenAI API failed")
        )
        resp = client.post("/api/audit/report", json=self._valid_body)
        assert resp.status_code == 500

    def test_empty_turns_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json={"turns": []})
        assert resp.status_code == 422

    def test_missing_turns_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json={})
        assert resp.status_code == 422

    def test_unexpected_error_returns_500(self) -> None:
        client = _make_test_client(
            audit_report_side_effect=RuntimeError("Unexpected error")
        )
        resp = client.post("/api/audit/report", json=self._valid_body)
        assert resp.status_code == 500

    def test_tool_version_in_report(self) -> None:
        client = _make_test_client()
        resp = client.post("/api/audit/report", json=self._valid_body)
        data = resp.json()
        assert "tool_version" in data
        assert len(data["tool_version"]) > 0


# ---------------------------------------------------------------------------
# POST /htmx/audit/prompt
# ---------------------------------------------------------------------------


class TestHtmxAuditPrompt:
    """Integration tests for POST /htmx/audit/prompt."""

    def test_valid_form_returns_200(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/htmx/audit/prompt",
            data={"prompt": "Tell me about online safety."},
        )
        assert resp.status_code == 200

    def test_response_is_html(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/htmx/audit/prompt",
            data={"prompt": "Hello"},
        )
        assert "text/html" in resp.headers["content-type"]

    def test_empty_prompt_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/htmx/audit/prompt",
            data={"prompt": ""},
        )
        assert resp.status_code == 422

    def test_missing_prompt_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post("/htmx/audit/prompt", data={})
        assert resp.status_code == 422

    def test_openai_error_returns_500_html(self) -> None:
        client = _make_test_client(
            audit_prompt_side_effect=OpenAICallError("OpenAI error")
        )
        resp = client.post(
            "/htmx/audit/prompt",
            data={"prompt": "Test"},
        )
        assert resp.status_code == 500
        assert "text/html" in resp.headers["content-type"]

    def test_with_response_field(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/htmx/audit/prompt",
            data={"prompt": "Question", "response": "Answer"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /htmx/audit/conversation
# ---------------------------------------------------------------------------


class TestHtmxAuditConversation:
    """Integration tests for POST /htmx/audit/conversation."""

    _valid_conversation_json: str = json.dumps([
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ])

    def test_valid_form_returns_200(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/htmx/audit/conversation",
            data={"conversation": self._valid_conversation_json},
        )
        assert resp.status_code == 200

    def test_response_is_html(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/htmx/audit/conversation",
            data={"conversation": self._valid_conversation_json},
        )
        assert "text/html" in resp.headers["content-type"]

    def test_empty_body_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/htmx/audit/conversation",
            data={"conversation": ""},
        )
        assert resp.status_code == 422

    def test_missing_conversation_field_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post("/htmx/audit/conversation", data={})
        assert resp.status_code == 422

    def test_invalid_json_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/htmx/audit/conversation",
            data={"conversation": "this is not json"},
        )
        assert resp.status_code == 422

    def test_non_array_json_returns_422(self) -> None:
        client = _make_test_client()
        resp = client.post(
            "/htmx/audit/conversation",
            data={"conversation": '{"role": "user", "content": "Hi"}'},
        )
        assert resp.status_code == 422

    def test_openai_error_returns_500_html(self) -> None:
        client = _make_test_client(
            audit_conversation_side_effect=OpenAICallError("OpenAI error")
        )
        resp = client.post(
            "/htmx/audit/conversation",
            data={"conversation": self._valid_conversation_json},
        )
        assert resp.status_code == 500
        assert "text/html" in resp.headers["content-type"]

    def test_no_user_turn_returns_422(self) -> None:
        """A conversation with only assistant turns should fail validation."""
        client = _make_test_client()
        bad_conv = json.dumps([{"role": "assistant", "content": "Hi"}])
        resp = client.post(
            "/htmx/audit/conversation",
            data={"conversation": bad_conv},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Engine unavailable (missing API key)
# ---------------------------------------------------------------------------


class TestEngineUnavailable:
    """Tests for 503 responses when the audit engine is not initialised."""

    def _make_unavailable_client(self) -> TestClient:
        """Create a TestClient where the engine is explicitly None."""
        app = create_app()
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            with patch("teen_safety_auditor.main.create_engine", side_effect=Exception("No key")):
                client = TestClient(app, raise_server_exceptions=False)
                # Force engine to None after startup
                client.app.state.engine = None  # type: ignore[union-attr]
                return client

    def test_audit_prompt_503_when_no_engine(self) -> None:
        client = self._make_unavailable_client()
        resp = client.post(
            "/api/audit/prompt",
            json={"prompt": "Test"},
        )
        assert resp.status_code == 503

    def test_audit_conversation_503_when_no_engine(self) -> None:
        client = self._make_unavailable_client()
        resp = client.post(
            "/api/audit/conversation",
            json={"turns": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 503

    def test_audit_report_503_when_no_engine(self) -> None:
        client = self._make_unavailable_client()
        resp = client.post(
            "/api/audit/report",
            json={"turns": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------


class TestAppMetadata:
    """Tests for FastAPI application metadata."""

    def test_openapi_docs_accessible(self) -> None:
        client = _make_test_client()
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_accessible(self) -> None:
        client = _make_test_client()
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_title(self) -> None:
        client = _make_test_client()
        resp = client.get("/openapi.json")
        data = resp.json()
        assert data["info"]["title"] == "Teen Safety Auditor"

    def test_openapi_version_matches_package(self) -> None:
        client = _make_test_client()
        resp = client.get("/openapi.json")
        data = resp.json()
        assert data["info"]["version"] == __version__

    def test_audit_prompt_route_in_schema(self) -> None:
        client = _make_test_client()
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/audit/prompt" in paths

    def test_audit_conversation_route_in_schema(self) -> None:
        client = _make_test_client()
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/audit/conversation" in paths

    def test_audit_report_route_in_schema(self) -> None:
        client = _make_test_client()
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/api/audit/report" in paths


# ---------------------------------------------------------------------------
# create_app factory
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_returns_fastapi_instance(self) -> None:
        from fastapi import FastAPI

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_title(self) -> None:
        app = create_app()
        assert app.title == "Teen Safety Auditor"

    def test_app_version(self) -> None:
        app = create_app()
        assert app.version == __version__

    def test_app_has_state(self) -> None:
        app = create_app()
        # State container should be attached
        assert hasattr(app, "state")
