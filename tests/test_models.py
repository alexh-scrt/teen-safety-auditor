"""tests/test_models.py — Unit tests for teen_safety_auditor/models.py.

Covers:
- SinglePromptRequest validation (required fields, blanks, coercions)
- ConversationTurn validation (role normalisation, content checks)
- ConversationRequest validation (min turns, user-turn requirement)
- CategoryScore metadata auto-population
- TurnAuditResult helpers and field constraints
- AuditResult and ConversationAuditResult construction
- AuditReport and sub-models construction
- AuditErrorResponse construction
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from teen_safety_auditor.models import (
    AuditErrorResponse,
    AuditReport,
    AuditReportSummary,
    AuditResult,
    CategoryScore,
    ConversationAuditResult,
    ConversationRequest,
    ConversationTurn,
    FlaggedTurnSummary,
    SinglePromptRequest,
    TurnAuditResult,
)
from teen_safety_auditor.policy import PolicyCategory, RiskLevel


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_category_score(
    category: PolicyCategory = PolicyCategory.VIOLENCE,
    score: float = 0.1,
    risk_level: RiskLevel = RiskLevel.SAFE,
) -> CategoryScore:
    return CategoryScore(category=category, score=score, risk_level=risk_level)


def _make_turn_result(
    turn_index: int = 0,
    role: str = "user",
    content_snippet: str = "Hello world",
    overall_score: float = 0.1,
    overall_risk_level: RiskLevel = RiskLevel.SAFE,
) -> TurnAuditResult:
    return TurnAuditResult(
        turn_index=turn_index,
        role=role,
        content_snippet=content_snippet,
        overall_score=overall_score,
        overall_risk_level=overall_risk_level,
    )


def _make_audit_report_summary(
    total_turns: int = 3,
    overall_score: float = 0.1,
    overall_risk_level: RiskLevel = RiskLevel.SAFE,
) -> AuditReportSummary:
    return AuditReportSummary(
        total_turns=total_turns,
        overall_score=overall_score,
        overall_risk_level=overall_risk_level,
    )


# ---------------------------------------------------------------------------
# SinglePromptRequest
# ---------------------------------------------------------------------------


class TestSinglePromptRequest:
    """Tests for SinglePromptRequest model."""

    def test_valid_prompt_only(self) -> None:
        req = SinglePromptRequest(prompt="Tell me about online safety.")
        assert req.prompt == "Tell me about online safety."
        assert req.response is None

    def test_valid_prompt_and_response(self) -> None:
        req = SinglePromptRequest(
            prompt="What is safe browsing?",
            response="Safe browsing means…",
        )
        assert req.response == "Safe browsing means…"

    def test_prompt_is_stripped(self) -> None:
        req = SinglePromptRequest(prompt="  hello world  ")
        assert req.prompt == "hello world"

    def test_response_is_stripped(self) -> None:
        req = SinglePromptRequest(prompt="Q", response="  answer  ")
        assert req.response == "answer"

    def test_empty_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            SinglePromptRequest(prompt="")

    def test_whitespace_only_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            SinglePromptRequest(prompt="   ")

    def test_blank_response_coerced_to_none(self) -> None:
        req = SinglePromptRequest(prompt="Q", response="   ")
        assert req.response is None

    def test_empty_response_coerced_to_none(self) -> None:
        req = SinglePromptRequest(prompt="Q", response="")
        assert req.response is None

    def test_prompt_too_long_raises(self) -> None:
        with pytest.raises(ValidationError):
            SinglePromptRequest(prompt="x" * 32_001)

    def test_response_too_long_raises(self) -> None:
        with pytest.raises(ValidationError):
            SinglePromptRequest(prompt="Q", response="x" * 32_001)

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            SinglePromptRequest()  # type: ignore[call-arg]

    def test_model_dump_contains_expected_keys(self) -> None:
        req = SinglePromptRequest(prompt="Test")
        data = req.model_dump()
        assert "prompt" in data
        assert "response" in data


# ---------------------------------------------------------------------------
# ConversationTurn
# ---------------------------------------------------------------------------


class TestConversationTurn:
    """Tests for ConversationTurn model."""

    def test_valid_user_turn(self) -> None:
        turn = ConversationTurn(role="user", content="Hello!")
        assert turn.role == "user"
        assert turn.content == "Hello!"

    def test_valid_assistant_turn(self) -> None:
        turn = ConversationTurn(role="assistant", content="Hi there!")
        assert turn.role == "assistant"

    def test_valid_system_turn(self) -> None:
        turn = ConversationTurn(role="system", content="You are helpful.")
        assert turn.role == "system"

    def test_role_normalised_to_lowercase(self) -> None:
        turn = ConversationTurn(role="User", content="Hello")
        assert turn.role == "user"

    def test_role_with_surrounding_whitespace_normalised(self) -> None:
        turn = ConversationTurn(role=" assistant ", content="Hi")
        assert turn.role == "assistant"

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConversationTurn(role="bot", content="Hello")

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConversationTurn(role="user", content="")

    def test_whitespace_only_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConversationTurn(role="user", content="   ")

    def test_content_stripped(self) -> None:
        turn = ConversationTurn(role="user", content="  hello  ")
        assert turn.content == "hello"

    def test_content_too_long_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConversationTurn(role="user", content="x" * 32_001)

    def test_model_dump_round_trip(self) -> None:
        turn = ConversationTurn(role="user", content="Test message")
        data = turn.model_dump()
        restored = ConversationTurn(**data)
        assert restored.role == turn.role
        assert restored.content == turn.content


# ---------------------------------------------------------------------------
# ConversationRequest
# ---------------------------------------------------------------------------


class TestConversationRequest:
    """Tests for ConversationRequest model."""

    def test_valid_single_user_turn(self) -> None:
        req = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hello")]
        )
        assert len(req.turns) == 1

    def test_valid_multi_turn(self) -> None:
        req = ConversationRequest(
            turns=[
                ConversationTurn(role="user", content="Hi"),
                ConversationTurn(role="assistant", content="Hello!"),
                ConversationTurn(role="user", content="How are you?"),
            ]
        )
        assert len(req.turns) == 3

    def test_empty_turns_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConversationRequest(turns=[])

    def test_no_user_turn_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConversationRequest(
                turns=[ConversationTurn(role="assistant", content="Hello!")]
            )

    def test_too_many_turns_raises(self) -> None:
        turns = [
            ConversationTurn(role="user", content=f"Message {i}")
            for i in range(101)
        ]
        with pytest.raises(ValidationError):
            ConversationRequest(turns=turns)

    def test_exactly_100_turns_allowed(self) -> None:
        turns = [
            ConversationTurn(role="user", content=f"Message {i}")
            for i in range(100)
        ]
        req = ConversationRequest(turns=turns)
        assert len(req.turns) == 100

    def test_system_turn_only_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConversationRequest(
                turns=[ConversationTurn(role="system", content="You are helpful.")]
            )

    def test_turns_are_ordered(self) -> None:
        turns = [
            ConversationTurn(role="user", content="First"),
            ConversationTurn(role="assistant", content="Second"),
        ]
        req = ConversationRequest(turns=turns)
        assert req.turns[0].content == "First"
        assert req.turns[1].content == "Second"


# ---------------------------------------------------------------------------
# CategoryScore
# ---------------------------------------------------------------------------


class TestCategoryScore:
    """Tests for CategoryScore model."""

    def test_basic_construction(self) -> None:
        cs = _make_category_score()
        assert cs.category == PolicyCategory.VIOLENCE
        assert cs.score == 0.1
        assert cs.risk_level == RiskLevel.SAFE

    def test_display_name_auto_populated(self) -> None:
        cs = _make_category_score(category=PolicyCategory.VIOLENCE)
        assert cs.display_name == "Violence"

    def test_description_auto_populated(self) -> None:
        cs = _make_category_score(category=PolicyCategory.SELF_HARM)
        assert len(cs.description) > 0

    def test_remediation_hint_auto_populated(self) -> None:
        cs = _make_category_score(category=PolicyCategory.GROOMING)
        assert len(cs.remediation_hint) > 0

    def test_all_categories_get_metadata(self) -> None:
        for cat in PolicyCategory:
            cs = CategoryScore(category=cat, score=0.1, risk_level=RiskLevel.SAFE)
            assert len(cs.display_name) > 0
            assert len(cs.description) > 0
            assert len(cs.remediation_hint) > 0

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            CategoryScore(category=PolicyCategory.VIOLENCE, score=-0.01, risk_level=RiskLevel.SAFE)

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            CategoryScore(category=PolicyCategory.VIOLENCE, score=1.01, risk_level=RiskLevel.SAFE)

    def test_score_zero_allowed(self) -> None:
        cs = CategoryScore(category=PolicyCategory.VIOLENCE, score=0.0, risk_level=RiskLevel.SAFE)
        assert cs.score == 0.0

    def test_score_one_allowed(self) -> None:
        cs = CategoryScore(category=PolicyCategory.VIOLENCE, score=1.0, risk_level=RiskLevel.VIOLATION)
        assert cs.score == 1.0

    def test_explicit_display_name_not_overwritten(self) -> None:
        cs = CategoryScore(
            category=PolicyCategory.VIOLENCE,
            score=0.1,
            risk_level=RiskLevel.SAFE,
            display_name="Custom Label",
        )
        assert cs.display_name == "Custom Label"

    def test_model_dump_serialisable(self) -> None:
        cs = _make_category_score()
        data = cs.model_dump()
        assert isinstance(data, dict)
        assert "category" in data
        assert "score" in data
        assert "risk_level" in data

    @pytest.mark.parametrize("cat", list(PolicyCategory))
    def test_every_category_constructs(self, cat: PolicyCategory) -> None:
        cs = CategoryScore(category=cat, score=0.5, risk_level=RiskLevel.CAUTION)
        assert cs.category == cat.value  # use_enum_values stores string


# ---------------------------------------------------------------------------
# TurnAuditResult
# ---------------------------------------------------------------------------


class TestTurnAuditResult:
    """Tests for TurnAuditResult model."""

    def test_basic_construction(self) -> None:
        turn = _make_turn_result()
        assert turn.turn_index == 0
        assert turn.role == "user"
        assert turn.overall_score == 0.1
        assert turn.overall_risk_level == RiskLevel.SAFE.value

    def test_negative_turn_index_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make_turn_result(turn_index=-1)

    def test_zero_turn_index_allowed(self) -> None:
        turn = _make_turn_result(turn_index=0)
        assert turn.turn_index == 0

    def test_overall_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _make_turn_result(overall_score=1.5)
        with pytest.raises(ValidationError):
            _make_turn_result(overall_score=-0.1)

    def test_flagged_categories_defaults_empty(self) -> None:
        turn = _make_turn_result()
        assert turn.flagged_categories == []

    def test_category_scores_defaults_empty(self) -> None:
        turn = _make_turn_result()
        assert turn.category_scores == []

    def test_reasoning_defaults_empty(self) -> None:
        turn = _make_turn_result()
        assert turn.reasoning == ""

    def test_make_content_snippet_short_string(self) -> None:
        snippet = TurnAuditResult.make_content_snippet("Hello world")
        assert snippet == "Hello world"

    def test_make_content_snippet_exactly_200_chars(self) -> None:
        content = "x" * 200
        snippet = TurnAuditResult.make_content_snippet(content)
        assert snippet == content
        assert len(snippet) == 200

    def test_make_content_snippet_truncates_at_200(self) -> None:
        content = "x" * 250
        snippet = TurnAuditResult.make_content_snippet(content)
        assert len(snippet) <= 204  # 200 chars + possible ellipsis
        assert snippet.endswith("…")

    def test_make_content_snippet_custom_max_length(self) -> None:
        content = "a" * 100
        snippet = TurnAuditResult.make_content_snippet(content, max_length=50)
        assert snippet.endswith("…")

    def test_with_category_scores(self) -> None:
        scores = [_make_category_score(cat) for cat in PolicyCategory]
        turn = TurnAuditResult(
            turn_index=0,
            role="user",
            content_snippet="Hi",
            overall_score=0.1,
            overall_risk_level=RiskLevel.SAFE,
            category_scores=scores,
        )
        assert len(turn.category_scores) == len(list(PolicyCategory))

    def test_with_flagged_categories(self) -> None:
        turn = TurnAuditResult(
            turn_index=1,
            role="assistant",
            content_snippet="Some risky content",
            overall_score=0.8,
            overall_risk_level=RiskLevel.VIOLATION,
            flagged_categories=[PolicyCategory.VIOLENCE, PolicyCategory.HATE_SPEECH],
        )
        assert len(turn.flagged_categories) == 2


# ---------------------------------------------------------------------------
# AuditResult
# ---------------------------------------------------------------------------


class TestAuditResult:
    """Tests for AuditResult model."""

    def test_basic_construction(self) -> None:
        result = AuditResult(
            prompt_snippet="Hello, how are you?",
            overall_score=0.05,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert result.prompt_snippet == "Hello, how are you?"
        assert result.overall_score == 0.05
        assert result.response_snippet is None

    def test_with_response_snippet(self) -> None:
        result = AuditResult(
            prompt_snippet="Q",
            response_snippet="A",
            overall_score=0.1,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert result.response_snippet == "A"

    def test_overall_score_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            AuditResult(
                prompt_snippet="Q",
                overall_score=1.1,
                overall_risk_level=RiskLevel.VIOLATION,
            )

    def test_flagged_categories_default_empty(self) -> None:
        result = AuditResult(
            prompt_snippet="Q",
            overall_score=0.0,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert result.flagged_categories == []

    def test_audit_timestamp_is_datetime(self) -> None:
        result = AuditResult(
            prompt_snippet="Q",
            overall_score=0.0,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert isinstance(result.audit_timestamp, datetime)

    def test_audit_timestamp_is_utc(self) -> None:
        result = AuditResult(
            prompt_snippet="Q",
            overall_score=0.0,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert result.audit_timestamp.tzinfo is not None

    def test_model_dump_serialisable(self) -> None:
        result = AuditResult(
            prompt_snippet="Q",
            overall_score=0.1,
            overall_risk_level=RiskLevel.SAFE,
        )
        data = result.model_dump()
        assert isinstance(data, dict)
        assert "prompt_snippet" in data
        assert "overall_score" in data
        assert "overall_risk_level" in data
        assert "audit_timestamp" in data

    def test_with_category_scores(self) -> None:
        scores = [_make_category_score()]
        result = AuditResult(
            prompt_snippet="Q",
            overall_score=0.1,
            overall_risk_level=RiskLevel.SAFE,
            category_scores=scores,
        )
        assert len(result.category_scores) == 1


# ---------------------------------------------------------------------------
# ConversationAuditResult
# ---------------------------------------------------------------------------


class TestConversationAuditResult:
    """Tests for ConversationAuditResult model."""

    def test_basic_construction(self) -> None:
        result = ConversationAuditResult(
            total_turns=2,
            overall_score=0.1,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert result.total_turns == 2
        assert result.flagged_turns == 0

    def test_flagged_turns_defaults_zero(self) -> None:
        result = ConversationAuditResult(
            total_turns=5,
            overall_score=0.2,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert result.flagged_turns == 0

    def test_turn_results_defaults_empty(self) -> None:
        result = ConversationAuditResult(
            total_turns=0,
            overall_score=0.0,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert result.turn_results == []

    def test_most_flagged_categories_defaults_empty(self) -> None:
        result = ConversationAuditResult(
            total_turns=1,
            overall_score=0.0,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert result.most_flagged_categories == []

    def test_audit_timestamp_set(self) -> None:
        result = ConversationAuditResult(
            total_turns=1,
            overall_score=0.0,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert isinstance(result.audit_timestamp, datetime)

    def test_with_turn_results(self) -> None:
        turns = [_make_turn_result(i) for i in range(3)]
        result = ConversationAuditResult(
            turn_results=turns,
            total_turns=3,
            flagged_turns=1,
            overall_score=0.2,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert len(result.turn_results) == 3
        assert result.flagged_turns == 1

    def test_negative_total_turns_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConversationAuditResult(
                total_turns=-1,
                overall_score=0.0,
                overall_risk_level=RiskLevel.SAFE,
            )

    def test_overall_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            ConversationAuditResult(
                total_turns=1,
                overall_score=1.5,
                overall_risk_level=RiskLevel.VIOLATION,
            )


# ---------------------------------------------------------------------------
# AuditReportSummary
# ---------------------------------------------------------------------------


class TestAuditReportSummary:
    """Tests for AuditReportSummary model."""

    def test_basic_construction(self) -> None:
        summary = _make_audit_report_summary()
        assert summary.total_turns == 3
        assert summary.overall_score == 0.1

    def test_defaults(self) -> None:
        summary = _make_audit_report_summary()
        assert summary.flagged_turns == 0
        assert summary.safe_turns == 0
        assert summary.violation_turns == 0
        assert summary.caution_turns == 0
        assert summary.compliance_rate == 100.0
        assert summary.most_flagged_categories == []

    def test_compliance_rate_range(self) -> None:
        summary = AuditReportSummary(
            total_turns=10,
            overall_score=0.5,
            overall_risk_level=RiskLevel.CAUTION,
            compliance_rate=75.0,
        )
        assert summary.compliance_rate == 75.0

    def test_compliance_rate_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            AuditReportSummary(
                total_turns=1,
                overall_score=0.0,
                overall_risk_level=RiskLevel.SAFE,
                compliance_rate=101.0,
            )
        with pytest.raises(ValidationError):
            AuditReportSummary(
                total_turns=1,
                overall_score=0.0,
                overall_risk_level=RiskLevel.SAFE,
                compliance_rate=-1.0,
            )

    def test_with_flagged_categories(self) -> None:
        summary = AuditReportSummary(
            total_turns=5,
            overall_score=0.4,
            overall_risk_level=RiskLevel.CAUTION,
            most_flagged_categories=[PolicyCategory.VIOLENCE, PolicyCategory.HATE_SPEECH],
        )
        assert PolicyCategory.VIOLENCE.value in summary.most_flagged_categories


# ---------------------------------------------------------------------------
# FlaggedTurnSummary
# ---------------------------------------------------------------------------


class TestFlaggedTurnSummary:
    """Tests for FlaggedTurnSummary model."""

    def test_basic_construction(self) -> None:
        summary = FlaggedTurnSummary(
            turn_index=2,
            role="assistant",
            content_snippet="Flagged content here",
            overall_score=0.8,
            overall_risk_level=RiskLevel.VIOLATION,
        )
        assert summary.turn_index == 2
        assert summary.role == "assistant"
        assert summary.overall_score == 0.8

    def test_defaults(self) -> None:
        summary = FlaggedTurnSummary(
            turn_index=0,
            role="user",
            content_snippet="Something",
            overall_score=0.4,
            overall_risk_level=RiskLevel.CAUTION,
        )
        assert summary.flagged_categories == []
        assert summary.category_scores == []
        assert summary.reasoning == ""
        assert summary.remediation_hints == {}

    def test_negative_turn_index_raises(self) -> None:
        with pytest.raises(ValidationError):
            FlaggedTurnSummary(
                turn_index=-1,
                role="user",
                content_snippet="X",
                overall_score=0.5,
                overall_risk_level=RiskLevel.CAUTION,
            )

    def test_with_remediation_hints(self) -> None:
        hints = {
            "violence": "Do not include violent content.",
            "hate_speech": "Avoid discriminatory language.",
        }
        summary = FlaggedTurnSummary(
            turn_index=0,
            role="assistant",
            content_snippet="Bad content",
            overall_score=0.9,
            overall_risk_level=RiskLevel.VIOLATION,
            remediation_hints=hints,
        )
        assert "violence" in summary.remediation_hints
        assert "hate_speech" in summary.remediation_hints


# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------


class TestAuditReport:
    """Tests for AuditReport model."""

    def _make_report(self, **kwargs: Any) -> AuditReport:
        summary = _make_audit_report_summary()
        return AuditReport(
            report_id="test-report-001",
            summary=summary,
            **kwargs,
        )

    def test_basic_construction(self) -> None:
        report = self._make_report()
        assert report.report_id == "test-report-001"
        assert report.schema_version == "1.0"

    def test_generated_at_is_datetime(self) -> None:
        report = self._make_report()
        assert isinstance(report.generated_at, datetime)

    def test_generated_at_is_utc(self) -> None:
        report = self._make_report()
        assert report.generated_at.tzinfo is not None

    def test_defaults(self) -> None:
        report = self._make_report()
        assert report.flagged_turns == []
        assert report.all_turn_results == []
        assert "openai.com" in report.policy_reference
        assert report.tool_version == "0.1.0"
        assert report.metadata == {}

    def test_with_flagged_turns(self) -> None:
        flagged = [
            FlaggedTurnSummary(
                turn_index=1,
                role="assistant",
                content_snippet="Problematic content",
                overall_score=0.9,
                overall_risk_level=RiskLevel.VIOLATION,
            )
        ]
        report = self._make_report(flagged_turns=flagged)
        assert len(report.flagged_turns) == 1

    def test_with_metadata(self) -> None:
        meta = {"app_name": "My Teen App", "tester": "dev@example.com"}
        report = self._make_report(metadata=meta)
        assert report.metadata["app_name"] == "My Teen App"

    def test_model_dump_serialisable(self) -> None:
        report = self._make_report()
        data = report.model_dump()
        assert isinstance(data, dict)
        assert "report_id" in data
        assert "summary" in data
        assert "generated_at" in data
        assert "schema_version" in data

    def test_with_all_turn_results(self) -> None:
        turns = [_make_turn_result(i) for i in range(5)]
        report = self._make_report(all_turn_results=turns)
        assert len(report.all_turn_results) == 5

    def test_policy_reference_is_url(self) -> None:
        report = self._make_report()
        assert report.policy_reference.startswith("http")


# ---------------------------------------------------------------------------
# AuditErrorResponse
# ---------------------------------------------------------------------------


class TestAuditErrorResponse:
    """Tests for AuditErrorResponse model."""

    def test_basic_construction(self) -> None:
        err = AuditErrorResponse(
            error="openai_error",
            message="The OpenAI API returned an error.",
        )
        assert err.error == "openai_error"
        assert err.message == "The OpenAI API returned an error."
        assert err.details is None

    def test_with_details(self) -> None:
        err = AuditErrorResponse(
            error="rate_limit",
            message="Rate limit exceeded.",
            details="Retry after 60 seconds.",
        )
        assert err.details == "Retry after 60 seconds."

    def test_missing_error_raises(self) -> None:
        with pytest.raises(ValidationError):
            AuditErrorResponse(message="Something went wrong.")  # type: ignore[call-arg]

    def test_missing_message_raises(self) -> None:
        with pytest.raises(ValidationError):
            AuditErrorResponse(error="some_error")  # type: ignore[call-arg]

    def test_model_dump(self) -> None:
        err = AuditErrorResponse(error="invalid_input", message="Bad request.")
        data = err.model_dump()
        assert data["error"] == "invalid_input"
        assert data["message"] == "Bad request."
        assert data["details"] is None

    def test_error_stripped(self) -> None:
        err = AuditErrorResponse(error="  openai_error  ", message="Error occurred.")
        assert err.error == "openai_error"


# ---------------------------------------------------------------------------
# Cross-model integration
# ---------------------------------------------------------------------------


class TestCrossModelIntegration:
    """Verify that models compose correctly for realistic audit scenarios."""

    def test_full_conversation_audit_structure(self) -> None:
        """Build a realistic ConversationAuditResult with populated sub-objects."""
        category_scores = [
            CategoryScore(
                category=cat,
                score=0.1,
                risk_level=RiskLevel.SAFE,
            )
            for cat in PolicyCategory
        ]
        turn_result = TurnAuditResult(
            turn_index=0,
            role="user",
            content_snippet="Hello, can you help me?",
            category_scores=category_scores,
            overall_score=0.05,
            overall_risk_level=RiskLevel.SAFE,
            reasoning="Content is benign and safe for teen audiences.",
        )
        conv_result = ConversationAuditResult(
            turn_results=[turn_result],
            total_turns=1,
            flagged_turns=0,
            overall_score=0.05,
            overall_risk_level=RiskLevel.SAFE,
        )
        assert len(conv_result.turn_results) == 1
        assert conv_result.turn_results[0].overall_risk_level == RiskLevel.SAFE.value
        assert len(conv_result.turn_results[0].category_scores) == len(list(PolicyCategory))

    def test_full_report_structure(self) -> None:
        """Build a realistic AuditReport from a flagged conversation."""
        category_scores = [
            CategoryScore(
                category=PolicyCategory.VIOLENCE,
                score=0.85,
                risk_level=RiskLevel.VIOLATION,
            ),
            CategoryScore(
                category=PolicyCategory.HATE_SPEECH,
                score=0.4,
                risk_level=RiskLevel.CAUTION,
            ),
        ]
        turn_result = TurnAuditResult(
            turn_index=0,
            role="assistant",
            content_snippet="Some violent content…",
            category_scores=category_scores,
            overall_score=0.85,
            overall_risk_level=RiskLevel.VIOLATION,
            flagged_categories=[PolicyCategory.VIOLENCE, PolicyCategory.HATE_SPEECH],
            reasoning="Response contains graphic violence.",
        )
        flagged_summary = FlaggedTurnSummary(
            turn_index=0,
            role="assistant",
            content_snippet="Some violent content…",
            overall_score=0.85,
            overall_risk_level=RiskLevel.VIOLATION,
            flagged_categories=[PolicyCategory.VIOLENCE],
            remediation_hints={
                "violence": "Do not include graphic violence for teen audiences."
            },
        )
        summary = AuditReportSummary(
            total_turns=1,
            flagged_turns=1,
            safe_turns=0,
            violation_turns=1,
            caution_turns=0,
            overall_score=0.85,
            overall_risk_level=RiskLevel.VIOLATION,
            most_flagged_categories=[PolicyCategory.VIOLENCE],
            compliance_rate=0.0,
        )
        report = AuditReport(
            report_id="integration-test-001",
            summary=summary,
            flagged_turns=[flagged_summary],
            all_turn_results=[turn_result],
            metadata={"test": "integration"},
        )
        assert report.summary.violation_turns == 1
        assert len(report.flagged_turns) == 1
        assert len(report.all_turn_results) == 1
        assert report.metadata["test"] == "integration"

    def test_json_round_trip_audit_result(self) -> None:
        """Verify AuditResult survives a model_dump → model_validate round trip."""
        original = AuditResult(
            prompt_snippet="Test prompt",
            response_snippet="Test response",
            category_scores=[_make_category_score()],
            overall_score=0.15,
            overall_risk_level=RiskLevel.SAFE,
            flagged_categories=[],
            reasoning="Content appears safe.",
        )
        data = original.model_dump(mode="json")
        restored = AuditResult.model_validate(data)
        assert restored.prompt_snippet == original.prompt_snippet
        assert restored.overall_score == original.overall_score
        assert len(restored.category_scores) == len(original.category_scores)
