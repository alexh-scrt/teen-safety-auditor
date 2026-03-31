"""tests/test_auditor.py — Unit tests for teen_safety_auditor/auditor.py.

Covers:
- AuditEngine initialisation (valid key, missing key)
- _strip_markdown_fences helper
- _coerce_score helper
- _fallback_response helper
- _build_category_scores helper
- _resolve_overall_score helper
- _collect_flagged_categories helper
- _mean_score helper
- _most_flagged_categories helper
- _build_remediation_hints helper
- _build_report_from_result helper
- audit_prompt with mocked OpenAI responses (safe, caution, violation, parse failure)
- audit_conversation with mocked OpenAI responses
- build_report / build_report_from_result
- OpenAI error propagation (RateLimitError, APITimeoutError, APIError)
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from teen_safety_auditor.auditor import (
    AuditEngine,
    AuditEngineError,
    OpenAICallError,
    ResponseParseError,
    create_engine,
)
from teen_safety_auditor.models import (
    AuditReport,
    AuditResult,
    CategoryScore,
    ConversationAuditResult,
    ConversationRequest,
    ConversationTurn,
    SinglePromptRequest,
    TurnAuditResult,
)
from teen_safety_auditor.policy import PolicyCategory, RiskLevel


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


FAKE_API_KEY = "sk-test-fake-key-for-unit-tests"


def _safe_scores_payload() -> dict[str, Any]:
    """Return a JSON-serialisable safeguard payload with all-safe scores."""
    scores = {cat.value: 0.05 for cat in PolicyCategory}
    return {
        "scores": scores,
        "overall_score": 0.05,
        "flagged_categories": [],
        "reasoning": "Content appears safe for teen audiences.",
    }


def _violation_scores_payload() -> dict[str, Any]:
    """Return a payload with violence and hate_speech in violation range."""
    scores = {cat.value: 0.05 for cat in PolicyCategory}
    scores[PolicyCategory.VIOLENCE.value] = 0.9
    scores[PolicyCategory.HATE_SPEECH.value] = 0.75
    return {
        "scores": scores,
        "overall_score": 0.9,
        "flagged_categories": ["violence", "hate_speech"],
        "reasoning": "Response contains graphic violence and hate speech.",
    }


def _caution_scores_payload() -> dict[str, Any]:
    """Return a payload with substance_abuse in caution range."""
    scores = {cat.value: 0.05 for cat in PolicyCategory}
    scores[PolicyCategory.SUBSTANCE_ABUSE.value] = 0.50
    return {
        "scores": scores,
        "overall_score": 0.50,
        "flagged_categories": ["substance_abuse"],
        "reasoning": "Response mentions drug use in an ambiguous context.",
    }


def _make_mock_completion(content: str) -> MagicMock:
    """Create a mock OpenAI ChatCompletion object with the given content."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


def _make_engine_with_mock_client(mock_completion_content: str) -> tuple[AuditEngine, AsyncMock]:
    """Create an AuditEngine with a mocked AsyncOpenAI client.

    Returns the engine and the mock create coroutine for assertion.
    """
    engine = AuditEngine(api_key=FAKE_API_KEY)
    mock_create = AsyncMock(
        return_value=_make_mock_completion(mock_completion_content)
    )
    engine._client.chat.completions.create = mock_create  # type: ignore[assignment]
    return engine, mock_create


# ---------------------------------------------------------------------------
# AuditEngine initialisation
# ---------------------------------------------------------------------------


class TestAuditEngineInit:
    """Tests for AuditEngine.__init__."""

    def test_valid_api_key_constructs(self) -> None:
        engine = AuditEngine(api_key=FAKE_API_KEY)
        assert engine is not None

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(AuditEngineError, match="API key"):
            AuditEngine(api_key="")

    def test_default_model_is_gpt4o_mini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SAFEGUARD_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        engine = AuditEngine(api_key=FAKE_API_KEY)
        assert engine._model == "gpt-4o-mini"

    def test_model_overridden_by_argument(self) -> None:
        engine = AuditEngine(api_key=FAKE_API_KEY, model="gpt-4-turbo")
        assert engine._model == "gpt-4-turbo"

    def test_model_overridden_by_safeguard_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SAFEGUARD_MODEL", "gpt-4o")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        engine = AuditEngine(api_key=FAKE_API_KEY)
        assert engine._model == "gpt-4o"

    def test_model_overridden_by_openai_model_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SAFEGUARD_MODEL", raising=False)
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        engine = AuditEngine(api_key=FAKE_API_KEY)
        assert engine._model == "gpt-4"

    def test_system_prompt_is_set(self) -> None:
        engine = AuditEngine(api_key=FAKE_API_KEY)
        assert isinstance(engine._system_prompt, str)
        assert len(engine._system_prompt) > 0

    def test_create_engine_factory(self) -> None:
        engine = create_engine(api_key=FAKE_API_KEY)
        assert isinstance(engine, AuditEngine)

    def test_create_engine_missing_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(AuditEngineError):
            create_engine(api_key="")


# ---------------------------------------------------------------------------
# Static helper methods
# ---------------------------------------------------------------------------


class TestStripMarkdownFences:
    """Tests for AuditEngine._strip_markdown_fences."""

    def test_no_fences_unchanged(self) -> None:
        text = '{"key": "value"}'
        result = AuditEngine._strip_markdown_fences(text)
        assert result == text

    def test_json_fence_stripped(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        result = AuditEngine._strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_plain_fence_stripped(self) -> None:
        text = '```\n{"key": "value"}\n```'
        result = AuditEngine._strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_multi_line_json_in_fence(self) -> None:
        text = '```json\n{\n  "key": "value"\n}\n```'
        result = AuditEngine._strip_markdown_fences(text)
        assert '"key": "value"' in result

    def test_fence_without_closing_stripped(self) -> None:
        text = '```json\n{"key": 1}'
        result = AuditEngine._strip_markdown_fences(text)
        # Should strip opening fence even without closing
        assert result.strip().startswith("{")

    def test_empty_string_unchanged(self) -> None:
        result = AuditEngine._strip_markdown_fences("")
        assert result == ""


class TestCoerceScore:
    """Tests for AuditEngine._coerce_score."""

    def test_float_passthrough(self) -> None:
        assert AuditEngine._coerce_score(0.75) == 0.75

    def test_int_converted(self) -> None:
        assert AuditEngine._coerce_score(1) == 1.0

    def test_string_float_converted(self) -> None:
        assert AuditEngine._coerce_score("0.5") == 0.5

    def test_none_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        assert AuditEngine._coerce_score(None) == _FALLBACK_SCORE

    def test_negative_clamped_to_zero(self) -> None:
        assert AuditEngine._coerce_score(-0.5) == 0.0

    def test_above_one_clamped_to_one(self) -> None:
        assert AuditEngine._coerce_score(1.5) == 1.0

    def test_invalid_string_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        assert AuditEngine._coerce_score("not_a_number") == _FALLBACK_SCORE

    def test_zero_passthrough(self) -> None:
        assert AuditEngine._coerce_score(0.0) == 0.0

    def test_one_passthrough(self) -> None:
        assert AuditEngine._coerce_score(1.0) == 1.0


class TestFallbackResponse:
    """Tests for AuditEngine._fallback_response."""

    def test_returns_dict(self) -> None:
        result = AuditEngine._fallback_response()
        assert isinstance(result, dict)

    def test_has_scores_key(self) -> None:
        result = AuditEngine._fallback_response()
        assert "scores" in result

    def test_scores_all_categories_present(self) -> None:
        result = AuditEngine._fallback_response()
        for cat in PolicyCategory:
            assert cat.value in result["scores"]

    def test_scores_are_fallback_value(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        result = AuditEngine._fallback_response()
        for score in result["scores"].values():
            assert score == _FALLBACK_SCORE

    def test_has_overall_score(self) -> None:
        result = AuditEngine._fallback_response()
        assert "overall_score" in result

    def test_has_flagged_categories(self) -> None:
        result = AuditEngine._fallback_response()
        assert "flagged_categories" in result
        assert isinstance(result["flagged_categories"], list)

    def test_reasoning_contains_reason(self) -> None:
        result = AuditEngine._fallback_response(reason="JSON error")
        assert "JSON error" in result["reasoning"]

    def test_empty_reason(self) -> None:
        result = AuditEngine._fallback_response(reason="")
        assert isinstance(result["reasoning"], str)


class TestBuildCategoryScores:
    """Tests for AuditEngine._build_category_scores."""

    def test_returns_list_of_category_scores(self) -> None:
        scores = {cat.value: 0.1 for cat in PolicyCategory}
        result = AuditEngine._build_category_scores(scores)
        assert isinstance(result, list)
        assert all(isinstance(cs, CategoryScore) for cs in result)

    def test_all_categories_represented(self) -> None:
        scores = {cat.value: 0.1 for cat in PolicyCategory}
        result = AuditEngine._build_category_scores(scores)
        result_cats = {cs.category for cs in result}
        expected_cats = {cat.value for cat in PolicyCategory}
        assert result_cats == expected_cats

    def test_score_values_preserved(self) -> None:
        scores = {cat.value: 0.8 for cat in PolicyCategory}
        result = AuditEngine._build_category_scores(scores)
        for cs in result:
            assert cs.score == 0.8

    def test_missing_category_gets_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        result = AuditEngine._build_category_scores({})
        for cs in result:
            assert cs.score == _FALLBACK_SCORE

    def test_risk_classification_applied(self) -> None:
        scores = {cat.value: 0.9 for cat in PolicyCategory}
        result = AuditEngine._build_category_scores(scores)
        for cs in result:
            assert cs.risk_level == RiskLevel.VIOLATION.value

    def test_safe_scores_classified_safe(self) -> None:
        scores = {cat.value: 0.0 for cat in PolicyCategory}
        result = AuditEngine._build_category_scores(scores)
        for cs in result:
            assert cs.risk_level == RiskLevel.SAFE.value

    def test_category_count_matches_enum(self) -> None:
        scores = {cat.value: 0.1 for cat in PolicyCategory}
        result = AuditEngine._build_category_scores(scores)
        assert len(result) == len(list(PolicyCategory))


class TestResolveOverallScore:
    """Tests for AuditEngine._resolve_overall_score."""

    def _make_scores(self, score: float) -> list[CategoryScore]:
        return [
            CategoryScore(
                category=PolicyCategory.VIOLENCE,
                score=score,
                risk_level=RiskLevel.SAFE,
            )
        ]

    def test_uses_model_score_when_valid(self) -> None:
        scores = self._make_scores(0.9)
        result = AuditEngine._resolve_overall_score(0.3, scores)
        assert result == 0.3

    def test_falls_back_to_max_when_none(self) -> None:
        scores = self._make_scores(0.7)
        result = AuditEngine._resolve_overall_score(None, scores)
        assert result == 0.7

    def test_falls_back_to_max_when_out_of_range_high(self) -> None:
        scores = self._make_scores(0.6)
        result = AuditEngine._resolve_overall_score(1.5, scores)
        assert result == 0.6

    def test_falls_back_to_max_when_out_of_range_low(self) -> None:
        scores = self._make_scores(0.4)
        result = AuditEngine._resolve_overall_score(-0.1, scores)
        assert result == 0.4

    def test_empty_scores_and_none_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        result = AuditEngine._resolve_overall_score(None, [])
        assert result == _FALLBACK_SCORE

    def test_zero_model_score_accepted(self) -> None:
        scores = self._make_scores(0.5)
        result = AuditEngine._resolve_overall_score(0.0, scores)
        assert result == 0.0

    def test_one_model_score_accepted(self) -> None:
        scores = self._make_scores(0.1)
        result = AuditEngine._resolve_overall_score(1.0, scores)
        assert result == 1.0


class TestCollectFlaggedCategories:
    """Tests for AuditEngine._collect_flagged_categories."""

    def _make_category_scores(
        self, overrides: dict[PolicyCategory, tuple[float, RiskLevel]]
    ) -> list[CategoryScore]:
        result = []
        for cat in PolicyCategory:
            if cat in overrides:
                score, risk = overrides[cat]
            else:
                score, risk = 0.05, RiskLevel.SAFE
            result.append(CategoryScore(category=cat, score=score, risk_level=risk))
        return result

    def test_all_safe_returns_empty(self) -> None:
        scores = self._make_category_scores({})
        flagged = AuditEngine._collect_flagged_categories(scores)
        assert flagged == []

    def test_violation_category_flagged(self) -> None:
        scores = self._make_category_scores(
            {PolicyCategory.VIOLENCE: (0.9, RiskLevel.VIOLATION)}
        )
        flagged = AuditEngine._collect_flagged_categories(scores)
        assert PolicyCategory.VIOLENCE in flagged

    def test_caution_category_flagged(self) -> None:
        scores = self._make_category_scores(
            {PolicyCategory.SUBSTANCE_ABUSE: (0.5, RiskLevel.CAUTION)}
        )
        flagged = AuditEngine._collect_flagged_categories(scores)
        assert PolicyCategory.SUBSTANCE_ABUSE in flagged

    def test_multiple_flagged(self) -> None:
        scores = self._make_category_scores(
            {
                PolicyCategory.VIOLENCE: (0.9, RiskLevel.VIOLATION),
                PolicyCategory.HATE_SPEECH: (0.5, RiskLevel.CAUTION),
            }
        )
        flagged = AuditEngine._collect_flagged_categories(scores)
        assert len(flagged) == 2
        assert PolicyCategory.VIOLENCE in flagged
        assert PolicyCategory.HATE_SPEECH in flagged

    def test_returns_list(self) -> None:
        scores = self._make_category_scores({})
        result = AuditEngine._collect_flagged_categories(scores)
        assert isinstance(result, list)


class TestMeanScore:
    """Tests for AuditEngine._mean_score."""

    def _make_turn(self, score: float) -> TurnAuditResult:
        return TurnAuditResult(
            turn_index=0,
            role="user",
            content_snippet="X",
            overall_score=score,
            overall_risk_level=RiskLevel.SAFE,
        )

    def test_empty_returns_zero(self) -> None:
        assert AuditEngine._mean_score([]) == 0.0

    def test_single_turn(self) -> None:
        turns = [self._make_turn(0.6)]
        assert AuditEngine._mean_score(turns) == pytest.approx(0.6)

    def test_multiple_turns(self) -> None:
        turns = [self._make_turn(0.2), self._make_turn(0.4), self._make_turn(0.6)]
        assert AuditEngine._mean_score(turns) == pytest.approx(0.4)

    def test_all_zeros(self) -> None:
        turns = [self._make_turn(0.0) for _ in range(5)]
        assert AuditEngine._mean_score(turns) == 0.0

    def test_all_ones(self) -> None:
        turns = [self._make_turn(1.0) for _ in range(3)]
        assert AuditEngine._mean_score(turns) == pytest.approx(1.0)


class TestMostFlaggedCategories:
    """Tests for AuditEngine._most_flagged_categories."""

    def _make_turn_with_flags(
        self, turn_index: int, flagged: list[PolicyCategory]
    ) -> TurnAuditResult:
        return TurnAuditResult(
            turn_index=turn_index,
            role="user",
            content_snippet="X",
            overall_score=0.8 if flagged else 0.1,
            overall_risk_level=RiskLevel.VIOLATION if flagged else RiskLevel.SAFE,
            flagged_categories=flagged,
        )

    def test_empty_turns_returns_empty(self) -> None:
        result = AuditEngine._most_flagged_categories([])
        assert result == []

    def test_no_flagged_categories_returns_empty(self) -> None:
        turns = [self._make_turn_with_flags(i, []) for i in range(3)]
        result = AuditEngine._most_flagged_categories(turns)
        assert result == []

    def test_single_flagged_category(self) -> None:
        turns = [self._make_turn_with_flags(0, [PolicyCategory.VIOLENCE])]
        result = AuditEngine._most_flagged_categories(turns)
        assert PolicyCategory.VIOLENCE in result

    def test_most_frequent_first(self) -> None:
        turns = [
            self._make_turn_with_flags(0, [PolicyCategory.VIOLENCE, PolicyCategory.HATE_SPEECH]),
            self._make_turn_with_flags(1, [PolicyCategory.VIOLENCE]),
            self._make_turn_with_flags(2, [PolicyCategory.HATE_SPEECH]),
            self._make_turn_with_flags(3, [PolicyCategory.VIOLENCE]),
        ]
        result = AuditEngine._most_flagged_categories(turns)
        assert result[0] == PolicyCategory.VIOLENCE

    def test_top_n_respected(self) -> None:
        turns = [
            self._make_turn_with_flags(
                i,
                [cat for cat in PolicyCategory],
            )
            for i in range(3)
        ]
        result = AuditEngine._most_flagged_categories(turns, top_n=3)
        assert len(result) <= 3

    def test_returns_policy_category_instances(self) -> None:
        turns = [self._make_turn_with_flags(0, [PolicyCategory.SELF_HARM])]
        result = AuditEngine._most_flagged_categories(turns)
        for cat in result:
            assert isinstance(cat, PolicyCategory)


class TestBuildRemediationHints:
    """Tests for AuditEngine._build_remediation_hints."""

    def test_empty_list_returns_empty_dict(self) -> None:
        result = AuditEngine._build_remediation_hints([])
        assert result == {}

    def test_single_category(self) -> None:
        result = AuditEngine._build_remediation_hints([PolicyCategory.VIOLENCE])
        assert "violence" in result
        assert isinstance(result["violence"], str)
        assert len(result["violence"]) > 0

    def test_multiple_categories(self) -> None:
        cats = [PolicyCategory.VIOLENCE, PolicyCategory.HATE_SPEECH, PolicyCategory.SELF_HARM]
        result = AuditEngine._build_remediation_hints(cats)
        assert len(result) == 3
        for cat in cats:
            assert cat.value in result

    def test_all_categories_have_hints(self) -> None:
        all_cats = list(PolicyCategory)
        result = AuditEngine._build_remediation_hints(all_cats)
        assert len(result) == len(all_cats)
        for hint in result.values():
            assert isinstance(hint, str)
            assert len(hint) > 0


# ---------------------------------------------------------------------------
# _parse_safeguard_response
# ---------------------------------------------------------------------------


class TestParseSafeguardResponse:
    """Tests for AuditEngine._parse_safeguard_response."""

    def setup_method(self) -> None:
        self.engine = AuditEngine(api_key=FAKE_API_KEY)

    def test_valid_json_parsed(self) -> None:
        payload = _safe_scores_payload()
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert "scores" in result
        assert "overall_score" in result
        assert "reasoning" in result

    def test_all_categories_in_scores(self) -> None:
        payload = _safe_scores_payload()
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        for cat in PolicyCategory:
            assert cat.value in result["scores"]

    def test_markdown_fenced_json_parsed(self) -> None:
        payload = _safe_scores_payload()
        raw = f"```json\n{json.dumps(payload)}\n```"
        result = self.engine._parse_safeguard_response(raw)
        assert "scores" in result

    def test_invalid_json_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        result = self.engine._parse_safeguard_response("not valid json")
        assert result["overall_score"] == _FALLBACK_SCORE
        for score in result["scores"].values():
            assert score == _FALLBACK_SCORE

    def test_non_object_json_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        result = self.engine._parse_safeguard_response("[1, 2, 3]")
        assert result["overall_score"] == _FALLBACK_SCORE

    def test_missing_scores_field_gets_fallback_scores(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        raw = json.dumps({"overall_score": 0.1, "flagged_categories": [], "reasoning": ""})
        result = self.engine._parse_safeguard_response(raw)
        for score in result["scores"].values():
            assert score == _FALLBACK_SCORE

    def test_overall_score_coerced(self) -> None:
        payload = _safe_scores_payload()
        payload["overall_score"] = "0.7"  # string instead of float
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert result["overall_score"] == pytest.approx(0.7)

    def test_scores_clamped_to_unit_interval(self) -> None:
        payload = _safe_scores_payload()
        payload["scores"][PolicyCategory.VIOLENCE.value] = 2.5  # out of range
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert result["scores"][PolicyCategory.VIOLENCE.value] == 1.0

    def test_reasoning_extracted(self) -> None:
        payload = _safe_scores_payload()
        payload["reasoning"] = "This is the reasoning."
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert result["reasoning"] == "This is the reasoning."

    def test_empty_string_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        result = self.engine._parse_safeguard_response("")
        assert result["overall_score"] == _FALLBACK_SCORE


# ---------------------------------------------------------------------------
# audit_prompt (async, with mocked OpenAI)
# ---------------------------------------------------------------------------


class TestAuditPrompt:
    """Integration tests for AuditEngine.audit_prompt with mocked OpenAI."""

    @pytest.mark.asyncio
    async def test_safe_content_returns_safe_result(self) -> None:
        payload = _safe_scores_payload()
        engine, mock_create = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Hello, how do I stay safe online?")
        result = await engine.audit_prompt(request)

        assert isinstance(result, AuditResult)
        assert result.overall_risk_level == RiskLevel.SAFE.value
        assert result.overall_score == pytest.approx(0.05)
        assert result.flagged_categories == []
        mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_violation_content_returns_violation_result(self) -> None:
        payload = _violation_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Describe graphic violence.")
        result = await engine.audit_prompt(request)

        assert result.overall_risk_level == RiskLevel.VIOLATION.value
        assert len(result.flagged_categories) > 0

    @pytest.mark.asyncio
    async def test_caution_content_returns_caution_result(self) -> None:
        payload = _caution_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Tell me about drug use.")
        result = await engine.audit_prompt(request)

        assert result.overall_risk_level in (
            RiskLevel.CAUTION.value,
            RiskLevel.VIOLATION.value,
        )

    @pytest.mark.asyncio
    async def test_result_has_all_category_scores(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Hi there!")
        result = await engine.audit_prompt(request)

        assert len(result.category_scores) == len(list(PolicyCategory))

    @pytest.mark.asyncio
    async def test_result_has_prompt_snippet(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Short prompt")
        result = await engine.audit_prompt(request)

        assert "Short prompt" in result.prompt_snippet

    @pytest.mark.asyncio
    async def test_result_with_response_has_response_snippet(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Q", response="The answer is 42.")
        result = await engine.audit_prompt(request)

        assert result.response_snippet is not None
        assert "42" in result.response_snippet

    @pytest.mark.asyncio
    async def test_result_without_response_has_none_snippet(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Solo prompt")
        result = await engine.audit_prompt(request)

        assert result.response_snippet is None

    @pytest.mark.asyncio
    async def test_audit_timestamp_is_set(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Test")
        result = await engine.audit_prompt(request)

        assert isinstance(result.audit_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_reasoning_propagated(self) -> None:
        payload = _safe_scores_payload()
        payload["reasoning"] = "Content is benign."
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Hi")
        result = await engine.audit_prompt(request)

        assert "benign" in result.reasoning

    @pytest.mark.asyncio
    async def test_parse_failure_returns_caution_fallback(self) -> None:
        """A malformed response should produce a conservative caution result."""
        engine, _ = _make_engine_with_mock_client("this is not valid json at all!!!")

        request = SinglePromptRequest(prompt="Test prompt")
        result = await engine.audit_prompt(request)

        # Fallback score is 0.5 → CAUTION
        assert result.overall_risk_level in (
            RiskLevel.CAUTION.value,
            RiskLevel.VIOLATION.value,
        )

    @pytest.mark.asyncio
    async def test_long_prompt_snippet_truncated(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        long_prompt = "A" * 500
        request = SinglePromptRequest(prompt=long_prompt)
        result = await engine.audit_prompt(request)

        assert len(result.prompt_snippet) <= 204  # 200 + possible ellipsis


# ---------------------------------------------------------------------------
# OpenAI error propagation
# ---------------------------------------------------------------------------


class TestOpenAIErrorPropagation:
    """Verify that OpenAI SDK exceptions are wrapped in OpenAICallError."""

    @pytest.mark.asyncio
    async def test_rate_limit_raises_openai_call_error(self) -> None:
        from openai import RateLimitError

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError(
                "Rate limit",
                response=MagicMock(status_code=429, headers={}),
                body={},
            )
        )

        with pytest.raises(OpenAICallError, match="rate limit"):
            await engine.audit_prompt(SinglePromptRequest(prompt="Test"))

    @pytest.mark.asyncio
    async def test_timeout_raises_openai_call_error(self) -> None:
        from openai import APITimeoutError

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )

        with pytest.raises(OpenAICallError, match="timed out"):
            await engine.audit_prompt(SinglePromptRequest(prompt="Test"))

    @pytest.mark.asyncio
    async def test_api_error_raises_openai_call_error(self) -> None:
        from openai import APIError

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=APIError(
                "API error",
                request=MagicMock(),
                body={},
            )
        )

        with pytest.raises(OpenAICallError):
            await engine.audit_prompt(SinglePromptRequest(prompt="Test"))

    @pytest.mark.asyncio
    async def test_unexpected_error_raises_openai_call_error(self) -> None:
        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("Network unreachable")
        )

        with pytest.raises(OpenAICallError):
            await engine.audit_prompt(SinglePromptRequest(prompt="Test"))


# ---------------------------------------------------------------------------
# audit_conversation
# ---------------------------------------------------------------------------


class TestAuditConversation:
    """Tests for AuditEngine.audit_conversation."""

    @pytest.mark.asyncio
    async def test_single_user_turn(self) -> None:
        payload = _safe_scores_payload()
        engine, mock_create = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hello!")]
        )
        result = await engine.audit_conversation(request)

        assert isinstance(result, ConversationAuditResult)
        assert result.total_turns == 1
        mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_user_assistant_pair_single_call(self) -> None:
        """A user+assistant adjacent pair should be evaluated in a single API call."""
        payload = _safe_scores_payload()
        engine, mock_create = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[
                ConversationTurn(role="user", content="Hi"),
                ConversationTurn(role="assistant", content="Hello!"),
            ]
        )
        result = await engine.audit_conversation(request)

        # One user+assistant pair = 1 API call, 1 turn result
        assert mock_create.await_count == 1
        assert result.total_turns == 1

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self) -> None:
        payload = _safe_scores_payload()
        engine, mock_create = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[
                ConversationTurn(role="user", content="Turn 1"),
                ConversationTurn(role="assistant", content="Response 1"),
                ConversationTurn(role="user", content="Turn 2"),
                ConversationTurn(role="assistant", content="Response 2"),
            ]
        )
        result = await engine.audit_conversation(request)

        # Two pairs → 2 API calls, 2 turn results
        assert mock_create.await_count == 2
        assert result.total_turns == 2

    @pytest.mark.asyncio
    async def test_system_turn_skipped(self) -> None:
        payload = _safe_scores_payload()
        engine, mock_create = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[
                ConversationTurn(role="system", content="You are helpful."),
                ConversationTurn(role="user", content="Hi!"),
            ]
        )
        result = await engine.audit_conversation(request)

        # System turn skipped; only user turn evaluated
        assert mock_create.await_count == 1
        assert result.total_turns == 1

    @pytest.mark.asyncio
    async def test_overall_score_is_mean(self) -> None:
        payload = _safe_scores_payload()
        payload["overall_score"] = 0.1
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[
                ConversationTurn(role="user", content="First"),
                ConversationTurn(role="user", content="Second"),
            ]
        )
        result = await engine.audit_conversation(request)

        assert result.overall_score == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_flagged_turns_counted(self) -> None:
        # Return violation for first call, safe for second
        violation_payload = _violation_scores_payload()
        safe_payload = _safe_scores_payload()

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=[
                _make_mock_completion(json.dumps(violation_payload)),
                _make_mock_completion(json.dumps(safe_payload)),
            ]
        )

        request = ConversationRequest(
            turns=[
                ConversationTurn(role="user", content="Violent content request"),
                ConversationTurn(role="user", content="Benign question"),
            ]
        )
        result = await engine.audit_conversation(request)

        assert result.flagged_turns == 1

    @pytest.mark.asyncio
    async def test_audit_timestamp_is_set(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hello")]
        )
        result = await engine.audit_conversation(request)

        assert isinstance(result.audit_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_turn_results_have_correct_indices(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[
                ConversationTurn(role="user", content="A"),
                ConversationTurn(role="user", content="B"),
            ]
        )
        result = await engine.audit_conversation(request)

        indices = [t.turn_index for t in result.turn_results]
        assert 0 in indices

    @pytest.mark.asyncio
    async def test_most_flagged_categories_populated(self) -> None:
        payload = _violation_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Bad content")]
        )
        result = await engine.audit_conversation(request)

        # Since violence is flagged, it should appear in most_flagged_categories
        cat_values = [c if isinstance(c, str) else c.value for c in result.most_flagged_categories]
        assert "violence" in cat_values or len(cat_values) > 0


# ---------------------------------------------------------------------------
# build_report and build_report_from_result
# ---------------------------------------------------------------------------


class TestBuildReport:
    """Tests for AuditEngine.build_report and build_report_from_result."""

    @pytest.mark.asyncio
    async def test_build_report_returns_audit_report(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hello")]
        )
        report = await engine.build_report(request)

        assert isinstance(report, AuditReport)

    @pytest.mark.asyncio
    async def test_report_has_unique_id(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hello")]
        )
        report1 = await engine.build_report(request)

        engine2, _ = _make_engine_with_mock_client(json.dumps(payload))
        report2 = await engine2.build_report(request)

        assert report1.report_id != report2.report_id

    @pytest.mark.asyncio
    async def test_report_summary_totals_correct(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Safe content")]
        )
        report = await engine.build_report(request)

        assert report.summary.total_turns == 1
        assert report.summary.safe_turns == 1
        assert report.summary.violation_turns == 0

    @pytest.mark.asyncio
    async def test_report_violation_counted_in_summary(self) -> None:
        payload = _violation_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Violent content")]
        )
        report = await engine.build_report(request)

        assert report.summary.violation_turns == 1
        assert report.summary.safe_turns == 0

    @pytest.mark.asyncio
    async def test_report_has_flagged_turns(self) -> None:
        payload = _violation_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Bad content")]
        )
        report = await engine.build_report(request)

        assert len(report.flagged_turns) > 0

    @pytest.mark.asyncio
    async def test_report_safe_content_no_flagged_turns(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Safe content")]
        )
        report = await engine.build_report(request)

        assert len(report.flagged_turns) == 0

    @pytest.mark.asyncio
    async def test_report_all_turn_results_populated(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Test")]
        )
        report = await engine.build_report(request)

        assert len(report.all_turn_results) == 1

    @pytest.mark.asyncio
    async def test_report_metadata_embedded(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Test")]
        )
        meta = {"app": "MyApp", "tester": "dev@test.com"}
        report = await engine.build_report(request, metadata=meta)

        assert report.metadata["app"] == "MyApp"

    @pytest.mark.asyncio
    async def test_report_policy_reference_is_url(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hi")]
        )
        report = await engine.build_report(request)

        assert report.policy_reference.startswith("http")

    @pytest.mark.asyncio
    async def test_report_compliance_rate_all_safe(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hi")]
        )
        report = await engine.build_report(request)

        assert report.summary.compliance_rate == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_report_compliance_rate_all_violations(self) -> None:
        payload = _violation_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Bad content")]
        )
        report = await engine.build_report(request)

        assert report.summary.compliance_rate == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_build_report_from_result(self) -> None:
        """build_report_from_result should produce same structure without extra API calls."""
        payload = _safe_scores_payload()
        engine, mock_create = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Test")]
        )
        conv_result = await engine.audit_conversation(request)
        call_count_after_conv = mock_create.await_count

        report = await engine.build_report_from_result(conv_result)

        # No additional API calls should have been made
        assert mock_create.await_count == call_count_after_conv
        assert isinstance(report, AuditReport)

    @pytest.mark.asyncio
    async def test_report_flagged_turn_has_remediation_hints(self) -> None:
        payload = _violation_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Bad request")]
        )
        report = await engine.build_report(request)

        if report.flagged_turns:
            flagged = report.flagged_turns[0]
            # Remediation hints should be present for flagged categories
            assert isinstance(flagged.remediation_hints, dict)

    @pytest.mark.asyncio
    async def test_report_tool_version_present(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hi")]
        )
        report = await engine.build_report(request)

        assert isinstance(report.tool_version, str)
        assert len(report.tool_version) > 0

    @pytest.mark.asyncio
    async def test_report_schema_version_is_one_dot_zero(self) -> None:
        payload = _safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hi")]
        )
        report = await engine.build_report(request)

        assert report.schema_version == "1.0"
