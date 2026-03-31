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
from tests.fixtures import (
    FAKE_API_KEY,
    make_mock_engine,
    make_mock_openai_completion,
    make_multi_response_mock_engine,
    safe_scores_payload,
    caution_scores_payload,
    violation_scores_payload,
    multi_category_violation_payload,
    malformed_json_response,
    markdown_fenced_payload,
    empty_scores_response,
    out_of_range_scores_payload,
    make_safe_conversation_audit_result,
    make_violation_conversation_audit_result,
    SAFE_PROMPT,
    SAFE_RESPONSE,
    CAUTION_PROMPT,
    VIOLATION_PROMPT,
    SAFE_CONVERSATION_TURNS,
    MIXED_CONVERSATION_TURNS,
    SYSTEM_PLUS_USER_TURNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_with_mock_client(
    mock_completion_content: str,
) -> tuple[AuditEngine, AsyncMock]:
    """Create an AuditEngine with a mocked AsyncOpenAI client.

    Returns the engine and the mock create coroutine for assertion.
    """
    engine = AuditEngine(api_key=FAKE_API_KEY)
    mock_create = AsyncMock(
        return_value=make_mock_openai_completion(mock_completion_content)
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

    def test_empty_string_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(AuditEngineError):
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

    def test_safeguard_model_takes_precedence_over_openai_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SAFEGUARD_MODEL", "safeguard-model")
        monkeypatch.setenv("OPENAI_MODEL", "other-model")
        engine = AuditEngine(api_key=FAKE_API_KEY)
        assert engine._model == "safeguard-model"

    def test_argument_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SAFEGUARD_MODEL", "env-model")
        engine = AuditEngine(api_key=FAKE_API_KEY, model="arg-model")
        assert engine._model == "arg-model"

    def test_system_prompt_is_set(self) -> None:
        engine = AuditEngine(api_key=FAKE_API_KEY)
        assert isinstance(engine._system_prompt, str)
        assert len(engine._system_prompt) > 0

    def test_client_is_created(self) -> None:
        engine = AuditEngine(api_key=FAKE_API_KEY)
        assert engine._client is not None

    def test_base_url_passed_to_client(self) -> None:
        # Just confirm construction does not raise; we cannot inspect AsyncOpenAI internals easily
        engine = AuditEngine(
            api_key=FAKE_API_KEY,
            base_url="http://localhost:8080/v1",
        )
        assert engine is not None

    def test_create_engine_factory(self) -> None:
        engine = create_engine(api_key=FAKE_API_KEY)
        assert isinstance(engine, AuditEngine)

    def test_create_engine_missing_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(AuditEngineError):
            create_engine(api_key="")

    def test_create_engine_with_model(self) -> None:
        engine = create_engine(api_key=FAKE_API_KEY, model="custom-model")
        assert engine._model == "custom-model"


# ---------------------------------------------------------------------------
# _strip_markdown_fences
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

    def test_non_fence_backticks_unchanged(self) -> None:
        # Inline code with single backtick should not be affected
        text = '`some code`'
        result = AuditEngine._strip_markdown_fences(text)
        assert result == text

    def test_fenced_payload_parseable_after_strip(self) -> None:
        payload = safe_scores_payload()
        fenced = markdown_fenced_payload(payload)
        stripped = AuditEngine._strip_markdown_fences(fenced)
        parsed = json.loads(stripped)
        assert "scores" in parsed

    def test_python_fence_stripped(self) -> None:
        text = '```python\nprint("hello")\n```'
        result = AuditEngine._strip_markdown_fences(text)
        assert 'print("hello")' in result
        assert '```' not in result


# ---------------------------------------------------------------------------
# _coerce_score
# ---------------------------------------------------------------------------


class TestCoerceScore:
    """Tests for AuditEngine._coerce_score."""

    def test_float_passthrough(self) -> None:
        assert AuditEngine._coerce_score(0.75) == 0.75

    def test_int_converted(self) -> None:
        assert AuditEngine._coerce_score(1) == 1.0

    def test_zero_int_converted(self) -> None:
        assert AuditEngine._coerce_score(0) == 0.0

    def test_string_float_converted(self) -> None:
        assert AuditEngine._coerce_score("0.5") == 0.5

    def test_string_int_converted(self) -> None:
        assert AuditEngine._coerce_score("1") == 1.0

    def test_none_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        assert AuditEngine._coerce_score(None) == _FALLBACK_SCORE

    def test_negative_clamped_to_zero(self) -> None:
        assert AuditEngine._coerce_score(-0.5) == 0.0

    def test_above_one_clamped_to_one(self) -> None:
        assert AuditEngine._coerce_score(1.5) == 1.0

    def test_far_above_one_clamped_to_one(self) -> None:
        assert AuditEngine._coerce_score(100.0) == 1.0

    def test_far_below_zero_clamped_to_zero(self) -> None:
        assert AuditEngine._coerce_score(-100.0) == 0.0

    def test_invalid_string_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        assert AuditEngine._coerce_score("not_a_number") == _FALLBACK_SCORE

    def test_empty_string_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        assert AuditEngine._coerce_score("") == _FALLBACK_SCORE

    def test_zero_passthrough(self) -> None:
        assert AuditEngine._coerce_score(0.0) == 0.0

    def test_one_passthrough(self) -> None:
        assert AuditEngine._coerce_score(1.0) == 1.0

    def test_exact_boundary_values(self) -> None:
        assert AuditEngine._coerce_score(0.35) == 0.35
        assert AuditEngine._coerce_score(0.65) == 0.65

    def test_dict_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        assert AuditEngine._coerce_score({"nested": 0.5}) == _FALLBACK_SCORE

    def test_list_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        assert AuditEngine._coerce_score([0.5]) == _FALLBACK_SCORE


# ---------------------------------------------------------------------------
# _fallback_response
# ---------------------------------------------------------------------------


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

    def test_overall_score_is_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        result = AuditEngine._fallback_response()
        assert result["overall_score"] == _FALLBACK_SCORE

    def test_has_flagged_categories(self) -> None:
        result = AuditEngine._fallback_response()
        assert "flagged_categories" in result
        assert isinstance(result["flagged_categories"], list)

    def test_flagged_categories_is_empty(self) -> None:
        result = AuditEngine._fallback_response()
        assert result["flagged_categories"] == []

    def test_has_reasoning(self) -> None:
        result = AuditEngine._fallback_response()
        assert "reasoning" in result
        assert isinstance(result["reasoning"], str)

    def test_reasoning_contains_reason(self) -> None:
        result = AuditEngine._fallback_response(reason="JSON error")
        assert "JSON error" in result["reasoning"]

    def test_empty_reason(self) -> None:
        result = AuditEngine._fallback_response(reason="")
        assert isinstance(result["reasoning"], str)

    def test_no_reason_arg(self) -> None:
        result = AuditEngine._fallback_response()
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0

    def test_scores_count_matches_categories(self) -> None:
        result = AuditEngine._fallback_response()
        assert len(result["scores"]) == len(list(PolicyCategory))

    def test_all_scores_in_unit_interval(self) -> None:
        result = AuditEngine._fallback_response()
        for score in result["scores"].values():
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _build_category_scores
# ---------------------------------------------------------------------------


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

    def test_partial_scores_fills_rest_with_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        scores = {PolicyCategory.VIOLENCE.value: 0.9}
        result = AuditEngine._build_category_scores(scores)
        violence_score = next(
            cs for cs in result if cs.category == PolicyCategory.VIOLENCE.value
        )
        assert violence_score.score == 0.9
        other_scores = [
            cs for cs in result if cs.category != PolicyCategory.VIOLENCE.value
        ]
        for cs in other_scores:
            assert cs.score == _FALLBACK_SCORE

    def test_mixed_risk_levels(self) -> None:
        scores = {cat.value: 0.05 for cat in PolicyCategory}
        scores[PolicyCategory.VIOLENCE.value] = 0.9
        scores[PolicyCategory.SELF_HARM.value] = 0.45
        result = AuditEngine._build_category_scores(scores)
        violence_cs = next(
            cs for cs in result if cs.category == PolicyCategory.VIOLENCE.value
        )
        self_harm_cs = next(
            cs for cs in result if cs.category == PolicyCategory.SELF_HARM.value
        )
        assert violence_cs.risk_level == RiskLevel.VIOLATION.value
        # 0.45 is between self_harm thresholds (0.30 safe_max, 0.55 violation_min) => caution
        assert self_harm_cs.risk_level == RiskLevel.CAUTION.value

    def test_display_name_populated(self) -> None:
        scores = {cat.value: 0.1 for cat in PolicyCategory}
        result = AuditEngine._build_category_scores(scores)
        for cs in result:
            assert len(cs.display_name) > 0

    def test_remediation_hint_populated(self) -> None:
        scores = {cat.value: 0.9 for cat in PolicyCategory}
        result = AuditEngine._build_category_scores(scores)
        for cs in result:
            assert len(cs.remediation_hint) > 0


# ---------------------------------------------------------------------------
# _resolve_overall_score
# ---------------------------------------------------------------------------


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

    def test_uses_max_of_multiple_category_scores(self) -> None:
        scores = [
            CategoryScore(category=PolicyCategory.VIOLENCE, score=0.2, risk_level=RiskLevel.SAFE),
            CategoryScore(category=PolicyCategory.HATE_SPEECH, score=0.8, risk_level=RiskLevel.VIOLATION),
            CategoryScore(category=PolicyCategory.SELF_HARM, score=0.1, risk_level=RiskLevel.SAFE),
        ]
        result = AuditEngine._resolve_overall_score(None, scores)
        assert result == pytest.approx(0.8)

    def test_model_score_boundary_zero_accepted(self) -> None:
        scores = self._make_scores(0.9)
        # 0.0 is valid (in [0.0, 1.0])
        result = AuditEngine._resolve_overall_score(0.0, scores)
        assert result == 0.0

    def test_empty_scores_with_valid_model_score(self) -> None:
        result = AuditEngine._resolve_overall_score(0.7, [])
        assert result == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# _collect_flagged_categories
# ---------------------------------------------------------------------------


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

    def test_safe_category_not_in_flagged(self) -> None:
        scores = self._make_category_scores(
            {
                PolicyCategory.VIOLENCE: (0.9, RiskLevel.VIOLATION),
                PolicyCategory.SELF_HARM: (0.05, RiskLevel.SAFE),
            }
        )
        flagged = AuditEngine._collect_flagged_categories(scores)
        assert PolicyCategory.SELF_HARM not in flagged

    def test_all_violation_all_flagged(self) -> None:
        scores = self._make_category_scores(
            {cat: (0.9, RiskLevel.VIOLATION) for cat in PolicyCategory}
        )
        flagged = AuditEngine._collect_flagged_categories(scores)
        assert len(flagged) == len(list(PolicyCategory))

    def test_empty_input_returns_empty(self) -> None:
        flagged = AuditEngine._collect_flagged_categories([])
        assert flagged == []


# ---------------------------------------------------------------------------
# _mean_score
# ---------------------------------------------------------------------------


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

    def test_two_turns_mean(self) -> None:
        turns = [self._make_turn(0.0), self._make_turn(1.0)]
        assert AuditEngine._mean_score(turns) == pytest.approx(0.5)

    def test_result_in_unit_interval(self) -> None:
        import random
        random.seed(42)
        turns = [self._make_turn(random.random()) for _ in range(10)]
        mean = AuditEngine._mean_score(turns)
        assert 0.0 <= mean <= 1.0


# ---------------------------------------------------------------------------
# _most_flagged_categories
# ---------------------------------------------------------------------------


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

    def test_top_n_zero_returns_empty(self) -> None:
        turns = [self._make_turn_with_flags(0, [PolicyCategory.VIOLENCE])]
        result = AuditEngine._most_flagged_categories(turns, top_n=0)
        assert result == []

    def test_default_top_n_is_five(self) -> None:
        # Create turns with 8 distinct flagged categories
        all_cats = list(PolicyCategory)
        turns = [self._make_turn_with_flags(i, [cat]) for i, cat in enumerate(all_cats)]
        result = AuditEngine._most_flagged_categories(turns)
        assert len(result) <= 5

    def test_frequency_counting(self) -> None:
        turns = [
            self._make_turn_with_flags(0, [PolicyCategory.VIOLENCE]),
            self._make_turn_with_flags(1, [PolicyCategory.VIOLENCE]),
            self._make_turn_with_flags(2, [PolicyCategory.VIOLENCE]),
            self._make_turn_with_flags(3, [PolicyCategory.HATE_SPEECH]),
            self._make_turn_with_flags(4, [PolicyCategory.HATE_SPEECH]),
            self._make_turn_with_flags(5, [PolicyCategory.SELF_HARM]),
        ]
        result = AuditEngine._most_flagged_categories(turns, top_n=3)
        assert result[0] == PolicyCategory.VIOLENCE
        assert result[1] == PolicyCategory.HATE_SPEECH
        assert PolicyCategory.SELF_HARM in result


# ---------------------------------------------------------------------------
# _build_remediation_hints
# ---------------------------------------------------------------------------


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

    def test_hints_are_strings(self) -> None:
        result = AuditEngine._build_remediation_hints([PolicyCategory.GROOMING])
        assert isinstance(result["grooming"], str)

    def test_sexual_content_hint(self) -> None:
        result = AuditEngine._build_remediation_hints([PolicyCategory.SEXUAL_CONTENT])
        assert "sexual_content" in result
        assert len(result["sexual_content"]) > 10  # non-trivial hint

    def test_returns_dict(self) -> None:
        result = AuditEngine._build_remediation_hints([PolicyCategory.VIOLENCE])
        assert isinstance(result, dict)

    def test_duplicate_categories_handled(self) -> None:
        # Passing the same category twice should produce one entry
        result = AuditEngine._build_remediation_hints(
            [PolicyCategory.VIOLENCE, PolicyCategory.VIOLENCE]
        )
        # dict will just overwrite with same value
        assert "violence" in result
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _parse_safeguard_response
# ---------------------------------------------------------------------------


class TestParseSafeguardResponse:
    """Tests for AuditEngine._parse_safeguard_response."""

    def setup_method(self) -> None:
        self.engine = AuditEngine(api_key=FAKE_API_KEY)

    def test_valid_json_parsed(self) -> None:
        payload = safe_scores_payload()
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert "scores" in result
        assert "overall_score" in result
        assert "reasoning" in result

    def test_all_categories_in_scores(self) -> None:
        payload = safe_scores_payload()
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        for cat in PolicyCategory:
            assert cat.value in result["scores"]

    def test_markdown_fenced_json_parsed(self) -> None:
        payload = safe_scores_payload()
        raw = markdown_fenced_payload(payload)
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
        raw = json.dumps(empty_scores_response())
        result = self.engine._parse_safeguard_response(raw)
        for score in result["scores"].values():
            assert score == _FALLBACK_SCORE

    def test_overall_score_coerced_from_string(self) -> None:
        payload = safe_scores_payload()
        payload["overall_score"] = "0.7"  # string instead of float
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert result["overall_score"] == pytest.approx(0.7)

    def test_scores_clamped_to_unit_interval(self) -> None:
        payload = out_of_range_scores_payload()
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        # 2.5 should clamp to 1.0
        assert result["scores"][PolicyCategory.VIOLENCE.value] == 1.0
        # -0.3 should clamp to 0.0
        assert result["scores"][PolicyCategory.SELF_HARM.value] == 0.0

    def test_reasoning_extracted(self) -> None:
        payload = safe_scores_payload()
        payload["reasoning"] = "This is the reasoning."
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert result["reasoning"] == "This is the reasoning."

    def test_empty_string_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        result = self.engine._parse_safeguard_response("")
        assert result["overall_score"] == _FALLBACK_SCORE

    def test_malformed_response_returns_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        result = self.engine._parse_safeguard_response(malformed_json_response())
        assert result["overall_score"] == _FALLBACK_SCORE

    def test_flagged_categories_extracted(self) -> None:
        payload = violation_scores_payload()
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert "flagged_categories" in result
        assert isinstance(result["flagged_categories"], list)
        assert len(result["flagged_categories"]) > 0

    def test_non_list_flagged_categories_becomes_empty(self) -> None:
        payload = safe_scores_payload()
        payload["flagged_categories"] = "not_a_list"
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert isinstance(result["flagged_categories"], list)

    def test_non_dict_scores_gets_fallback(self) -> None:
        from teen_safety_auditor.auditor import _FALLBACK_SCORE
        payload = safe_scores_payload()
        payload["scores"] = "not_a_dict"
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        for score in result["scores"].values():
            assert score == _FALLBACK_SCORE

    def test_violation_payload_has_correct_scores(self) -> None:
        payload = violation_scores_payload([PolicyCategory.VIOLENCE])
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert result["scores"][PolicyCategory.VIOLENCE.value] == pytest.approx(0.92)

    def test_multi_category_payload_parsed(self) -> None:
        payload = multi_category_violation_payload()
        raw = json.dumps(payload)
        result = self.engine._parse_safeguard_response(raw)
        assert result["scores"][PolicyCategory.SEXUAL_CONTENT.value] == pytest.approx(0.95)
        assert result["scores"][PolicyCategory.GROOMING.value] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# audit_prompt (async, with mocked OpenAI)
# ---------------------------------------------------------------------------


class TestAuditPrompt:
    """Integration tests for AuditEngine.audit_prompt with mocked OpenAI."""

    @pytest.mark.asyncio
    async def test_safe_content_returns_safe_result(self) -> None:
        payload = safe_scores_payload()
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
        payload = violation_scores_payload([PolicyCategory.VIOLENCE, PolicyCategory.HATE_SPEECH])
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Describe graphic violence.")
        result = await engine.audit_prompt(request)

        assert result.overall_risk_level == RiskLevel.VIOLATION.value
        assert len(result.flagged_categories) > 0

    @pytest.mark.asyncio
    async def test_caution_content_returns_caution_or_above(self) -> None:
        payload = caution_scores_payload(
            caution_category=PolicyCategory.SUBSTANCE_ABUSE,
            caution_score=0.50,
        )
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Tell me about drug use.")
        result = await engine.audit_prompt(request)

        assert result.overall_risk_level in (
            RiskLevel.CAUTION.value,
            RiskLevel.VIOLATION.value,
        )

    @pytest.mark.asyncio
    async def test_result_has_all_category_scores(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Hi there!")
        result = await engine.audit_prompt(request)

        assert len(result.category_scores) == len(list(PolicyCategory))

    @pytest.mark.asyncio
    async def test_result_has_prompt_snippet(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Short prompt")
        result = await engine.audit_prompt(request)

        assert "Short prompt" in result.prompt_snippet

    @pytest.mark.asyncio
    async def test_result_with_response_has_response_snippet(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Q", response="The answer is 42.")
        result = await engine.audit_prompt(request)

        assert result.response_snippet is not None
        assert "42" in result.response_snippet

    @pytest.mark.asyncio
    async def test_result_without_response_has_none_snippet(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Solo prompt")
        result = await engine.audit_prompt(request)

        assert result.response_snippet is None

    @pytest.mark.asyncio
    async def test_audit_timestamp_is_set(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Test")
        result = await engine.audit_prompt(request)

        assert isinstance(result.audit_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_audit_timestamp_is_utc(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Test")
        result = await engine.audit_prompt(request)

        assert result.audit_timestamp.tzinfo is not None

    @pytest.mark.asyncio
    async def test_reasoning_propagated(self) -> None:
        payload = safe_scores_payload()
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

        # Fallback score is 0.5 => CAUTION
        assert result.overall_risk_level in (
            RiskLevel.CAUTION.value,
            RiskLevel.VIOLATION.value,
        )

    @pytest.mark.asyncio
    async def test_long_prompt_snippet_truncated(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        long_prompt = "A" * 500
        request = SinglePromptRequest(prompt=long_prompt)
        result = await engine.audit_prompt(request)

        assert len(result.prompt_snippet) <= 204  # 200 + possible ellipsis

    @pytest.mark.asyncio
    async def test_flagged_categories_populated_for_violation(self) -> None:
        payload = violation_scores_payload([PolicyCategory.SEXUAL_CONTENT])
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt=VIOLATION_PROMPT)
        result = await engine.audit_prompt(request)

        assert len(result.flagged_categories) > 0

    @pytest.mark.asyncio
    async def test_mock_called_with_messages(self) -> None:
        payload = safe_scores_payload()
        engine, mock_create = _make_engine_with_mock_client(json.dumps(payload))

        request = SinglePromptRequest(prompt="Hello!")
        await engine.audit_prompt(request)

        call_kwargs = mock_create.call_args
        # Should have been called with messages
        assert call_kwargs is not None

    @pytest.mark.asyncio
    async def test_uses_fixture_safe_engine(self, safe_engine: AuditEngine) -> None:
        """Verify the safe_engine pytest fixture works."""
        request = SinglePromptRequest(prompt=SAFE_PROMPT)
        result = await safe_engine.audit_prompt(request)
        assert result.overall_risk_level == RiskLevel.SAFE.value

    @pytest.mark.asyncio
    async def test_uses_fixture_violation_engine(self, violation_engine: AuditEngine) -> None:
        """Verify the violation_engine pytest fixture works."""
        request = SinglePromptRequest(prompt=VIOLATION_PROMPT)
        result = await violation_engine.audit_prompt(request)
        assert result.overall_risk_level == RiskLevel.VIOLATION.value

    @pytest.mark.asyncio
    async def test_uses_fixture_parse_error_engine(self, parse_error_engine: AuditEngine) -> None:
        """Verify the parse_error_engine fixture produces a caution fallback."""
        request = SinglePromptRequest(prompt="Any prompt")
        result = await parse_error_engine.audit_prompt(request)
        assert result.overall_risk_level in (
            RiskLevel.CAUTION.value,
            RiskLevel.VIOLATION.value,
        )


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

    @pytest.mark.asyncio
    async def test_rate_limit_error_message_preserved(self) -> None:
        from openai import RateLimitError

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError(
                "You have exceeded your rate limit",
                response=MagicMock(status_code=429, headers={}),
                body={},
            )
        )

        with pytest.raises(OpenAICallError) as exc_info:
            await engine.audit_prompt(SinglePromptRequest(prompt="Test"))

        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_openai_call_error_is_audit_engine_error(self) -> None:
        from openai import APIError

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=APIError("Fail", request=MagicMock(), body={})
        )

        with pytest.raises(AuditEngineError):
            await engine.audit_prompt(SinglePromptRequest(prompt="Test"))


# ---------------------------------------------------------------------------
# audit_conversation
# ---------------------------------------------------------------------------


class TestAuditConversation:
    """Tests for AuditEngine.audit_conversation."""

    @pytest.mark.asyncio
    async def test_single_user_turn(self) -> None:
        payload = safe_scores_payload()
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
        payload = safe_scores_payload()
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
        payload = safe_scores_payload()
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

        # Two pairs => 2 API calls, 2 turn results
        assert mock_create.await_count == 2
        assert result.total_turns == 2

    @pytest.mark.asyncio
    async def test_system_turn_skipped(self) -> None:
        payload = safe_scores_payload()
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
    async def test_system_plus_user_conversation(self) -> None:
        payload = safe_scores_payload()
        engine, mock_create = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(turns=SYSTEM_PLUS_USER_TURNS)  # type: ignore[arg-type]
        result = await engine.audit_conversation(request)

        # System turn is skipped, user+assistant pair is one call
        assert result.total_turns >= 1

    @pytest.mark.asyncio
    async def test_overall_score_is_mean(self) -> None:
        payload = safe_scores_payload()
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
        violation_payload = violation_scores_payload([PolicyCategory.VIOLENCE])
        safe_payload = safe_scores_payload()

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=[
                make_mock_openai_completion(json.dumps(violation_payload)),
                make_mock_openai_completion(json.dumps(safe_payload)),
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
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hello")]
        )
        result = await engine.audit_conversation(request)

        assert isinstance(result.audit_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_turn_results_have_correct_indices(self) -> None:
        payload = safe_scores_payload()
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
        payload = violation_scores_payload([PolicyCategory.VIOLENCE])
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Bad content")]
        )
        result = await engine.audit_conversation(request)

        # Since violence is flagged, it should appear in most_flagged_categories
        cat_values = [
            c if isinstance(c, str) else c.value
            for c in result.most_flagged_categories
        ]
        assert "violence" in cat_values or len(cat_values) > 0

    @pytest.mark.asyncio
    async def test_multi_response_engine_different_per_turn(self) -> None:
        """Use make_multi_response_mock_engine to test per-turn differentiation."""
        payloads = [
            violation_scores_payload([PolicyCategory.HATE_SPEECH]),
            safe_scores_payload(),
        ]
        engine, mock_create = make_multi_response_mock_engine(payloads)

        request = ConversationRequest(
            turns=[
                ConversationTurn(role="user", content="Hateful request"),
                ConversationTurn(role="user", content="Safe request"),
            ]
        )
        result = await engine.audit_conversation(request)

        assert result.total_turns == 2
        assert mock_create.await_count == 2
        # First turn should be flagged
        flagged_count = sum(
            1 for t in result.turn_results if len(t.flagged_categories) > 0
        )
        assert flagged_count >= 1

    @pytest.mark.asyncio
    async def test_safe_conversation_from_fixture(
        self, safe_conversation_request: ConversationRequest, safe_engine: AuditEngine
    ) -> None:
        result = await safe_engine.audit_conversation(safe_conversation_request)
        assert result.overall_risk_level == RiskLevel.SAFE.value

    @pytest.mark.asyncio
    async def test_turn_results_role_preserved(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hello")]
        )
        result = await engine.audit_conversation(request)

        assert result.turn_results[0].role == "user"

    @pytest.mark.asyncio
    async def test_content_snippet_in_turn_result(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hello, can you help me?")]
        )
        result = await engine.audit_conversation(request)

        assert "Hello" in result.turn_results[0].content_snippet

    @pytest.mark.asyncio
    async def test_openai_error_propagated_in_conversation(self) -> None:
        from openai import RateLimitError

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError(
                "Rate limit",
                response=MagicMock(status_code=429, headers={}),
                body={},
            )
        )

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Test")]
        )
        with pytest.raises(OpenAICallError):
            await engine.audit_conversation(request)


# ---------------------------------------------------------------------------
# build_report and build_report_from_result
# ---------------------------------------------------------------------------


class TestBuildReport:
    """Tests for AuditEngine.build_report and build_report_from_result."""

    @pytest.mark.asyncio
    async def test_build_report_returns_audit_report(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hello")]
        )
        report = await engine.build_report(request)

        assert isinstance(report, AuditReport)

    @pytest.mark.asyncio
    async def test_report_has_unique_id(self) -> None:
        payload = safe_scores_payload()
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
        payload = safe_scores_payload()
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
        payload = violation_scores_payload([PolicyCategory.VIOLENCE])
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Violent content")]
        )
        report = await engine.build_report(request)

        assert report.summary.violation_turns == 1
        assert report.summary.safe_turns == 0

    @pytest.mark.asyncio
    async def test_report_has_flagged_turns(self) -> None:
        payload = violation_scores_payload([PolicyCategory.VIOLENCE])
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Bad content")]
        )
        report = await engine.build_report(request)

        assert len(report.flagged_turns) > 0

    @pytest.mark.asyncio
    async def test_report_safe_content_no_flagged_turns(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Safe content")]
        )
        report = await engine.build_report(request)

        assert len(report.flagged_turns) == 0

    @pytest.mark.asyncio
    async def test_report_all_turn_results_populated(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Test")]
        )
        report = await engine.build_report(request)

        assert len(report.all_turn_results) == 1

    @pytest.mark.asyncio
    async def test_report_metadata_embedded(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Test")]
        )
        meta = {"app": "MyApp", "tester": "dev@test.com"}
        report = await engine.build_report(request, metadata=meta)

        assert report.metadata["app"] == "MyApp"
        assert report.metadata["tester"] == "dev@test.com"

    @pytest.mark.asyncio
    async def test_report_policy_reference_is_url(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hi")]
        )
        report = await engine.build_report(request)

        assert report.policy_reference.startswith("http")
        assert "openai.com" in report.policy_reference

    @pytest.mark.asyncio
    async def test_report_compliance_rate_all_safe(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hi")]
        )
        report = await engine.build_report(request)

        assert report.summary.compliance_rate == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_report_compliance_rate_all_violations(self) -> None:
        payload = violation_scores_payload([PolicyCategory.VIOLENCE])
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Bad content")]
        )
        report = await engine.build_report(request)

        assert report.summary.compliance_rate == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_build_report_from_result(self) -> None:
        """build_report_from_result should produce same structure without extra API calls."""
        payload = safe_scores_payload()
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
        payload = violation_scores_payload([PolicyCategory.VIOLENCE])
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Bad request")]
        )
        report = await engine.build_report(request)

        if report.flagged_turns:
            flagged = report.flagged_turns[0]
            # Remediation hints should be present for flagged categories
            assert isinstance(flagged.remediation_hints, dict)
            assert len(flagged.remediation_hints) > 0

    @pytest.mark.asyncio
    async def test_report_tool_version_present(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hi")]
        )
        report = await engine.build_report(request)

        assert isinstance(report.tool_version, str)
        assert len(report.tool_version) > 0

    @pytest.mark.asyncio
    async def test_report_schema_version_is_one_dot_zero(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hi")]
        )
        report = await engine.build_report(request)

        assert report.schema_version == "1.0"

    @pytest.mark.asyncio
    async def test_report_generated_at_is_datetime(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Hi")]
        )
        report = await engine.build_report(request)

        assert isinstance(report.generated_at, datetime)

    @pytest.mark.asyncio
    async def test_build_report_from_result_with_metadata(self) -> None:
        payload = safe_scores_payload()
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Test")]
        )
        conv_result = await engine.audit_conversation(request)
        meta = {"env": "staging"}
        report = await engine.build_report_from_result(conv_result, metadata=meta)

        assert report.metadata["env"] == "staging"

    @pytest.mark.asyncio
    async def test_report_from_violation_fixture(
        self, violation_conversation_result: ConversationAuditResult
    ) -> None:
        """Verify building a report from a pre-computed violation result."""
        engine = AuditEngine(api_key=FAKE_API_KEY)
        report = await engine.build_report_from_result(violation_conversation_result)
        assert isinstance(report, AuditReport)
        assert report.summary.flagged_turns > 0

    @pytest.mark.asyncio
    async def test_caution_turn_counted_correctly(self) -> None:
        payload = caution_scores_payload(
            caution_category=PolicyCategory.SELF_HARM,
            caution_score=0.48,
        )
        engine, _ = _make_engine_with_mock_client(json.dumps(payload))

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content=CAUTION_PROMPT)]
        )
        report = await engine.build_report(request)

        # 0.48 for self_harm is between thresholds => CAUTION
        # Caution turns are not safe, so safe_turns should be 0
        assert report.summary.safe_turns == 0
        assert report.summary.violation_turns == 0
        assert report.summary.caution_turns == 1

    @pytest.mark.asyncio
    async def test_report_openai_error_propagated(self) -> None:
        from openai import APIError

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=APIError("Fail", request=MagicMock(), body={})
        )

        request = ConversationRequest(
            turns=[ConversationTurn(role="user", content="Test")]
        )
        with pytest.raises(OpenAICallError):
            await engine.build_report(request)

    @pytest.mark.asyncio
    async def test_multi_turn_report_compliance_rate(self) -> None:
        """50% compliance rate when 1 of 2 turns is safe."""
        violation_payload = violation_scores_payload([PolicyCategory.VIOLENCE])
        safe_payload = safe_scores_payload()

        engine = AuditEngine(api_key=FAKE_API_KEY)
        engine._client.chat.completions.create = AsyncMock(
            side_effect=[
                make_mock_openai_completion(json.dumps(safe_payload)),
                make_mock_openai_completion(json.dumps(violation_payload)),
            ]
        )

        request = ConversationRequest(
            turns=[
                ConversationTurn(role="user", content="Safe question"),
                ConversationTurn(role="user", content="Violent question"),
            ]
        )
        report = await engine.build_report(request)

        assert report.summary.total_turns == 2
        assert report.summary.safe_turns == 1
        assert report.summary.violation_turns == 1
        assert report.summary.compliance_rate == pytest.approx(50.0)
