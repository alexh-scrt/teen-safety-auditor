"""tests/test_policy.py — Unit tests for teen_safety_auditor/policy.py.

Covers:
- All PolicyCategory and RiskLevel enum members exist and have correct values
- Threshold constant ordering invariant
- classify_risk function correctness across boundary values
- CategoryMeta metadata completeness
- build_system_prompt and build_evaluation_prompt output correctness
"""

from __future__ import annotations

import json
import re

import pytest

from teen_safety_auditor.policy import (
    CATEGORY_METADATA,
    CATEGORY_THRESHOLDS,
    RISK_LEVEL_COLORS,
    SAFE_MAX,
    SYSTEM_PROMPT_TEMPLATE,
    THRESHOLDS,
    VIOLATION_MIN,
    CategoryMeta,
    PolicyCategory,
    RiskLevel,
    build_evaluation_prompt,
    build_system_prompt,
    classify_risk,
)


# ---------------------------------------------------------------------------
# PolicyCategory enum
# ---------------------------------------------------------------------------


class TestPolicyCategoryEnum:
    """Verify all expected policy categories are present with correct values."""

    EXPECTED_MEMBERS: dict[str, str] = {
        "SEXUAL_CONTENT": "sexual_content",
        "SELF_HARM": "self_harm",
        "GROOMING": "grooming",
        "VIOLENCE": "violence",
        "DANGEROUS_ACTIVITIES": "dangerous_activities",
        "SUBSTANCE_ABUSE": "substance_abuse",
        "PRIVACY_VIOLATION": "privacy_violation",
        "HATE_SPEECH": "hate_speech",
    }

    def test_all_expected_members_exist(self) -> None:
        for name, value in self.EXPECTED_MEMBERS.items():
            assert hasattr(PolicyCategory, name), f"Missing PolicyCategory.{name}"
            assert PolicyCategory[name].value == value

    def test_total_count(self) -> None:
        assert len(PolicyCategory) == len(self.EXPECTED_MEMBERS)

    def test_members_are_strings(self) -> None:
        for cat in PolicyCategory:
            assert isinstance(cat.value, str)
            assert len(cat.value) > 0

    def test_is_str_subclass(self) -> None:
        """PolicyCategory should subclass str for easy JSON serialisation."""
        for cat in PolicyCategory:
            assert isinstance(cat, str)


# ---------------------------------------------------------------------------
# RiskLevel enum
# ---------------------------------------------------------------------------


class TestRiskLevelEnum:
    """Verify RiskLevel members."""

    def test_has_safe(self) -> None:
        assert RiskLevel.SAFE.value == "safe"

    def test_has_caution(self) -> None:
        assert RiskLevel.CAUTION.value == "caution"

    def test_has_violation(self) -> None:
        assert RiskLevel.VIOLATION.value == "violation"

    def test_exactly_three_levels(self) -> None:
        assert len(RiskLevel) == 3

    def test_is_str_subclass(self) -> None:
        for level in RiskLevel:
            assert isinstance(level, str)


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------


class TestThresholdConstants:
    """Verify threshold constants are well-formed."""

    def test_safe_max_is_float(self) -> None:
        assert isinstance(SAFE_MAX, float)

    def test_violation_min_is_float(self) -> None:
        assert isinstance(VIOLATION_MIN, float)

    def test_ordering_invariant(self) -> None:
        assert SAFE_MAX < VIOLATION_MIN, "SAFE_MAX must be less than VIOLATION_MIN"

    def test_safe_max_in_unit_interval(self) -> None:
        assert 0.0 < SAFE_MAX < 1.0

    def test_violation_min_in_unit_interval(self) -> None:
        assert 0.0 < VIOLATION_MIN < 1.0

    def test_thresholds_dict_keys(self) -> None:
        assert "safe_max" in THRESHOLDS
        assert "violation_min" in THRESHOLDS

    def test_thresholds_dict_values_match_constants(self) -> None:
        assert THRESHOLDS["safe_max"] == SAFE_MAX
        assert THRESHOLDS["violation_min"] == VIOLATION_MIN

    def test_risk_level_colors_has_all_levels(self) -> None:
        for level in RiskLevel:
            assert level in RISK_LEVEL_COLORS

    def test_risk_level_colors_are_strings(self) -> None:
        for level, color in RISK_LEVEL_COLORS.items():
            assert isinstance(color, str) and len(color) > 0

    def test_category_thresholds_all_categories_covered(self) -> None:
        """Every PolicyCategory should have per-category thresholds defined."""
        for cat in PolicyCategory:
            assert cat in CATEGORY_THRESHOLDS, f"{cat} missing from CATEGORY_THRESHOLDS"

    def test_category_thresholds_values_in_range(self) -> None:
        for cat, t in CATEGORY_THRESHOLDS.items():
            assert 0.0 < t["safe_max"] < 1.0, f"{cat} safe_max out of range"
            assert 0.0 < t["violation_min"] < 1.0, f"{cat} violation_min out of range"
            assert t["safe_max"] < t["violation_min"], f"{cat} threshold ordering violated"


# ---------------------------------------------------------------------------
# classify_risk function
# ---------------------------------------------------------------------------


class TestClassifyRisk:
    """Unit tests for the classify_risk helper function."""

    def test_zero_score_is_safe(self) -> None:
        assert classify_risk(0.0) == RiskLevel.SAFE

    def test_one_score_is_violation(self) -> None:
        assert classify_risk(1.0) == RiskLevel.VIOLATION

    def test_safe_max_boundary_is_safe(self) -> None:
        assert classify_risk(SAFE_MAX) == RiskLevel.SAFE

    def test_just_above_safe_max_is_caution(self) -> None:
        assert classify_risk(SAFE_MAX + 0.001) == RiskLevel.CAUTION

    def test_violation_min_boundary_is_violation(self) -> None:
        assert classify_risk(VIOLATION_MIN) == RiskLevel.VIOLATION

    def test_just_below_violation_min_is_caution(self) -> None:
        assert classify_risk(VIOLATION_MIN - 0.001) == RiskLevel.CAUTION

    def test_midpoint_is_caution(self) -> None:
        midpoint = (SAFE_MAX + VIOLATION_MIN) / 2
        assert classify_risk(midpoint) == RiskLevel.CAUTION

    def test_negative_score_raises(self) -> None:
        with pytest.raises(ValueError, match="Score must be in"):
            classify_risk(-0.01)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="Score must be in"):
            classify_risk(1.01)

    def test_with_sexual_content_category(self) -> None:
        """Sexual content has tighter thresholds; score 0.3 should be CAUTION there."""
        # 0.3 > sexual_content safe_max (0.25) => at least CAUTION
        result = classify_risk(0.3, PolicyCategory.SEXUAL_CONTENT)
        assert result in (RiskLevel.CAUTION, RiskLevel.VIOLATION)

    def test_with_category_below_safe_max_is_safe(self) -> None:
        result = classify_risk(0.1, PolicyCategory.VIOLENCE)
        assert result == RiskLevel.SAFE

    def test_with_category_above_violation_min_is_violation(self) -> None:
        result = classify_risk(0.9, PolicyCategory.SELF_HARM)
        assert result == RiskLevel.VIOLATION

    @pytest.mark.parametrize("score", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_returns_risk_level_for_various_scores(self, score: float) -> None:
        result = classify_risk(score)
        assert isinstance(result, RiskLevel)

    @pytest.mark.parametrize("cat", list(PolicyCategory))
    def test_with_every_category_at_midpoint(self, cat: PolicyCategory) -> None:
        """classify_risk should return a valid RiskLevel for every category."""
        result = classify_risk(0.5, cat)
        assert isinstance(result, RiskLevel)


# ---------------------------------------------------------------------------
# CategoryMeta and CATEGORY_METADATA
# ---------------------------------------------------------------------------


class TestCategoryMetadata:
    """Verify CATEGORY_METADATA completeness and correctness."""

    def test_all_categories_have_metadata(self) -> None:
        for cat in PolicyCategory:
            assert cat in CATEGORY_METADATA, f"Missing metadata for {cat}"

    def test_metadata_is_category_meta_instances(self) -> None:
        for cat, meta in CATEGORY_METADATA.items():
            assert isinstance(meta, CategoryMeta), (
                f"{cat} metadata is not a CategoryMeta instance"
            )

    def test_display_names_are_non_empty(self) -> None:
        for cat, meta in CATEGORY_METADATA.items():
            assert meta.display_name.strip(), f"{cat} has empty display_name"

    def test_descriptions_are_non_empty(self) -> None:
        for cat, meta in CATEGORY_METADATA.items():
            assert meta.description.strip(), f"{cat} has empty description"

    def test_examples_are_lists(self) -> None:
        for cat, meta in CATEGORY_METADATA.items():
            assert isinstance(meta.examples, list), f"{cat} examples is not a list"

    def test_examples_not_empty(self) -> None:
        for cat, meta in CATEGORY_METADATA.items():
            assert len(meta.examples) > 0, f"{cat} has no examples"

    def test_remediation_hints_non_empty(self) -> None:
        for cat, meta in CATEGORY_METADATA.items():
            assert meta.remediation_hint.strip(), f"{cat} has empty remediation_hint"

    def test_category_field_matches_key(self) -> None:
        for cat, meta in CATEGORY_METADATA.items():
            assert meta.category == cat, (
                f"{cat} metadata.category field does not match dict key"
            )

    def test_meta_is_frozen(self) -> None:
        """CategoryMeta is a frozen dataclass; mutation should raise."""
        meta = CATEGORY_METADATA[PolicyCategory.VIOLENCE]
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(meta, "display_name", "hacked")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for the build_system_prompt() factory function."""

    def test_returns_non_empty_string(self) -> None:
        result = build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_matches_template_constant(self) -> None:
        assert build_system_prompt() == SYSTEM_PROMPT_TEMPLATE

    def test_contains_all_category_keys(self) -> None:
        prompt = build_system_prompt()
        for cat in PolicyCategory:
            assert cat.value in prompt, f"System prompt missing category key: {cat.value}"

    def test_contains_json_structure_hint(self) -> None:
        prompt = build_system_prompt()
        assert "scores" in prompt
        assert "overall_score" in prompt
        assert "flagged_categories" in prompt
        assert "reasoning" in prompt

    def test_instructs_json_only_output(self) -> None:
        prompt = build_system_prompt()
        # Should explicitly tell model to return only JSON
        assert "JSON" in prompt or "json" in prompt

    def test_mentions_teen_age_range(self) -> None:
        prompt = build_system_prompt()
        # Should reference the teen age range
        assert "13" in prompt or "teen" in prompt.lower()

    def test_contains_score_range_description(self) -> None:
        prompt = build_system_prompt()
        assert "0.0" in prompt
        assert "1.0" in prompt


# ---------------------------------------------------------------------------
# build_evaluation_prompt
# ---------------------------------------------------------------------------


class TestBuildEvaluationPrompt:
    """Tests for the build_evaluation_prompt() helper."""

    def test_basic_user_message_only(self) -> None:
        result = build_evaluation_prompt("Hello, how are you?")
        assert "Hello, how are you?" in result

    def test_includes_user_content(self) -> None:
        content = "Tell me about online safety."
        result = build_evaluation_prompt(content)
        assert content in result

    def test_includes_assistant_content_when_provided(self) -> None:
        result = build_evaluation_prompt("Question", assistant_content="Some answer")
        assert "Some answer" in result

    def test_no_assistant_content_note(self) -> None:
        result = build_evaluation_prompt("Just a prompt")
        assert "none" in result.lower() or "no assistant" in result.lower() or "(none" in result

    def test_turn_index_included_when_provided(self) -> None:
        result = build_evaluation_prompt("Hi", turn_index=2)
        # turn_index=2 => turn 3 (1-indexed)
        assert "3" in result

    def test_turn_index_zero_produces_turn_1(self) -> None:
        result = build_evaluation_prompt("Hi", turn_index=0)
        assert "1" in result

    def test_empty_user_content_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_evaluation_prompt("")

    def test_whitespace_only_user_content_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_evaluation_prompt("   ")

    def test_returns_string(self) -> None:
        result = build_evaluation_prompt("Test prompt")
        assert isinstance(result, str)

    def test_whitespace_stripped_from_user_content(self) -> None:
        result = build_evaluation_prompt("  Test content  ")
        assert "Test content" in result

    def test_whitespace_stripped_from_assistant_content(self) -> None:
        result = build_evaluation_prompt("Q", assistant_content="  Answer  ")
        assert "Answer" in result

    def test_empty_assistant_content_treated_as_none(self) -> None:
        result_none = build_evaluation_prompt("Q", assistant_content=None)
        result_empty = build_evaluation_prompt("Q", assistant_content="")
        # Both should produce same result pattern (no assistant content)
        assert "(none" in result_none.lower() or "none" in result_none.lower()
        assert "(none" in result_empty.lower() or "none" in result_empty.lower()

    def test_json_instruction_present(self) -> None:
        result = build_evaluation_prompt("Some content")
        assert "json" in result.lower() or "JSON" in result

    @pytest.mark.parametrize("turn_idx", [0, 1, 5, 10, 99])
    def test_various_turn_indices(self, turn_idx: int) -> None:
        result = build_evaluation_prompt("content", turn_index=turn_idx)
        # Should reference turn_idx + 1 somewhere
        assert str(turn_idx + 1) in result
