"""teen_safety_auditor/models.py — Pydantic request and response models for the audit API.

This module defines all data-transfer objects used across the FastAPI endpoints:

- **Request models**: ``SinglePromptRequest``, ``ConversationRequest``
- **Per-turn result models**: ``CategoryScore``, ``TurnAuditResult``
- **Top-level response models**: ``AuditResult``, ``ConversationAuditResult``
- **Downloadable report model**: ``AuditReport``

All models use Pydantic v2 semantics (``model_config``, ``model_validator``, etc.).

Typical usage::

    from teen_safety_auditor.models import SinglePromptRequest, AuditResult

    req = SinglePromptRequest(prompt="Hello", response="Hi there!")
    # ... call auditor ...
    result: AuditResult = auditor.audit_prompt(req)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from teen_safety_auditor.policy import (
    CATEGORY_METADATA,
    PolicyCategory,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Shared configuration mixin
# ---------------------------------------------------------------------------


class _BaseAuditModel(BaseModel):
    """Base class for all audit models with shared Pydantic configuration."""

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
        "use_enum_values": True,
    }


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SinglePromptRequest(_BaseAuditModel):
    """Request body for the ``POST /api/audit/prompt`` endpoint.

    Attributes
    ----------
    prompt:
        The user-side message or prompt text to be evaluated.  Required and
        must not be empty after whitespace stripping.
    response:
        Optional AI-generated response to evaluate alongside the prompt.  When
        omitted, only the user prompt is scored.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="The user prompt or message to audit.",
        examples=["Tell me how to make friends online."],
    )
    response: str | None = Field(
        default=None,
        max_length=32_000,
        description="Optional AI response to evaluate alongside the prompt.",
        examples=["Making friends online can be fun and safe if you follow these tips…"],
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, v: str) -> str:
        """Ensure prompt is not blank after whitespace stripping."""
        if not v.strip():
            raise ValueError("prompt must not be blank or whitespace-only")
        return v

    @field_validator("response")
    @classmethod
    def response_none_if_blank(cls, v: str | None) -> str | None:
        """Coerce blank-string response to None for consistent downstream handling."""
        if v is not None and not v.strip():
            return None
        return v


class ConversationTurn(_BaseAuditModel):
    """A single turn in a multi-turn conversation, following the OpenAI chat format.

    Attributes
    ----------
    role:
        Message role — one of ``"user"``, ``"assistant"``, or ``"system"``.
    content:
        The textual content of the message.  Must be non-empty.
    """

    role: str = Field(
        ...,
        description="Message role: 'user', 'assistant', or 'system'.",
        examples=["user"],
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="Textual content of the message.",
        examples=["Can you help me with my homework?"],
    )

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        """Ensure role is one of the accepted OpenAI chat roles."""
        allowed = {"user", "assistant", "system"}
        normalised = v.strip().lower()
        if normalised not in allowed:
            raise ValueError(
                f"role must be one of {sorted(allowed)!r}, got {v!r}"
            )
        return normalised

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, v: str) -> str:
        """Ensure content is not blank after whitespace stripping."""
        if not v.strip():
            raise ValueError("content must not be blank or whitespace-only")
        return v


class ConversationRequest(_BaseAuditModel):
    """Request body for the ``POST /api/audit/conversation`` endpoint.

    Attributes
    ----------
    turns:
        Ordered list of conversation turns to audit.  Must contain at least one
        turn.  Maximum of 100 turns per request to guard against abuse.
    """

    turns: list[ConversationTurn] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Ordered conversation turns in OpenAI chat format.",
    )

    @model_validator(mode="after")
    def conversation_must_have_user_turn(self) -> "ConversationRequest":
        """Validate that at least one user turn is present in the conversation."""
        roles = {t.role for t in self.turns}
        if "user" not in roles:
            raise ValueError(
                "Conversation must contain at least one turn with role='user'."
            )
        return self


# ---------------------------------------------------------------------------
# Per-category score model
# ---------------------------------------------------------------------------


class CategoryScore(_BaseAuditModel):
    """Compliance score and risk classification for a single policy category.

    Attributes
    ----------
    category:
        The :class:`~teen_safety_auditor.policy.PolicyCategory` being scored.
    score:
        Normalised float in [0.0, 1.0] returned by the safeguard model.
    risk_level:
        Classified :class:`~teen_safety_auditor.policy.RiskLevel` derived from
        *score* using per-category thresholds.
    display_name:
        Human-readable category label, populated from
        :data:`~teen_safety_auditor.policy.CATEGORY_METADATA`.
    description:
        One-sentence policy description for this category.
    remediation_hint:
        Actionable advice for developers when this category is flagged.
    """

    category: PolicyCategory = Field(
        ...,
        description="Policy category identifier.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised compliance score between 0.0 (safe) and 1.0 (violation).",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Classified risk level: safe, caution, or violation.",
    )
    display_name: str = Field(
        default="",
        description="Human-readable category label.",
    )
    description: str = Field(
        default="",
        description="Brief policy description for this category.",
    )
    remediation_hint: str = Field(
        default="",
        description="Developer guidance for remediating a flagged category.",
    )

    @model_validator(mode="after")
    def populate_metadata_fields(self) -> "CategoryScore":
        """Auto-populate display_name, description, and remediation_hint from policy metadata."""
        # Use the PolicyCategory enum member (handle str values from use_enum_values)
        cat_key: PolicyCategory
        if isinstance(self.category, PolicyCategory):
            cat_key = self.category
        else:
            cat_key = PolicyCategory(self.category)

        meta = CATEGORY_METADATA.get(cat_key)
        if meta is not None:
            if not self.display_name:
                object.__setattr__(self, "display_name", meta.display_name)
            if not self.description:
                object.__setattr__(self, "description", meta.description)
            if not self.remediation_hint:
                object.__setattr__(self, "remediation_hint", meta.remediation_hint)
        return self


# ---------------------------------------------------------------------------
# Per-turn audit result
# ---------------------------------------------------------------------------


class TurnAuditResult(_BaseAuditModel):
    """Audit result for a single conversation turn.

    Attributes
    ----------
    turn_index:
        Zero-based index of this turn within the conversation.
    role:
        Role of the message author (``"user"``, ``"assistant"``, or ``"system"``).
    content_snippet:
        Truncated preview of the evaluated content (first 200 characters) for
        display in the UI without exposing full message text in list views.
    category_scores:
        Per-category compliance scores and risk classifications.
    overall_score:
        Aggregate compliance score across all categories for this turn.  This
        is either the value returned by the model or the max of category scores
        as a fallback.
    overall_risk_level:
        Overall risk classification derived from *overall_score*.
    flagged_categories:
        Subset of categories whose *risk_level* is ``CAUTION`` or ``VIOLATION``.
    reasoning:
        One-sentence explanation from the safeguard model for the most
        significant finding in this turn.
    """

    turn_index: int = Field(
        ...,
        ge=0,
        description="Zero-based turn index within the conversation.",
    )
    role: str = Field(
        ...,
        description="Message role: 'user', 'assistant', or 'system'.",
    )
    content_snippet: str = Field(
        ...,
        description="Truncated content preview (first 200 characters).",
    )
    category_scores: list[CategoryScore] = Field(
        default_factory=list,
        description="Per-category compliance scores.",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate compliance score for this turn.",
    )
    overall_risk_level: RiskLevel = Field(
        ...,
        description="Overall risk classification for this turn.",
    )
    flagged_categories: list[PolicyCategory] = Field(
        default_factory=list,
        description="Categories flagged as CAUTION or VIOLATION.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the most significant finding.",
    )

    @classmethod
    def make_content_snippet(cls, content: str, max_length: int = 200) -> str:
        """Truncate *content* to *max_length* characters, appending ellipsis if needed.

        Parameters
        ----------
        content:
            Full message content string.
        max_length:
            Maximum number of characters to retain (default: 200).

        Returns
        -------
        str
            Possibly truncated content string.
        """
        if len(content) <= max_length:
            return content
        return content[:max_length].rstrip() + "…"


# ---------------------------------------------------------------------------
# Single-prompt audit result
# ---------------------------------------------------------------------------


class AuditResult(_BaseAuditModel):
    """Response model for the ``POST /api/audit/prompt`` endpoint.

    Attributes
    ----------
    prompt_snippet:
        Truncated preview of the evaluated prompt (first 200 characters).
    response_snippet:
        Truncated preview of the evaluated AI response, or ``None`` if no
        response was provided.
    category_scores:
        Per-category compliance scores and risk classifications.
    overall_score:
        Aggregate compliance score across all categories.
    overall_risk_level:
        Overall risk classification derived from *overall_score*.
    flagged_categories:
        Categories whose risk level is ``CAUTION`` or ``VIOLATION``.
    reasoning:
        One-sentence model explanation of the most significant finding.
    audit_timestamp:
        ISO-8601 UTC timestamp of when the audit was performed.
    """

    prompt_snippet: str = Field(
        ...,
        description="Truncated prompt preview (first 200 characters).",
    )
    response_snippet: str | None = Field(
        default=None,
        description="Truncated AI response preview, or None if not provided.",
    )
    category_scores: list[CategoryScore] = Field(
        default_factory=list,
        description="Per-category compliance scores.",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate compliance score.",
    )
    overall_risk_level: RiskLevel = Field(
        ...,
        description="Overall risk classification.",
    )
    flagged_categories: list[PolicyCategory] = Field(
        default_factory=list,
        description="Categories flagged as CAUTION or VIOLATION.",
    )
    reasoning: str = Field(
        default="",
        description="Brief model explanation of the most significant finding.",
    )
    audit_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this audit was performed.",
    )


# ---------------------------------------------------------------------------
# Multi-turn conversation audit result
# ---------------------------------------------------------------------------


class ConversationAuditResult(_BaseAuditModel):
    """Response model for the ``POST /api/audit/conversation`` endpoint.

    Attributes
    ----------
    turn_results:
        Ordered list of per-turn audit results.
    total_turns:
        Total number of turns evaluated.
    flagged_turns:
        Count of turns that have at least one CAUTION or VIOLATION category.
    overall_score:
        Mean overall score across all turns.
    overall_risk_level:
        Risk classification derived from *overall_score*.
    most_flagged_categories:
        Policy categories that appeared most frequently in flagged turns,
        ordered by frequency descending.
    audit_timestamp:
        ISO-8601 UTC timestamp of when the audit was performed.
    """

    turn_results: list[TurnAuditResult] = Field(
        default_factory=list,
        description="Ordered per-turn audit results.",
    )
    total_turns: int = Field(
        ...,
        ge=0,
        description="Total number of conversation turns evaluated.",
    )
    flagged_turns: int = Field(
        default=0,
        ge=0,
        description="Number of turns with at least one CAUTION or VIOLATION.",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean compliance score across all turns.",
    )
    overall_risk_level: RiskLevel = Field(
        ...,
        description="Overall conversation risk classification.",
    )
    most_flagged_categories: list[PolicyCategory] = Field(
        default_factory=list,
        description="Most-frequently flagged categories across all turns.",
    )
    audit_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this audit was performed.",
    )


# ---------------------------------------------------------------------------
# Audit report models
# ---------------------------------------------------------------------------


class FlaggedTurnSummary(_BaseAuditModel):
    """Compact summary of a single flagged turn included in the downloadable report.

    Attributes
    ----------
    turn_index:
        Zero-based index of the flagged turn.
    role:
        Message role of the flagged turn.
    content_snippet:
        Truncated content preview.
    overall_score:
        Aggregate score for this turn.
    overall_risk_level:
        Overall risk classification.
    flagged_categories:
        List of categories flagged in this turn.
    category_scores:
        Full per-category scoring details.
    reasoning:
        Model reasoning for the most significant finding.
    remediation_hints:
        Mapping of flagged category key → remediation hint string.
    """

    turn_index: int = Field(..., ge=0)
    role: str = Field(...)
    content_snippet: str = Field(...)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    overall_risk_level: RiskLevel = Field(...)
    flagged_categories: list[PolicyCategory] = Field(default_factory=list)
    category_scores: list[CategoryScore] = Field(default_factory=list)
    reasoning: str = Field(default="")
    remediation_hints: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of category key to remediation advice.",
    )


class AuditReportSummary(_BaseAuditModel):
    """High-level summary block included at the top of an :class:`AuditReport`.

    Attributes
    ----------
    total_turns:
        Total number of turns evaluated.
    flagged_turns:
        Number of turns with at least one CAUTION or VIOLATION.
    safe_turns:
        Number of turns with all categories rated SAFE.
    violation_turns:
        Number of turns with at least one VIOLATION.
    caution_turns:
        Number of turns with CAUTION but no VIOLATION.
    overall_score:
        Mean compliance score across all turns.
    overall_risk_level:
        Overall conversation risk classification.
    most_flagged_categories:
        Policy categories most frequently flagged.
    compliance_rate:
        Fraction of turns that are fully SAFE, expressed as a percentage
        (0–100).
    """

    total_turns: int = Field(..., ge=0)
    flagged_turns: int = Field(default=0, ge=0)
    safe_turns: int = Field(default=0, ge=0)
    violation_turns: int = Field(default=0, ge=0)
    caution_turns: int = Field(default=0, ge=0)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    overall_risk_level: RiskLevel = Field(...)
    most_flagged_categories: list[PolicyCategory] = Field(default_factory=list)
    compliance_rate: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of turns rated fully SAFE.",
    )


class AuditReport(_BaseAuditModel):
    """Downloadable JSON audit report returned by ``POST /api/audit/report``.

    This is the top-level structure serialised to JSON and served as a file
    download.  It contains everything a developer needs to review compliance
    results offline.

    Attributes
    ----------
    report_id:
        Unique identifier for this report (UUID-like string).
    generated_at:
        ISO-8601 UTC timestamp when the report was generated.
    schema_version:
        Report format version string for forward compatibility.
    summary:
        High-level compliance summary.
    flagged_turns:
        List of compact summaries for all flagged turns.
    all_turn_results:
        Full per-turn audit results for every turn evaluated.
    policy_reference:
        URL pointing to the OpenAI teen safety policy documentation.
    tool_version:
        Version of the teen_safety_auditor tool that generated this report.
    metadata:
        Optional freeform key-value metadata (e.g. app name, tester name).
    """

    report_id: str = Field(
        ...,
        description="Unique report identifier.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this report was generated.",
    )
    schema_version: str = Field(
        default="1.0",
        description="Report schema version for forward compatibility.",
    )
    summary: AuditReportSummary = Field(
        ...,
        description="High-level compliance summary.",
    )
    flagged_turns: list[FlaggedTurnSummary] = Field(
        default_factory=list,
        description="Compact summaries of all flagged turns.",
    )
    all_turn_results: list[TurnAuditResult] = Field(
        default_factory=list,
        description="Full per-turn audit results for every evaluated turn.",
    )
    policy_reference: str = Field(
        default="https://openai.com/policies/usage-policies#child-safety",
        description="URL to the OpenAI teen/child safety policy documentation.",
    )
    tool_version: str = Field(
        default="0.1.0",
        description="Version of teen_safety_auditor that generated this report.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional freeform metadata (app name, tester, etc.).",
    )


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------


class AuditErrorResponse(_BaseAuditModel):
    """Standard error response returned by audit endpoints on failure.

    Attributes
    ----------
    error:
        Machine-readable error code (e.g. ``"openai_error"``, ``"invalid_input"``).
    message:
        Human-readable error description.
    details:
        Optional additional context (e.g. upstream error message).
    """

    error: str = Field(
        ...,
        description="Machine-readable error code.",
        examples=["openai_error"],
    )
    message: str = Field(
        ...,
        description="Human-readable error description.",
    )
    details: str | None = Field(
        default=None,
        description="Optional additional context.",
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # Request models
    "SinglePromptRequest",
    "ConversationTurn",
    "ConversationRequest",
    # Score / result models
    "CategoryScore",
    "TurnAuditResult",
    "AuditResult",
    "ConversationAuditResult",
    # Report models
    "FlaggedTurnSummary",
    "AuditReportSummary",
    "AuditReport",
    # Error model
    "AuditErrorResponse",
]
