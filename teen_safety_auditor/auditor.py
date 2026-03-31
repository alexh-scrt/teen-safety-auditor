"""teen_safety_auditor/auditor.py — Core auditing engine.

This module implements the :class:`AuditEngine` class, which is responsible for:

1. Calling the OpenAI API (using the ``gpt-4o-mini`` model as a safeguard evaluator)
   with a structured system prompt and per-turn evaluation prompt.
2. Parsing the JSON response returned by the model into category scores.
3. Classifying each category score into a :class:`~teen_safety_auditor.policy.RiskLevel`
   using per-category thresholds.
4. Building :class:`~teen_safety_auditor.models.AuditResult`,
   :class:`~teen_safety_auditor.models.ConversationAuditResult`, and
   :class:`~teen_safety_auditor.models.AuditReport` response objects.

The engine is intentionally stateless — it holds only the OpenAI client and
configuration.  No prompt content is stored between calls.

Typical usage::

    import os
    from teen_safety_auditor.auditor import AuditEngine
    from teen_safety_auditor.models import SinglePromptRequest, ConversationRequest

    engine = AuditEngine(api_key=os.environ["OPENAI_API_KEY"])

    # Single-prompt audit
    result = await engine.audit_prompt(SinglePromptRequest(prompt="Hello!"))

    # Multi-turn conversation audit
    result = await engine.audit_conversation(ConversationRequest(turns=[...]))

    # Full downloadable report
    report = await engine.build_report(ConversationRequest(turns=[...]))
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from teen_safety_auditor.models import (
    AuditReport,
    AuditReportSummary,
    AuditResult,
    CategoryScore,
    ConversationAuditResult,
    ConversationRequest,
    FlaggedTurnSummary,
    SinglePromptRequest,
    TurnAuditResult,
)
from teen_safety_auditor.policy import (
    CATEGORY_METADATA,
    PolicyCategory,
    RiskLevel,
    build_evaluation_prompt,
    build_system_prompt,
    classify_risk,
)
from teen_safety_auditor import __version__

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default model used for safety evaluation.  Override via OPENAI_MODEL or
# SAFEGUARD_MODEL environment variables.
_DEFAULT_MODEL: str = "gpt-4o-mini"

# Maximum tokens to request from the model.  The JSON response is compact, so
# 512 tokens is more than sufficient.
_MAX_TOKENS: int = 512

# Temperature for evaluation calls — we want deterministic, consistent scoring.
_TEMPERATURE: float = 0.0

# Fallback score used when the model fails to return a parseable response.
_FALLBACK_SCORE: float = 0.5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AuditEngineError(Exception):
    """Base exception for errors raised by :class:`AuditEngine`."""


class OpenAICallError(AuditEngineError):
    """Raised when the OpenAI API call fails after exhausting retries."""


class ResponseParseError(AuditEngineError):
    """Raised when the safeguard model returns a response that cannot be parsed."""


# ---------------------------------------------------------------------------
# AuditEngine
# ---------------------------------------------------------------------------


class AuditEngine:
    """Stateless auditing engine that evaluates content against teen safety policies.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Defaults to the ``OPENAI_API_KEY`` environment variable.
    model:
        Name of the OpenAI model used as the safety evaluator.  Defaults to
        ``SAFEGUARD_MODEL`` env var, then ``OPENAI_MODEL`` env var, then
        ``gpt-4o-mini``.
    base_url:
        Optional custom OpenAI-compatible base URL (useful for testing with a
        local proxy or mock server).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise AuditEngineError(
                "OpenAI API key not provided.  Set the OPENAI_API_KEY environment "
                "variable or pass api_key= to AuditEngine()."
            )

        self._model: str = (
            model
            or os.environ.get("SAFEGUARD_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or _DEFAULT_MODEL
        )

        client_kwargs: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._system_prompt: str = build_system_prompt()

        logger.info(
            "AuditEngine initialised with model=%r",
            self._model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def audit_prompt(
        self,
        request: SinglePromptRequest,
    ) -> AuditResult:
        """Audit a single prompt/response pair.

        Parameters
        ----------
        request:
            A :class:`~teen_safety_auditor.models.SinglePromptRequest` containing
            the user prompt and optional AI response.

        Returns
        -------
        AuditResult
            Populated result with per-category scores, overall risk level, and
            flagged categories.

        Raises
        ------
        OpenAICallError
            If the OpenAI API call fails.
        ResponseParseError
            If the model response cannot be parsed into the expected structure.
        """
        evaluation_prompt = build_evaluation_prompt(
            user_content=request.prompt,
            assistant_content=request.response,
            turn_index=None,
        )

        raw = await self._call_safeguard(evaluation_prompt)
        parsed = self._parse_safeguard_response(raw)

        category_scores = self._build_category_scores(parsed["scores"])
        overall_score = self._resolve_overall_score(
            parsed.get("overall_score"), category_scores
        )
        overall_risk_level = classify_risk(overall_score)
        flagged_categories = self._collect_flagged_categories(category_scores)

        return AuditResult(
            prompt_snippet=TurnAuditResult.make_content_snippet(request.prompt),
            response_snippet=(
                TurnAuditResult.make_content_snippet(request.response)
                if request.response
                else None
            ),
            category_scores=category_scores,
            overall_score=overall_score,
            overall_risk_level=overall_risk_level,
            flagged_categories=flagged_categories,
            reasoning=parsed.get("reasoning", ""),
            audit_timestamp=datetime.now(timezone.utc),
        )

    async def audit_conversation(
        self,
        request: ConversationRequest,
    ) -> ConversationAuditResult:
        """Audit every turn in a multi-turn conversation.

        Each turn is evaluated independently.  User and assistant turns are
        paired when adjacent (user turn followed immediately by an assistant
        turn) so that the model can score the exchange holistically.

        Parameters
        ----------
        request:
            A :class:`~teen_safety_auditor.models.ConversationRequest` containing
            an ordered list of conversation turns.

        Returns
        -------
        ConversationAuditResult
            Aggregated per-turn results with overall compliance metrics.

        Raises
        ------
        OpenAICallError
            If any OpenAI API call fails.
        ResponseParseError
            If any model response cannot be parsed.
        """
        turn_results: list[TurnAuditResult] = []
        turns = request.turns

        # Pair adjacent user→assistant turns for holistic evaluation.
        # Any unpaired user or non-assistant turns are evaluated alone.
        evaluated_indices: set[int] = set()
        i = 0
        while i < len(turns):
            current = turns[i]

            # System turns are skipped for individual scoring (low information
            # value for teen-safety auditing) but included in the index count.
            if current.role == "system":
                evaluated_indices.add(i)
                i += 1
                continue

            user_content = current.content
            assistant_content: str | None = None

            # Look ahead for a paired assistant response
            if (
                current.role == "user"
                and i + 1 < len(turns)
                and turns[i + 1].role == "assistant"
            ):
                assistant_content = turns[i + 1].content
                pair_consumed = 2
            else:
                pair_consumed = 1

            evaluation_prompt = build_evaluation_prompt(
                user_content=user_content,
                assistant_content=assistant_content,
                turn_index=i,
            )

            raw = await self._call_safeguard(evaluation_prompt)
            parsed = self._parse_safeguard_response(raw)

            category_scores = self._build_category_scores(parsed["scores"])
            overall_score = self._resolve_overall_score(
                parsed.get("overall_score"), category_scores
            )
            overall_risk_level = classify_risk(overall_score)
            flagged_categories = self._collect_flagged_categories(category_scores)

            turn_result = TurnAuditResult(
                turn_index=i,
                role=current.role,
                content_snippet=TurnAuditResult.make_content_snippet(user_content),
                category_scores=category_scores,
                overall_score=overall_score,
                overall_risk_level=overall_risk_level,
                flagged_categories=flagged_categories,
                reasoning=parsed.get("reasoning", ""),
            )
            turn_results.append(turn_result)

            for offset in range(pair_consumed):
                evaluated_indices.add(i + offset)
            i += pair_consumed

        # Aggregate metrics
        total_turns = len(turn_results)
        flagged_turn_count = sum(
            1 for t in turn_results if len(t.flagged_categories) > 0
        )
        overall_score = self._mean_score(turn_results)
        overall_risk = classify_risk(overall_score)
        most_flagged = self._most_flagged_categories(turn_results)

        return ConversationAuditResult(
            turn_results=turn_results,
            total_turns=total_turns,
            flagged_turns=flagged_turn_count,
            overall_score=overall_score,
            overall_risk_level=overall_risk,
            most_flagged_categories=most_flagged,
            audit_timestamp=datetime.now(timezone.utc),
        )

    async def build_report(
        self,
        request: ConversationRequest,
        metadata: dict[str, Any] | None = None,
    ) -> AuditReport:
        """Generate a full downloadable audit report for a conversation.

        Internally calls :meth:`audit_conversation` and then assembles the
        report structure from the results.

        Parameters
        ----------
        request:
            The conversation to audit.
        metadata:
            Optional freeform key-value metadata to embed in the report (e.g.
            app name, developer email, environment).

        Returns
        -------
        AuditReport
            Fully populated report ready for JSON serialisation and download.

        Raises
        ------
        OpenAICallError
            If any OpenAI API call fails.
        ResponseParseError
            If any model response cannot be parsed.
        """
        conv_result = await self.audit_conversation(request)
        return self._build_report_from_result(conv_result, metadata=metadata)

    async def build_report_from_result(
        self,
        conv_result: ConversationAuditResult,
        metadata: dict[str, Any] | None = None,
    ) -> AuditReport:
        """Assemble an :class:`AuditReport` from a pre-computed conversation result.

        Use this when you already have a :class:`ConversationAuditResult` from
        :meth:`audit_conversation` and want to avoid re-calling the API.

        Parameters
        ----------
        conv_result:
            The conversation audit result to convert into a report.
        metadata:
            Optional freeform metadata to embed in the report.

        Returns
        -------
        AuditReport
            Fully populated report ready for JSON serialisation.
        """
        return self._build_report_from_result(conv_result, metadata=metadata)

    # ------------------------------------------------------------------
    # Private helpers — OpenAI interaction
    # ------------------------------------------------------------------

    async def _call_safeguard(self, evaluation_prompt: str) -> str:
        """Call the safeguard model and return the raw response text.

        Parameters
        ----------
        evaluation_prompt:
            The user-turn message to send for evaluation.

        Returns
        -------
        str
            Raw text content of the model's response.

        Raises
        ------
        OpenAICallError
            If the API call fails due to a network error, rate limit, or API
            error.
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": evaluation_prompt},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
            )
        except RateLimitError as exc:
            logger.error("OpenAI rate limit exceeded: %s", exc)
            raise OpenAICallError(
                f"OpenAI rate limit exceeded. Please wait and try again. Detail: {exc}"
            ) from exc
        except APITimeoutError as exc:
            logger.error("OpenAI API timeout: %s", exc)
            raise OpenAICallError(
                f"OpenAI API request timed out. Detail: {exc}"
            ) from exc
        except APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise OpenAICallError(
                f"OpenAI API returned an error. Detail: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error calling OpenAI API: %s", exc)
            raise OpenAICallError(
                f"Unexpected error when calling OpenAI API: {exc}"
            ) from exc

        content = response.choices[0].message.content or ""
        logger.debug("Safeguard raw response: %r", content[:500])
        return content

    # ------------------------------------------------------------------
    # Private helpers — response parsing
    # ------------------------------------------------------------------

    def _parse_safeguard_response(
        self, raw: str
    ) -> dict[str, Any]:
        """Parse the raw model response into a structured dictionary.

        The model is instructed to return only a JSON object.  This method
        handles minor formatting issues (e.g. markdown code fences) and
        validates that all required keys are present.

        Parameters
        ----------
        raw:
            Raw string returned by the safeguard model.

        Returns
        -------
        dict[str, Any]
            Parsed dictionary with keys:
            - ``scores``: dict mapping category key → float
            - ``overall_score``: float
            - ``flagged_categories``: list of str
            - ``reasoning``: str

        Raises
        ------
        ResponseParseError
            If the response cannot be parsed or is missing required fields.
        """
        cleaned = self._strip_markdown_fences(raw.strip())

        try:
            data: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse safeguard response as JSON: %s | raw=%r",
                exc,
                raw[:300],
            )
            # Attempt a best-effort fallback: return neutral scores
            return self._fallback_response(reason=f"JSON parse error: {exc}")

        if not isinstance(data, dict):
            logger.warning("Safeguard response is not a JSON object: %r", raw[:300])
            return self._fallback_response(reason="Response is not a JSON object")

        # Validate and coerce the scores sub-dict
        scores_raw = data.get("scores", {})
        if not isinstance(scores_raw, dict):
            logger.warning("'scores' field is not a dict in response: %r", raw[:300])
            scores_raw = {}

        scores: dict[str, float] = {}
        for cat in PolicyCategory:
            raw_val = scores_raw.get(cat.value)
            scores[cat.value] = self._coerce_score(raw_val)

        overall_score_raw = data.get("overall_score")
        overall_score = self._coerce_score(overall_score_raw)

        flagged_raw = data.get("flagged_categories", [])
        flagged_categories: list[str] = (
            [str(f) for f in flagged_raw]
            if isinstance(flagged_raw, list)
            else []
        )

        reasoning = str(data.get("reasoning", "")).strip()

        return {
            "scores": scores,
            "overall_score": overall_score,
            "flagged_categories": flagged_categories,
            "reasoning": reasoning,
        }

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences if the model wrapped its JSON output.

        Parameters
        ----------
        text:
            Potentially fence-wrapped text.

        Returns
        -------
        str
            Text with leading/trailing fences removed.
        """
        if text.startswith("```"):
            # Strip the opening fence line (e.g. ```json)
            lines = text.splitlines()
            # Drop first line (fence open) and last line if it's a fence close
            start = 1
            end = len(lines)
            if lines and lines[-1].strip() == "```":
                end -= 1
            return "\n".join(lines[start:end]).strip()
        return text

    @staticmethod
    def _coerce_score(value: Any) -> float:
        """Coerce an arbitrary value to a float score in [0.0, 1.0].

        Parameters
        ----------
        value:
            Raw value from the parsed JSON (may be int, float, str, or None).

        Returns
        -------
        float
            A clamped float in [0.0, 1.0].  Returns :data:`_FALLBACK_SCORE` if
            the value cannot be coerced.
        """
        if value is None:
            return _FALLBACK_SCORE
        try:
            score = float(value)
        except (TypeError, ValueError):
            return _FALLBACK_SCORE
        return max(0.0, min(1.0, score))

    @staticmethod
    def _fallback_response(reason: str = "") -> dict[str, Any]:
        """Build a neutral fallback response when parsing fails.

        All category scores are set to :data:`_FALLBACK_SCORE` (0.5, caution
        territory) so that a parse failure results in a conservative CAUTION
        classification rather than silently passing unsafe content.

        Parameters
        ----------
        reason:
            Human-readable reason for the fallback, stored in reasoning.

        Returns
        -------
        dict[str, Any]
            Structured dict compatible with the normal parse output.
        """
        scores = {cat.value: _FALLBACK_SCORE for cat in PolicyCategory}
        return {
            "scores": scores,
            "overall_score": _FALLBACK_SCORE,
            "flagged_categories": [],
            "reasoning": (
                f"Audit could not parse model response; conservative score applied."
                + (f" ({reason})" if reason else "")
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers — score classification and aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_category_scores(
        scores: dict[str, float],
    ) -> list[CategoryScore]:
        """Convert a raw scores dict into a list of :class:`CategoryScore` objects.

        Parameters
        ----------
        scores:
            Mapping of category key string → float score in [0.0, 1.0].

        Returns
        -------
        list[CategoryScore]
            One :class:`CategoryScore` per :class:`PolicyCategory`, in enum
            declaration order.
        """
        result: list[CategoryScore] = []
        for cat in PolicyCategory:
            score = scores.get(cat.value, _FALLBACK_SCORE)
            risk_level = classify_risk(score, cat)
            result.append(
                CategoryScore(
                    category=cat,
                    score=score,
                    risk_level=risk_level,
                )
            )
        return result

    @staticmethod
    def _resolve_overall_score(
        model_overall: float | None,
        category_scores: list[CategoryScore],
    ) -> float:
        """Determine the final overall score for a turn.

        Prefers the model-provided ``overall_score`` if it is a valid float.
        Falls back to the maximum of all category scores, which is the most
        conservative safe choice (a single high-risk category should dominate).

        Parameters
        ----------
        model_overall:
            The ``overall_score`` value from the parsed model response, or None.
        category_scores:
            The list of :class:`CategoryScore` objects for this turn.

        Returns
        -------
        float
            Resolved overall score in [0.0, 1.0].
        """
        if model_overall is not None and 0.0 <= model_overall <= 1.0:
            return model_overall
        if category_scores:
            return max(cs.score for cs in category_scores)
        return _FALLBACK_SCORE

    @staticmethod
    def _collect_flagged_categories(
        category_scores: list[CategoryScore],
    ) -> list[PolicyCategory]:
        """Return categories whose risk level is CAUTION or VIOLATION.

        Parameters
        ----------
        category_scores:
            Per-category scores for a single turn.

        Returns
        -------
        list[PolicyCategory]
            Categories whose risk level is not SAFE, in declaration order.
        """
        flagged: list[PolicyCategory] = []
        for cs in category_scores:
            # cs.risk_level is stored as a string due to use_enum_values
            risk_val = cs.risk_level
            if isinstance(risk_val, RiskLevel):
                risk_val = risk_val.value
            if risk_val in (RiskLevel.CAUTION.value, RiskLevel.VIOLATION.value):
                cat_val = cs.category
                if isinstance(cat_val, PolicyCategory):
                    flagged.append(cat_val)
                else:
                    try:
                        flagged.append(PolicyCategory(cat_val))
                    except ValueError:
                        pass
        return flagged

    @staticmethod
    def _mean_score(turn_results: list[TurnAuditResult]) -> float:
        """Compute the mean overall score across all turn results.

        Parameters
        ----------
        turn_results:
            List of evaluated turns.

        Returns
        -------
        float
            Mean score in [0.0, 1.0], or 0.0 if *turn_results* is empty.
        """
        if not turn_results:
            return 0.0
        return sum(t.overall_score for t in turn_results) / len(turn_results)

    @staticmethod
    def _most_flagged_categories(
        turn_results: list[TurnAuditResult],
        top_n: int = 5,
    ) -> list[PolicyCategory]:
        """Return the most frequently flagged policy categories across all turns.

        Parameters
        ----------
        turn_results:
            List of evaluated turns.
        top_n:
            Maximum number of categories to return (default: 5).

        Returns
        -------
        list[PolicyCategory]
            Up to *top_n* categories ordered by frequency descending.
        """
        counter: Counter[str] = Counter()
        for turn in turn_results:
            for cat in turn.flagged_categories:
                if isinstance(cat, PolicyCategory):
                    counter[cat.value] += 1
                else:
                    counter[str(cat)] += 1

        result: list[PolicyCategory] = []
        for cat_val, _ in counter.most_common(top_n):
            try:
                result.append(PolicyCategory(cat_val))
            except ValueError:
                pass
        return result

    # ------------------------------------------------------------------
    # Private helpers — report assembly
    # ------------------------------------------------------------------

    def _build_report_from_result(
        self,
        conv_result: ConversationAuditResult,
        metadata: dict[str, Any] | None = None,
    ) -> AuditReport:
        """Assemble an :class:`AuditReport` from a :class:`ConversationAuditResult`.

        Parameters
        ----------
        conv_result:
            Pre-computed conversation audit result.
        metadata:
            Optional freeform metadata to embed.

        Returns
        -------
        AuditReport
            Fully populated report.
        """
        turn_results = conv_result.turn_results
        total = conv_result.total_turns

        # Classify each turn
        safe_turns = 0
        caution_turns = 0
        violation_turns = 0
        flagged_summaries: list[FlaggedTurnSummary] = []

        for turn in turn_results:
            risk_val = turn.overall_risk_level
            if isinstance(risk_val, RiskLevel):
                risk_str = risk_val.value
            else:
                risk_str = str(risk_val)

            if risk_str == RiskLevel.SAFE.value and len(turn.flagged_categories) == 0:
                safe_turns += 1
            elif risk_str == RiskLevel.VIOLATION.value:
                violation_turns += 1
            else:
                caution_turns += 1

            # Build flagged summary for any non-safe turn
            if len(turn.flagged_categories) > 0:
                remediation_hints = self._build_remediation_hints(turn.flagged_categories)
                flagged_summaries.append(
                    FlaggedTurnSummary(
                        turn_index=turn.turn_index,
                        role=turn.role,
                        content_snippet=turn.content_snippet,
                        overall_score=turn.overall_score,
                        overall_risk_level=turn.overall_risk_level,  # type: ignore[arg-type]
                        flagged_categories=[
                            PolicyCategory(c) if not isinstance(c, PolicyCategory) else c
                            for c in turn.flagged_categories
                        ],
                        category_scores=turn.category_scores,
                        reasoning=turn.reasoning,
                        remediation_hints=remediation_hints,
                    )
                )

        compliance_rate = (safe_turns / total * 100.0) if total > 0 else 100.0

        summary = AuditReportSummary(
            total_turns=total,
            flagged_turns=conv_result.flagged_turns,
            safe_turns=safe_turns,
            violation_turns=violation_turns,
            caution_turns=caution_turns,
            overall_score=conv_result.overall_score,
            overall_risk_level=conv_result.overall_risk_level,  # type: ignore[arg-type]
            most_flagged_categories=conv_result.most_flagged_categories,  # type: ignore[arg-type]
            compliance_rate=compliance_rate,
        )

        return AuditReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.now(timezone.utc),
            schema_version="1.0",
            summary=summary,
            flagged_turns=flagged_summaries,
            all_turn_results=turn_results,
            policy_reference="https://openai.com/policies/usage-policies#child-safety",
            tool_version=__version__,
            metadata=metadata or {},
        )

    @staticmethod
    def _build_remediation_hints(
        flagged_categories: list[PolicyCategory],
    ) -> dict[str, str]:
        """Build a dict of category key → remediation hint for flagged categories.

        Parameters
        ----------
        flagged_categories:
            List of policy categories that were flagged in a turn.

        Returns
        -------
        dict[str, str]
            Mapping of category value string → remediation hint text.
        """
        hints: dict[str, str] = {}
        for cat in flagged_categories:
            if not isinstance(cat, PolicyCategory):
                try:
                    cat = PolicyCategory(cat)
                except ValueError:
                    continue
            meta = CATEGORY_METADATA.get(cat)
            if meta:
                hints[cat.value] = meta.remediation_hint
        return hints


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def create_engine(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> AuditEngine:
    """Factory function to create a configured :class:`AuditEngine` instance.

    Provides a clean single-call interface for application startup code and
    dependency injection in the FastAPI layer.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Defaults to the ``OPENAI_API_KEY`` environment variable.
    model:
        Safety evaluation model name.  Defaults to environment variables or
        ``gpt-4o-mini``.
    base_url:
        Optional custom base URL for the OpenAI-compatible API.

    Returns
    -------
    AuditEngine
        Configured engine instance.

    Raises
    ------
    AuditEngineError
        If the API key is not available.
    """
    return AuditEngine(api_key=api_key, model=model, base_url=base_url)


__all__: list[str] = [
    "AuditEngine",
    "AuditEngineError",
    "OpenAICallError",
    "ResponseParseError",
    "create_engine",
]
