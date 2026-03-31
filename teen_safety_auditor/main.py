"""teen_safety_auditor/main.py — FastAPI application factory, route registration, and Uvicorn entrypoint.

This module is the top-level entry point for the Teen Safety Auditor web
application.  It:

- Creates and configures the :class:`fastapi.FastAPI` instance via
  :func:`create_app`.
- Mounts static files and wires up Jinja2 template rendering.
- Registers all HTTP routes:
    - ``GET  /``                      — Renders the main single-page UI.
    - ``POST /api/audit/prompt``      — Audits a single prompt/response pair.
    - ``POST /api/audit/conversation``— Audits a multi-turn conversation.
    - ``POST /api/audit/report``      — Generates and downloads a JSON report.
    - ``GET  /health``                — Simple health-check endpoint.
- Provides a :func:`run` function used by the ``teen-safety-auditor`` CLI
  script and ``python -m teen_safety_auditor.main``.

Typical usage::

    # Start via module invocation
    python -m teen_safety_auditor.main

    # Or via installed script
    teen-safety-auditor

    # Or programmatically
    from teen_safety_auditor.main import create_app
    app = create_app()
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from teen_safety_auditor import __version__
from teen_safety_auditor.auditor import (
    AuditEngine,
    AuditEngineError,
    OpenAICallError,
    create_engine,
)
from teen_safety_auditor.models import (
    AuditErrorResponse,
    AuditReport,
    AuditResult,
    ConversationAuditResult,
    ConversationRequest,
    SinglePromptRequest,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_PACKAGE_DIR: Path = Path(__file__).parent
_TEMPLATES_DIR: Path = _PACKAGE_DIR / "templates"
_STATIC_DIR: Path = _PACKAGE_DIR / "static"


# ---------------------------------------------------------------------------
# Application state helpers
# ---------------------------------------------------------------------------


class _AppState:
    """Container for application-wide shared state attached to ``app.state``.

    Attributes
    ----------
    engine:
        The :class:`~teen_safety_auditor.auditor.AuditEngine` instance shared
        across all requests.  Initialised during application startup and torn
        down (if needed) during shutdown.
    """

    def __init__(self) -> None:
        self.engine: AuditEngine | None = None


# ---------------------------------------------------------------------------
# Lifespan context manager (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler — runs startup logic before yielding and
    shutdown logic after.

    On startup:
    - Loads ``.env`` file if present.
    - Instantiates the :class:`~teen_safety_auditor.auditor.AuditEngine`.

    On shutdown:
    - Closes the underlying ``AsyncOpenAI`` HTTP client to avoid resource
      leak warnings.
    """
    # Load .env from the working directory (silently ignored if absent)
    load_dotenv(override=False)

    state: _AppState = app.state  # type: ignore[assignment]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY is not set.  Audit endpoints will return errors "
            "until the key is configured."
        )
        state.engine = None
    else:
        try:
            state.engine = create_engine(
                api_key=api_key,
                model=(
                    os.environ.get("SAFEGUARD_MODEL")
                    or os.environ.get("OPENAI_MODEL")
                ),
            )
            logger.info("AuditEngine started successfully (model=%r).", state.engine._model)
        except AuditEngineError as exc:
            logger.error("Failed to initialise AuditEngine: %s", exc)
            state.engine = None

    yield

    # Shutdown: close the async HTTP client
    if state.engine is not None:
        try:
            await state.engine._client.close()
        except Exception as exc:  # pragma: no cover
            logger.debug("Error closing OpenAI client: %s", exc)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    This factory function is the single place where all application-level
    configuration is applied: metadata, lifespan, middleware, static files,
    templates, and route registration.

    Returns
    -------
    FastAPI
        A fully configured FastAPI application ready to be served by Uvicorn.
    """
    app = FastAPI(
        title="Teen Safety Auditor",
        description=(
            "A developer tool for testing AI prompts and conversation flows "
            "against OpenAI teen safety policy guidelines."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # Attach mutable state container
    app.state = _AppState()  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # ------------------------------------------------------------------
    # Dependency: resolved AuditEngine
    # ------------------------------------------------------------------

    def _get_engine(request: Request) -> AuditEngine:
        """FastAPI dependency that resolves the shared :class:`AuditEngine`.

        Raises
        ------
        HTTPException (503)
            If the engine is not available (e.g. missing API key).
        """
        state: _AppState = request.app.state  # type: ignore[assignment]
        if state.engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Audit engine is not available.  Ensure OPENAI_API_KEY is set "
                    "and restart the server."
                ),
            )
        return state.engine

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    # --- Health check ---------------------------------------------------

    @app.get(
        "/health",
        summary="Health check",
        tags=["meta"],
        response_model=dict,
    )
    async def health_check(request: Request) -> JSONResponse:
        """Return a simple health-check payload.

        Indicates whether the service is running and whether the audit engine
        has been successfully initialised.

        Returns
        -------
        JSONResponse
            ``{"status": "ok", "engine_ready": bool, "version": str}``
        """
        state: _AppState = request.app.state  # type: ignore[assignment]
        return JSONResponse(
            content={
                "status": "ok",
                "engine_ready": state.engine is not None,
                "version": __version__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    # --- Main UI --------------------------------------------------------

    @app.get(
        "/",
        response_class=HTMLResponse,
        summary="Main UI",
        tags=["ui"],
        include_in_schema=False,
    )
    async def index(request: Request) -> HTMLResponse:
        """Render the main single-page auditor UI.

        Returns
        -------
        HTMLResponse
            The rendered ``index.html`` template.
        """
        state: _AppState = request.app.state  # type: ignore[assignment]
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "version": __version__,
                "engine_ready": state.engine is not None,
            },
        )

    # --- Single-prompt audit --------------------------------------------

    @app.post(
        "/api/audit/prompt",
        response_model=AuditResult,
        summary="Audit a single prompt/response pair",
        tags=["audit"],
        responses={
            200: {"description": "Audit result with per-category scores"},
            422: {"description": "Validation error in request body"},
            500: {"model": AuditErrorResponse, "description": "OpenAI API error"},
            503: {"description": "Audit engine not available"},
        },
    )
    async def audit_prompt(
        body: SinglePromptRequest,
        engine: AuditEngine = Depends(_get_engine),
    ) -> AuditResult:
        """Audit a single prompt and optional AI response for teen safety compliance.

        The request is evaluated against all configured policy categories and
        scored using the safeguard model.  Results include per-category risk
        levels and an overall compliance score.

        Parameters
        ----------
        body:
            Request body containing the prompt and optional AI response.
        engine:
            Injected :class:`~teen_safety_auditor.auditor.AuditEngine`.

        Returns
        -------
        AuditResult
            Per-category scores, overall risk level, flagged categories, and
            reasoning.

        Raises
        ------
        HTTPException (500)
            If the OpenAI API call fails.
        HTTPException (503)
            If the audit engine is not available.
        """
        try:
            result = await engine.audit_prompt(body)
        except OpenAICallError as exc:
            logger.error("audit_prompt OpenAI error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error in audit_prompt: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {exc}",
            ) from exc
        return result

    # --- Multi-turn conversation audit ----------------------------------

    @app.post(
        "/api/audit/conversation",
        response_model=ConversationAuditResult,
        summary="Audit a multi-turn conversation",
        tags=["audit"],
        responses={
            200: {"description": "Per-turn compliance breakdown"},
            422: {"description": "Validation error in request body"},
            500: {"model": AuditErrorResponse, "description": "OpenAI API error"},
            503: {"description": "Audit engine not available"},
        },
    )
    async def audit_conversation(
        body: ConversationRequest,
        engine: AuditEngine = Depends(_get_engine),
    ) -> ConversationAuditResult:
        """Audit every turn in a multi-turn conversation for teen safety compliance.

        Each user+assistant turn pair is evaluated holistically.  The response
        includes a per-turn breakdown and aggregated compliance metrics.

        Parameters
        ----------
        body:
            Request body containing an ordered list of conversation turns in
            OpenAI chat format.
        engine:
            Injected :class:`~teen_safety_auditor.auditor.AuditEngine`.

        Returns
        -------
        ConversationAuditResult
            Per-turn results, flagged turn count, overall score, and most
            frequently flagged categories.

        Raises
        ------
        HTTPException (500)
            If any OpenAI API call fails.
        HTTPException (503)
            If the audit engine is not available.
        """
        try:
            result = await engine.audit_conversation(body)
        except OpenAICallError as exc:
            logger.error("audit_conversation OpenAI error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error in audit_conversation: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {exc}",
            ) from exc
        return result

    # --- Report download ------------------------------------------------

    @app.post(
        "/api/audit/report",
        summary="Generate and download a JSON audit report",
        tags=["audit"],
        responses={
            200: {
                "description": "Downloadable JSON audit report",
                "content": {"application/json": {}},
            },
            422: {"description": "Validation error in request body"},
            500: {"model": AuditErrorResponse, "description": "OpenAI API error"},
            503: {"description": "Audit engine not available"},
        },
    )
    async def generate_report(
        body: ConversationRequest,
        engine: AuditEngine = Depends(_get_engine),
    ) -> Response:
        """Generate a full audit report for a conversation and return it as a
        downloadable JSON file attachment.

        The report contains per-turn scores, flagged turn summaries with
        remediation hints, and an overall compliance summary.

        Parameters
        ----------
        body:
            Request body containing an ordered list of conversation turns.
        engine:
            Injected :class:`~teen_safety_auditor.auditor.AuditEngine`.

        Returns
        -------
        Response
            A ``200 OK`` response with ``Content-Type: application/json`` and
            ``Content-Disposition: attachment`` so browsers trigger a file
            download.

        Raises
        ------
        HTTPException (500)
            If any OpenAI API call fails.
        HTTPException (503)
            If the audit engine is not available.
        """
        try:
            report: AuditReport = await engine.build_report(body)
        except OpenAICallError as exc:
            logger.error("generate_report OpenAI error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error in generate_report: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {exc}",
            ) from exc

        # Serialise the report to JSON using Pydantic's model_dump with
        # mode="json" to ensure datetimes are rendered as ISO strings.
        report_data: dict[str, Any] = report.model_dump(mode="json")
        report_json: str = json.dumps(report_data, indent=2, ensure_ascii=False)

        timestamp_str = (
            report.generated_at.strftime("%Y%m%d_%H%M%S")
            if report.generated_at
            else "report"
        )
        filename = f"teen_safety_audit_{timestamp_str}.json"

        return Response(
            content=report_json,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    # --- HTMX partial: audit prompt ------------------------------------
    # This endpoint is called by the HTMX form on the main UI and returns
    # an HTML partial (result_card) rather than JSON.

    @app.post(
        "/htmx/audit/prompt",
        response_class=HTMLResponse,
        summary="HTMX endpoint: audit a single prompt and return result card HTML",
        tags=["htmx"],
        include_in_schema=False,
    )
    async def htmx_audit_prompt(
        request: Request,
        engine: AuditEngine = Depends(_get_engine),
    ) -> HTMLResponse:
        """Handle form submission from the HTMX single-prompt UI.

        Reads ``prompt`` and ``response`` from the multipart/form-data body,
        runs the audit, and returns a rendered ``result_card.html`` partial
        suitable for HTMX ``hx-swap`` injection.

        Returns
        -------
        HTMLResponse
            The rendered ``partials/result_card.html`` template.
        """
        form = await request.form()
        prompt_text = str(form.get("prompt", "")).strip()
        response_text = str(form.get("response", "")).strip() or None

        if not prompt_text:
            return templates.TemplateResponse(
                request=request,
                name="partials/result_card.html",
                context={
                    "error": "Prompt must not be empty.",
                    "result": None,
                    "is_conversation": False,
                },
                status_code=422,
            )

        try:
            req = SinglePromptRequest(prompt=prompt_text, response=response_text)
            result = await engine.audit_prompt(req)
        except OpenAICallError as exc:
            logger.error("htmx_audit_prompt error: %s", exc)
            return templates.TemplateResponse(
                request=request,
                name="partials/result_card.html",
                context={
                    "error": f"OpenAI API error: {exc}",
                    "result": None,
                    "is_conversation": False,
                },
                status_code=500,
            )
        except Exception as exc:
            logger.exception("Unexpected htmx_audit_prompt error: %s", exc)
            return templates.TemplateResponse(
                request=request,
                name="partials/result_card.html",
                context={
                    "error": f"Unexpected error: {exc}",
                    "result": None,
                    "is_conversation": False,
                },
                status_code=500,
            )

        return templates.TemplateResponse(
            request=request,
            name="partials/result_card.html",
            context={
                "error": None,
                "result": result.model_dump(mode="json"),
                "is_conversation": False,
            },
        )

    # --- HTMX partial: audit conversation ------------------------------

    @app.post(
        "/htmx/audit/conversation",
        response_class=HTMLResponse,
        summary="HTMX endpoint: audit a conversation and return result card HTML",
        tags=["htmx"],
        include_in_schema=False,
    )
    async def htmx_audit_conversation(
        request: Request,
        engine: AuditEngine = Depends(_get_engine),
    ) -> HTMLResponse:
        """Handle form submission from the HTMX multi-turn conversation UI.

        Reads a JSON-encoded ``conversation`` field from the form body, parses
        the turns, runs the audit, and returns rendered HTML.

        Returns
        -------
        HTMLResponse
            The rendered ``partials/result_card.html`` template.
        """
        form = await request.form()
        conversation_raw = str(form.get("conversation", "")).strip()

        if not conversation_raw:
            return templates.TemplateResponse(
                request=request,
                name="partials/result_card.html",
                context={
                    "error": "Conversation JSON must not be empty.",
                    "result": None,
                    "is_conversation": True,
                },
                status_code=422,
            )

        # Parse the JSON conversation array
        try:
            raw_turns: list[dict[str, Any]] = json.loads(conversation_raw)
            if not isinstance(raw_turns, list):
                raise ValueError("Conversation must be a JSON array.")
        except (json.JSONDecodeError, ValueError) as exc:
            return templates.TemplateResponse(
                request=request,
                name="partials/result_card.html",
                context={
                    "error": f"Invalid JSON: {exc}",
                    "result": None,
                    "is_conversation": True,
                },
                status_code=422,
            )

        try:
            conv_request = ConversationRequest(turns=raw_turns)  # type: ignore[arg-type]
        except Exception as exc:
            return templates.TemplateResponse(
                request=request,
                name="partials/result_card.html",
                context={
                    "error": f"Invalid conversation format: {exc}",
                    "result": None,
                    "is_conversation": True,
                },
                status_code=422,
            )

        try:
            result = await engine.audit_conversation(conv_request)
        except OpenAICallError as exc:
            logger.error("htmx_audit_conversation error: %s", exc)
            return templates.TemplateResponse(
                request=request,
                name="partials/result_card.html",
                context={
                    "error": f"OpenAI API error: {exc}",
                    "result": None,
                    "is_conversation": True,
                },
                status_code=500,
            )
        except Exception as exc:
            logger.exception("Unexpected htmx_audit_conversation error: %s", exc)
            return templates.TemplateResponse(
                request=request,
                name="partials/result_card.html",
                context={
                    "error": f"Unexpected error: {exc}",
                    "result": None,
                    "is_conversation": True,
                },
                status_code=500,
            )

        return templates.TemplateResponse(
            request=request,
            name="partials/result_card.html",
            context={
                "error": None,
                "result": result.model_dump(mode="json"),
                "is_conversation": True,
            },
        )

    return app


# ---------------------------------------------------------------------------
# Application singleton — used by Uvicorn when invoked as a module
# ---------------------------------------------------------------------------

app: FastAPI = create_app()
"""
The FastAPI application instance.  Import this when running with Uvicorn directly::

    uvicorn teen_safety_auditor.main:app --reload
"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run() -> None:
    """Start the Uvicorn development server.

    This function is registered as the ``teen-safety-auditor`` CLI script in
    ``pyproject.toml`` and is also invoked when the module is run directly
    (``python -m teen_safety_auditor.main``).

    Configuration is read from environment variables:

    - ``HOST`` — bind address (default: ``127.0.0.1``)
    - ``PORT`` — bind port (default: ``8000``)
    - ``DEBUG`` — enable Uvicorn auto-reload (default: ``false``)
    """
    import uvicorn

    # Load .env before reading config so that environment overrides work
    load_dotenv(override=False)

    host: str = os.environ.get("HOST", "127.0.0.1")
    port: int = int(os.environ.get("PORT", "8000"))
    debug: bool = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")

    log_level = "debug" if debug else "info"

    logger.info(
        "Starting Teen Safety Auditor v%s on %s:%d (reload=%s)",
        __version__,
        host,
        port,
        debug,
    )

    uvicorn.run(
        "teen_safety_auditor.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=log_level,
    )


if __name__ == "__main__":
    run()
