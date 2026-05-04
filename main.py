"""
FastAPI application entry point for the ClinicalTrials.gov
Query-to-Visualization Agent.

This module configures the web server, defines the API endpoints,
and mounts the static frontend. The primary endpoint (POST /query)
delegates all processing to the agentic pipeline in ``agent.py``.
"""

from __future__ import annotations

import logging
import pathlib

from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from agent import run_pipeline
from models import QueryRequest, VisualizationResponse

STATIC_DIR = pathlib.Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ClinicalTrials.gov Query-to-Visualization Agent",
    description="Natural-language query → structured JSON visualization spec with deep citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect the root URL to the interactive dashboard UI."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint for monitoring and container orchestration."""
    return {"status": "ok"}


@app.post("/query", response_model=VisualizationResponse)
async def query(request: QueryRequest) -> VisualizationResponse:
    """Main endpoint: accept a natural-language query and return a visualization spec.

    Delegates to the agentic pipeline which performs:
      LLM intent extraction → CT.gov API fetch → Pandas aggregation → JSON response.

    Raises:
        HTTPException 422: If the query cannot be processed (e.g., invalid encoding).
        HTTPException 500: If an internal error occurs during pipeline execution.
    """
    try:
        return await run_pipeline(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled pipeline error")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


# Mount static files AFTER API routes so /query takes priority
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
