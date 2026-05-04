"""
Pipeline orchestrator for the ClinicalTrials.gov Query-to-Visualization Agent.

This module implements the end-to-end agentic pipeline:

  1. **Intent Extraction** — The LLM (via Instructor) translates the user's
     natural-language query into a structured ``QueryIntent``.
  2. **Field Merging** — User-supplied explicit filters override LLM-extracted
     values, ensuring user intent always takes precedence.
  3. **Data Fetching** — ``client.py`` retrieves raw study records from the
     ClinicalTrials.gov v2 API.
  4. **Aggregation** — ``aggregator.py`` performs deterministic Pandas-based
     grouping and counting with deep citations.

Anti-Hallucination Guarantee:
  The LLM is invoked **exactly once** (step 1). All numerical values in the
  final response are computed deterministically by Pandas in step 4.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import instructor
from openai import AsyncOpenAI

from aggregator import aggregate, build_network
from client import fetch_studies
from models import (
    Encoding,
    MetaInfo,
    NetworkData,
    QueryIntent,
    QueryRequest,
    VisualizationPayload,
    VisualizationResponse,
    VisualizationType,
)

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# LLM Client Initialisation
# ───────────────────────────────────────────────────────────────

_client: Optional[instructor.AsyncInstructor] = None


def _get_llm_client() -> instructor.AsyncInstructor:
    """Return a cached, Instructor-patched AsyncOpenAI client.

    The client is lazily initialised on first call to avoid import-time
    side-effects and to ensure the API key is available at runtime.

    Raises:
        RuntimeError: If the OPENAI_API_KEY environment variable is not set.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please export it before starting the server."
            )
        _client = instructor.from_openai(AsyncOpenAI(api_key=api_key))
    return _client


SYSTEM_PROMPT = """\
You are a clinical-trials search assistant.  Your ONLY job is to translate a
user's natural-language question into structured search parameters and a
visualization specification.

RULES — read carefully:
• You must NEVER perform any counting, math, or aggregation.
• You must NEVER invent data or statistics.
• For the axis mapping, use dot-notated paths that exist in the ClinicalTrials.gov
  v2 study JSON.  Common paths:
    - protocolSection.statusModule.overallStatus
    - protocolSection.statusModule.startDateStruct.date  (YYYY-MM or YYYY-MM-DD)
    - protocolSection.designModule.phases               (list of phase strings)
    - protocolSection.designModule.studyType
    - protocolSection.conditionsModule.conditions        (list)
    - protocolSection.armsInterventionsModule.interventions (list of objects)
    - protocolSection.sponsorCollaboratorsModule.leadSponsor.name
    - protocolSection.designModule.enrollmentInfo.count  (integer)

VISUALIZATION TYPE SELECTION — you MUST follow these rules strictly:

1. Use "time_series" when the question contains ANY of these signals:
   - "per year", "each year", "over time", "trend", "timeline", "yearly",
     "annual", "since", "over the last", "started each", "growth", "changed over"
   - Set x to: protocolSection.statusModule.startDateStruct.date

2. Use "network_graph" when the question asks about RELATIONSHIPS:
   - "relationships", "connections", "links between", "network",
     "which drugs treat", "drug-condition", "sponsor-condition"

3. Use "pie_chart" when the question asks about PROPORTIONS:
   - "proportion", "percentage", "share", "pie", "makeup"

4. Use "scatter_plot" when the question asks to CORRELATE two fields:
   - "correlate", "relationship between enrollment and", "scatter"

5. Use "bar_chart" for all other questions:
   - "distribution", "breakdown", "by status", "by phase", "compare",
     "top sponsors", "how many"

• Keep `search_expression` focused and concise.
• List any assumptions you make in the `assumptions` field.
• When the user says "by status", do NOT put status values in the `status` filter —
  leave it empty so the aggregator can group by all statuses.
"""



async def extract_intent(query: str) -> QueryIntent:
    """Invoke the LLM to extract a structured QueryIntent from the user's query.

    This is the **only** LLM call in the entire pipeline. The LLM receives
    a system prompt constraining it to act purely as a router/extractor
    and returns a Pydantic-validated ``QueryIntent`` via the Instructor library.

    Args:
        query: The user's natural-language question.

    Returns:
        A validated QueryIntent containing search parameters and
        visualization specification.
    """
    client = _get_llm_client()
    intent: QueryIntent = await client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        response_model=QueryIntent,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_retries=2,
    )
    logger.info("Extracted intent: %s", intent.model_dump_json(indent=2))
    return intent

# ───────────────────────────────────────────────────────────────
# Main Pipeline
# ───────────────────────────────────────────────────────────────


async def run_pipeline(request: QueryRequest) -> VisualizationResponse:
    """Execute the end-to-end query-to-visualization pipeline.

    Pipeline stages:
      1. LLM extracts intent (the ONLY LLM call in the system).
      2. User-supplied explicit filters are merged into the intent.
      3. Raw studies are fetched from ClinicalTrials.gov.
      4. Pandas performs deterministic aggregation with deep citations.
      5. The result is assembled into a VisualizationResponse.

    Args:
        request: Validated user request from the API endpoint.

    Returns:
        A complete VisualizationResponse with chart data and metadata.
    """

    # Step 1 — LLM extracts intent (the ONLY LLM call)
    intent = await extract_intent(request.query)

    # Step 1b — Merge user-supplied structured fields into intent
    _merge_user_fields(intent, request)

    # Step 2 — Deterministic data fetch
    studies = await fetch_studies(
        intent=intent,
        max_results=request.max_results,
        extra_filters=request.filters or None,
        start_year=request.start_year,
        end_year=request.end_year,
    )
    logger.info("Fetched %d studies from ClinicalTrials.gov", len(studies))

    if not studies:
        return _empty_response(intent, request)

    # Step 3 — Deterministic aggregation (NO LLM)
    network_data = None
    if intent.visualization_type == VisualizationType.NETWORK_GRAPH:
        network_data = build_network(studies)

    data_points = aggregate(studies, intent.encoding)

    # Step 4 — Assemble response
    encoding = Encoding(
        x=_humanise(intent.encoding.x),
        y=f"{intent.encoding.y_aggregation}({_humanise(intent.encoding.y or 'studies')})",
        color=_humanise(intent.encoding.color) if intent.encoding.color else None,
    )

    filters_applied: Dict[str, Any] = {}
    if intent.condition:
        filters_applied["condition"] = intent.condition
    if intent.intervention:
        filters_applied["intervention"] = intent.intervention
    if intent.phase:
        filters_applied["phase"] = intent.phase
    if intent.status:
        filters_applied["status"] = intent.status
    if request.drug_name:
        filters_applied["drug_name"] = request.drug_name
    if request.sponsor:
        filters_applied["sponsor"] = request.sponsor
    if request.start_year:
        filters_applied["start_year"] = request.start_year
    if request.end_year:
        filters_applied["end_year"] = request.end_year
    if request.filters:
        filters_applied["raw_overrides"] = request.filters

    return VisualizationResponse(
        visualization=VisualizationPayload(
            type=intent.visualization_type,
            title=intent.title,
            encoding=encoding,
            data=data_points,
            network_data=network_data,
        ),
        meta=MetaInfo(
            search_expression=intent.search_expression,
            filters_applied=filters_applied,
            total_studies_fetched=len(studies),
            assumptions=intent.assumptions,
        ),
    )


# ───────────────────────────────────────────────────────────────
# Private Helpers
# ───────────────────────────────────────────────────────────────


def _humanise(dotpath: str) -> str:
    """Extract the final segment of a dot-notated JSON path for display.

    Example:
        'protocolSection.statusModule.overallStatus' → 'overallStatus'
    """
    return dotpath.rsplit(".", maxsplit=1)[-1] if "." in dotpath else dotpath


def _merge_user_fields(intent: QueryIntent, request: QueryRequest) -> None:
    """Override LLM-extracted intent fields with explicit user-supplied values.

    User-provided structured filters (drug_name, condition, trial_phase,
    sponsor) take precedence over LLM interpretation because they represent
    deliberate, explicit user intent. This is a key anti-hallucination measure.

    Args:
        intent:  Mutable QueryIntent to update in place.
        request: The original user request containing optional filter fields.
    """
    PHASE_MAP = {
        "phase 1": "PHASE1", "phase1": "PHASE1", "1": "PHASE1",
        "phase 2": "PHASE2", "phase2": "PHASE2", "2": "PHASE2",
        "phase 3": "PHASE3", "phase3": "PHASE3", "3": "PHASE3",
        "phase 4": "PHASE4", "phase4": "PHASE4", "4": "PHASE4",
    }
    if request.condition:
        intent.condition = request.condition
    if request.drug_name:
        intent.intervention = request.drug_name
    if request.trial_phase:
        mapped = PHASE_MAP.get(request.trial_phase.lower(), request.trial_phase.upper())
        intent.phase = [mapped]
    if request.sponsor:
        # Sponsor is appended to the search expression
        intent.search_expression = f"{intent.search_expression} {request.sponsor}".strip()


def _empty_response(intent: QueryIntent, request: QueryRequest) -> VisualizationResponse:
    """Construct a well-formed but empty response when no studies match the query.

    Returns a valid VisualizationResponse with an empty data array and
    an additional assumption noting that no studies were found.
    """
    return VisualizationResponse(
        visualization=VisualizationPayload(
            type=intent.visualization_type,
            title=intent.title,
            encoding=Encoding(
                x=_humanise(intent.encoding.x),
                y=f"{intent.encoding.y_aggregation}({_humanise(intent.encoding.y or 'studies')})",
                color=_humanise(intent.encoding.color) if intent.encoding.color else None,
            ),
            data=[],
        ),
        meta=MetaInfo(
            search_expression=intent.search_expression,
            filters_applied=request.filters,
            total_studies_fetched=0,
            assumptions=intent.assumptions + ["No studies matched the query."],
        ),
    )
