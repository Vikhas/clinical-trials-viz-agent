"""
Asynchronous HTTP client for the ClinicalTrials.gov v2 API.

This module handles all communication with the external API. It is fully
deterministic — the LLM never touches the network. Key responsibilities:

  - Mapping structured QueryIntent fields to CT.gov query parameters.
  - Cursor-based pagination to retrieve up to ``max_results`` studies.
  - Building advanced filter expressions (phase, status, date range).
  - Graceful degradation on 400 errors (retries without advanced filters).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from models import QueryIntent

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Fields requested from the API to minimise payload size while covering
# the most commonly visualised dimensions (identification, status, design,
# conditions, interventions, sponsor, and eligibility).
_DEFAULT_FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.statusModule.startDateStruct",
    "protocolSection.statusModule.completionDateStruct",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.enrollmentInfo",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
    "protocolSection.eligibilityModule.sex",
    "protocolSection.eligibilityModule.minimumAge",
    "protocolSection.eligibilityModule.maximumAge",
]


async def fetch_studies(
    intent: QueryIntent,
    max_results: int = 100,
    extra_filters: Optional[Dict[str, str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch raw study records from the ClinicalTrials.gov v2 API.

    Translates the structured QueryIntent into API query parameters,
    handles cursor-based pagination, and returns up to ``max_results``
    study records.

    If the API returns a 400 error due to an invalid advanced filter
    expression, the request is retried without advanced filters as a
    graceful degradation strategy.

    Args:
        intent:        Structured search parameters extracted by the LLM.
        max_results:   Upper bound on the number of studies to retrieve.
        extra_filters: Additional query-string params forwarded verbatim.
        start_year:    Include only trials starting on or after this year.
        end_year:      Include only trials starting on or before this year.

    Returns:
        A list of raw study dictionaries from the API, trimmed to
        ``max_results``.
    """
    params: Dict[str, Any] = {
        "format": "json",
        "pageSize": min(max_results, 100),  # API cap per page
        "fields": "|".join(_DEFAULT_FIELDS),
    }

    # Map intent fields → CT.gov query params
    if intent.search_expression:
        params["query.term"] = intent.search_expression
    if intent.condition:
        params["query.cond"] = intent.condition
    if intent.intervention:
        params["query.intr"] = intent.intervention

    # Build advanced filter: phases use AND (can combine), statuses use OR
    # (a trial has exactly one status).
    advanced_parts = _build_advanced_filter(
        intent.phase, intent.status, start_year, end_year
    )
    if advanced_parts:
        params["filter.advanced"] = advanced_parts

    # Merge any hard user-provided filters (override collisions)
    if extra_filters:
        params.update(extra_filters)

    studies: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(studies) < max_results:
            if next_page_token:
                params["pageToken"] = next_page_token

            logger.info("GET %s  params=%s", BASE_URL, params)
            try:
                response = await client.get(BASE_URL, params=params)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # If the advanced filter caused a 400, retry without it
                if exc.response.status_code == 400 and "filter.advanced" in params:
                    logger.warning(
                        "CT.gov returned 400 with filter.advanced='%s'. "
                        "Retrying without advanced filters.",
                        params["filter.advanced"],
                    )
                    params.pop("filter.advanced", None)
                    response = await client.get(BASE_URL, params=params)
                    response.raise_for_status()
                else:
                    raise

            payload = response.json()

            batch = payload.get("studies", [])
            if not batch:
                break

            studies.extend(batch)
            next_page_token = payload.get("nextPageToken")
            if not next_page_token:
                break

    # Trim to requested size
    return studies[:max_results]


# ───────────────────────────────────────────────────────────────
# Private Helpers
# ───────────────────────────────────────────────────────────────


def _build_advanced_filter(
    phases: List[str],
    statuses: List[str],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> str:
    """Build a CT.gov v2 advanced filter expression.

    Combines phase, status, and date range filters into a single
    expression string. Phases are AND-joined (narrowing), statuses
    are OR-joined (a study has exactly one status at a time).

    Args:
        phases:     Phase filter values (e.g., ['PHASE3']).
        statuses:   Status filter values (e.g., ['RECRUITING', 'COMPLETED']).
        start_year: Minimum trial start year (inclusive).
        end_year:   Maximum trial start year (inclusive).

    Returns:
        A filter expression string, or empty string if no filters apply.
    """
    parts: List[str] = []

    # Phase filters (AND — each narrows results)
    for phase in phases:
        parts.append(f"AREA[Phase]{phase}")

    # Status filters (OR — a study has exactly one status)
    if statuses:
        status_exprs = [f"AREA[OverallStatus]{s}" for s in statuses]
        if len(status_exprs) == 1:
            parts.append(status_exprs[0])
        else:
            parts.append("(" + " OR ".join(status_exprs) + ")")

    # Date range filter
    if start_year or end_year:
        date_min = f"{start_year}-01-01" if start_year else "MIN"
        date_max = f"{end_year}-12-31" if end_year else "MAX"
        parts.append(f"AREA[StartDate]RANGE[{date_min},{date_max}]")

    return " AND ".join(parts) if parts else ""
