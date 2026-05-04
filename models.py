"""
Pydantic schema definitions for the ClinicalTrials.gov Query-to-Visualization Agent.

This module defines the data contracts at every stage of the agentic pipeline:

  1. **Request**  — User-facing input schema with explicit optional filters.
  2. **Intent**   — Structured LLM output representing extracted query intent.
  3. **Response** — Final visualization payload with deep citations.

Design Rationale:
  - Three-layer schema ensures strict separation between the external API
    contract and internal LLM output format, allowing either to evolve
    independently without breaking the other.
  - All models use Pydantic v2 for runtime validation and automatic
    JSON Schema generation (visible at /docs).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# 1. Request models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """User-facing request schema for the POST /query endpoint.

    Supports both free-text natural-language queries and explicit structured
    filters. When both are provided, the explicit fields take priority over
    LLM-extracted values (see agent._merge_user_fields).
    """

    query: str = Field(
        ...,
        description="Natural-language question about clinical trials.",
        min_length=3,
        examples=["How many Phase 3 cancer trials started each year?"],
    )
    drug_name: Optional[str] = Field(
        default=None,
        description="Filter by drug / intervention name (e.g. 'Pembrolizumab').",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Filter by medical condition or disease (e.g. 'breast cancer').",
    )
    trial_phase: Optional[str] = Field(
        default=None,
        description="Filter by trial phase: 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'.",
    )
    sponsor: Optional[str] = Field(
        default=None,
        description="Filter by lead sponsor name.",
    )
    start_year: Optional[int] = Field(
        default=None,
        ge=1990,
        le=2030,
        description="Include only trials starting on or after this year.",
    )
    end_year: Optional[int] = Field(
        default=None,
        ge=1990,
        le=2030,
        description="Include only trials starting on or before this year.",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of studies to retrieve from ClinicalTrials.gov.",
    )
    filters: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional raw filters forwarded directly to the CT.gov API.",
    )


# ──────────────────────────────────────────────
# 2. LLM structured-output models (Query Intent)
# ──────────────────────────────────────────────

class VisualizationType(str, Enum):
    """Enumeration of supported visualization types.

    The LLM selects one of these based on the user's question semantics.
    Each type maps to a distinct rendering strategy in the frontend.
    """

    BAR_CHART = "bar_chart"
    TIME_SERIES = "time_series"
    PIE_CHART = "pie_chart"
    HISTOGRAM = "histogram"
    SCATTER_PLOT = "scatter_plot"
    NETWORK_GRAPH = "network_graph"
    TABLE = "table"


class AxisMapping(BaseModel):
    """Specification of how raw data fields map to chart axes.

    The LLM populates this using dot-notated paths into the ClinicalTrials.gov
    v2 study JSON structure (e.g., 'protocolSection.statusModule.overallStatus').
    """

    x: str = Field(
        ...,
        description="Dot-notated path in the CT.gov study JSON to use as the x-axis "
                    "or grouping dimension (e.g. 'protocolSection.statusModule.overallStatus').",
    )
    y_aggregation: str = Field(
        default="count",
        description="Aggregation function to apply: 'count', 'sum', 'mean', 'nunique'.",
    )
    y: Optional[str] = Field(
        default=None,
        description="Dot-notated path for the value to aggregate (required for sum/mean). "
                    "Leave None when y_aggregation='count'.",
    )
    color: Optional[str] = Field(
        default=None,
        description="Optional dot-notated path for a secondary grouping (stacked / colored).",
    )


class QueryIntent(BaseModel):
    """Structured output the LLM produces via the Instructor library.

    The LLM acts exclusively as a *router/extractor*: it translates the
    user's free-text query into deterministic search parameters and a
    visualization specification. It must NOT compute any numerical values.

    This model is internal to the pipeline and never exposed to the user.
    """

    search_expression: str = Field(
        ...,
        description="Free-text search string to pass as 'query.term' to the CT.gov API.",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Medical condition filter for 'query.cond'.",
    )
    intervention: Optional[str] = Field(
        default=None,
        description="Intervention/treatment filter for 'query.intr'.",
    )
    phase: List[str] = Field(
        default_factory=list,
        description="Phase filters, e.g. ['PHASE3']. "
                    "Valid values: EARLY_PHASE1, PHASE1, PHASE2, PHASE3, PHASE4, NA.",
    )
    status: List[str] = Field(
        default_factory=list,
        description="Overall status filters, e.g. ['RECRUITING', 'COMPLETED'].",
    )
    visualization_type: VisualizationType = Field(
        ...,
        description="The chart type best suited for answering the user's question.",
    )
    title: str = Field(
        ...,
        description="A concise, human-readable chart title.",
    )
    encoding: AxisMapping = Field(
        ...,
        description="Mapping of data fields to chart axes.",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Any assumptions the LLM made when interpreting the query.",
    )


# ──────────────────────────────────────────────
# 3. Response models (Visualization payload)
# ──────────────────────────────────────────────

class Citation(BaseModel):
    """Deep citation linking an aggregated data point back to its source study.

    Every data point in the response carries a list of citations, enabling
    full traceability from any visualized number to the underlying NCT records.
    """

    nct_id: str = Field(..., description="ClinicalTrials.gov NCT identifier.")
    excerpt: str = Field(
        ...,
        description="Short text excerpt from the raw API response justifying inclusion.",
    )


class DataPoint(BaseModel):
    """Single aggregated data point in the visualization output.

    Represents one bar, slice, or line point in the chart, along with
    the list of source studies that contributed to this value.
    """

    label: str = Field(..., description="The category or time-bucket label.")
    value: Union[float, int] = Field(..., description="The computed metric value.")
    color_group: Optional[str] = Field(
        default=None,
        description="Secondary grouping label (for stacked / multi-series charts).",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Studies that contributed to this data point.",
    )


# ── Network Graph Models ────────────────────────


class NetworkNode(BaseModel):
    """A node in the entity co-occurrence network graph.

    Nodes represent clinical entities (conditions, interventions, sponsors)
    extracted from the fetched studies.
    """

    id: str = Field(..., description="Unique node identifier.")
    label: str = Field(..., description="Display label.")
    group: str = Field(..., description="Node type: 'condition', 'intervention', 'sponsor'.")
    size: int = Field(default=1, description="Weight / count for sizing.")


class NetworkEdge(BaseModel):
    """An edge representing co-occurrence between two entities.

    Edges are created when two entities (e.g., a drug and a condition)
    appear in the same study. The weight reflects how many studies
    link these entities, and each edge carries deep citations.
    """

    source: str = Field(..., description="Source node id.")
    target: str = Field(..., description="Target node id.")
    weight: int = Field(default=1, description="Number of studies linking these entities.")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Studies that contribute to this edge.",
    )


class NetworkData(BaseModel):
    """Node + edge data for network graph visualizations."""

    nodes: List[NetworkNode] = Field(default_factory=list)
    edges: List[NetworkEdge] = Field(default_factory=list)


class Encoding(BaseModel):
    """Axis encodings in the response (human-readable)."""

    x: str
    y: str
    color: Optional[str] = None


class VisualizationPayload(BaseModel):
    """The visualization section of the response."""

    type: VisualizationType
    title: str
    encoding: Encoding
    data: List[DataPoint]
    network_data: Optional[NetworkData] = Field(
        default=None,
        description="Node and edge data (populated only when type=network_graph).",
    )


class MetaInfo(BaseModel):
    """Request metadata included in every response for transparency.

    Provides the user with full visibility into what was searched,
    which filters were applied, and any assumptions the LLM made.
    """

    source: str = Field(default="clinicaltrials.gov", description="Data source.")
    search_expression: str
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    total_studies_fetched: int
    assumptions: List[str] = Field(default_factory=list)


class VisualizationResponse(BaseModel):
    """Top-level response returned to the user."""

    visualization: VisualizationPayload
    meta: MetaInfo
