"""
Deterministic aggregation layer for the ClinicalTrials.gov visualization agent.

This module is responsible for ALL counting, grouping, and mathematical
operations in the pipeline. The LLM never performs aggregation — it only
specifies *what* to aggregate via axis mappings.

Key Responsibilities:
  - Dynamic Pandas groupby/aggregation based on LLM-selected axis mappings.
  - Deep citation generation linking every data point to its source studies.
  - Entity co-occurrence network graph construction for relationship queries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from models import (
    AxisMapping,
    Citation,
    DataPoint,
    NetworkData,
    NetworkEdge,
    NetworkNode,
)

logger = logging.getLogger(__name__)


# ── public API ──────────────────────────────────────────────


def aggregate(
    studies: List[Dict[str, Any]],
    encoding: AxisMapping,
) -> List[DataPoint]:
    """Deterministically aggregate raw study records into visualization-ready data points.

    This function flattens nested study JSON, performs Pandas groupby/aggregation
    based on the LLM-selected axis mapping, and attaches deep citations linking
    each aggregated value back to its contributing studies.

    Args:
        studies:  Raw study JSON objects from the ClinicalTrials.gov API.
        encoding: Axis mapping specifying the grouping field, aggregation
                  function, and optional secondary grouping (color).

    Returns:
        A list of DataPoint objects sorted by value in descending order,
        each carrying citations to the contributing NCT records.
    """
    if not studies:
        return []

    # 1. Flatten nested JSON into a tabular DataFrame
    #    e.g., {"protocolSection": {"statusModule": {"overallStatus": "COMPLETED"}}}
    #    becomes {"protocolSection.statusModule.overallStatus": "COMPLETED"}
    flat_rows = [_flatten(s) for s in studies]
    df = pd.DataFrame(flat_rows)

    # Ensure the x column exists
    x_col = encoding.x
    if x_col not in df.columns:
        logger.warning("x column '%s' not found. Available: %s", x_col, list(df.columns))
        x_col = _best_match(x_col, df.columns)
        if x_col is None:
            raise ValueError(
                f"Cannot find column '{encoding.x}' in flattened data. "
                f"Available columns: {sorted(df.columns)}"
            )

    # Coerce missing x values
    df[x_col] = df[x_col].fillna("Unknown")

    # Handle list-valued x columns (e.g., phases, conditions) by exploding
    if df[x_col].apply(lambda v: isinstance(v, list)).any():
        df = df.explode(x_col)
        df[x_col] = df[x_col].fillna("Unknown")

    # Convert any remaining unhashable types (dicts) to strings for groupby
    df[x_col] = df[x_col].apply(lambda v: str(v) if isinstance(v, (dict, list)) else v)

    # 2. Determine grouping keys for the Pandas aggregation
    group_keys = [x_col]
    color_col = encoding.color
    if color_col and color_col in df.columns:
        df[color_col] = df[color_col].fillna("Unknown")
        if df[color_col].apply(lambda v: isinstance(v, list)).any():
            df = df.explode(color_col)
            df[color_col] = df[color_col].fillna("Unknown")
        group_keys.append(color_col)

    # 3. Build citation index: maps each group label to its contributing studies
    #    This enables deep traceability from any visualized number to source data.
    nct_col = "protocolSection.identificationModule.nctId"
    title_col = "protocolSection.identificationModule.briefTitle"
    citation_map: Dict[Tuple, List[Citation]] = {}

    for idx, row in df.iterrows():
        key = tuple(str(row.get(k, "Unknown")) for k in group_keys)
        nct_id = str(row.get(nct_col, "N/A"))
        excerpt = str(row.get(title_col, ""))[:200]
        # Include the x-value context in the excerpt
        x_val = str(row.get(x_col, ""))
        citation_text = f"[{x_val}] {excerpt}"
        citation_map.setdefault(key, []).append(
            Citation(nct_id=nct_id, excerpt=citation_text)
        )

    # 4. Perform the actual aggregation (count, sum, mean, or nunique)
    agg_func = encoding.y_aggregation.lower()
    y_col = encoding.y

    if agg_func == "count":
        agg_df = df.groupby(group_keys, sort=True).size().reset_index(name="value")
    elif y_col and y_col in df.columns:
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        agg_df = (
            df.groupby(group_keys, sort=True)[y_col]
            .agg(agg_func)
            .reset_index(name="value")
        )
    else:
        # Fallback to count if y column missing
        logger.warning(
            "y column '%s' not found or agg='%s' unsupported; falling back to count.",
            y_col,
            agg_func,
        )
        agg_df = df.groupby(group_keys, sort=True).size().reset_index(name="value")

    # 5. Convert aggregated rows to DataPoint objects with attached citations
    data_points: List[DataPoint] = []
    for _, row in agg_df.iterrows():
        label = str(row[x_col])
        value = row["value"]
        color_group = str(row[color_col]) if color_col and color_col in row.index else None

        key = tuple(str(row.get(k, "Unknown")) for k in group_keys)
        citations = citation_map.get(key, [])

        data_points.append(
            DataPoint(
                label=label,
                value=int(value) if float(value) == int(value) else round(float(value), 2),
                color_group=color_group,
                citations=citations,
            )
        )

    # Sort by value descending for readability
    data_points.sort(key=lambda dp: dp.value, reverse=True)

    return data_points


# ───────────────────────────────────────────────────────────────
# Private Helpers
# ───────────────────────────────────────────────────────────────


def _flatten(obj: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten a nested dictionary using dot-separated keys.

    Example:
        {"a": {"b": {"c": 1}}} → {"a.b.c": 1}

    Args:
        obj:        Nested dictionary to flatten.
        parent_key: Prefix for the current recursion level.
        sep:        Separator character for key segments.

    Returns:
        A flat dictionary with dot-notated keys.
    """
    items: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key, sep).items())
            else:
                items.append((new_key, v))
    return dict(items)


def _best_match(target: str, columns: pd.Index) -> Optional[str]:
    """Attempt fuzzy column matching by comparing tail segments.

    Handles minor path mismatches between what the LLM outputs
    (e.g., 'startDateStruct.date') and the actual flattened column name
    (e.g., 'protocolSection.statusModule.startDateStruct.date').

    Args:
        target:  The column name the LLM specified.
        columns: Available columns in the flattened DataFrame.

    Returns:
        The first matching column name, or None if no match is found.
    """
    target_tail = target.rsplit(".", maxsplit=1)[-1].lower()
    for col in columns:
        if col.lower().endswith(target_tail):
            logger.info("Fuzzy-matched '%s' → '%s'", target, col)
            return col
    return None


# ───────────────────────────────────────────────────────────────
# Network Graph Builder
# ───────────────────────────────────────────────────────────────


def build_network(studies: List[Dict[str, Any]]) -> NetworkData:
    """Build an entity co-occurrence network graph from raw study data.

    Extracts three entity types (conditions, interventions, sponsors) from
    each study and creates nodes and edges representing their co-occurrence
    within clinical trials. Edges are only created between *different* entity
    types (e.g., drug↔condition, sponsor↔drug) to produce a meaningful
    bipartite-like graph.

    Each edge carries deep citations back to the contributing studies,
    maintaining full data traceability.

    Args:
        studies: Raw study JSON objects from the ClinicalTrials.gov API.

    Returns:
        A NetworkData object containing the top 100 edges (by weight)
        and their referenced nodes.
    """
    nodes_map: Dict[str, NetworkNode] = {}
    edges_map: Dict[Tuple[str, str], NetworkEdge] = {}

    nct_col = "protocolSection.identificationModule.nctId"
    title_col = "protocolSection.identificationModule.briefTitle"
    cond_col = "protocolSection.conditionsModule.conditions"
    intr_col = "protocolSection.armsInterventionsModule.interventions"
    sponsor_col = "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"

    for study in studies:
        flat = _flatten(study)
        nct_id = str(flat.get(nct_col, "N/A"))
        excerpt = str(flat.get(title_col, ""))[:200]
        citation = Citation(nct_id=nct_id, excerpt=excerpt)

        # Extract entities
        conditions = flat.get(cond_col, []) or []
        if isinstance(conditions, str):
            conditions = [conditions]

        raw_intrs = flat.get(intr_col, []) or []
        interventions: List[str] = []
        if isinstance(raw_intrs, list):
            for item in raw_intrs:
                if isinstance(item, dict):
                    name = item.get("name", item.get("interventionName", ""))
                    if name:
                        interventions.append(str(name))
                elif isinstance(item, str):
                    interventions.append(item)

        sponsor = flat.get(sponsor_col)
        if isinstance(sponsor, str) and sponsor:
            sponsors = [sponsor]
        else:
            sponsors = []

        # Register nodes
        study_entities: List[Tuple[str, str]] = []  # (id, group)
        for c in conditions[:5]:  # cap to avoid explosion
            nid = f"cond:{c}"
            if nid not in nodes_map:
                nodes_map[nid] = NetworkNode(id=nid, label=c, group="condition", size=0)
            nodes_map[nid].size += 1
            study_entities.append((nid, "condition"))

        for i in interventions[:5]:
            nid = f"intr:{i}"
            if nid not in nodes_map:
                nodes_map[nid] = NetworkNode(id=nid, label=i, group="intervention", size=0)
            nodes_map[nid].size += 1
            study_entities.append((nid, "intervention"))

        for s in sponsors:
            nid = f"sponsor:{s}"
            if nid not in nodes_map:
                nodes_map[nid] = NetworkNode(id=nid, label=s, group="sponsor", size=0)
            nodes_map[nid].size += 1
            study_entities.append((nid, "sponsor"))

        # Create edges between entities that co-occur in the same study
        for i in range(len(study_entities)):
            for j in range(i + 1, len(study_entities)):
                a, grp_a = study_entities[i]
                b, grp_b = study_entities[j]
                if grp_a == grp_b:
                    continue  # only link across entity types
                edge_key = (min(a, b), max(a, b))
                if edge_key not in edges_map:
                    edges_map[edge_key] = NetworkEdge(
                        source=edge_key[0], target=edge_key[1], weight=0
                    )
                edges_map[edge_key].weight += 1
                edges_map[edge_key].citations.append(citation)

    # Sort edges by weight and limit to top 100 for readability
    sorted_edges = sorted(edges_map.values(), key=lambda e: e.weight, reverse=True)[:100]

    # Keep only nodes referenced by the top edges
    referenced_ids = set()
    for e in sorted_edges:
        referenced_ids.add(e.source)
        referenced_ids.add(e.target)
    filtered_nodes = [n for n in nodes_map.values() if n.id in referenced_ids]

    return NetworkData(nodes=filtered_nodes, edges=sorted_edges)
