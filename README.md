# ClinicalTrials.gov Query-to-Visualization Agent

An agentic backend that accepts a **natural-language question** about clinical trials and returns a **strict JSON visualization specification** — complete with deep citations linking every data point back to the source studies on [ClinicalTrials.gov](https://clinicaltrials.gov).

## Demo

https://github.com/user-attachments/assets/demo.mp4

<video src="demo.mp4" controls width="100%"></video>

> *If the video doesn't render above, download and view [`demo.mp4`](demo.mp4) directly.*

---

## Table of Contents

1. [System Design](#system-design)
2. [How to Run](#how-to-run)
3. [Request Schema (Input)](#request-schema-input)
4. [Response Schema (Output)](#response-schema-output)
5. [Supported Visualizations](#supported-visualizations)
6. [Pagination](#pagination)
7. [Example Runs](#example-runs)
8. [Key Design Decisions & Tradeoffs](#key-design-decisions--tradeoffs)
9. [Limitations & Future Improvements](#limitations--future-improvements)
10. [File Structure](#file-structure)
11. [Tools & Integrity Note](#tools--integrity-note)

---

## System Design

### High-Level Architecture

```
┌───────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  User Query   │────▶│  LLM (Router)   │────▶│  CT.gov v2 API   │────▶│  Pandas Agg   │
│  (POST /query)│     │  Extract intent  │     │  Fetch raw data  │     │  Deterministic│
└───────────────┘     └─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                                                │
                                                                    ┌───────────▼───────────┐
                                                                    │  VisualizationResponse │
                                                                    │  (JSON + citations)    │
                                                                    └────────────────────────┘
```

### Anti-Hallucination Architecture

The system enforces a strict **separation of concerns** to prevent the LLM from hallucinating numbers:

| Layer | Role | What It Does NOT Do |
|-------|------|---------------------|
| **LLM** (`agent.py`) | Router / intent extractor only | Never counts, sums, or aggregates data |
| **Client** (`client.py`) | Deterministic HTTP fetch with pagination | Never interprets or transforms data |
| **Aggregator** (`aggregator.py`) | Pandas groupby / aggregation | Never calls the LLM |

The LLM is called **exactly once** per request to produce a `QueryIntent` (a Pydantic model via the `instructor` library). Every number in the response is computed deterministically by Pandas from real API data.

### Pipeline Steps

1. **Extract Intent** — The LLM translates the user's natural-language query into a structured `QueryIntent` containing search parameters, visualization type, and axis mappings. This is the **only** LLM call in the system.
2. **Merge User Fields** — Explicit user-provided structured fields (`drug_name`, `condition`, `trial_phase`, `sponsor`, `start_year`, `end_year`) override LLM-extracted values, ensuring user intent always takes precedence.
3. **Fetch Data** — The deterministic HTTP client maps intent fields to ClinicalTrials.gov v2 API parameters and paginates through results using cursor-based tokens.
4. **Aggregate** — Pandas performs all grouping, counting, and math. For `network_graph` queries, a separate builder creates node/edge data from entity co-occurrences across studies.
5. **Respond** — Results are packaged into a strictly-typed Pydantic `VisualizationResponse` with deep citations linking every data point to its source NCT records.

---

## How to Run

### Prerequisites

- Python 3.9 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Optionally choose a different model (defaults to `gpt-4o-mini`):

```bash
export OPENAI_MODEL="gpt-4o"
```

### 3. Start the Server

```bash
python main.py
```

The server starts at `http://0.0.0.0:8000` with hot-reload enabled.

### Available URLs

| URL | Description |
|-----|-------------|
| http://localhost:8000/ | Interactive dashboard UI |
| http://localhost:8000/docs | Swagger API documentation |
| http://localhost:8000/health | Health check endpoint |
| http://localhost:8000/openapi.json | OpenAPI specification |

### Quick Test (cURL)

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Distribution of Alzheimer trials by status"}' | python3 -m json.tool
```

---

## Request Schema (Input)

**Endpoint:** `POST /query`  
**Content-Type:** `application/json`

| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| `query` | `string` | ✅ Yes | — | min 3 chars | Natural-language question about clinical trials |
| `drug_name` | `string` | No | `null` | — | Filter by drug / intervention name (e.g., `"Pembrolizumab"`) |
| `condition` | `string` | No | `null` | — | Filter by medical condition (e.g., `"breast cancer"`) |
| `trial_phase` | `string` | No | `null` | — | Filter by trial phase: `"Phase 1"` through `"Phase 4"` |
| `sponsor` | `string` | No | `null` | — | Filter by lead sponsor name |
| `start_year` | `integer` | No | `null` | 1990–2030 | Include only trials starting on or after this year |
| `end_year` | `integer` | No | `null` | 1990–2030 | Include only trials starting on or before this year |
| `max_results` | `integer` | No | `100` | 1–1000 | Maximum number of studies to retrieve from ClinicalTrials.gov |
| `filters` | `object` | No | `{}` | — | Additional raw CT.gov API query-string parameters |

> **Note on structured filters:** When both a natural-language query and explicit structured fields are provided, the structured fields **always override** the LLM's interpretation. This is a deliberate anti-hallucination measure.

### Example Requests

```json
// Minimal — natural-language only
{
  "query": "Distribution of Alzheimer trials by status"
}

// With a drug filter (matching the problem statement example)
{
  "query": "How has the number of trials for this drug changed over time?",
  "drug_name": "Pembrolizumab"
}

// With date range and phase filter
{
  "query": "Phase 3 oncology trials by status",
  "trial_phase": "Phase 3",
  "start_year": 2020,
  "end_year": 2025,
  "max_results": 500
}
```

---

## Response Schema (Output)

Every response has two top-level sections: `visualization` and `meta`.

### `visualization` Object

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | One of: `bar_chart`, `time_series`, `pie_chart`, `histogram`, `scatter_plot`, `network_graph`, `table` |
| `title` | `string` | Human-readable chart title generated by the LLM |
| `encoding.x` | `string` | Field mapped to the x-axis (human-readable) |
| `encoding.y` | `string` | Aggregation function and field (e.g., `"count(studies)"`) |
| `encoding.color` | `string\|null` | Optional field for color/series grouping |
| `data` | `array` | Array of aggregated `DataPoint` objects |
| `network_data` | `object\|null` | Node and edge data (populated only when `type=network_graph`) |

### `data[]` — DataPoint

| Field | Type | Description |
|-------|------|-------------|
| `label` | `string` | Category or time-bucket label (e.g., `"COMPLETED"`, `"2023-01"`) |
| `value` | `number` | Computed metric value (deterministically aggregated by Pandas) |
| `color_group` | `string\|null` | Secondary grouping label for stacked/multi-series charts |
| `citations` | `array` | List of source studies that contributed to this data point |
| `citations[].nct_id` | `string` | ClinicalTrials.gov NCT identifier (e.g., `"NCT03402659"`) |
| `citations[].excerpt` | `string` | Brief text excerpt from the source study for verification |

### `network_data` (when `type=network_graph`)

| Field | Type | Description |
|-------|------|-------------|
| `nodes[].id` | `string` | Unique node identifier (prefixed by type, e.g., `"cond:Lung Cancer"`) |
| `nodes[].label` | `string` | Display label |
| `nodes[].group` | `string` | Entity type: `condition`, `intervention`, or `sponsor` |
| `nodes[].size` | `integer` | Number of studies containing this entity |
| `edges[].source` | `string` | Source node ID |
| `edges[].target` | `string` | Target node ID |
| `edges[].weight` | `integer` | Number of studies linking these two entities |
| `edges[].citations` | `array` | Studies that contribute to this edge |

### `meta` Object

| Field | Type | Description |
|-------|------|-------------|
| `source` | `string` | Always `"clinicaltrials.gov"` |
| `search_expression` | `string` | The search term sent to the CT.gov API |
| `filters_applied` | `object` | All filters used (phase, status, drug_name, date range, etc.) |
| `total_studies_fetched` | `integer` | Number of raw studies retrieved from the API |
| `assumptions` | `array` | LLM's interpretation notes and assumptions |

### Example Response

```json
{
  "visualization": {
    "type": "bar_chart",
    "title": "Distribution of Alzheimer Trials by Status",
    "encoding": { "x": "overallStatus", "y": "count(studies)", "color": null },
    "data": [
      {
        "label": "COMPLETED",
        "value": 56,
        "color_group": null,
        "citations": [
          { "nct_id": "NCT03402659", "excerpt": "[COMPLETED] Proof-of-Concept Study of a Selective p38 MAPK..." },
          { "nct_id": "NCT01078636", "excerpt": "[COMPLETED] Alzheimer's Disease Neuroimaging Initiative..." }
        ]
      },
      {
        "label": "RECRUITING",
        "value": 18,
        "color_group": null,
        "citations": [
          { "nct_id": "NCT05655650", "excerpt": "[RECRUITING] Identifying Biomarkers in Alzheimer's Disease" }
        ]
      }
    ],
    "network_data": null
  },
  "meta": {
    "source": "clinicaltrials.gov",
    "search_expression": "Alzheimer",
    "filters_applied": {},
    "total_studies_fetched": 100,
    "assumptions": ["The query is interpreting 'Alzheimer' as a broad condition search."]
  }
}
```

---

## Supported Visualizations

| Type | When Selected | Encoding |
|------|--------------|----------|
| `bar_chart` | Distribution / breakdown queries ("by status", "by phase", "top sponsors") | x = category, y = count |
| `time_series` | Trend over time queries ("per year", "over time", "since 2020") | x = date, y = count (line chart) |
| `pie_chart` | Proportion / share queries ("proportion", "percentage", "makeup") | x = category, y = count |
| `histogram` | Numeric distribution queries | x = numeric bins, y = count |
| `scatter_plot` | Correlation queries ("correlate", "relationship between") | x = field1, y = field2 |
| `network_graph` | Entity relationship queries ("relationships", "connections", "which drugs treat") | nodes = entities, edges = co-occurrence |
| `table` | Tabular listing queries | x = row key, y = value |

The LLM selects the visualization type based on keyword analysis in the system prompt. **Grouped bar charts** are supported via the `color_group` field on each data point.

---

## Pagination

The ClinicalTrials.gov v2 API caps results at **100 per page**. The client implements **cursor-based token pagination** to retrieve larger result sets:

```
Page 1:  GET /studies?pageSize=100&query.term=Alzheimer
         → { studies: [...100], nextPageToken: "abc123" }

Page 2:  GET /studies?pageSize=100&query.term=Alzheimer&pageToken=abc123
         → { studies: [...100], nextPageToken: "def456" }

Page 3:  GET /studies?pageSize=100&query.term=Alzheimer&pageToken=def456
         → { studies: [...50], nextPageToken: null }
         → STOP
```

**Three stop conditions:**
1. Accumulated studies ≥ `max_results` — we have enough data.
2. API returns an empty batch — no more studies match.
3. `nextPageToken` is missing — we're on the last page.

The final list is trimmed to exactly `max_results` to prevent overshoot from the last page.

---

## Example Runs

Full JSON outputs are saved in the [`examples/`](examples/) directory:

| File | Query | Viz Type | Studies | Data Points |
|------|-------|----------|---------|-------------|
| [`example_1_bar_chart.json`](examples/example_1_bar_chart.json) | Distribution of Alzheimer trials by status | `bar_chart` | 100 | 8 |
| [`example_2_time_series.json`](examples/example_2_time_series.json) | How many COVID-19 vaccine trials started each year? | `time_series` | 200 | 183 |
| [`example_3_drug_filter.json`](examples/example_3_drug_filter.json) | Drug-specific query with `drug_name: "Pembrolizumab"` | `time_series` | 150 | 148 |
| [`example_4_phase_breakdown.json`](examples/example_4_phase_breakdown.json) | Breakdown of diabetes trials by phase | `bar_chart` | 100 | 7 |
| [`example_5_network_graph.json`](examples/example_5_network_graph.json) | Relationships between drugs and conditions in breast cancer | `network_graph` | 50 | 52 nodes, 100 edges |

### Quick cURL Tests

```bash
# 1. Bar chart — status distribution
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Distribution of Alzheimer trials by status"}' | python3 -m json.tool

# 2. Time series — trials over time
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many COVID-19 vaccine trials started each year?", "max_results": 200}' | python3 -m json.tool

# 3. Pie chart — proportion query
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What proportion of diabetes trials are in each phase?"}' | python3 -m json.tool

# 4. Network graph — entity relationships
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show relationships between drugs and conditions in breast cancer trials", "max_results": 50}' | python3 -m json.tool

# 5. Structured filters — drug + date range
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How has the number of trials for this drug changed over time?", "drug_name": "Pembrolizumab", "start_year": 2015}' | python3 -m json.tool

# 6. Filtered by phase + year range
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Phase 3 oncology trials by status", "trial_phase": "Phase 3", "start_year": 2020}' | python3 -m json.tool
```

---

## Key Design Decisions & Tradeoffs

### 1. LLM as Router, Not Calculator (Anti-Hallucination)

**Decision:** The LLM is used exclusively for intent extraction — translating natural language into structured search parameters and selecting the visualization type. All data aggregation is performed deterministically by Pandas.

**Tradeoff:** This limits the system's ability to handle nuanced subjective queries (e.g., "which trials had surprising enrollment numbers?"), but completely eliminates the risk of hallucinated statistics. Every number in the output is verifiable.

### 2. Instructor + Pydantic for Structured Outputs

**Decision:** The `instructor` library enforces strict Pydantic schema validation on LLM outputs, ensuring type-safe, predictable structured data with automatic retries on validation failures.

**Tradeoff:** Adds a dependency and marginally increases latency (~200ms for schema enforcement), but guarantees that every LLM response conforms to the `QueryIntent` model — no JSON parsing failures or malformed output.

### 3. Explicit User Fields Override LLM Extraction

**Decision:** When a user provides structured fields (e.g., `drug_name`, `trial_phase`), these always override the LLM's extraction from the query text.

**Tradeoff:** The LLM's interpretation may conflict with the explicit fields, but deterministic user input should always prevail over probabilistic extraction. This is a key anti-hallucination guardrail.

### 4. OR-Joined Status Filters vs AND-Joined Phase Filters

**Decision:** Status values in the CT.gov advanced filter are OR-joined because a trial has exactly one status. Phase values are AND-joined to narrow results.

**Tradeoff:** Discovered during testing — AND-joining statuses returns 0 results since no trial can simultaneously be `RECRUITING` and `COMPLETED`.

### 5. Graceful 400 Recovery

**Decision:** If the CT.gov API returns a 400 error (likely due to an LLM-generated filter expression that doesn't match the API's syntax), the client retries without the advanced filter.

**Tradeoff:** May return broader results than intended, but prevents complete failure. The Pandas aggregator still groups correctly, so the visualization remains accurate.

### 6. Network Graph via Entity Co-occurrence

**Decision:** Network graphs are built by extracting entities (conditions, interventions, sponsors) from each study and creating edges between entities that co-occur in the same trial. Edges are only created **across** entity types (drug↔condition, sponsor↔drug), not within the same type.

**Tradeoff:** The graph can become dense for broad queries. Mitigated by capping to the top 100 edges by weight and keeping only referenced nodes.

---

## Limitations & Future Improvements

### Current Limitations

1. **LLM Dependency** — Requires an OpenAI API key; no offline fallback exists for intent extraction.
2. **Rate Limits** — The ClinicalTrials.gov API may rate-limit heavy usage. No caching layer is currently implemented.
3. **Single-Shot Interpretation** — Complex multi-step queries (e.g., "compare last year's recruiting trials to this year's") are interpreted in a single LLM call; iterative refinement is not supported.
4. **Network Graph Density** — Broad queries can produce very dense graphs. The UI uses a simple force-directed layout that may not scale well beyond ~100 nodes.
5. **Scatter Plot Coverage** — The schema supports `scatter_plot` but the aggregator currently falls back to count-based grouping for most scatter queries, since scatter requires two numeric fields.

### What I Would Improve With More Time

1. **Caching Layer** — Add Redis or in-memory caching for CT.gov API responses to reduce latency and respect rate limits.
2. **Multi-Turn Conversation** — Allow users to refine queries iteratively ("now filter to Phase 3 only") using conversation history.
3. **Advanced Network Visualizations** — Add community detection (Louvain clustering) and centrality metrics to identify key entities.
4. **Streaming Responses** — For large datasets, stream aggregation results using Server-Sent Events.
5. **Docker Deployment** — Containerize with a multi-stage Dockerfile and add health monitoring with observability (e.g., LangSmith for LLM call tracking).
6. **Formal Test Suite** — Convert the manual test cases into pytest with mocked API responses for CI/CD integration.

---

## File Structure

```
assignment/
├── main.py              # FastAPI entry point, endpoint definitions, static file serving
├── agent.py             # Pipeline orchestrator: LLM intent → merge → fetch → aggregate → respond
├── client.py            # Async HTTP client for ClinicalTrials.gov v2 API with pagination
├── aggregator.py        # Deterministic Pandas aggregation + network graph builder + citations
├── models.py            # All Pydantic schemas: QueryRequest, QueryIntent, VisualizationResponse
├── requirements.txt     # Python dependencies (FastAPI, Instructor, Pandas, HTTPX, etc.)
├── static/
│   └── index.html       # Interactive frontend dashboard (Chart.js + Canvas network graph)
└── examples/
    ├── example_1_bar_chart.json
    ├── example_2_time_series.json
    ├── example_3_drug_filter.json
    ├── example_4_phase_breakdown.json
    └── example_5_network_graph.json
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI | Async API with auto-generated Swagger docs |
| LLM Integration | OpenAI + Instructor | Structured output extraction with Pydantic validation |
| Data Aggregation | Pandas | Deterministic groupby/count/sum operations |
| HTTP Client | HTTPX | Async requests to ClinicalTrials.gov v2 API |
| Schema Validation | Pydantic v2 | Request/response type safety and JSON Schema generation |
| Frontend | Chart.js + Canvas | Bar/line/pie/doughnut charts + force-directed network graph |

---

## Tools & Integrity Note

### Tools Used

| Tool | How It Was Used |
|------|-----------------|
| **AI Coding Assistant (GitHub Copilot)** | Scaffolding boilerplate: Pydantic field declarations, FastAPI route wiring, Chart.js config objects, CSS styling. All generated code was reviewed, tested, and modified. |
| **ClinicalTrials.gov API v2 Documentation** | Referenced directly for JSON field paths (`protocolSection.statusModule.*`), advanced filter syntax (`AREA[Phase]PHASE3`), and pagination token semantics. |
| **Instructor Library Docs** | Used for understanding the `response_model` pattern and retry configuration for structured LLM outputs. |

### How I Validated Correctness

Correctness was validated through a combination of end-to-end testing against the live API and targeted debugging of specific failure modes:

1. **End-to-end query testing** — 6 diverse queries were tested covering all visualization types (bar chart, time series, pie chart, network graph) and filter combinations (drug name, phase, date range). Results were manually verified against the ClinicalTrials.gov website.

2. **Citation spot-checking** — For each visualization type, I randomly selected 3–5 citations from the response and verified the NCT IDs existed on ClinicalTrials.gov with matching statuses/phases. This confirmed the aggregation was correctly counting real studies.

3. **Pagination verification** — Tested with `max_results: 200` and confirmed the client made exactly 2 API calls (100 per page) by inspecting server logs. Verified the `nextPageToken` loop terminates correctly.

4. **Filter logic debugging** — Discovered that AND-joining status filters (`AREA[OverallStatus]RECRUITING AND AREA[OverallStatus]COMPLETED`) returns 0 results because a trial has exactly one status. Fixed by OR-joining statuses and AND-joining phases.

5. **Fuzzy column matching** — Found that the LLM sometimes outputs partial field paths (e.g., `startDateStruct.date` instead of `protocolSection.statusModule.startDateStruct.date`). Implemented tail-segment matching in `aggregator._best_match()` and verified with logging.

6. **400 error recovery** — Deliberately triggered a 400 error by sending an invalid filter expression, confirmed the client retried without filters and still produced a valid (broader) visualization.

### What Was Designed/Implemented Deliberately vs Generated/Adapted

**Designed and implemented from scratch:**

| Component | Rationale |
|-----------|-----------|
| **Anti-hallucination pipeline architecture** | The three-layer separation (LLM router → deterministic fetch → deterministic aggregation) was the core design decision. The LLM never sees aggregate numbers, and Pandas never calls the LLM. This guarantees every number in the output is verifiable. |
| **Deep citation system** | Every `DataPoint` carries a `citations[]` array linking it to specific NCT IDs. This was designed to meet the "evidence of correctness" requirement — a user can click any bar in the chart and see exactly which studies contributed to that count. |
| **Network graph builder** (`aggregator.build_network()`) | Entity co-occurrence logic that extracts conditions, interventions, and sponsors from each study and builds cross-type edges. The decision to only link *across* entity types (drug↔condition, not drug↔drug) was deliberate to produce meaningful graphs. |
| **User field override mechanism** (`agent._merge_user_fields()`) | Explicit structured fields always override LLM extraction. This prevents the LLM from misinterpreting a drug name or phase and ensures deterministic behavior for API users who know exactly what they want. |
| **Graceful 400 recovery** (`client.py`) | After hitting a real API error during testing, I implemented a retry-without-filter fallback. This was an iterative improvement from a production-like failure, not pre-planned. |
| **Cursor-based pagination loop** (`client.py`) | Implemented after discovering the CT.gov API caps at 100 results per page. The three stop conditions (enough data, empty batch, no token) were refined through testing. |

**Generated/adapted (with review and modification):**

| Component | What Was Generated | What I Modified |
|-----------|-------------------|-----------------|
| **Pydantic model fields** | Field declarations with basic types | Added custom validators (`start_year` range), descriptions, and the three-layer schema architecture |
| **FastAPI boilerplate** | Route decorators, CORS middleware | Added error handling with specific exception types, health endpoint, static file mounting |
| **Chart.js rendering** | Basic chart configuration | Customized color palette, added gradient fills, tooltip formatting, and responsive options |
| **CSS styling** | Base layout structure | Implemented glassmorphism, animated gradients, micro-animations, and dark theme from scratch |
| **Canvas network graph** | — (fully custom) | Wrote the force-directed simulation, collision detection, and interactive hover system entirely by hand |

### Evidence of Iteration

The codebase evolved through several rounds of testing and refinement:

- **Iteration 1:** All queries returned `bar_chart`. Root cause: the system prompt lacked explicit keyword-to-visualization-type mapping. Fixed by adding a strict keyword table to the prompt.
- **Iteration 2:** Network graphs showed edges between nodes of the same type (condition↔condition), producing uninformative clusters. Fixed by adding a `grp_a == grp_b: continue` guard.
- **Iteration 3:** The LLM occasionally output empty `x` fields. Added fuzzy matching (`_best_match`) and a fallback to the first available column.
- **Iteration 4:** Unhashable dict values in Pandas columns caused `groupby` to crash. Fixed by adding `str(v)` conversion for dict/list values before grouping.

