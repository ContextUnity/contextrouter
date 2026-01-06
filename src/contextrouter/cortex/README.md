# Brain (LangGraph) pipeline

This package contains the shared agent “cortex” used by both the web UI and Telegram.

The cortex is implemented as a LangGraph `StateGraph`:

- **Graph wiring**: `contextrouter/cortex/graphs/brain.py` (central router) and `contextrouter/cortex/graphs/rag_retrieval.py` (RAG retrieval graph)
- **Nodes (graph steps)**: `contextrouter/cortex/nodes/`
- **Runners (host entrypoints)**: `contextrouter/cortex/runners/` (stream/invoke helpers)
- **State schema**: `contextrouter/cortex/state.py`
- **Generic prompts (cortex-owned)**: `contextrouter/cortex/prompting/`

## Ingestion graphs (dynamic recipes)

Ingestion can be wired as a full pipeline (`cortex/graphs/rag_ingestion.py`) or built dynamically from
a declarative recipe (`IngestionRecipe`) which selects a subset of stages (e.g. `preprocess` only,
`preprocess+taxonomy`, etc.).

The recipe is intentionally *not code* and is safe to accept from transports.

## High-level flow

The graph executes the following steps for each user message:

1) `extract_query`
2) `detect_intent`
3) Conditional routing via `should_retrieve`
4) `retrieve` (optional)
5) `suggest`
6) `generate`

In short:

- If the intent is **rag_and_web**, we retrieve from **Vertex AI Search** and (optionally) from **Google CSE** (site-limited).
- If retrieval yields **zero docs**, we return a **no-results response** generated with a dedicated no-results prompt.

## Nodes (what each one does)

### `extract_query`

- Reads the latest `HumanMessage` from `state.messages`.
- Writes:
  - `user_query` (string)
  - `should_retrieve` (bool)
  - initializes defaults (`intent=rag_and_web`, empty `retrieved_docs`, etc.)

### `detect_intent`

- Uses `gemini-2.0-flash-lite` to classify intent.
- Possible intents are:
  - `rag_and_web` - questions requiring retrieval from sources
  - `translate` - translation requests
  - `summarize` - summarization requests
  - `rewrite` - rewriting/editing requests
  - `identity` - questions about the assistant itself ("Who are you?", "What can you do?")

**Identity intent** (no retrieval):

- Detects self-referential questions about the assistant: "Who are you?", "What can you do?", "Tell me about yourself", "Are you an AI?"
- Skips RAG retrieval entirely (no Vertex Search, no web search)
- Uses `IDENTITY_PROMPT` with `style_prompt` context to generate a contextual response
- Prevents irrelevant RAG results when user asks about the assistant vs. philosophical concepts

Taxonomy + Graph + Ontology integration (runtime):

**Taxonomy enrichment**:

- **Before** the LLM call, `detect_intent` loads `taxonomy.json` (cached) and appends a **small taxonomy context** to the system prompt:
  - up to N top-level category names (default N=20)
  - a few example synonym mappings (from `canonical_map`)
- **After** the LLM call, `detect_intent` derives per-request taxonomy tags by matching the current user query against the taxonomy `canonical_map`:
  - `taxonomy_concepts`: canonical concepts detected in the query
  - `taxonomy_categories`: categories for those concepts (via graph service lookup)

**Graph facts (Path B retrieval)**:

- `retrieve` uses `GraphService.get_facts(taxonomy_concepts)` to fetch explicit relationship facts
- Facts are **ontology-filtered**: only relations marked as `runtime_fact_labels` in `ontology.json` are emitted
- Facts are **non-citation** background knowledge added to the RAG prompt (separate from Vertex Search citations)

**Retrieval query strengthening**:

- `detect_intent` returns `retrieval_queries` (1-3 short queries derived from `cleaned_query`)
- When `taxonomy_concepts` are detected, a compact concept query is added to strengthen retrieval
- This ensures Vertex Search benefits from taxonomy normalization even if the user query uses synonyms

Example (how it works):

- **taxonomy canonical_map**: `"pma" -> "Positive Mental Attitude"`, `"autosuggestion" -> "Autosuggestion"`
- **user query**: "How do I build PMA and use autosuggestion daily?"
- **result**:
  - `taxonomy_concepts=["Positive Mental Attitude", "Autosuggestion"]`
  - `taxonomy_categories=[...]` (resolved via GraphService)
  - `retrieval_queries=["How do I build Positive Mental Attitude", "use Autosuggestion daily", "PMA autosuggestion"]` (original + concept-strengthened)
  - `graph_facts=["Fact: Positive Mental Attitude CAUSES Success", "Fact: Autosuggestion REQUIRES Repetition"]` (ontology-filtered)

What if there’s no match with “top 20 categories”?

- The “top 20 categories” list is only a **prompt hint** for the LLM.
- Concept matching uses the **full** `canonical_map` (not limited to 20), so a concept can still be detected even if its category is not listed in the top 20.

### `should_retrieve` (routing)

This is the conditional router for the graph.

- If `intent!=rag_and_web` -> route directly to `suggest` (then `generate`)
- If `intent=rag_and_web` and `should_retrieve=True` and we have no docs yet -> route to `retrieve`
- Otherwise -> route to `suggest` (then `generate`)

### `retrieve` (Vertex AI Search + Graph Facts + Reranking)

**Path A (Vector Search)**:
- Runs searches in Vertex AI Search (book/video/qa) for each query in `retrieval_queries`.
- If web_allowed_domains is defined, runs web search for all domains.
- **Reranking**: After retrieval, documents are reranked per source type using the Vertex AI Ranking API. Reranking runs in parallel for all source types (book, video, qa, web).

**Path B (Graph Facts)**:
- Uses `GraphService.get_facts(taxonomy_concepts)` to fetch explicit relationship facts
- Facts are **ontology-filtered** (only `runtime_fact_labels` from `ontology.json` are emitted)
- Facts are added to state as `graph_facts` (non-citation background knowledge)
- Facts are injected into the RAG prompt via `build_rag_prompt(graph_facts=...)`

Web citations:

- Web search results are represented as `source_type="web"` docs.
- Citations include `type="web"` with:
  - `title`
  - `url`
  - `summary` (snippet or extracted page text)
- The citations builder limits web citations to **3 unique URLs** per run.

### `suggest`

- Generates optional “search suggestions” (used only for `intent=rag_and_web`).

### `generate`

- Produces the final assistant message.

Key behaviors:

- If `intent=rag_and_web` and there are **no retrieved docs**, we generate a **no-results response** using an LLM call with suggestions for alternative queries.

Web sources:

- We do **not** append a `Sources:` section to the answer text.
- URLs/titles are carried via `web` citations and rendered in the UI (Web tab).

No-results prompt:

- When `intent=rag_and_web` and there are 0 retrieved docs, the brain generates a no-results message using `gemini-2.0-flash-lite`.
- The prompt template can be overridden by the host via the input state field `no_results_prompt`.
- Host applications are expected to supply product tone/theme in `no_results_prompt`.

## Debugging

### `DEBUG_PIPELINE=1`

Emits structured per-node logs so you can see how the query is processed:

- `PIPELINE extract_query | ...`
- `PIPELINE detect_intent.in/out | taxonomy_concepts=... taxonomy_categories=... retrieval_queries=...`
- `PIPELINE route | ...`
- `PIPELINE retrieve.in | user_query=... retrieval_queries=...`
- `PIPELINE retrieve.graph_facts | facts=N concepts=[...] sample_facts=[...]`
- `PIPELINE retrieve.out | docs=N books=N videos=N qa=N web=N citations=N`
- `PIPELINE retrieve.fallback_to_web | ...`
- `PIPELINE suggest.in | ...`
- `PIPELINE generate.in | intent=... retrieved_docs=N citations=N`
- `PIPELINE generate.out | assistant_chars=N web_sources=N`
- `PIPELINE generate.no_results | ...`

### `DEBUG_WEB_SEARCH=1`

Logs a safe preview of raw CSE results (title/url/snippet) and the kept list.

### Filter-to-zero diagnostics (always on)

If CSE returns results but we keep 0 after filtering, we log a warning with:

- sample rejected host/url pairs
- sample invalid/missing links

This is logged even when `DEBUG_WEB_SEARCH=0`.
