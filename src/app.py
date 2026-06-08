"""Streamlit demo UI for the RAG-after-Build-2026 utility-ops walkthrough.

Three tabs, three punchlines:

  1. Reliable retrieval   — same paraphrased question, four modes.
                            Keyword misses; semantic nails it.
  2. Semantic configs     — same query, three configs, see which doc each
                            one promotes to #1. Tune retrieval without
                            retraining anything.
  3. Agentic retrieval    — one REST call decomposes a compound question,
                            runs parallel subqueries, reranks, and a chat
                            model synthesizes a cited answer.

    streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `import common` etc. when streamlit launches us from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from common import Settings  # noqa: E402
from agentic_demo import _extract_answer, retrieve, synthesize_answer  # noqa: E402
from retrieval_lab import (  # noqa: E402
    RunResult,
    run_hybrid,
    run_keyword,
    run_semantic,
    run_vector,
)


# ---------------------------------------------------------------------------
# RAG maturity model — single source of truth for the Overview tab
# ---------------------------------------------------------------------------


MATURITY_LEVELS: list[dict] = [
    {
        "id": "L0",
        "name": "L0 — Just an LLM",
        "tagline": "Send the question straight to the model. No retrieval, no grounding.",
        "resources": [
            ("Azure OpenAI Service", "S0 (the only SKU). Region with capacity for your chosen model."),
            ("Model deployment", "e.g. `gpt-5-mini`, **GlobalStandard** for cheapest TPS; **Standard** if you need data residency."),
        ],
        "pros": [
            "Trivial to build (one API call).",
            "Great for general-knowledge or creative tasks.",
        ],
        "cons": [
            "**Hallucinates** on company-specific questions.",
            "No traceability — you can't tell what the answer was based on.",
            "Can't be updated with new policies without retraining or fine-tuning.",
        ],
        "docs": [
            ("What is Azure OpenAI Service?", "https://learn.microsoft.com/azure/ai-services/openai/overview"),
            ("Create and deploy an Azure OpenAI resource", "https://learn.microsoft.com/azure/ai-services/openai/how-to/create-resource"),
            ("Model and region availability", "https://learn.microsoft.com/azure/ai-services/openai/concepts/models"),
        ],
        "diagram": """
flowchart LR
    U([User]) -- question --> APP[Your app]
    APP -- prompt --> AOAI[Azure OpenAI<br/>gpt-5-mini]
    AOAI -- answer --> APP --> U
    classDef ai fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#000
    classDef app fill:#f3f4f6,stroke:#6b7280,color:#000
    class AOAI ai
    class APP app
""",
    },
    {
        "id": "L1",
        "name": "L1 — Naive (keyword) RAG",
        "tagline": "Add a keyword search before the LLM. Stuff the top matches into the prompt.",
        "resources": [
            ("Azure OpenAI", "S0 + chat deployment (as in L0)."),
            ("Azure AI Search", "**Basic** tier is the practical floor. Free works for prototypes (50 MB index limit, no SLA)."),
            ("Azure Blob Storage", "Standard (LRS or ZRS) to hold the source documents."),
            ("Search indexer (optional)", "Blob indexer pulls docs, cracks them (PDF/DOCX/HTML), and pushes to the index."),
            ("Identity", "User-assigned managed identity on Search → **Storage Blob Data Reader** on the storage account."),
        ],
        "pros": [
            "Grounds answers in your own documents.",
            "Cheap and predictable — BM25 is a 1970s algorithm.",
            "Good when users ask using the document's exact terminology.",
        ],
        "cons": [
            "Breaks the moment the user paraphrases (\"protective gear\" vs \"PPE\").",
            "No relevance signal you can gate on — only ranking *within* a query.",
            "No defense against an LLM that confidently summarizes the wrong chunk.",
        ],
        "docs": [
            ("What is Azure AI Search?", "https://learn.microsoft.com/azure/search/search-what-is-azure-search"),
            ("Full-text search and BM25 ranking", "https://learn.microsoft.com/azure/search/search-lucene-query-architecture"),
            ("Choose a service SKU/tier", "https://learn.microsoft.com/azure/search/search-sku-tier"),
            ("Index Azure Blob Storage documents", "https://learn.microsoft.com/azure/search/search-howto-indexing-azure-blob-storage"),
            ("RAG pattern in Azure AI Search", "https://learn.microsoft.com/azure/search/retrieval-augmented-generation-overview"),
        ],
        "diagram": """
flowchart LR
    DOCS[(Documents<br/>in Blob)] -. indexed .-> SEARCH
    U([User]) -- question --> APP[Your app]
    APP -- BM25 query --> SEARCH[Azure AI Search<br/>keyword index]
    SEARCH -- top-k chunks --> APP
    APP -- chunks + question --> AOAI[Azure OpenAI<br/>gpt-5-mini]
    AOAI -- grounded answer --> APP --> U
    classDef ai fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#000
    classDef search fill:#fef3c7,stroke:#ca8a04,stroke-width:2px,color:#000
    classDef app fill:#f3f4f6,stroke:#6b7280,color:#000
    classDef data fill:#fce7f3,stroke:#be185d,color:#000
    class AOAI ai
    class SEARCH search
    class APP app
    class DOCS data
""",
    },
    {
        "id": "L2",
        "name": "L2 — Vector RAG",
        "tagline": "Replace (or augment) keyword search with embeddings. Now you understand meaning, not just words.",
        "resources": [
            ("Azure OpenAI", "S0 + **two** deployments: a chat model and an **embedding** model (e.g. `text-embedding-3-large`, 3072 dims)."),
            ("Azure AI Search", "**Basic** tier works; vector storage counts against your index size quota."),
            ("Azure Blob Storage", "Same as L1."),
            ("Index schema", "Add a `Collection(Edm.Single)` vector field + an `AzureOpenAIVectorizer` so query-time embedding is automatic."),
            ("Identity", "Search MSI → **Cognitive Services OpenAI User** on the AOAI resource (so the vectorizer can call embeddings)."),
        ],
        "pros": [
            "Handles paraphrasing, synonyms, and conceptual queries.",
            "Built-in `AzureOpenAIVectorizer` does query-time embedding for you.",
            "Cosine similarity gives a relative similarity number per result.",
        ],
        "cons": [
            "Can latch onto **topically similar but irrelevant** chunks.",
            "Embedding scores are not calibrated — you can rank but not gate.",
            "Loses on exact-match cases (product codes, IDs, acronyms).",
        ],
        "docs": [
            ("Vector search overview", "https://learn.microsoft.com/azure/search/vector-search-overview"),
            ("Integrated vectorization (AzureOpenAIVectorizer)", "https://learn.microsoft.com/azure/search/vector-search-integrated-vectorization"),
            ("Azure OpenAI embedding models", "https://learn.microsoft.com/azure/ai-services/openai/concepts/models"),
            ("Chunking strategies for RAG", "https://learn.microsoft.com/azure/search/vector-search-how-to-chunk-documents"),
        ],
        "diagram": """
flowchart LR
    DOCS[(Documents)] -. chunked + embedded .-> SEARCH
    U([User]) -- question --> APP[Your app]
    APP -- vectorize query --> EMB[Azure OpenAI<br/>text-embedding-3-large]
    EMB -- 3072-dim vector --> APP
    APP -- vector query --> SEARCH[Azure AI Search<br/>vector index]
    SEARCH -- top-k chunks --> APP
    APP -- chunks + question --> AOAI[Azure OpenAI<br/>gpt-5-mini]
    AOAI -- answer --> APP --> U
    classDef ai fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#000
    classDef search fill:#fef3c7,stroke:#ca8a04,stroke-width:2px,color:#000
    classDef app fill:#f3f4f6,stroke:#6b7280,color:#000
    classDef data fill:#fce7f3,stroke:#be185d,color:#000
    class AOAI,EMB ai
    class SEARCH search
    class APP app
    class DOCS data
""",
    },
    {
        "id": "L3",
        "name": "L3 — Hybrid + Semantic Ranking  (production baseline)",
        "tagline": "Combine keyword + vector, then re-rank with a deep model. The reranker score is the gate that lets you refuse to answer.",
        "resources": [
            ("Azure OpenAI", "S0 + chat + embedding deployments (as in L2)."),
            ("Azure AI Search", "**Standard (S1)** or higher — **semantic ranker requires Standard tier** and must be enabled on the service."),
            ("Semantic configuration", "Defined on the index; tells the L2 reranker which fields carry meaning (title / content / keywords)."),
            ("Azure Blob Storage", "Same as L1/L2."),
            ("Identity", "App's MSI → **Search Index Data Reader** on the index for keyless query auth."),
            ("What this app uses", "This is the architecture deployed by `infra/main.bicep` — see tabs 1 & 2."),
        ],
        "pros": [
            "Best recall (hybrid) **and** best precision (L2 reranker).",
            "**Calibrated 0–4 score** lets you build a 'should I trust this?' gate.",
            "Extractive captions/answers give you LLM-free fallback grounding.",
            "**This is what tabs 1 + 2 of this app demonstrate.**",
        ],
        "cons": [
            "Requires Standard SKU or higher on Search.",
            "Multi-part questions still need orchestration code you write yourself.",
        ],
        "docs": [
            ("Hybrid search (BM25 + vector + RRF)", "https://learn.microsoft.com/azure/search/hybrid-search-overview"),
            ("Semantic ranking overview", "https://learn.microsoft.com/azure/search/semantic-search-overview"),
            ("Enable semantic ranker on a service", "https://learn.microsoft.com/azure/search/semantic-how-to-enable-disable"),
            ("Configure semantic ranking on an index", "https://learn.microsoft.com/azure/search/semantic-how-to-configure"),
            ("Reciprocal Rank Fusion (RRF) ranking", "https://learn.microsoft.com/azure/search/hybrid-search-ranking"),
        ],
        "diagram": """
flowchart LR
    DOCS[(Documents)] -. chunked + embedded .-> SEARCH
    U([User]) -- question --> APP[Your app]
    APP -- hybrid query --> SEARCH[Azure AI Search<br/>HYBRID:<br/>BM25 + vector + L2 reranker]
    SEARCH -- top-k + rerankerScore --> APP
    APP -- gate: score ≥ 2.0? --> AOAI[Azure OpenAI<br/>gpt-5-mini]
    AOAI -- cited answer --> APP --> U
    classDef ai fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#000
    classDef search fill:#bbf7d0,stroke:#16a34a,stroke-width:3px,color:#000
    classDef app fill:#f3f4f6,stroke:#6b7280,color:#000
    classDef data fill:#fce7f3,stroke:#be185d,color:#000
    class AOAI ai
    class SEARCH search
    class APP app
    class DOCS data
""",
    },
    {
        "id": "L4",
        "name": "L4 — Agentic RAG  (new in Build 2026)",
        "tagline": "One REST call replaces your orchestration layer. The Knowledge Base decomposes, plans, retrieves, and reranks.",
        "resources": [
            ("Azure OpenAI", "S0 + chat + embedding (the KB calls chat on your behalf for query planning)."),
            ("Azure AI Search", "**Standard (S1)** + semantic ranker enabled (same as L3)."),
            ("Knowledge Source", "A pointer to an existing index, declared via `PUT /knowledgesources/{name}` at api-version `2026-04-01`."),
            ("Knowledge Base", "The managed agent itself: `PUT /knowledgebases/{name}`. References one or more Knowledge Sources + an AOAI deployment."),
            ("Identity — critical extra", "Search MSI → **Cognitive Services OpenAI User** on the AOAI resource (the KB calls AOAI for its own planner)."),
            ("Azure Blob Storage", "Same as L3."),
        ],
        "pros": [
            "Handles **multi-part questions** out of the box.",
            "Built-in query decomposition, parallel subqueries, semantic reranking.",
            "Returns an **activity trace** (sub-queries, timings) for observability.",
            "Less code to write, less code to maintain.",
            "**This is what tab 3 of this app demonstrates.**",
        ],
        "cons": [
            "Newer API surface (REST `2026-04-01`).",
            "Less control over each stage than rolling your own.",
        ],
        "docs": [
            ("Agentic retrieval concept", "https://learn.microsoft.com/azure/search/search-agentic-retrieval-concept"),
            ("Build a knowledge source", "https://learn.microsoft.com/azure/search/search-knowledge-source-how-to"),
            ("Build a knowledge base", "https://learn.microsoft.com/azure/search/search-knowledge-base-how-to"),
            ("REST: Knowledge bases", "https://learn.microsoft.com/rest/api/searchservice/knowledge-bases"),
            ("REST: /retrieve endpoint", "https://learn.microsoft.com/rest/api/searchservice/knowledge-retrieval/retrieve"),
            ("What's new in Azure AI Search", "https://learn.microsoft.com/azure/search/whats-new"),
        ],
        "diagram": """
flowchart LR
    DOCS[(Documents)] -. chunked + embedded .-> KS
    U([User]) -- multi-part<br/>question --> APP[Your app]
    APP -- "POST /retrieve" --> KB[Knowledge Base<br/>managed agent]
    KB -- decompose<br/>+ plan --> KB
    KB -- parallel subqueries --> KS[Knowledge Source<br/>= AI Search index<br/>hybrid + L2 reranker]
    KS -- reranked chunks --> KB
    KB -- grounding payload<br/>+ activity trace --> APP
    APP -- synthesis prompt --> AOAI[Azure OpenAI<br/>gpt-5-mini]
    AOAI -- cited answer --> APP --> U
    classDef ai fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#000
    classDef search fill:#bbf7d0,stroke:#16a34a,stroke-width:2px,color:#000
    classDef kb fill:#e9d5ff,stroke:#7c3aed,stroke-width:3px,color:#000
    classDef app fill:#f3f4f6,stroke:#6b7280,color:#000
    classDef data fill:#fce7f3,stroke:#be185d,color:#000
    class AOAI ai
    class KS search
    class KB kb
    class APP app
    class DOCS data
""",
    },
]


def _mermaid(diagram: str, height: int = 360) -> None:
    """Render a Mermaid diagram inside a Streamlit component."""
    html = f"""
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <style>
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
        .mermaid {{ display: flex; justify-content: center; }}
      </style>
    </head>
    <body>
      <div class="mermaid">
{diagram}
      </div>
      <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default',
                              flowchart: {{ curve: 'basis', htmlLabels: true }} }});
      </script>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=False)


def _render_level(level: dict, *, height: int = 360) -> None:
    st.markdown(f"### {level['name']}")
    st.markdown(f"*{level['tagline']}*")
    _mermaid(level["diagram"], height=height)

    col_r, col_pc = st.columns([1.1, 1])
    with col_r:
        st.markdown("**Azure resources needed**")
        for name, detail in level["resources"]:
            st.markdown(f"- **{name}** — {detail}")
    with col_pc:
        st.markdown("**What you get**")
        st.markdown("\n".join(f"- {s}" for s in level["pros"]))
        st.markdown("**What it still can't do**")
        st.markdown("\n".join(f"- {s}" for s in level["cons"]))

    with st.expander(f"📚  Microsoft Learn docs for {level['id']}", expanded=False):
        st.markdown(
            "\n".join(f"- [{title}]({url})" for title, url in level["docs"])
        )



# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="RAG after Build 2026 — utility demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_settings() -> Settings:
    return Settings.from_env()


# Queries written to PARAPHRASE the corpus, not quote it. That's the point:
# keyword loses when the user doesn't speak the document's vocabulary.
SAMPLE_QUERIES_RELIABILITY = [
    "what gear protects me from a really bad arc flash",
    "how fast do we have to tell the feds about a cyber incident",
    "when can we shut off heat to a homeowner during a cold snap",
    "is it ok if the dispatcher tells customers power is coming back sooner",
    "what do we do if the gas stops smelling like gas",
]

SAMPLE_QUERIES_SEMANTIC = [
    "transformer oil sample interpretation",
    "tree trimming clearance requirements",
    "rooftop solar approval process",
    "smart meter installation steps",
]

SAMPLE_QUERIES_AGENTIC = [
    (
        "If a Grade 1 gas leak is detected next to a substation that is also "
        "experiencing a CIP-008 reportable cyber incident, what are our "
        "concurrent notification obligations and field actions in the first hour?"
    ),
    (
        "Compare what we tell a residential customer about a planned PSPS "
        "de-energization vs. an unplanned storm outage — same template or "
        "different?"
    ),
    (
        "A customer disputes a bill that covers a period including their AMI "
        "meter install. What waivers apply and what evidence do we keep?"
    ),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sync_textbox_from_pick(pick_key: str, text_key: str) -> None:
    """on_change callback: when the dropdown changes, push its value into
    the bound text input via session_state. Needed because Streamlit widgets
    with a key own their state and ignore subsequent `value=` defaults."""
    pick_val = st.session_state.get(pick_key, "")
    if pick_val and pick_val != "(choose one)":
        st.session_state[text_key] = pick_val


def _score_color(reranker: float | None, search: float | None) -> tuple[str, str, float]:
    if reranker is not None:
        color = "#16a34a" if reranker >= 2.0 else "#ca8a04" if reranker >= 1.5 else "#dc2626"
        return "reranker score (0–4)", color, reranker
    return "BM25 search score (unbounded)", "#6b7280", search or 0.0


def _content_snippet(text: str, max_chars: int = 320) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " …"


def _result_card(result: RunResult, *, color: str, show_snippet: bool = True) -> None:
    """Card: mode name + score badge + top doc + matched snippet + expander."""
    st.markdown(f"#### {result.mode}")
    st.caption(result.note)
    if not result.hits:
        st.error("No results")
        return

    top = result.hits[0]
    label, score_color, score_val = _score_color(top.reranker_score, top.search_score)

    st.markdown(
        f"<div style='padding:12px;border-left:6px solid {color};"
        f"background:#f9fafb;border-radius:4px;margin-bottom:8px'>"
        f"<div style='font-weight:600;font-size:1.05em'>{top.title}</div>"
        f"<div style='color:#6b7280;font-size:0.85em'>"
        f"<code>{top.source}</code> · chunk {top.chunk_index}</div>"
        f"<div style='margin-top:6px;color:{score_color};font-weight:600'>"
        f"{label}: {score_val:.3f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if show_snippet:
        snippet_source = (top.captions[0] if top.captions else top.content)
        st.markdown(
            "<div style='font-size:0.85em;color:#374151;padding:8px 12px;"
            "background:#ffffff;border:1px solid #e5e7eb;border-radius:4px;"
            f"min-height:120px'><b>Matched content:</b><br/>{_content_snippet(snippet_source)}</div>",
            unsafe_allow_html=True,
        )

    if result.answer:
        st.success(f"Extractive answer: {result.answer}")

    with st.expander("Show all results"):
        rows = []
        for rank, h in enumerate(result.hits, 1):
            rows.append(
                {
                    "rank": rank,
                    "reranker": round(h.reranker_score, 3) if h.reranker_score else None,
                    "search": round(h.search_score, 3) if h.search_score else None,
                    "title": h.title,
                    "source": h.source,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


with st.sidebar:
    st.markdown("### Demo configuration")
    settings = get_settings()
    st.write(f"**Index:** `{settings.search_index_name}`")
    st.write(f"**Embedding:** `{settings.aoai_embedding_deployment}`")
    st.write(f"**Chat model:** `{settings.aoai_chat_deployment}`")
    st.markdown("---")
    st.markdown("### Post-Build 2026 highlights")
    st.markdown(
        "- Agentic retrieval (GA in REST `2026-04-01`)\n"
        "- Knowledge Bases + Knowledge Sources\n"
        "- Query decomposition, parallel subqueries\n"
        "- Answer synthesis with citations\n"
        "- Semantic ranker as the reliability backstop\n"
        "- Integrated `AzureOpenAIVectorizer` — no client-side embedding at query time"
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


st.title("RAG after Build 2026")

PUBLIC_DEMO = os.environ.get("PUBLIC_DEMO", "").lower() == "true"
TAB3_LIMIT_PER_SESSION = 10

if PUBLIC_DEMO:
    st.info(
        f"👋 **Public demo.** This site is shared and rate-limited — "
        f"each browser session can run **up to {TAB3_LIMIT_PER_SESSION} "
        f"agentic queries (Tab 3)** before Azure OpenAI calls are paused. "
        f"Refresh the page to reset, or [clone the repo]"
        f"(https://github.com/) and run it against your own subscription "
        f"to lift the cap."
    )

st.markdown(
    "**Retrieval-Augmented Generation — in 30 seconds.** Large language "
    "models can't be trusted to *remember* your company's docs, and they "
    "hallucinate when they try. The fix is **RAG**: when a user asks a "
    "question, *first* search a private index of your documents, *then* "
    "hand the matching snippets to the LLM and tell it 'answer only from "
    "these'. The quality of the final answer is bounded by the quality of "
    "that search step."
)

with st.expander("📖  RAG vocabulary cheat-sheet (open me before the demo)"):
    st.markdown(
        "- **Index** — the searchable copy of your documents, broken into "
        "small *chunks* (a paragraph or two each).\n"
        "- **Keyword search (BM25)** — classic word-matching. Fast, exact, "
        "but breaks the moment the user uses different vocabulary than the "
        "document.\n"
        "- **Vector search** — each chunk is turned into a list of numbers "
        "(an *embedding*) that captures its meaning. The query gets the "
        "same treatment, and we return the closest matches by cosine "
        "distance. Handles paraphrasing well but can be fooled by topical "
        "similarity that isn't actually relevant.\n"
        "- **Hybrid search** — keyword + vector, results merged with "
        "*Reciprocal Rank Fusion*. Catches both exact matches and "
        "paraphrases.\n"
        "- **Semantic ranker (L2 reranker)** — a separate ML model from "
        "Microsoft Research that re-scores the top 50 hybrid results using "
        "deep language understanding. Outputs a `reranker score` from 0–4.\n"
        "- **Semantic configuration** — the recipe that tells the reranker "
        "which fields of your document carry the meaning (title? body? "
        "keywords?). You can define several per index.\n"
        "- **Knowledge Base / Knowledge Source** — *new in Build 2026*: a "
        "managed agent built into Azure AI Search that takes a complex "
        "question, breaks it into sub-questions, runs them in parallel, "
        "and returns the reranked grounding chunks for an LLM to answer "
        "from."
    )

st.markdown(
    "This demo runs the same utility-operations corpus (47 chunks: gas leaks, "
    "transformers, wildfire mitigation, FERC/NERC compliance, customer FAQs) "
    "through each technique. Each tab makes **one** concrete point."
)

tab_overview, tab_reliability, tab_semantic, tab_agentic = st.tabs(
    [
        "0.  Overview & maturity model",
        "1.  Why retrieval matters",
        "2.  Tuning what 'relevant' means",
        "3.  Agentic retrieval  (new in Build 2026)",
    ]
)


# ---------------------------------------------------------------------------
# Tab 0 — Overview & RAG maturity model
# ---------------------------------------------------------------------------


with tab_overview:
    st.subheader("The RAG maturity model")
    st.markdown(
        "RAG isn't one thing — it's a **ladder of techniques** that each "
        "trade more Azure surface area for more reliability. Most teams "
        "land somewhere on this ladder; the question is whether they "
        "*know* which rung they're on and what the next one buys them."
    )
    st.markdown(
        "Use the picker below to explore each level and the Azure "
        "architecture it requires. Turn on **compare mode** to put two "
        "levels side-by-side so you can see exactly what each upgrade adds."
    )

    compare = st.toggle("Compare two levels side-by-side", value=False, key="ov_compare")

    level_names = [lvl["name"] for lvl in MATURITY_LEVELS]
    name_to_level = {lvl["name"]: lvl for lvl in MATURITY_LEVELS}

    if not compare:
        chosen = st.select_slider(
            "Maturity level",
            options=level_names,
            value=level_names[3],  # default to L3 (what this app demos)
            key="ov_pick_single",
        )
        _render_level(name_to_level[chosen], height=380)
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            left = st.selectbox(
                "Left", level_names, index=1, key="ov_pick_left"
            )
            _render_level(name_to_level[left], height=340)
        with col_right:
            right = st.selectbox(
                "Right", level_names, index=4, key="ov_pick_right"
            )
            _render_level(name_to_level[right], height=340)

    st.markdown("---")
    st.markdown(
        "**Where does this app sit on the ladder?**\n"
        "- Tabs **1 & 2** demonstrate **L3 — Hybrid + Semantic Ranking**. "
        "Same Azure AI Search index, same `text-embedding-3-large` model, "
        "Standard tier with the semantic ranker on.\n"
        "- Tab **3** demonstrates **L4 — Agentic RAG**. Same index "
        "(now exposed as a *Knowledge Source*), wrapped by a *Knowledge "
        "Base* that takes a compound question and orchestrates the "
        "decomposition, parallel subqueries, and reranking for you."
    )
    st.info(
        "**Practical advice:** start at L1 to validate the user problem, "
        "jump to L3 the moment you have real users (you'll need that "
        "reranker score to decide when to refuse to answer), and adopt L4 "
        "when you start writing your second or third orchestration layer."
    )


# ---------------------------------------------------------------------------
# Tab 1 — Reliable retrieval: paraphrase test
# ---------------------------------------------------------------------------


with tab_reliability:
    st.subheader("The same question, four ways to search for it")
    st.markdown(
        "We're going to run **one** user question through four different "
        "search modes and see which one returns the right document.\n\n"
        "The trick: the sample questions are written in **plain English**, "
        "the way a real field tech would ask — not in the corpus's exact "
        "vocabulary. (Example: *'what gear protects me from a really bad "
        "arc flash'* instead of the document's actual wording, *'PPE for "
        "incident energy >25 cal/cm²'*.)"
    )
    st.markdown(
        "**What to watch for in the four cards below:**\n"
        "- 🔴 **Keyword** — will often miss because the user's words don't "
        "appear in the doc.\n"
        "- 🟡 **Vector** — will usually find *something topically related*, "
        "but not always the right thing.\n"
        "- 🔵 **Hybrid** — combines both. Better recall, but no judgement "
        "about which result is most authoritative.\n"
        "- 🟢 **Semantic** — hybrid + the semantic reranker. Ranks like a "
        "human would and gives you a **confidence score** you can gate on."
    )
    st.markdown(
        "**The score on each card matters.** Reranker score is 0–4; my rule "
        "of thumb: "
        "<span style='color:#16a34a;font-weight:600'>≥ 2.0 ship it</span>, "
        "<span style='color:#ca8a04;font-weight:600'>1.5–2.0 caveat it</span>, "
        "<span style='color:#dc2626;font-weight:600'>&lt; 1.5 say I don't know</span>. "
        "This is the gate that lets you build a RAG app that *refuses to "
        "answer* when it shouldn't.",
        unsafe_allow_html=True,
    )

    pick = st.selectbox(
        "Sample paraphrased queries",
        ["(choose one)"] + SAMPLE_QUERIES_RELIABILITY,
        key="reliability_pick",
        on_change=_sync_textbox_from_pick,
        args=("reliability_pick", "reliability_input"),
    )
    typed = st.text_input(
        "…or type your own",
        key="reliability_input",
    )

    if st.button("Run all four", key="reliability_go", type="primary"):
        if not typed:
            st.warning("Pick a sample query or type one above first.")
            st.stop()
        with st.spinner("Querying …"):
            keyword = run_keyword(settings, typed)
            vector = run_vector(settings, typed)
            hybrid = run_hybrid(settings, typed)
            semantic = run_semantic(settings, typed)

        st.markdown(f"### Question  \n> _{typed}_")

        with st.expander("❔  How do I read these scores?", expanded=True):
            st.markdown(
                "Each card below shows **one** score. The score type "
                "depends on the search mode:\n\n"
                "- **BM25 search score** (keyword, vector, hybrid) — a raw "
                "relevance number with **no upper bound**. Higher = better, "
                "but the scale changes per query and per index, so a value "
                "of `13.2` on one query is not comparable to `13.2` on "
                "another. You can rank *within* a query but you can't set a "
                "global 'trust this' threshold.\n"
                "- **Reranker score** (semantic) — a **calibrated 0–4** "
                "score from a deep model trained on human relevance "
                "judgments. The scale is **stable across queries and "
                "indexes**, which is what lets you build a 'should I trust "
                "this enough to feed an LLM?' gate. Rule of thumb:"
            )
            st.markdown(
                "<table style='font-size:0.9em'>"
                "<tr><td style='color:#16a34a;font-weight:600;padding-right:16px'>≥ 2.0</td>"
                "<td>High confidence — safe to ground an LLM answer on it.</td></tr>"
                "<tr><td style='color:#ca8a04;font-weight:600;padding-right:16px'>1.5 – 2.0</td>"
                "<td>Moderate — useful, but the LLM should hedge.</td></tr>"
                "<tr><td style='color:#dc2626;font-weight:600;padding-right:16px'>&lt; 1.5</td>"
                "<td>Low — better to say 'I don't have a sourced answer for that.'</td></tr>"
                "</table>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Vector results also have a `@search.score`, but it's "
                "derived from cosine distance and likewise unbounded by "
                "index — don't compare it across modes."
            )

        col_k, col_v, col_h, col_s = st.columns(4)
        with col_k:
            _result_card(keyword, color="#dc2626")
        with col_v:
            _result_card(vector, color="#ca8a04")
        with col_h:
            _result_card(hybrid, color="#2563eb")
        with col_s:
            _result_card(semantic, color="#16a34a")

        # Winner = mode with highest reranker score; fall back to search
        def _score(r: RunResult) -> float:
            if not r.hits:
                return -1.0
            return r.hits[0].reranker_score or (r.hits[0].search_score or 0.0) / 10

        runs = {
            "Keyword": keyword,
            "Vector": vector,
            "Hybrid": hybrid,
            "Semantic (hybrid + L2 reranker)": semantic,
        }
        winner_name, winner = max(runs.items(), key=lambda kv: _score(kv[1]))
        winner_top = winner.hits[0]
        st.success(
            f"**Winner: {winner_name}** — top doc *{winner_top.title}* "
            f"(reranker {winner_top.reranker_score:.3f})"
            if winner_top.reranker_score is not None
            else f"**Winner: {winner_name}** — top doc *{winner_top.title}*"
        )

        titles = {name: (r.hits[0].title if r.hits else None) for name, r in runs.items()}
        if len({t for t in titles.values() if t}) > 1:
            st.warning(
                "The four modes disagreed on the top result — read across the "
                "cards. The reranker score is what your downstream LLM should "
                "gate on, not the raw search score."
            )
        else:
            st.info(
                "All four modes converged on the same top document. The "
                "differences will show up in the **score** — and that's what "
                "matters for a 'should I trust this enough to feed an LLM' "
                "decision."
            )


# ---------------------------------------------------------------------------
# Tab 2 — Semantic configurations
# ---------------------------------------------------------------------------


with tab_semantic:
    st.subheader("Tuning the reranker by telling it where 'meaning' lives")
    st.markdown(
        "Most documents have multiple fields — a **title**, a **body**, "
        "a list of **keywords/tags**, sometimes a summary. The semantic "
        "reranker can't read all of them equally; you have to tell it which "
        "ones matter most for *your* users' questions. That recipe is called "
        "a **semantic configuration**, and you can define more than one on "
        "the same index."
    )
    st.markdown(
        "We pre-defined three configurations on our index:"
    )
    st.markdown(
        "| Config | What it tells the ranker |\n"
        "|---|---|\n"
        "| `default-balanced` | Look at title, body, and keywords roughly equally. |\n"
        "| `content-only` | Ignore the title and keywords — only the body matters. |\n"
        "| `title-and-keywords` | Title and keywords get the most weight; body is supporting. |"
    )
    st.markdown(
        "**What to watch for:** pick a question whose answer is in a "
        "document's **title** (e.g. *'transformer oil sample interpretation'* "
        "— there's a doc literally titled that) and see how the title-weighted "
        "config rockets it to #1. Then try a question that lives only in the "
        "body and watch `content-only` win. **Same data, same index, no "
        "retraining — just a different recipe.**"
    )

    pick2 = st.selectbox(
        "Sample queries",
        ["(choose one)"] + SAMPLE_QUERIES_SEMANTIC,
        key="semantic_pick",
        on_change=_sync_textbox_from_pick,
        args=("semantic_pick", "semantic_input"),
    )
    typed2 = st.text_input(
        "…or type your own",
        key="semantic_input",
    )

    if st.button("Compare configurations", key="semantic_go", type="primary"):
        if not typed2:
            st.warning("Pick a sample query or type one above first.")
            st.stop()
        with st.spinner("Querying three configurations …"):
            r_default = run_semantic(
                settings, typed2, semantic_configuration="default-balanced"
            )
            r_content = run_semantic(
                settings, typed2, semantic_configuration="content-only"
            )
            r_titlekw = run_semantic(
                settings, typed2, semantic_configuration="title-and-keywords"
            )

        st.markdown(f"### Question  \n> _{typed2}_")
        st.caption(
            "All three cards below show **reranker scores (0–4)**, since every "
            "config runs through the semantic ranker. Compare the scores "
            "directly — same scale, same query."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            _result_card(r_default, color="#2563eb")
        with c2:
            _result_card(r_content, color="#0891b2")
        with c3:
            _result_card(r_titlekw, color="#7c3aed")

        titles = {
            r.mode: (r.hits[0].title if r.hits else None)
            for r in (r_default, r_content, r_titlekw)
        }
        if len({t for t in titles.values() if t}) > 1:
            st.warning(
                "The three configurations disagreed on the top result. "
                "This is exactly the dial you tune to match your users' "
                "language: are they asking by equipment name (title-heavy) "
                "or by symptom (content-heavy)?"
            )
        else:
            st.info(
                "All three configs picked the same top doc — but the scores "
                "below show by how much. Pick a query where title vs. body "
                "matters more to see them split."
            )


# ---------------------------------------------------------------------------
# Tab 3 — Agentic retrieval
# ---------------------------------------------------------------------------


with tab_agentic:
    st.subheader("One REST call replaces an orchestration layer")
    st.markdown(
        "Until Build 2026, building a RAG app that could handle a **multi-part "
        "question** meant writing your own orchestration code: detect that the "
        "question has multiple parts, split it, fire a query for each part, "
        "merge the results, dedupe, rerank, and *then* call the LLM. That's a "
        "few hundred lines of code per app, and every team rewrites it."
    )
    st.markdown(
        "A **Knowledge Base** is a managed agent that does all of that for "
        "you. You point it at a **Knowledge Source** (your index) and a chat "
        "model, then you `POST /retrieve` with the user's question. It:\n"
        "1. Decomposes the question into sub-queries\n"
        "2. Runs them in parallel against the index\n"
        "3. Reranks everything with the semantic ranker\n"
        "4. Returns the grounding chunks + a trace of what it did\n\n"
        "Then your code makes a normal chat-completion call with those "
        "chunks. The result is a **cited prose answer** — you'll see it "
        "highlighted in green below."
    )
    st.markdown(
        "**What to watch for:** the *'How the planner decomposed it'* table "
        "— that's the part that used to be your code. Now it's a built-in "
        "feature of Azure AI Search."
    )
    st.caption(
        "Prerequisite (one time, already done in this environment): "
        "`python src/agentic_demo.py --setup`"
    )

    pick3 = st.selectbox(
        "Sample multi-part questions",
        ["(choose one)"] + SAMPLE_QUERIES_AGENTIC,
        key="agentic_pick",
        on_change=_sync_textbox_from_pick,
        args=("agentic_pick", "agentic_input"),
    )
    typed3 = st.text_area(
        "…or type your own",
        key="agentic_input",
        height=110,
    )

    if st.button("Ask the knowledge base", key="agentic_go", type="primary"):
        if not typed3:
            st.warning("Pick a sample question or type one above first.")
            st.stop()
        if PUBLIC_DEMO:
            used = st.session_state.get("tab3_calls", 0)
            if used >= TAB3_LIMIT_PER_SESSION:
                st.warning(
                    f"⚠️ Per-session cap reached "
                    f"({TAB3_LIMIT_PER_SESSION} agentic queries). This bounds "
                    "Azure OpenAI cost on the shared deploy. **Refresh the "
                    "page to reset**, or run the demo locally against your "
                    "own subscription for unlimited use."
                )
                st.stop()
            st.session_state["tab3_calls"] = used + 1
            st.caption(
                f"Session usage: {st.session_state['tab3_calls']} / "
                f"{TAB3_LIMIT_PER_SESSION} agentic queries."
            )
        with st.spinner("Decomposing, retrieving, reranking …"):
            try:
                resp = retrieve(settings, typed3)
            except Exception as exc:  # surface auth / setup failures clearly
                st.error(
                    "Retrieve failed. Confirm `python src/agentic_demo.py "
                    f"--setup` has been run and your identity has Search "
                    f"Index Data Reader.\n\n{exc}"
                )
                st.stop()

        grounding = _extract_answer(resp)

        with st.spinner("Synthesizing answer with gpt-5-mini ..."):
            try:
                answer = synthesize_answer(settings, typed3, grounding)
            except Exception as exc:
                answer = None
                st.error(f"Answer synthesis failed: {exc}")

        # --- Headline: the synthesized prose answer ---
        st.markdown("### Synthesized answer (cited)")
        if answer:
            st.markdown(
                f"<div style='padding:16px;background:#f0fdf4;"
                f"border-left:6px solid #16a34a;border-radius:4px'>{answer}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("(no synthesized answer available)")

        # --- The planning trace ---
        st.markdown("### How the planner decomposed it")
        rows = []
        for step in resp.get("activity", []):
            args = step.get("searchIndexArguments") or step.get("SearchIndexArguments")
            if args:
                rows.append(
                    {
                        "subquery": args.get("search") or args.get("Search"),
                        "semantic_config": (
                            args.get("semanticConfigurationName")
                            or args.get("SemanticConfigurationName")
                        ),
                        "elapsed_ms": step.get("elapsedMs", step.get("ElapsedMs")),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("(No subquery activity reported — the planner answered directly.)")

        # --- Grounding chunks ---
        st.markdown("### Grounding chunks (what the answer is built from)")
        ref_rows = []
        for i, ref in enumerate(resp.get("references", [])):
            source_data = ref.get("sourceData") or ref.get("SourceData") or {}
            ref_rows.append(
                {
                    "ref_id": i,
                    "reranker": ref.get("rerankerScore", ref.get("RerankerScore")),
                    "doc_key": ref.get("docKey", ref.get("DocKey")),
                    "title": source_data.get("title") if isinstance(source_data, dict) else "",
                    "source": (
                        source_data.get("source") if isinstance(source_data, dict) else ""
                    ),
                }
            )
        if ref_rows:
            st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

        with st.expander("Raw retrieve response (for the curious)"):
            st.json(resp)
        with st.expander("Raw grounding payload fed to the chat model"):
            st.code(grounding, language="json")
