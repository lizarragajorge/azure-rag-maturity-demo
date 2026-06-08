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
        "code": {
            "caption": "There's no retrieval. The whole 'integration' is one chat call. Adapted from `src/agentic_demo.py` — same SDK pattern, no grounding payload.",
            "file": "src/agentic_demo.py",
            "language": "python",
            "body": (
                'from openai import AzureOpenAI\n'
                '\n'
                'aoai = AzureOpenAI(\n'
                '    azure_endpoint=settings.aoai_endpoint,\n'
                '    azure_ad_token_provider=get_aoai_token_provider(),\n'
                '    api_version="2024-10-21",\n'
                ')\n'
                '\n'
                'resp = aoai.chat.completions.create(\n'
                '    model=settings.aoai_chat_deployment,  # gpt-5-mini\n'
                '    messages=[\n'
                '        {"role": "system", "content": "You answer questions."},\n'
                '        {"role": "user",   "content": question},\n'
                '    ],\n'
                ')\n'
                'print(resp.choices[0].message.content)\n'
                '# ⚠️  The model has no idea what *your* docs say. It will\n'
                '#    confidently make up gas-leak procedures and FERC numbers.'
            ),
        },
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
        "code": {
            "caption": (
                "Verbatim from `src/retrieval_lab.py::run_keyword`. "
                "`client.search(search_text=...)` with no vector_queries and "
                "no query_type defaults to **BM25 keyword search** — this is "
                "what Azure AI Search does out of the box."
            ),
            "file": "src/retrieval_lab.py",
            "language": "python",
            "body": (
                'def run_keyword(settings, query, top=5):\n'
                '    """Plain BM25 keyword search. Strong on exact terms,\n'
                '    weak on paraphrase.\"\"\"\n'
                '    client = SearchClient(\n'
                '        endpoint=settings.search_endpoint,\n'
                '        index_name=settings.search_index_name,\n'
                '        credential=get_credential(),\n'
                '    )\n'
                '    results = client.search(\n'
                '        search_text=query,   # ← BM25; no vector, no semantic\n'
                '        top=top,\n'
                '        select=["id", "title", "content", "source"],\n'
                '    )\n'
                '    return list(results)\n'
                '# Then you stuff results into the prompt and hope the LLM\n'
                '# picks the right chunk. Breaks on paraphrase.'
            ),
        },
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
    DOCS[(Documents)] -. chunked .-> EMB
    EMB[Azure OpenAI<br/>text-embedding-3-large] -. vectors .-> SEARCH
    U([User]) -- question --> APP[Your app]
    APP -- vector query<br/>VectorizableTextQuery --> SEARCH[Azure AI Search<br/>vector index]
    SEARCH -- integrated vectorizer<br/>embeds query --> EMB
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
        "code": {
            "caption": (
                "Verbatim from `src/retrieval_lab.py::run_vector`. "
                "`VectorizableTextQuery` is the *integrated vectorizer* path: "
                "the index embeds the query for you using the "
                "`AzureOpenAIVectorizer` you declared on the field — no "
                "client-side embedding call required."
            ),
            "file": "src/retrieval_lab.py",
            "language": "python",
            "body": (
                'from azure.search.documents.models import VectorizableTextQuery\n'
                '\n'
                'def run_vector(settings, query, top=5):\n'
                '    client = SearchClient(...)\n'
                '    vector_query = VectorizableTextQuery(\n'
                '        text=query,             # AI Search embeds it for you\n'
                '        k_nearest_neighbors=top,\n'
                '        fields="content_vector",\n'
                '    )\n'
                '    results = client.search(\n'
                '        search_text=None,       # ← vector only, no BM25\n'
                '        vector_queries=[vector_query],\n'
                '        top=top,\n'
                '    )\n'
                '    return list(results)\n'
                '# Handles paraphrase. Can drift on rare proper nouns or\n'
                '# numeric thresholds ("25 cal/cm²") with no semantic neighborhood.'
            ),
        },
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
    DOCS[(Documents)] -. chunked .-> EMB
    EMB[Azure OpenAI<br/>text-embedding-3-large] -. vectors .-> SEARCH
    U([User]) -- question --> APP[Your app]
    APP -- hybrid query<br/>text + VectorizableTextQuery --> SEARCH[Azure AI Search<br/>HYBRID:<br/>BM25 + vector + L2 reranker]
    SEARCH -- integrated vectorizer<br/>embeds query --> EMB
    SEARCH -- top-k + rerankerScore --> APP
    APP -- gate: score ≥ 2.0? --> AOAI[Azure OpenAI<br/>gpt-5-mini]
    AOAI -- cited answer --> APP --> U
    classDef ai fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#000
    classDef search fill:#bbf7d0,stroke:#16a34a,stroke-width:3px,color:#000
    classDef app fill:#f3f4f6,stroke:#6b7280,color:#000
    classDef data fill:#fce7f3,stroke:#be185d,color:#000
    class AOAI,EMB ai
    class SEARCH search
    class APP app
    class DOCS data
""",
        "code": {
            "caption": (
                "Verbatim from `src/retrieval_lab.py::run_semantic` — this is "
                "what Tabs 1 & 2 of this app actually call. Send **both** "
                "`search_text` and `vector_queries` (= hybrid + RRF), add "
                "`query_type=SEMANTIC` + a `semantic_configuration_name` (= L2 "
                "reranker). The `rerankerScore` on each hit is your gate."
            ),
            "file": "src/retrieval_lab.py",
            "language": "python",
            "body": (
                'from azure.search.documents.models import (\n'
                '    QueryType, QueryCaptionType, QueryAnswerType,\n'
                '    VectorizableTextQuery,\n'
                ')\n'
                '\n'
                'def run_semantic(settings, query, top=5,\n'
                '                 semantic_configuration="default-balanced"):\n'
                '    client = SearchClient(...)\n'
                '    vector_query = VectorizableTextQuery(\n'
                '        text=query, k_nearest_neighbors=50,\n'
                '        fields="content_vector",\n'
                '    )\n'
                '    results = client.search(\n'
                '        search_text=query,             # BM25\n'
                '        vector_queries=[vector_query], # + vector → hybrid via RRF\n'
                '        top=top,\n'
                '        query_type=QueryType.SEMANTIC,         # ← L2 reranker on\n'
                '        semantic_configuration_name=semantic_configuration,\n'
                '        query_caption=QueryCaptionType.EXTRACTIVE,\n'
                '        query_answer=QueryAnswerType.EXTRACTIVE,\n'
                '    )\n'
                '    return list(results)\n'
                '\n'
                '# Each hit now has @search.rerankerScore (0–4).\n'
                '# Three-band gate:  ≥ 2.0 ground   •   1.5–2.0 caveat   •   < 1.5 refuse.'
            ),
        },
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
        "code": {
            "caption": (
                "Verbatim from `src/agentic_demo.py::retrieve` — this is what "
                "Tab 3 calls. **One REST call** to `POST /knowledgebases/{name}/retrieve` "
                "on api-version `2026-05-01-preview`. Sending `messages` "
                "(instead of `intents`) flips the KB into **modelQueryPlanning** "
                "mode: the chat model registered on the KB decomposes the "
                "question, runs the sub-queries in parallel, and returns the "
                "full activity trace (planner → N × searchIndex → reasoning)."
            ),
            "file": "src/agentic_demo.py",
            "language": "python",
            "body": (
                'RETRIEVE_API_VERSION = "2026-05-01-preview"  # enables LLM planning\n'
                '\n'
                'def retrieve(settings, question, reasoning_effort="low"):\n'
                '    url = (\n'
                '        f"{settings.search_endpoint}"\n'
                '        f"/knowledgebases/{settings.knowledge_base_name}"\n'
                '        f"/retrieve?api-version={RETRIEVE_API_VERSION}"\n'
                '    )\n'
                '    body = {\n'
                '        "messages": [{                  # ← messages, not intents,\n'
                '            "role": "user",            #   triggers the planner\n'
                '            "content": [{"type": "text", "text": question}],\n'
                '        }],\n'
                '        "includeActivity": True,\n'
                '        "outputMode": "extractiveData",     # we synthesize ourselves\n'
                '        "retrievalReasoningEffort": {"kind": reasoning_effort},\n'
                '        "maxRuntimeInSeconds": 60,\n'
                '        "knowledgeSourceParams": [{\n'
                '            "knowledgeSourceName": settings.knowledge_source_name,\n'
                '            "kind": "searchIndex",\n'
                '            "includeReferences": True,\n'
                '            "includeReferenceSourceData": True,\n'
                '        }],\n'
                '    }\n'
                '    r = requests.post(url, headers=_headers(...),\n'
                '                      data=json.dumps(body), timeout=120)\n'
                '    r.raise_for_status()\n'
                '    return r.json()\n'
                '# activity → [modelQueryPlanning, searchIndex × N, agenticReasoning]'
            ),
        },
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

    code = level.get("code")
    if code:
        with st.expander(
            f"👩‍💻  Show the code for {level['id']} (from this repo)",
            expanded=False,
        ):
            st.markdown(
                f"{code['caption']}  \n"
                f"_See [{code['file']}](https://github.com/lizarragajorge/azure-rag-maturity-demo/blob/main/{code['file']}) in the repo._"
            )
            st.code(code["body"], language=code.get("language", "python"))

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
    initial_sidebar_state="collapsed",
)

# Hide Streamlit's built-in chrome (Deploy button → Streamlit Community Cloud /
# Snowflake; main menu → Streamlit branding & share dialogs; footer).
st.markdown(
    """
    <style>
      [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
      [data-testid="stDecoration"] {display: none;}
      [data-testid="stStatusWidget"] {visibility: hidden; height: 0; position: fixed;}
      #MainMenu {visibility: hidden;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_settings() -> Settings:
    return Settings.from_env()


# Each query is engineered to make ONE method shine and the others either
# pull the wrong doc or rank the right doc lower. That divergence is the
# entire point of Tab 1 — if all four methods agree, the demo is boring.
SAMPLE_QUERIES_RELIABILITY = [
    # Pure paraphrase: target doc 011 (mercaptan/odorization) but query uses
    # ZERO of its tokens. Vector should win; BM25 will latch onto any doc that
    # mentions "gas" or "leak".
    "the stuff they add to natural gas so people can smell leaks",
    # Exact regulation cite: target doc 012 (FAC-003). BM25 nails this on the
    # first token. Vector will pull other transmission/right-of-way docs.
    "FAC-003 minimum vegetation clearance for transmission",
    # Hybrid sweet spot: rare numeric threshold ("25 cal/cm²") + paraphrased
    # question ("what category"). BM25 catches the number, vector catches the
    # intent, hybrid fuses both, reranker confirms.
    "what category of PPE do I need at 25 cal per cm squared",
    # Reranker disambiguates: "de-energize during fire season" is paraphrase
    # for PSPS (doc 003), but "fire" also appears in 012 (veg mgmt). Vector
    # may split; only the reranker reads the QUESTION and picks PSPS.
    "when should we proactively de-energize lines during fire season",
    # Ambiguity stress test: multiple docs share "customer" + "power" +
    # "notify". BM25 fans out; vector splits; only the reranker picks the
    # outage-communication template (doc 004).
    "what do we have to tell residential customers when their power is out",
]

# Each query targets a SPECIFIC semantic config:
#   • a near-title match → `title-and-keywords` wins by a wide margin
#   • a body-only detail (title is misleading) → `content-only` wins
#   • a mixed query (title language + body intent) → `default-balanced` wins
SAMPLE_QUERIES_SEMANTIC = [
    # → title-and-keywords (nearly the literal title of doc 002)
    "pad-mount transformer preventive maintenance schedule",
    # → title-and-keywords (nearly the literal title of doc 010)
    "substation arc-flash PPE selection matrix",
    # → content-only: DGA / dissolved gas analysis lives in BODY of doc 002,
    #   but doc 002's title is just "PM Schedule" — useless for ranking
    "what lab test tells us if a transformer has internal damage",
    # → content-only: "sniff test failure" lives in BODY of doc 011, title is
    #   "Mercaptan Injection Requirements" — actively misleading
    "what do we do if the rotten-egg smell fades from the gas in the line",
    # → default-balanced: "Itron" is in keywords, "swap-out" is body language,
    #   title is "Field Procedure". Each config picks up a different signal.
    "Itron meter swap-out gotchas in the field",
]

SAMPLE_QUERIES_AGENTIC = [
    # Spans docs 003 (PSPS), 004 (outage templates). Planner should split into
    # "planned PSPS notification", "unplanned storm outage notification",
    # "ETR-update timing requirements".
    (
        "What's the difference between how we notify residential customers "
        "about a planned PSPS de-energization versus an unplanned storm "
        "outage, and which one has stricter ETR-update requirements?"
    ),
    # Spans docs 001 (gas leak Grade 1), 014 (CIP-008 cyber response). Two
    # different incident types in the same scenario — must be decomposed.
    (
        "Compare the first-hour notification obligations for a Grade 1 gas "
        "leak versus a CIP-008 cyber incident at a substation — who do we "
        "call, in what order, and how long do we have for each?"
    ),
    # Spans docs 008 (AMI install procedure), 009 (billing dispute SOP).
    # "What was logged during the install" vs "what the dispute SOP requires".
    (
        "A residential customer says their bill spiked right after their AMI "
        "meter was installed. What field-procedure steps should have been "
        "logged during the install, and what does our billing-dispute SOP "
        "require us to do?"
    ),
    # Spans docs 013 (cold-weather load shed), 004 (outage comms). Three
    # natural sub-questions joined by "how does that interact" and "what's the".
    (
        "Cold snap forecast: what triggers firm-gas curtailment, how does "
        "that interact with rolling-blackout load shedding, and what's the "
        "customer communication template we'd use?"
    ),
    # Spans docs 016 (small-gen interconnection / IEEE 1547), 007 (rate case
    # tariff). Engineering rules + tariff/legal evidence — different domains.
    (
        "Before approving a residential rooftop solar interconnection, what "
        "IEEE 1547 / Rule 21 requirements apply, AND what evidence does the "
        "2026 rate-case tariff require us to keep for net-metering disputes?"
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


def _render_rank_shift(
    per_method: dict[str, RunResult],
    *,
    sort_priority: tuple[str, ...],
) -> None:
    """Render a per-doc rank table across methods, highlighting top ranks.

    `per_method` maps method label → its `RunResult`. `sort_priority` is the
    column order used to sort rows so the "best by the method we trust most"
    floats to the top.
    """
    def _best_rank_per_doc(result: RunResult) -> dict[str, tuple[int, str]]:
        ranks: dict[str, tuple[int, str]] = {}
        for rank, h in enumerate(result.hits, 1):
            if h.parent_id not in ranks:
                ranks[h.parent_id] = (rank, h.title)
        return ranks

    per_lookup = {name: _best_rank_per_doc(r) for name, r in per_method.items()}
    all_docs: dict[str, str] = {}
    for ranks in per_lookup.values():
        for pid, (_, title) in ranks.items():
            all_docs[pid] = title

    if not all_docs:
        st.info("No results from any method.")
        return

    method_names = list(per_method.keys())
    rows = []
    for pid, title in all_docs.items():
        row: dict[str, object] = {"Document": title}
        for name in method_names:
            ranks = per_lookup[name]
            row[name] = ranks[pid][0] if pid in ranks else None
        rows.append(row)

    rank_df = pd.DataFrame(rows)
    rank_df["_sort"] = rank_df[list(sort_priority)].min(axis=1, skipna=True)
    rank_df = (
        rank_df.sort_values("_sort", na_position="last")
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

    def _highlight_rank(val):
        if pd.isna(val):
            return "color:#9ca3af"
        if val == 1:
            return "background-color:#dcfce7;color:#166534;font-weight:700"
        if val == 2:
            return "background-color:#fef3c7;color:#854d0e"
        if val == 3:
            return "background-color:#fef9c3;color:#854d0e"
        return ""

    styled = (
        rank_df.style
        .map(_highlight_rank, subset=method_names)
        .format(na_rep="—", precision=0)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


st.title("RAG after Build 2026")

settings = get_settings()
st.caption(
    f"Live against Azure: index `{settings.search_index_name}` • "
    f"embedding `{settings.aoai_embedding_deployment}` • "
    f"chat `{settings.aoai_chat_deployment}`"
)

PUBLIC_DEMO = os.environ.get("PUBLIC_DEMO", "").lower() == "true"
TAB3_LIMIT_PER_SESSION = 10

if PUBLIC_DEMO:
    st.info(
        f"👋 **Public demo.** This site is shared and rate-limited — "
        f"each browser session can run **up to {TAB3_LIMIT_PER_SESSION} "
        f"agentic queries (Tab 3)** before Azure OpenAI calls are paused. "
        f"Refresh the page to reset, or [clone the repo]"
        f"(https://github.com/lizarragajorge/azure-rag-maturity-demo) and run it against your own subscription "
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

with st.expander("📖  RAG vocabulary cheat-sheet"):
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

(
    tab_overview,
    tab_reliability,
    tab_semantic,
    tab_agentic,
    tab_playbook,
) = st.tabs(
    [
        "0.  Overview & maturity model",
        "1.  Why retrieval matters",
        "2.  Tuning what 'relevant' means",
        "3.  Agentic retrieval  (new in Build 2026)",
        "4.  Quality playbook  (12 levers)",
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
                "differences will show up in the **rank shift** and **score "
                "scale** tables below — that's where the value of layering on "
                "hybrid + reranker actually pays off."
            )

        # -- Rank-shift table: same docs, different positions per method ----
        st.markdown("### 📊  Rank shift — same docs, different positions")
        st.markdown(
            "Each row is a document that appeared in **at least one method's "
            "top 5**. The number = the rank that method gave it. A blank cell "
            "means the method didn't surface that doc at all. **Watch the "
            "columns disagree** — that's the value of layering methods."
        )
        _render_rank_shift(
            {
                "BM25": keyword,
                "Vector": vector,
                "Hybrid": hybrid,
                "Semantic": semantic,
            },
            sort_priority=("Semantic", "Hybrid", "Vector", "BM25"),
        )

        # -- Score-scale table: prove the reranker is the only gateable score
        st.markdown("### 📈  Top-1 scores — different scales, only one is gateable")
        st.caption(
            "Each method emits a score on a **different scale**. You can rank "
            "*within* a column but you can't compare *across* columns — except "
            "the **semantic reranker (0–4)**, which is calibrated across "
            "queries and indexes. That's the score your LLM-trust gate uses."
        )
        score_rows = [
            {
                "Method": "BM25 keyword",
                "Top-1 score": round(float(keyword.hits[0].search_score or 0.0), 3) if keyword.hits else 0.0,
                "Scale": "unbounded · per-query",
                "Gateable across queries?": "No",
            },
            {
                "Method": "Vector (cosine)",
                "Top-1 score": round(float(vector.hits[0].search_score or 0.0), 3) if vector.hits else 0.0,
                "Scale": "~0–1 · per-query",
                "Gateable across queries?": "No",
            },
            {
                "Method": "Hybrid (RRF fusion)",
                "Top-1 score": round(float(hybrid.hits[0].search_score or 0.0), 3) if hybrid.hits else 0.0,
                "Scale": "~0–1 · per-query",
                "Gateable across queries?": "No",
            },
            {
                "Method": "Semantic raw search",
                "Top-1 score": round(float(semantic.hits[0].search_score or 0.0), 3) if semantic.hits else 0.0,
                "Scale": "~0–1 · per-query",
                "Gateable across queries?": "No",
            },
            {
                "Method": "Semantic RERANKER",
                "Top-1 score": round(float(semantic.hits[0].reranker_score or 0.0), 3) if semantic.hits else 0.0,
                "Scale": "0–4 · stable across queries",
                "Gateable across queries?": "Yes ✓",
            },
        ]
        st.dataframe(
            pd.DataFrame(score_rows),
            use_container_width=True,
            hide_index=True,
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
                "All three configs picked the same top doc — but the **rank "
                "shift table** below shows how the *runner-up* docs moved. "
                "That's where tuning the config buys you precision."
            )

        # -- Rank-shift table for the three semantic configs ---------------
        st.markdown("### 📊  Rank shift across configs — same data, different recipes")
        st.markdown(
            "Each row is a document that appeared in at least one config's "
            "top 5. Watch how a title-heavy recipe pushes title-matching "
            "docs up, while content-only pushes body-matching docs up. "
            "All scores here are on the same 0–4 reranker scale."
        )
        _render_rank_shift(
            {
                "default-balanced": r_default,
                "content-only": r_content,
                "title-and-keywords": r_titlekw,
            },
            sort_priority=(
                "default-balanced",
                "content-only",
                "title-and-keywords",
            ),
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


# ---------------------------------------------------------------------------
# Tab 4 — Quality playbook: 11 levers to improve AI Search retrieval
# ---------------------------------------------------------------------------


RETRIEVAL_QUALITY_LEVERS = [
    # ---------- Phase 1: Index-time ----------
    {
        "phase": "🏗️ Index-time — make the data findable before queries arrive",
        "title": "1. Chunking strategy",
        "symptom": "The top-ranked chunk doesn't contain the answer even though the full document does.",
        "lever": (
            "Chunks that are too big bury the relevant sentence in noise; "
            "chunks that are too small lose the surrounding context that makes "
            "them embeddable. Target **800–1500 chars with 10–20% overlap** "
            "on natural boundaries (paragraphs, headings). For long structured "
            "docs (policies, runbooks, contracts), use **semantic chunking** "
            "that splits on the heading hierarchy instead of fixed windows."
        ),
        "in_repo": "✅ See `src/common.py::_split_into_chunks` — 1200 chars + 200 overlap, paragraph-aware.",
        "code": {
            "language": "python",
            "file": "src/common.py",
            "body": (
                "def _split_into_chunks(text, target_chars=1200, overlap=200):\n"
                "    paras = re.split(r'\\n\\s*\\n', text.strip())\n"
                "    chunks, buf = [], ''\n"
                "    for p in paras:\n"
                "        if len(buf) + len(p) > target_chars and buf:\n"
                "            chunks.append(buf)\n"
                "            buf = buf[-overlap:] + '\\n\\n' + p  # carry overlap\n"
                "        else:\n"
                "            buf = (buf + '\\n\\n' + p) if buf else p\n"
                "    if buf: chunks.append(buf)\n"
                "    return chunks"
            ),
        },
        "effort": "S",
        "impact": "🔥 High — the single biggest knob. Re-chunking often beats switching embedding models.",
    },
    {
        "phase": "🏗️ Index-time — make the data findable before queries arrive",
        "title": "2. Field design + metadata enrichment",
        "symptom": "BM25 ranks the wrong chunk because the right one doesn't repeat the literal keyword in its body.",
        "lever": (
            "Don't index a single `content` field — give the ranker more signal:\n\n"
            "- **`title`** (searchable, boosted) — the document name or section heading.\n"
            "- **`keywords`** (searchable, Collection(Edm.String)) — domain terms extracted by a skillset (e.g. *PSPS, CIP-008, AMI, mercaptan*).\n"
            "- **`summary`** (searchable) — 1–2 sentence LLM-generated abstract.\n"
            "- **`docType`, `regulator`, `revisionDate`** (filterable + facetable) — for filters and scoring profiles below.\n\n"
            "Use the **AI enrichment pipeline** (skillsets) to populate `keywords` "
            "and `summary` at index time with one LLM call per doc. Then point "
            "your semantic config at title+keywords (see Tab 2)."
        ),
        "in_repo": "✅ Index has `title`, `parent_id`, `chunk`, `content_vector`. 🛠️ Adding `keywords` + `summary` via a skillset is the next obvious lever.",
        "code": {
            "language": "python",
            "file": "src/build_index.py",
            "body": (
                "# Add to the SearchIndex field list:\n"
                "SearchableField(name='title', analyzer_name='en.microsoft'),\n"
                "SearchableField(\n"
                "    name='keywords', collection=True,\n"
                "    analyzer_name='keyword',  # exact-match for jargon\n"
                "),\n"
                "SearchableField(name='summary', analyzer_name='en.microsoft'),\n"
                "SimpleField(name='docType',  filterable=True, facetable=True),\n"
                "SimpleField(name='revisionDate', type=SearchFieldDataType.DateTimeOffset,\n"
                "            filterable=True, sortable=True),\n"
                "\n"
                "# In the semantic config, prioritize title + keywords:\n"
                "SemanticPrioritizedFields(\n"
                "    title_field=SemanticField(field_name='title'),\n"
                "    keywords_fields=[SemanticField(field_name='keywords')],\n"
                "    content_fields=[SemanticField(field_name='chunk')],\n"
                ")"
            ),
        },
        "effort": "M",
        "impact": "🔥 High — typically +10–20% NDCG, and unlocks every other lever (filters, scoring profiles, semantic ranker).",
    },
    {
        "phase": "🏗️ Index-time — make the data findable before queries arrive",
        "title": "3. Synonym maps",
        "symptom": "Users type *'rolling blackout'* but the docs say *'load shedding'* — BM25 misses it; vector sort-of catches it.",
        "lever": (
            "A **SynonymMap** rewrites query terms before BM25 sees them. Perfect "
            "for industry jargon and abbreviation expansion. One-time setup, "
            "near-zero query cost, and it stacks on top of vector + semantic."
        ),
        "in_repo": "🛠️ Not yet in this repo. Add 4–6 maps for the utility domain (PSPS↔public safety power shutoff, AMI↔smart meter, CIP↔NERC CIP, etc.).",
        "code": {
            "language": "python",
            "file": "src/build_index.py (new)",
            "body": (
                "from azure.search.documents.indexes.models import SynonymMap\n"
                "\n"
                "smap = SynonymMap(\n"
                "    name='utility-jargon',\n"
                "    # equivalent terms; one rule per line\n"
                "    synonyms='\\n'.join([\n"
                "        'psps, public safety power shutoff, de-energization',\n"
                "        'load shedding, rolling blackout, controlled outage',\n"
                "        'ami, smart meter, advanced metering infrastructure',\n"
                "        'mercaptan, odorant, thiol',\n"
                "        'cip-008, nerc cip-008, cyber incident response',\n"
                "    ]),\n"
                ")\n"
                "index_client.create_or_update_synonym_map(smap)\n"
                "\n"
                "# Then attach to the searchable fields you care about:\n"
                "SearchableField(name='chunk', synonym_map_names=['utility-jargon'])"
            ),
        },
        "effort": "S",
        "impact": "Medium — surgical win for vocabulary mismatch, especially in regulated industries.",
    },
    # ---------- Phase 2: Query-time ----------
    {
        "phase": "🔍 Query-time — shape each call so the index returns the right slice",
        "title": "4. Filters (narrow the corpus before scoring)",
        "symptom": "The reranker has to fight through 50 chunks from the wrong document type before finding the right one.",
        "lever": (
            "Filters are free and aggressive — use them. If the user (or the "
            "agent) gives you a regulator, doc type, region, or date range, "
            "filter on it. Filters happen **before** scoring, so they shrink "
            "the candidate pool the ranker has to consider, which improves "
            "both quality and latency."
        ),
        "in_repo": "🛠️ Not used in this repo yet — corpus is tiny. Real deployments at 100k+ docs essentially require filters to stay performant.",
        "code": {
            "language": "python",
            "file": "src/retrieval_lab.py",
            "body": (
                "# In your search call:\n"
                "results = client.search(\n"
                "    search_text=query,\n"
                "    filter=(\n"
                "        \"docType eq 'runbook' \"\n"
                "        \"and regulator eq 'NERC' \"\n"
                "        \"and revisionDate ge 2025-01-01T00:00:00Z\"\n"
                "    ),\n"
                "    select=['parent_id','title','chunk'],\n"
                "    top=5,\n"
                ")"
            ),
        },
        "effort": "S",
        "impact": "🔥 High at scale — filters routinely cut latency 5–10x and bump precision because the ranker isn't distracted.",
    },
    {
        "phase": "🔍 Query-time — shape each call so the index returns the right slice",
        "title": "5. Scoring profiles (boost by recency, source authority, doc type)",
        "symptom": "Two chunks tie on text relevance but one is from a 2026 revision and one is from a 2019 obsolete spec — the obsolete one wins half the time.",
        "lever": (
            "A **scoring profile** is a function that multiplies the base "
            "search score by configurable signals. Common boosts: **recency** "
            "(newer revisions rank higher), **source authority** (regulator-issued > "
            "internal-wiki), **doc-type weight** (runbooks > marketing). "
            "Attached to the index, applied per-query."
        ),
        "in_repo": "🛠️ Not in this repo. Highest-value addition for any docs-with-revisions corpus.",
        "code": {
            "language": "python",
            "file": "src/build_index.py (new)",
            "body": (
                "from azure.search.documents.indexes.models import (\n"
                "    ScoringProfile, FreshnessScoringFunction,\n"
                "    FreshnessScoringParameters, TextWeights,\n"
                ")\n"
                "\n"
                "profile = ScoringProfile(\n"
                "    name='prefer-recent-authoritative',\n"
                "    text_weights=TextWeights(weights={\n"
                "        'title':    5.0,   # title hits worth 5x\n"
                "        'keywords': 3.0,\n"
                "        'chunk':    1.0,\n"
                "    }),\n"
                "    functions=[\n"
                "        FreshnessScoringFunction(\n"
                "            field_name='revisionDate', boost=2.0,\n"
                "            parameters=FreshnessScoringParameters(\n"
                "                boosting_duration='P730D',  # 2-year half-life\n"
                "            ),\n"
                "            interpolation='quadratic',\n"
                "        ),\n"
                "    ],\n"
                ")\n"
                "index.scoring_profiles.append(profile)\n"
                "index.default_scoring_profile = 'prefer-recent-authoritative'"
            ),
        },
        "effort": "M",
        "impact": "🔥 High — directly fixes the *'stale doc kept winning'* failure mode.",
    },
    {
        "phase": "🔍 Query-time — shape each call so the index returns the right slice",
        "title": "6. Hybrid weight + RRF tuning",
        "symptom": "Hybrid is *worse* than vector-only on some queries because BM25 is dragging in noise.",
        "lever": (
            "Hybrid search merges BM25 + vector results using **Reciprocal "
            "Rank Fusion (RRF)**. The defaults are reasonable but tunable. "
            "Two knobs:\n\n"
            "- **Vector weight** (`VectorizableTextQuery(weight=...)`) — bump above "
            "1.0 when your queries are paraphrastic; lower below 1.0 when "
            "users type exact identifiers (model numbers, regulation cites).\n"
            "- **`top` per modality** — increase the candidate pool before "
            "RRF picks final winners (e.g. `k_nearest_neighbors=50` even "
            "though you only return top=5)."
        ),
        "in_repo": "✅ Tab 1 → Hybrid uses the default weight. Try bumping vector to 2.0 for paraphrase-heavy queries.",
        "code": {
            "language": "python",
            "file": "src/retrieval_lab.py",
            "body": (
                "from azure.search.documents.models import VectorizableTextQuery\n"
                "\n"
                "results = client.search(\n"
                "    search_text=query,           # BM25 side\n"
                "    vector_queries=[VectorizableTextQuery(\n"
                "        text=query,\n"
                "        fields='content_vector',\n"
                "        k_nearest_neighbors=50,  # ← wide candidate pool\n"
                "        weight=2.0,              # ← bias toward vector\n"
                "    )],\n"
                "    query_type='semantic',\n"
                "    semantic_configuration_name='default',\n"
                "    top=5,\n"
                ")"
            ),
        },
        "effort": "S",
        "impact": "Medium — small but reliable; A/B test before committing.",
    },
    # ---------- Phase 3: Rerank & gate ----------
    {
        "phase": "🎯 Rerank & gate — make the system honest about what it doesn't know",
        "title": "7. Semantic reranker config (title vs keywords vs content)",
        "symptom": "Two configs (same data, different recipes) return different top-1 docs — Tab 2 makes this concrete.",
        "lever": (
            "The semantic ranker re-scores the top ~50 BM25/vector candidates "
            "using a deep learning model. You control which fields it pays "
            "attention to via the **semantic configuration**. Three named "
            "configs you should always have:\n\n"
            "- **title-and-keywords** — title is the strongest signal; perfect for FAQ-style queries.\n"
            "- **content-only** — best when titles are generic or boilerplate.\n"
            "- **default-balanced** — title + keywords + content; safest default.\n\n"
            "Pick at query time based on query shape (or A/B test)."
        ),
        "in_repo": "✅ Tab 2 demos all three side-by-side. See `src/retrieval_lab.py::run_semantic` and the three configs in `src/build_index.py`.",
        "code": {
            "language": "python",
            "file": "src/build_index.py",
            "body": (
                "SemanticConfiguration(\n"
                "    name='title-and-keywords',\n"
                "    prioritized_fields=SemanticPrioritizedFields(\n"
                "        title_field=SemanticField(field_name='title'),\n"
                "        keywords_fields=[SemanticField(field_name='keywords')],\n"
                "        # NO content_fields — force the model to trust\n"
                "        # title + keywords for matching\n"
                "    ),\n"
                ")"
            ),
        },
        "effort": "S",
        "impact": "Medium — biggest gain when your titles are clean and meaningful.",
    },
    {
        "phase": "🎯 Rerank & gate — make the system honest about what it doesn't know",
        "title": "8. Anatomy of a good semantic configuration (cross-encoder design)",
        "symptom": "You enabled semantic ranking and… it barely moved the needle. Or it moved it the wrong way on long documents.",
        "lever": (
            "A **semantic configuration** is the contract you sign with the "
            "cross-encoder ranker: *'here are the fields you should read, in "
            "this order of importance, when you decide who wins.'* Getting "
            "this contract right is what separates *'we turned on semantic'* "
            "from *'semantic actually helps.'*\n\n"
            "**What the ranker actually sees (and its limits):**\n\n"
            "- A deep-learning **cross-encoder** model re-scores the top "
            "~50 candidates from BM25/vector. It reads each candidate "
            "*together with* the query — that's why it catches paraphrase "
            "and intent in a way BM25 alone can't.\n"
            "- It has a **~2,000 token budget per document**. Long docs get "
            "truncated *in the order you specified*. Title first, then "
            "keywords, then content. If your chunks are huge, the bottom "
            "half is invisible to the ranker — yet another reason to chunk "
            "well (lever #1).\n"
            "- It outputs a calibrated **`rerankerScore` (0–4)** — the only "
            "score in the system you can confidently threshold on "
            "(lever #9 below). BM25 and vector scores are not comparable "
            "across queries.\n\n"
            "**Design rules for the prioritized-fields contract:**\n\n"
            "1. **`title_field`** → the cleanest one-line summary of the doc "
            "(or section). If your titles are autogenerated junk like "
            "`Page1.pdf`, fix that *first* — it's the strongest signal.\n"
            "2. **`keywords_fields`** → dense, domain-specific terms. Best "
            "populated by a skillset that extracts entities + acronyms "
            "(e.g. *PSPS, CIP-008-6, IEEE 1547*). Three to ten per chunk.\n"
            "3. **`content_fields`** → the chunk body. Order matters if you "
            "list multiple — the ranker reads top-to-bottom until its 2k "
            "budget is exhausted.\n\n"
            "**Free bonuses you get when you enable semantic:**\n\n"
            "- **`@search.captions`** — extractive snippets with the matching "
            "passages **highlighted**. Show these in your UI; they're better "
            "than truncating the chunk yourself.\n"
            "- **`@search.answers`** — if the query is a direct question and "
            "the answer is a short span in a doc, the ranker extracts it "
            "for you. Set `query_answer='extractive|count-3'` on the search "
            "call. Great for FAQ surfaces; useless for synthesis-style queries.\n\n"
            "**When NOT to enable semantic ranking:** exact-identifier "
            "lookups (model numbers, regulation cites, customer IDs) where "
            "BM25 + filter is already 100% precise — you're just paying the "
            "~300ms latency + extra unit cost for nothing."
        ),
        "in_repo": "✅ `src/build_index.py` defines three configs (`default-balanced`, `content-only`, `title-and-keywords`) and Tab 2 demos how the same query ranks differently across them. 🛠️ Next step: enable captions/answers in the query call so the UI can show extractive highlights instead of full chunks.",
        "code": {
            "language": "python",
            "file": "src/build_index.py + src/retrieval_lab.py",
            "body": (
                "# 1) Define the configuration on the index (one-time)\n"
                "from azure.search.documents.indexes.models import (\n"
                "    SemanticConfiguration, SemanticPrioritizedFields,\n"
                "    SemanticField, SemanticSearch,\n"
                ")\n"
                "\n"
                "config = SemanticConfiguration(\n"
                "    name='default-balanced',\n"
                "    prioritized_fields=SemanticPrioritizedFields(\n"
                "        title_field=SemanticField(field_name='title'),\n"
                "        keywords_fields=[SemanticField(field_name='keywords')],\n"
                "        content_fields=[SemanticField(field_name='chunk')],\n"
                "        # order matters: ranker reads top-to-bottom until\n"
                "        # ~2k token budget per doc is exhausted\n"
                "    ),\n"
                ")\n"
                "index.semantic_search = SemanticSearch(configurations=[config])\n"
                "\n"
                "# 2) Use it at query time — and ask for captions + answers\n"
                "results = client.search(\n"
                "    search_text=query,\n"
                "    query_type='semantic',\n"
                "    semantic_configuration_name='default-balanced',\n"
                "    query_caption='extractive|highlight-true',  # ← free\n"
                "    query_answer='extractive|count-3',           # ← free\n"
                "    top=5,\n"
                ")\n"
                "\n"
                "for r in results:\n"
                "    print(r['@search.rerankerScore'])     # 0–4, gateable\n"
                "    for cap in r['@search.captions']:\n"
                "        print(cap.text, cap.highlights)   # show in UI\n"
                "for ans in results.get_answers() or []:\n"
                "    print(ans.text, ans.score)            # direct-answer UI"
            ),
        },
        "effort": "S",
        "impact": "🔥 High — typically +5–15% NDCG once title and keywords are populated. Captions/answers are free UX wins on top.",
    },
    {
        "phase": "🎯 Rerank & gate — make the system honest about what it doesn't know",
        "title": "9. Score gating (refuse-to-answer threshold)",
        "symptom": "The chat model hallucinates an answer when retrieval returned no truly relevant chunks.",
        "lever": (
            "**The single most important lever for production RAG.** The "
            "semantic reranker score is calibrated (0–4); BM25 / vector scores "
            "are not. Pick a threshold (we use **≥2.0** for the green band; "
            "1.5–2.0 = yellow; <1.5 = refuse) and **gate synthesis** on it. "
            "Below the threshold, the chat model is told *'no good evidence — "
            "say you don't know'* rather than handed weak chunks."
        ),
        "in_repo": "✅ Tab 1's *'Top-1 scores — only one is gateable'* table demonstrates this. The reranker scale is the only one you can threshold on.",
        "code": {
            "language": "python",
            "file": "src/agentic_demo.py",
            "body": (
                "REFUSE_THRESHOLD = 1.5  # reranker score below this → refuse\n"
                "\n"
                "top1 = max(r.get('rerankerScore', 0) for r in references) \\\n"
                "       if references else 0\n"
                "\n"
                "if top1 < REFUSE_THRESHOLD:\n"
                "    return (\n"
                "        \"I don't have enough evidence in the knowledge base \"\n"
                "        \"to answer that confidently. Try rephrasing, or \"\n"
                "        \"escalate to a subject-matter expert.\"\n"
                "    )\n"
                "\n"
                "# Otherwise, hand the (good) chunks to the chat model:\n"
                "answer = synthesize_answer(settings, question, grounding)"
            ),
        },
        "effort": "S",
        "impact": "🔥🔥 Highest possible — turns a 'sometimes hallucinates' demo into a production-safe system.",
    },
    # ---------- Phase 4: Architecture & ops ----------
    {
        "phase": "🏛️ Architecture & ops — once the basics are solid, scale and observe",
        "title": "10. Agentic decomposition (already shown in Tab 3)",
        "symptom": "User asks a multi-part question; one retrieval call covers half of it.",
        "lever": (
            "Use a **Knowledge Base** (preview API `2026-05-01-preview` with "
            "`messages` body) to let the chat model decompose the question into "
            "sub-queries, run them in parallel, and rerank the union. You get "
            "an `activity` trace showing exactly which sub-queries fired — "
            "perfect for observability and debugging."
        ),
        "in_repo": "✅ Tab 3 demos this end-to-end. See `src/agentic_demo.py::retrieve`.",
        "code": {
            "language": "python",
            "file": "src/agentic_demo.py",
            "body": (
                "RETRIEVE_API_VERSION = '2026-05-01-preview'\n"
                "\n"
                "body = {\n"
                "    'messages': [{'role': 'user',\n"
                "                  'content': [{'type': 'text', 'text': question}]}],\n"
                "    'includeActivity': True,         # ← get the planner trace\n"
                "    'outputMode': 'extractiveData',  # we synthesize ourselves\n"
                "    'retrievalReasoningEffort': {'kind': 'low'},\n"
                "    'knowledgeSourceParams': [{\n"
                "        'knowledgeSourceName': ks_name,\n"
                "        'kind': 'searchIndex',\n"
                "        'includeReferences': True,\n"
                "    }],\n"
                "}\n"
                "# activity → [modelQueryPlanning, searchIndex × N, agenticReasoning]"
            ),
        },
        "effort": "M",
        "impact": "🔥 High for multi-part queries; near-zero for single-intent queries (don't pay the planner tax unnecessarily).",
    },
    {
        "phase": "🏛️ Architecture & ops — once the basics are solid, scale and observe",
        "title": "11. Multi-index strategy (one index per domain)",
        "symptom": "Single mega-index returns a public-relations doc when the question was about a NERC compliance procedure.",
        "lever": (
            "When your corpus spans clearly distinct domains (operations, "
            "compliance, customer-facing, HR), split into multiple indexes "
            "and route the query to the right one(s). Routing can be:\n\n"
            "- **Filter-based** (one index, but a `domain` filter — easiest).\n"
            "- **Multi-index** (separate indexes, classifier picks one).\n"
            "- **Knowledge agent with multiple sources** (the planner picks per sub-query — best for cross-domain questions like Tab 3's queries).\n\n"
            "Multi-index also lets each domain have its own chunking, fields, "
            "scoring profile, and refresh cadence."
        ),
        "in_repo": "🛠️ Single index today. Worth splitting once corpus passes a few thousand docs across 3+ domains.",
        "code": {
            "language": "python",
            "file": "(architectural pattern)",
            "body": (
                "# Option A: route by classifier (cheapest)\n"
                "domain = classify(question)   # tiny LLM call or rules\n"
                "client = clients_by_domain[domain]\n"
                "results = client.search(...)\n"
                "\n"
                "# Option B: knowledge base with multiple knowledge sources\n"
                "kb.knowledge_sources = [\n"
                "    {'name': 'ks-operations', 'kind': 'searchIndex', ...},\n"
                "    {'name': 'ks-compliance', 'kind': 'searchIndex', ...},\n"
                "    {'name': 'ks-customer',   'kind': 'searchIndex', ...},\n"
                "]\n"
                "# planner decides which to query per sub-question"
            ),
        },
        "effort": "L",
        "impact": "🔥 High at scale; overkill below ~5k docs.",
    },
    {
        "phase": "🏛️ Architecture & ops — once the basics are solid, scale and observe",
        "title": "12. Evaluation harness (golden Q&A + NDCG/MRR + LLM-as-judge)",
        "symptom": "You shipped lever #7, but nobody can tell whether retrieval actually got better.",
        "lever": (
            "**You cannot improve what you don't measure.** Build a small "
            "golden set (50–200 Q&A pairs with expected `parent_id`s) and "
            "run it as a smoke test on every index change. Track:\n\n"
            "- **Recall@k** — does the right chunk appear in the top-k?\n"
            "- **NDCG / MRR** — is it ranked high?\n"
            "- **LLM-as-judge** — does the synthesized answer cite correctly and avoid hallucinations?\n\n"
            "For utilities specifically, add **adversarial cases**: queries "
            "that should be refused (out-of-scope), queries with regulatory "
            "ambiguity, queries where the answer is *'it depends — escalate'*."
        ),
        "in_repo": "🛠️ Not in this repo. Easy weekend project; gates every future change.",
        "code": {
            "language": "python",
            "file": "tests/eval_retrieval.py (new)",
            "body": (
                "GOLDEN = [\n"
                "    {'q': 'What's a Grade 1 gas leak response time?',\n"
                "     'expected_parents': ['doc-001'], 'must_refuse': False},\n"
                "    {'q': 'When can we de-energize for wildfire risk?',\n"
                "     'expected_parents': ['doc-003'], 'must_refuse': False},\n"
                "    {'q': 'What's our HR vacation policy?',  # adversarial\n"
                "     'expected_parents': [], 'must_refuse': True},\n"
                "    # ... 50–200 of these, curated with SMEs\n"
                "]\n"
                "\n"
                "def evaluate():\n"
                "    hits, mrr, refuse_correct = 0, 0.0, 0\n"
                "    for case in GOLDEN:\n"
                "        results = run_hybrid_semantic(case['q'], top=5)\n"
                "        parents = [r['parent_id'] for r in results]\n"
                "        if case['must_refuse']:\n"
                "            top_score = results[0]['rerankerScore'] if results else 0\n"
                "            refuse_correct += int(top_score < 1.5)\n"
                "            continue\n"
                "        for rank, p in enumerate(parents, 1):\n"
                "            if p in case['expected_parents']:\n"
                "                hits += 1\n"
                "                mrr += 1 / rank\n"
                "                break\n"
                "    print(f'recall@5={hits/len(GOLDEN):.2%}  MRR={mrr/len(GOLDEN):.3f}')"
            ),
        },
        "effort": "M",
        "impact": "🔥🔥 Highest leverage long-term — every other lever becomes objectively measurable.",
    },
]


EFFORT_LABEL = {
    "S": ("Small", "🟢"),  # < 1 day
    "M": ("Medium", "🟡"),  # 1–3 days
    "L": ("Large", "🟠"),  # > 1 week
}


def _render_lever(lever: dict) -> None:
    title = f"**{lever['title']}**  ·  effort: {EFFORT_LABEL[lever['effort']][1]} {EFFORT_LABEL[lever['effort']][0]}"
    with st.expander(title, expanded=False):
        st.markdown(f"**🩺 Symptom** — {lever['symptom']}")
        st.markdown(f"**🔧 The lever** — {lever['lever']}")
        st.markdown(f"**📂 In this repo** — {lever['in_repo']}")
        st.caption(f"`{lever['code']['file']}`")
        st.code(lever["code"]["body"], language=lever["code"]["language"])
        st.markdown(f"**📈 Impact** — {lever['impact']}")


with tab_playbook:
    st.subheader("12 levers to push AI Search retrieval quality higher")
    st.markdown(
        "You've now seen four retrieval methods (Tab 1), three semantic "
        "configs (Tab 2), and agentic decomposition (Tab 3). So **what do you "
        "actually do** to make your own AI Search deployment better?"
    )
    st.markdown(
        "This playbook organizes the levers by **pipeline phase** — fix things "
        "early (at index time) before fixing them late (at query time), and "
        "fix things late before adding architectural complexity. Each lever "
        "lists the **symptom** it addresses, the **fix**, what it would look "
        "like **in this codebase**, and an **effort/impact** estimate."
    )
    st.info(
        "💡 **Reading order suggestion:** start with Phase 3 lever #8 (score "
        "gating). It's the single biggest win and lowest effort for any "
        "production RAG system."
    )

    phases_seen = set()
    for lever in RETRIEVAL_QUALITY_LEVERS:
        if lever["phase"] not in phases_seen:
            st.markdown(f"### {lever['phase']}")
            phases_seen.add(lever["phase"])
        _render_lever(lever)

    st.divider()
    st.markdown(
        "### 🗺️ Where to go after this"
    )
    st.markdown(
        "- **[Azure AI Search docs — relevance tuning overview](https://learn.microsoft.com/azure/search/search-relevance-overview)**\n"
        "- **[Scoring profiles reference](https://learn.microsoft.com/azure/search/index-add-scoring-profiles)**\n"
        "- **[Semantic ranking deep-dive](https://learn.microsoft.com/azure/search/semantic-search-overview)**\n"
        "- **[Knowledge bases (Build 2026 preview)](https://learn.microsoft.com/azure/search/search-knowledge-overview)**\n"
        "- **[RAG eval with Azure AI Foundry](https://learn.microsoft.com/azure/ai-foundry/concepts/evaluation-approach-gen-ai)**\n"
    )
