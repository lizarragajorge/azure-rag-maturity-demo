# RAG Pattern — Post-Build 2026 (utility-ops demo)

A focused RAG demo built around three questions:

1. **Where am I on the RAG maturity ladder** (L0 — just-an-LLM, L1 — keyword RAG, L2 — vector, L3 — hybrid+semantic, L4 — agentic)?
2. **How do I make retrieval reliable** at L3?
3. **What does L4 (agentic retrieval, new at Build 2026) actually buy me** over L3?

The Streamlit UI walks all three with the same corpus + the same index, side-by-side.

## 🚀 Deploy your own copy to Azure

This repo is `azd`-ready. From a clean machine:

```powershell
# Prereqs: az CLI, azd, docker, gh CLI (optional), Python 3.11+

azd auth login
az login

# Create an azd environment (picks region + subscription)
azd env new rag-maturity --location eastus2

# REQUIRED: tell Bicep who is deploying (for the user-scope role assignments)
azd env set AZURE_PRINCIPAL_ID $(az ad signed-in-user show --query id -o tsv)

# OPTIONAL: reuse an existing resource group (otherwise azd creates one)
azd env set AZURE_RESOURCE_GROUP rg-rag-after-build

# Provisions Search + AOAI + Container Apps + ACR, builds the image,
# pushes it, and prints the public URL.
azd up

# One-time corpus + Knowledge Base setup (run from your laptop, hits the
# resources azd just provisioned)
$env:SEARCH_ENDPOINT  = $(azd env get-values | Select-String '^SEARCH_ENDPOINT' | %{$_.ToString().Split('=')[1].Trim('"')})
$env:AOAI_ENDPOINT    = $(azd env get-values | Select-String '^AOAI_ENDPOINT'   | %{$_.ToString().Split('=')[1].Trim('"')})
python src/build_index.py
python src/agentic_demo.py --setup
```

The Container App is gated by an in-app **per-session cap of 10 agentic
queries** (Tab 3 is the only AOAI-spending tab) so you can leave the public URL
running without worrying about runaway cost. The cap is controlled by the
`PUBLIC_DEMO=true` env var Bicep sets on the Container App; locally it's off.

## What it shows

| Capability | What it gives you | Where in this repo |
|---|---|---|
| Hybrid search (BM25 + vector, fused with RRF) | Recall on rare terms *and* paraphrases | [src/retrieval_lab.py](src/retrieval_lab.py) |
| Semantic ranker (L2 re-ranker) | Precision: re-orders top results by deep semantic relevance, returns `@search.rerankerScore` | [src/retrieval_lab.py](src/retrieval_lab.py) |
| **Semantic configurations** (multiple per index) | Tell the ranker which fields carry the meaning — title vs body vs keywords | [src/build_index.py](src/build_index.py) |
| **Agentic retrieval** — Knowledge Base + Knowledge Source (GA, REST `2026-04-01`) | Decomposes a compound question into subqueries, runs them in parallel, reranks, then synthesizes a grounded answer with citations | [src/agentic_demo.py](src/agentic_demo.py) |
| Integrated vectorizer (`AzureOpenAIVectorizer`) | Embeddings happen *inside* the index — no separate embedding pipeline | [src/build_index.py](src/build_index.py) |
| Citations + "I don't know" guardrail | Reliability: every answer cites the chunk; no hallucination on out-of-corpus questions | [src/agentic_demo.py](src/agentic_demo.py), [src/app.py](src/app.py) |

## Run the demo locally

```powershell
# 1. Provision Azure (one-time) — uses any existing AI Search + Foundry resource
cp .env.example .env
# fill SEARCH_ENDPOINT, AOAI_ENDPOINT, deployment names

# 2. Install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Build the index and load the sample corpus
python src/build_index.py

# 4. (Optional) Stand up the agentic Knowledge Base
python src/agentic_demo.py --setup

# 5. Launch the demo UI
streamlit run src/app.py
```

## Security posture

This is a **public demo**, not a production template. See [SECURITY.md](SECURITY.md)
for the full threat-by-threat write-up. The short version:

- Keyless auth end-to-end (user-assigned managed identity + RBAC). `disableLocalAuth: true` on both Search and Azure OpenAI.
- Streamlit XSRF protection stays on. Container runs as a non-root user.
- Per-session cap of 10 agentic queries + `max_completion_tokens=600` on the LLM call bounds the cost of cost-DoS attempts.
- User-controlled text is XML-delimited inside the LLM prompt to blunt direct prompt-injection.
- Bicep param `ingressAllowedIpRanges` is an empty array by default (open). Pass a non-empty array of `ipSecurityRestrictions` objects to restrict to known IPs without changing code.

## License

[MIT](LICENSE) — fork it, modify it, deploy it.

## Sample corpus

[data/utility_corpus/](data/utility_corpus/) contains 16 short documents
modeled on generic utility content: gas-leak response procedures, transformer
maintenance, wildfire mitigation, FERC/NERC compliance notes, outage comms
templates, and customer-facing FAQs. Everything is **synthetic** and uses a
fictitious utility name ("Northern Grid Utility / NGU") with reserved
`.example` domains and 555-01xx telephone numbers per RFC 2606 / NANP
fiction-use conventions.

## Prereqs

- Python 3.11+
- An Azure AI Search service, Basic tier or higher (for managed identity & semantic ranker)
- A Microsoft Foundry / Azure OpenAI resource with deployments for:
  - An embedding model (default: `text-embedding-3-large`, 3072 dims)
  - A chat model for answer synthesis (default: `gpt-5-mini`)
- `az login` (the SDKs use `DefaultAzureCredential`)
