"""Agentic retrieval demo — the headline post-Build 2026 feature.

A **Knowledge Base** sits on top of one or more **Knowledge Sources** (an
index, in our case) and exposes a single `retrieve` endpoint that:

  1. Reads the conversation so far and infers the user's information need.
  2. Decomposes a compound question into focused subqueries.
  3. Runs those subqueries IN PARALLEL against the underlying knowledge source.
  4. Reranks results with the semantic ranker.
  5. (Optionally) synthesizes a grounded answer with citations.

This file talks to the REST API at `2026-04-01` (GA) directly so the wire
shape is obvious to the audience.

    python src/agentic_demo.py --setup           # create knowledge source + base
    python src/agentic_demo.py --ask "..."       # ask a question
    python src/agentic_demo.py --teardown        # delete both
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Optional

import requests
from azure.core.credentials import TokenCredential
from openai import AzureOpenAI

from common import Settings, get_aoai_token_provider, get_credential

API_VERSION = "2026-04-01"
SEARCH_SCOPE = "https://search.azure.com/.default"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("agentic_demo")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _bearer(credential: TokenCredential) -> str:
    return credential.get_token(SEARCH_SCOPE).token


def _headers(credential: TokenCredential) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_bearer(credential)}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Knowledge source + knowledge base lifecycle
# ---------------------------------------------------------------------------


def create_or_update_knowledge_source(settings: Settings) -> None:
    credential = get_credential()
    url = (
        f"{settings.search_endpoint}/knowledgesources/{settings.knowledge_source_name}"
        f"?api-version={API_VERSION}"
    )
    body = {
        "name": settings.knowledge_source_name,
        "kind": "searchIndex",
        "description": (
            "Utility operations corpus: gas leaks, transformers, "
            "wildfire mitigation, FERC/NERC, rate cases, customer FAQs."
        ),
        "searchIndexParameters": {
            "searchIndexName": settings.search_index_name,
            "sourceDataFields": [
                {"name": "id"},
                {"name": "title"},
                {"name": "content"},
                {"name": "source"},
                {"name": "document_type"},
                {"name": "last_updated"},
            ],
        },
    }
    log.info("PUT  %s", url)
    r = requests.put(url, headers=_headers(credential), data=json.dumps(body), timeout=60)
    if not r.ok:
        log.error("Body: %s", r.text)
    r.raise_for_status()
    log.info("Knowledge source %r ready.", settings.knowledge_source_name)


def create_or_update_knowledge_base(settings: Settings) -> None:
    credential = get_credential()
    url = (
        f"{settings.search_endpoint}/knowledgebases/{settings.knowledge_base_name}"
        f"?api-version={API_VERSION}"
    )
    # Per REST 2026-04-01 GA schema. Property names that differ from the
    # pre-GA shape:  deploymentId (was deploymentName); no outputMode / no
    # answerInstructions / no retrievalReasoningEffort at the KB level.
    body = {
        "name": settings.knowledge_base_name,
        "description": (
            "Utility ops Q&A. Decomposes multi-part questions, runs "
            "parallel retrievals, and synthesizes grounded answers with citations."
        ),
        "knowledgeSources": [{"name": settings.knowledge_source_name}],
        "models": [
            {
                "kind": "azureOpenAI",
                "azureOpenAIParameters": {
                    "resourceUri": settings.aoai_endpoint,
                    "deploymentId": settings.aoai_chat_deployment,
                    "modelName": settings.aoai_chat_model,
                },
            }
        ],
    }
    log.info("PUT  %s", url)
    r = requests.put(url, headers=_headers(credential), data=json.dumps(body), timeout=60)
    if not r.ok:
        log.error("Body: %s", r.text)
    r.raise_for_status()
    log.info("Knowledge base  %r ready.", settings.knowledge_base_name)


def delete_knowledge_base(settings: Settings) -> None:
    credential = get_credential()
    for kind, name in [
        ("knowledgebases", settings.knowledge_base_name),
        ("knowledgesources", settings.knowledge_source_name),
    ]:
        url = f"{settings.search_endpoint}/{kind}/{name}?api-version={API_VERSION}"
        log.info("DEL  %s", url)
        r = requests.delete(url, headers=_headers(credential), timeout=30)
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()
    log.info("Cleanup complete.")


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


def retrieve(
    settings: Settings,
    question: str,
    conversation: Optional[list[dict]] = None,
    reasoning_effort: str = "low",  # kept for CLI compat; not used in GA body
) -> dict[str, Any]:
    """Send a user query to the knowledge base and return the raw response.

    The GA 2026-04-01 retrieve body takes `intents` (one or more semantic
    queries) and `knowledgeSourceParams` (per-source runtime options).  We
    pass the user's question as a single semantic intent, ask for activity
    and references, and let the KB do its agentic decomposition + ranking.

    Response shape:
      - `response`   : the synthesized answer (text content)
      - `activity`   : the planning + subquery trace
      - `references` : grounded chunks with rerankerScore and docKey
    """
    del conversation, reasoning_effort  # accepted for forward-compat
    credential = get_credential()
    url = (
        f"{settings.search_endpoint}/knowledgebases/{settings.knowledge_base_name}"
        f"/retrieve?api-version={API_VERSION}"
    )

    body = {
        "intents": [{"search": question, "type": "semantic"}],
        "includeActivity": True,
        "knowledgeSourceParams": [
            {
                "knowledgeSourceName": settings.knowledge_source_name,
                "kind": "searchIndex",
                "includeReferences": True,
                "includeReferenceSourceData": True,
            }
        ],
    }

    r = requests.post(url, headers=_headers(credential), data=json.dumps(body), timeout=120)
    if not r.ok:
        log.error("Body: %s", r.text)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------


def _extract_answer(response: dict[str, Any]) -> str:
    try:
        first = response["response"][0]
        for content in first.get("content", []):
            if content.get("type") == "text":
                return content.get("text", "")
    except (KeyError, IndexError, TypeError):
        pass
    return "(no grounding)"


def synthesize_answer(settings: Settings, question: str, grounding: str) -> str:
    """Feed the KB grounding payload to gpt-5-mini for a final cited answer."""
    aoai = AzureOpenAI(
        azure_endpoint=settings.aoai_endpoint,
        azure_ad_token_provider=get_aoai_token_provider(),
        api_version="2024-10-21",
    )
    system = (
        "You answer questions for utility field operations, compliance, and "
        "customer-service teams using ONLY the JSON grounding provided. Cite "
        "each fact with the ref_id in square brackets, e.g. [0]. If the "
        "grounding does not contain enough information, say 'I don't have a "
        "sourced answer for that' instead of guessing. Keep the answer to "
        "4 sentences unless asked for more."
    )
    user = f"Question: {question}\n\nGrounding (JSON):\n{grounding}"
    resp = aoai.chat.completions.create(
        model=settings.aoai_chat_deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _print_response(response: dict[str, Any], synthesized: Optional[str] = None) -> None:
    bar = "=" * 78
    if synthesized is not None:
        print(f"\n{bar}\nSYNTHESIZED ANSWER (grounded in references below)\n{bar}")
        print(synthesized)
    print(f"\n{bar}\nGROUNDING PAYLOAD (returned by /retrieve, fed to the LLM)\n{bar}")
    grounding = _extract_answer(response)
    # Truncate for readability
    print(grounding[:1500] + ("\n... [truncated]" if len(grounding) > 1500 else ""))

    print(f"\n{bar}\nACTIVITY (planning + subqueries)\n{bar}")
    for step in response.get("activity", []):
        step_type = step.get("type") or step.get("@type") or "Activity"
        elapsed = step.get("elapsedMs", step.get("ElapsedMs", "?"))
        tokens_in = step.get("inputTokens", step.get("InputTokens"))
        tokens_out = step.get("outputTokens", step.get("OutputTokens"))
        line = f"  · {step_type}  elapsed={elapsed}ms"
        if tokens_in is not None or tokens_out is not None:
            line += f"  tokens(in/out)={tokens_in}/{tokens_out}"
        print(line)
        args = step.get("searchIndexArguments") or step.get("SearchIndexArguments")
        if args:
            print(f"      subquery     : {args.get('search') or args.get('Search')}")
            print(
                "      semantic cfg : "
                f"{args.get('semanticConfigurationName') or args.get('SemanticConfigurationName')}"
            )

    print(f"\n{bar}\nREFERENCES (grounding)\n{bar}")
    for i, ref in enumerate(response.get("references", []), 1):
        score = ref.get("rerankerScore", ref.get("RerankerScore"))
        doc_key = ref.get("docKey", ref.get("DocKey"))
        source = (ref.get("sourceData") or ref.get("SourceData") or {})
        title = source.get("title") if isinstance(source, dict) else None
        line = f"  [{i}] reranker={score}  id={doc_key}"
        if title:
            line += f"\n      {title}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Agentic retrieval demo")
    parser.add_argument("--setup", action="store_true", help="Create KS + KB")
    parser.add_argument("--teardown", action="store_true", help="Delete KS + KB")
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Ask a question (after --setup has been run at least once)",
    )
    parser.add_argument(
        "--effort",
        default="low",
        choices=["low", "medium", "high"],
        help="Retrieval reasoning effort (more effort = more subqueries, more tokens)",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()

    if args.setup:
        create_or_update_knowledge_source(settings)
        create_or_update_knowledge_base(settings)

    if args.ask:
        response = retrieve(settings, args.ask, reasoning_effort=args.effort)
        grounding = _extract_answer(response)
        synthesized = synthesize_answer(settings, args.ask, grounding)
        _print_response(response, synthesized=synthesized)

    if args.teardown:
        delete_knowledge_base(settings)

    if not any([args.setup, args.ask, args.teardown]):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
