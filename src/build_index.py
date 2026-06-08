"""Create the Azure AI Search index for the demo.

The index defines THREE semantic configurations so we can demonstrate, in the
meeting, how different field-prioritization choices change ranker behavior on
the same corpus and the same query:

1. `default-balanced`      — title + keywords as title-class signal,
                             content as the primary content field
2. `content-only`           — only the body content is considered; title and
                             keywords are ignored (illustrates what happens
                             when you fail to tell the ranker what matters)
3. `title-and-keywords`     — heavy emphasis on title and keywords with
                             content as supporting evidence (useful for
                             structured catalogs, equipment indexes, FAQs)

Run:
    python src/build_index.py
"""

from __future__ import annotations

import logging
import sys

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from openai import AzureOpenAI

from common import Settings, get_aoai_token_provider, get_credential, load_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("build_index")


# ---------------------------------------------------------------------------
# Index definition
# ---------------------------------------------------------------------------


def build_index(settings: Settings) -> SearchIndex:
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="parent_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        SimpleField(
            name="document_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            sortable=True,
        ),
        SearchableField(
            name="keywords",
            collection=True,
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="last_updated",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=settings.aoai_embedding_dimensions,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    # Integrated vectorizer: AI Search calls AOAI on indexing AND at query time
    # when you pass a `VectorizableTextQuery`.  No client-side embedding code.
    vectorizer = AzureOpenAIVectorizer(
        vectorizer_name="aoai-vectorizer",
        parameters=AzureOpenAIVectorizerParameters(
            resource_url=settings.aoai_endpoint,
            deployment_name=settings.aoai_embedding_deployment,
            model_name=settings.aoai_embedding_model,
        ),
    )

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-alg",
                vectorizer_name="aoai-vectorizer",
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-alg",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )
        ],
        vectorizers=[vectorizer],
    )

    # Three semantic configurations on the SAME fields, so we can compare
    # how field prioritization changes ranking on the same query.
    semantic_search = SemanticSearch(
        default_configuration_name="default-balanced",
        configurations=[
            SemanticConfiguration(
                name="default-balanced",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="keywords")],
                ),
            ),
            SemanticConfiguration(
                name="content-only",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                ),
            ),
            SemanticConfiguration(
                name="title-and-keywords",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    keywords_fields=[SemanticField(field_name="keywords")],
                    content_fields=[SemanticField(field_name="content")],
                ),
            ),
        ],
    )

    return SearchIndex(
        name=settings.search_index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def create_or_update_index(settings: Settings) -> None:
    credential = get_credential()
    index_client = SearchIndexClient(endpoint=settings.search_endpoint, credential=credential)
    index = build_index(settings)
    log.info("Creating or updating index %r ...", settings.search_index_name)
    index_client.create_or_update_index(index)
    log.info(
        "Index ready with %d semantic configurations: %s",
        len(index.semantic_search.configurations),
        ", ".join(c.name for c in index.semantic_search.configurations),
    )


def upload_documents(settings: Settings) -> int:
    credential = get_credential()
    search_client = SearchClient(
        endpoint=settings.search_endpoint,
        index_name=settings.search_index_name,
        credential=credential,
    )
    aoai = AzureOpenAI(
        azure_endpoint=settings.aoai_endpoint,
        azure_ad_token_provider=get_aoai_token_provider(),
        api_version="2024-10-21",
    )

    chunks = list(load_corpus())
    log.info("Embedding %d chunks via %s ...", len(chunks), settings.aoai_embedding_deployment)
    # Embed in batches; AOAI embeddings supports many inputs per call.
    BATCH_EMBED = 16
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), BATCH_EMBED):
        batch = chunks[i : i + BATCH_EMBED]
        resp = aoai.embeddings.create(
            model=settings.aoai_embedding_deployment,
            input=[c.content for c in batch],
        )
        vectors.extend(item.embedding for item in resp.data)

    docs: list[dict] = []
    for chunk, vector in zip(chunks, vectors):
        docs.append(
            {
                "id": chunk.id,
                "parent_id": chunk.parent_id,
                "title": chunk.title,
                "document_type": chunk.document_type,
                "keywords": chunk.keywords,
                "content": chunk.content,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "last_updated": chunk.last_updated,
                "content_vector": vector,
            }
        )

    log.info("Uploading %d chunks with embeddings ...", len(docs))
    BATCH = 500
    for i in range(0, len(docs), BATCH):
        result = search_client.upload_documents(documents=docs[i : i + BATCH])
        failed = [r for r in result if not r.succeeded]
        if failed:
            for f in failed[:5]:
                log.error("  failed id=%s status=%s msg=%s", f.key, f.status_code, f.error_message)
            raise RuntimeError(f"{len(failed)} documents failed to upload")
    return len(docs)


def main(argv: list[str]) -> int:
    settings = Settings.from_env()
    create_or_update_index(settings)
    count = upload_documents(settings)
    log.info("Loaded %d chunks into %r.", count, settings.search_index_name)
    log.info("Next step: `python src/agentic_demo.py --setup` to stand up the Knowledge Base.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
