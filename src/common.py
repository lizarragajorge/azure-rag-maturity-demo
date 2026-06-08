"""Shared helpers: env, clients, and corpus loading."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "utility_corpus"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    search_endpoint: str
    search_index_name: str
    aoai_endpoint: str
    aoai_embedding_deployment: str
    aoai_embedding_model: str
    aoai_embedding_dimensions: int
    aoai_chat_deployment: str
    aoai_chat_model: str
    knowledge_source_name: str
    knowledge_base_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        def req(name: str) -> str:
            value = os.environ.get(name)
            if not value:
                raise RuntimeError(
                    f"Missing required environment variable {name}. "
                    "Copy .env.example to .env and fill it in."
                )
            return value

        return cls(
            search_endpoint=req("SEARCH_ENDPOINT"),
            search_index_name=os.environ.get("SEARCH_INDEX_NAME", "utility-ops"),
            aoai_endpoint=req("AOAI_ENDPOINT"),
            aoai_embedding_deployment=os.environ.get(
                "AOAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
            ),
            aoai_embedding_model=os.environ.get(
                "AOAI_EMBEDDING_MODEL", "text-embedding-3-large"
            ),
            aoai_embedding_dimensions=int(
                os.environ.get("AOAI_EMBEDDING_DIMENSIONS", "3072")
            ),
            aoai_chat_deployment=os.environ.get("AOAI_CHAT_DEPLOYMENT", "gpt-5-mini"),
            aoai_chat_model=os.environ.get("AOAI_CHAT_MODEL", "gpt-5-mini"),
            knowledge_source_name=os.environ.get(
                "KNOWLEDGE_SOURCE_NAME", "utility-knowledge-source"
            ),
            knowledge_base_name=os.environ.get(
                "KNOWLEDGE_BASE_NAME", "utility-knowledge-base"
            ),
        )


def get_credential() -> TokenCredential:
    return DefaultAzureCredential()


def get_aoai_token_provider():
    """Bearer-token provider for the openai SDK (Azure AD auth, keyless)."""
    return get_bearer_token_provider(
        get_credential(),
        "https://cognitiveservices.azure.com/.default",
    )


# ---------------------------------------------------------------------------
# Corpus loading and chunking
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    parent_id: str
    title: str
    document_type: str
    keywords: list[str]
    content: str
    source: str
    chunk_index: int
    last_updated: str


_FRONTMATTER_RE = re.compile(r"^---\n(?P<fm>.*?)\n---\n(?P<body>.*)$", re.DOTALL)
_KV_RE = re.compile(r"^([A-Za-z0-9_]+):\s*(.*)$")
_LIST_RE = re.compile(r"^\[(.*)\]$")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("Document is missing YAML frontmatter")
    fm: dict = {}
    for raw_line in m.group("fm").splitlines():
        kv = _KV_RE.match(raw_line)
        if not kv:
            continue
        key, value = kv.group(1), kv.group(2).strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        list_match = _LIST_RE.match(value)
        if list_match:
            items = [
                item.strip().strip('"').strip("'")
                for item in list_match.group(1).split(",")
                if item.strip()
            ]
            fm[key] = items
        else:
            fm[key] = value
    return fm, m.group("body").strip()


def _split_into_chunks(body: str, target_chars: int = 1200) -> list[str]:
    """Split on markdown headings; merge small sections to ~target size."""
    sections = re.split(r"(?m)^(?=##\s)", body)
    chunks: list[str] = []
    buffer = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(buffer) + len(section) + 2 <= target_chars or not buffer:
            buffer = f"{buffer}\n\n{section}".strip()
        else:
            chunks.append(buffer)
            buffer = section
    if buffer:
        chunks.append(buffer)
    return chunks


def load_corpus() -> Iterator[Chunk]:
    for md_path in sorted(CORPUS_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        parent_id = fm["id"]
        title = fm["title"]
        for idx, chunk_body in enumerate(_split_into_chunks(body)):
            yield Chunk(
                id=f"{parent_id}-c{idx:02d}",
                parent_id=parent_id,
                title=title,
                document_type=fm.get("document_type", ""),
                keywords=fm.get("keywords", []) if isinstance(fm.get("keywords"), list) else [],
                content=chunk_body,
                source=fm.get("source", ""),
                chunk_index=idx,
                last_updated=fm.get("last_updated", ""),
            )
