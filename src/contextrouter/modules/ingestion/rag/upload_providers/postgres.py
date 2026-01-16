"""Postgres upload provider: load JSONL into pgvector store."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from psycopg_pool import AsyncConnectionPool

from contextrouter.core import get_core_config
from contextrouter.core.types import coerce_struct_data
from contextrouter.modules.models import model_registry

from .base import UploadProvider, UploadResult

logger = logging.getLogger(__name__)


class PostgresUploadProvider(UploadProvider):
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._pool: AsyncConnectionPool | None = None

    @property
    def name(self) -> str:
        return "postgres"

    def upload_and_index(self, local_path: Path, *, wait: bool = False) -> UploadResult:
        _ = wait
        try:
            asyncio.run(self._upload_async(local_path))
            return UploadResult(success=True, provider=self.name)
        except Exception as e:
            logger.exception("Postgres upload failed")
            return UploadResult(success=False, provider=self.name, error=str(e))

    def get_config_summary(self) -> dict[str, str]:
        return {"provider": self.name, "dsn": "***" if self._config.get("dsn") else "missing"}

    async def _upload_async(self, local_path: Path) -> None:
        cfg = get_core_config()
        dsn = self._config.get("dsn") or cfg.postgres.dsn
        if not dsn:
            raise ValueError("Postgres DSN is required")

        pool = await self._get_pool(dsn=dsn)
        tenant_id = str(self._config.get("tenant_id") or "public")
        user_id = self._config.get("user_id")
        embeddings_key = self._config.get("embeddings_model") or cfg.models.default_embeddings
        embedder = model_registry.get_embeddings(embeddings_key, config=cfg)

        with open(local_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        batch_size = 64
        for batch in _chunks(records, batch_size):
            texts, payloads = self._prepare_batch(batch)
            embeddings = await embedder.embed_documents(texts)
            await self._insert_batch(
                pool=pool,
                tenant_id=tenant_id,
                user_id=user_id,
                payloads=payloads,
                embeddings=embeddings,
            )

    async def _get_pool(self, *, dsn: str) -> AsyncConnectionPool:
        if self._pool is None:
            self._pool = AsyncConnectionPool(dsn, min_size=2, max_size=10, open=False)
        if not self._pool.opened:
            await self._pool.open()
        return self._pool

    def _prepare_batch(self, batch: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        texts: list[str] = []
        payloads: list[dict[str, Any]] = []
        for record in batch:
            raw = record.get("content", {}).get("rawBytes")
            if not raw:
                continue
            content = base64.b64decode(raw).decode("utf-8", errors="ignore")
            struct_data = record.get("structData") or {}
            struct_data = coerce_struct_data(struct_data)
            if not isinstance(struct_data, dict):
                struct_data = {}
            keywords_text = _flatten_keywords(struct_data)
            payloads.append(
                {
                    "id": record.get("id"),
                    "content": content,
                    "struct_data": struct_data,
                    "source_type": struct_data.get("source_type"),
                    "source_id": struct_data.get("source_id"),
                    "title": struct_data.get("title"),
                    "keywords_text": keywords_text,
                    "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )
            texts.append(content)
        return texts, payloads

    async def _insert_batch(
        self,
        *,
        pool: AsyncConnectionPool,
        tenant_id: str,
        user_id: str | None,
        payloads: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if not payloads:
            return
        async with pool.connection() as conn:
            async with conn.transaction():
                for payload, embedding in zip(payloads, embeddings):
                    await conn.execute(
                        """
                        INSERT INTO knowledge_nodes (
                            id, tenant_id, user_id, node_kind, source_type, source_id, title,
                            content, struct_data, keywords_text, content_hash, embedding
                        )
                        VALUES (
                            %(id)s, %(tenant_id)s, %(user_id)s, 'chunk', %(source_type)s, %(source_id)s,
                            %(title)s, %(content)s, %(struct_data)s, %(keywords_text)s, %(content_hash)s, %(embedding)s
                        )
                        ON CONFLICT (node_kind, content_hash) DO NOTHING
                        """,
                        {
                            "id": payload.get("id"),
                            "tenant_id": tenant_id,
                            "user_id": user_id,
                            "source_type": payload.get("source_type"),
                            "source_id": payload.get("source_id"),
                            "title": payload.get("title"),
                            "content": payload.get("content"),
                            "struct_data": payload.get("struct_data"),
                            "keywords_text": payload.get("keywords_text"),
                            "content_hash": payload.get("content_hash"),
                            "embedding": "[" + ",".join(f"{float(x):.8f}" for x in embedding) + "]",
                        },
                    )


def _chunks(seq: Iterable[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _flatten_keywords(struct_data: dict[str, Any]) -> str | None:
    keywords = struct_data.get("keywords")
    keyphrases = struct_data.get("keyphrase_texts")
    parts: list[str] = []
    for raw in (keywords, keyphrases):
        if isinstance(raw, list):
            for item in raw:
                text = str(item).strip()
                if text:
                    parts.append(text)
    if not parts:
        return None
    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return " ".join(uniq)


__all__ = ["PostgresUploadProvider"]
