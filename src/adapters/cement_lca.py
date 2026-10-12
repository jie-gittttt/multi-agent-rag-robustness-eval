"""水泥 LCA RAG 系统适配器。

将现有 RAG 项目的 RetrievalService + GenerationService 包装为
BaseRAGAdapter 接口，供评估工具调用。
"""

from .base import BaseRAGAdapter, RetrievedChunk, GenerationResult


class CementLCARAGAdapter(BaseRAGAdapter):
    """包装水泥 LCA RAG 的检索和生成服务"""

    def __init__(self, retrieval_service, generation_service):
        self._retrieval = retrieval_service
        self._generation = generation_service

    @property
    def name(self) -> str:
        return "水泥 LCA 混合检索 RAG"

    def retrieve(self, query: str, top_k: int = 5,
                 dense_enabled: bool = True,
                 sparse_enabled: bool = True) -> list[RetrievedChunk]:
        chunks = self._retrieval.retrieve(
            query, top_k=top_k,
            dense_enabled=dense_enabled,
            sparse_enabled=sparse_enabled,
        )
        return [
            RetrievedChunk(
                content=c.content,
                metadata=c.metadata,
                score=c.score,
            )
            for c in chunks
        ]

    def generate(self, query: str,
                 contexts: list[RetrievedChunk]) -> GenerationResult:
        # 将适配器的 RetrievedChunk 转回项目内部的 RetrievedChunk
        from ..services.retrieval.retrieval_service import (
            RetrievedChunk as InternalChunk,
        )
        internal_chunks = [
            InternalChunk(
                chunk_id=c.metadata.get("chunk_id", ""),
                content=c.content,
                score=c.score,
                metadata=c.metadata,
            )
            for c in contexts
        ]
        result = self._generation.generate_sync(query, internal_chunks)
        return GenerationResult(
            answer=result.answer,
            sources=result.sources,
        )
