from dataclasses import dataclass, field
from typing import List, Dict

from ..vectorstore.base import BaseVectorStore
from ..embedding.base import BaseEmbeddingProvider
from .bm25 import BM25Index
from .hybrid_retriever import ReciprocalRankFusion, FusedResult


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


class RetrievalService:
    """混合检索引擎：Dense + Sparse → RRF 融合 → Top-K"""

    def __init__(self, vector_store: BaseVectorStore,
                 bm25_index: BM25Index,
                 embedding_provider: BaseEmbeddingProvider,
                 rrf_fusion: ReciprocalRankFusion,
                 top_k: int = 5):
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._embedding_provider = embedding_provider
        self._rrf = rrf_fusion
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int = None,
                 dense_enabled: bool = True,
                 sparse_enabled: bool = True) -> List[RetrievedChunk]:
        k = top_k or self._top_k

        # --- 仅稠密检索 ---
        if dense_enabled and not sparse_enabled:
            q_embedding = self._embedding_provider.embed_query(query)
            results = self._vector_store.query(q_embedding, top_k=k)
            return [
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ]

        # --- 仅稀疏检索 ---
        if sparse_enabled and not dense_enabled:
            results = self._bm25_index.search(query, top_k=k)
            return [
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ]

        # --- 都不开启 ---
        if not dense_enabled and not sparse_enabled:
            return []

        # --- 混合检索（默认）---
        q_embedding = self._embedding_provider.embed_query(query)
        dense_results = self._vector_store.query(q_embedding, top_k=20)
        sparse_results = self._bm25_index.search(query, top_k=20)

        fused = self._rrf.fuse(dense_results, sparse_results, top_k=k)

        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.fusion_score,
                metadata=r.metadata,
            )
            for r in fused
        ]
