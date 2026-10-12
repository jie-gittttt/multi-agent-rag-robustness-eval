from typing import List, Optional
from .base import BaseVectorStore, VectorSearchResult


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_dir: str, collection_name: str):
        import chromadb
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: List[str], embeddings: List[List[float]],
            documents: List[str], metadatas: List[dict]) -> None:
        self._collection.add(
            ids=ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas,
        )

    def query(self, embedding: List[float], top_k: int = 20,
              where: Optional[dict] = None) -> List[VectorSearchResult]:
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        dists_list = results.get("distances", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        for i, chunk_id in enumerate(ids_list):
            out.append(VectorSearchResult(
                chunk_id=chunk_id,
                content=docs_list[i] if i < len(docs_list) else "",
                score=1.0 - dists_list[i] if i < len(dists_list) else 0.0,
                metadata=metas_list[i] if i < len(metas_list) else {},
            ))
        return out

    def delete(self, ids: List[str]) -> None:
        self._collection.delete(ids=ids)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按文档 ID 删除所有向量分块，返回删除数量"""
        before = self._collection.count()
        self._collection.delete(where={"doc_id": doc_id})
        after = self._collection.count()
        return before - after

    def count(self) -> int:
        return self._collection.count()

    def get_collection_stats(self) -> dict:
        return {
            "count": self._collection.count(),
            "name": self._collection.name,
        }
