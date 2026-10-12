import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .loaders import LoaderFactory, Document
from .chunkers import BaseChunker


@dataclass
class IngestionResult:
    doc_id: str
    filename: str
    chunk_count: int
    status: str


class IngestionManager:
    """管理已摄入文档的元数据（不依赖向量库）"""

    def __init__(self, data_dir: str):
        self._meta_file = os.path.join(data_dir, "ingested_docs.json")
        os.makedirs(data_dir, exist_ok=True)
        self._docs: dict = self._load_meta()

    def add(self, result: IngestionResult) -> None:
        self._docs[result.doc_id] = {
            "doc_id": result.doc_id,
            "filename": result.filename,
            "chunk_count": result.chunk_count,
        }
        self._save()

    def remove(self, doc_id: str) -> bool:
        if doc_id not in self._docs:
            return False
        del self._docs[doc_id]
        self._save()
        return True

    def list_all(self) -> list:
        return list(self._docs.values())

    def _load_meta(self) -> dict:
        if os.path.exists(self._meta_file):
            with open(self._meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False, indent=2)


class IngestionPipeline:
    """文档摄入流水线：加载 → 分块 → 向量化 → 双索引写入"""

    def __init__(self, loader_factory, chunker: BaseChunker,
                 embedding_provider, vector_store, bm25_index,
                 data_dir: str = "./data"):
        self._loader_factory = loader_factory
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._manager = IngestionManager(data_dir)

    def ingest_file(self, file_path: Path) -> IngestionResult:
        loader = self._loader_factory.get_loader(file_path)
        document: Document = loader.load(file_path)

        chunks = self._chunker.chunk(document)
        if not chunks:
            return IngestionResult(
                doc_id=document.id, filename=document.filename,
                chunk_count=0, status="empty",
            )

        chunk_texts = [c.content for c in chunks]
        chunk_ids = [c.id for c in chunks]
        chunk_metas = [c.metadata for c in chunks]

        embeddings = self._embedding_provider.batch_embed(chunk_texts)

        self._vector_store.add(
            ids=chunk_ids, embeddings=embeddings,
            documents=chunk_texts, metadatas=chunk_metas,
        )

        self._bm25_index.add_batch([
            (chunk_ids[i], chunk_texts[i], chunk_metas[i])
            for i in range(len(chunks))
        ])
        if self._bm25_index._index_path:
            self._bm25_index.save(self._bm25_index._index_path)

        result = IngestionResult(
            doc_id=document.id,
            filename=document.filename,
            chunk_count=len(chunks),
            status="ingested",
        )
        self._manager.add(result)
        return result

    def ingest_directory(self, dir_path: Path) -> List[IngestionResult]:
        results = []
        for ext in LoaderFactory.supported_extensions():
            for fp in dir_path.glob(f"*{ext}"):
                results.append(self.ingest_file(fp))
        return results

    def remove_document(self, doc_id: str) -> bool:
        if not self._manager.remove(doc_id):
            return False
        self._vector_store.delete_by_doc_id(doc_id)
        self._bm25_index.remove_by_doc_id(doc_id)
        if self._bm25_index._index_path:
            self._bm25_index.save(self._bm25_index._index_path)
        return True

    def list_documents(self) -> list:
        return self._manager.list_all()
