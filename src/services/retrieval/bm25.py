import pickle
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import jieba
from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    chunk_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


class BM25Index:
    """中文 BM25 稀疏检索索引，基于 jieba 分词 + rank_bm25，注入水泥 LCA 领域词典"""

    LCA_DICT = [
        "碳排放", "碳足迹", "生命周期", "熟料", "石灰石",
        "替代燃料", "余热发电", "粉磨", "均化", "窑炉",
        "单位产品", "排放因子", "碳捕集", "碳封存", "CCUS",
        "低碳水泥", "矿渣", "粉煤灰", "硅灰", "石膏",
        "碳酸钙", "煅烧", "生料", "分解炉", "预热器",
        "筒仓", "包装", "散装", "运输", "碳标签",
        "EPD", "PCR", "ISO14040", "ISO14044", "碳配额",
    ]

    def __init__(self, index_path: Optional[str] = None):
        self._chunks: List[Tuple[str, str, dict]] = []
        self._tokenized_corpus: List[List[str]] = []
        self._model: Optional[BM25Okapi] = None
        self._index_path = index_path

        for word in self.LCA_DICT:
            jieba.add_word(word)

    def add(self, chunk_id: str, content: str, metadata: dict) -> None:
        self._chunks.append((chunk_id, content, metadata))
        self._tokenized_corpus.append(self._tokenize(content))
        self._rebuild()

    def add_batch(self, chunks: List[Tuple[str, str, dict]]) -> None:
        for cid, content, meta in chunks:
            self._chunks.append((cid, content, meta))
            self._tokenized_corpus.append(self._tokenize(content))
        self._rebuild()

    def search(self, query: str, top_k: int = 20) -> List[BM25Result]:
        if not self._model:
            return []
        tokens = self._tokenize(query)
        scores = self._model.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            BM25Result(
                chunk_id=self._chunks[i][0],
                content=self._chunks[i][1],
                score=float(scores[i]),
                metadata=self._chunks[i][2],
            )
            for i in top_indices if scores[i] > 0
        ]

    def remove(self, chunk_id: str) -> None:
        indices = [i for i, c in enumerate(self._chunks) if c[0] == chunk_id]
        for i in reversed(indices):
            self._chunks.pop(i)
            self._tokenized_corpus.pop(i)
        self._rebuild()

    def remove_by_doc_id(self, doc_id: str) -> int:
        indices = [i for i, c in enumerate(self._chunks)
                   if c[2].get("doc_id") == doc_id]
        for i in reversed(indices):
            self._chunks.pop(i)
            self._tokenized_corpus.pop(i)
        self._rebuild()
        return len(indices)

    def doc_count(self) -> int:
        doc_ids = {c[2].get("doc_id") for c in self._chunks}
        return len(doc_ids)

    def chunk_count(self) -> int:
        return len(self._chunks)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self._chunks,
                "tokenized": self._tokenized_corpus,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._chunks = data["chunks"]
            self._tokenized_corpus = data["tokenized"]
        self._rebuild()

    def _tokenize(self, text: str) -> List[str]:
        return [w for w in jieba.cut(text) if w.strip()]

    def _rebuild(self) -> None:
        if self._tokenized_corpus:
            self._model = BM25Okapi(self._tokenized_corpus)
        else:
            self._model = None
