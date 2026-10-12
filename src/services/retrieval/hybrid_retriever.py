from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class FusedResult:
    chunk_id: str
    content: str
    fusion_score: float
    metadata: dict = field(default_factory=dict)


class ReciprocalRankFusion:
    """RRF 融合：score(d) = sum(1 / (k + rank(d))) — 默认 k=60"""

    def __init__(self, k: int = 60):
        self._k = k

    def fuse(self, dense_results: list, sparse_results: list,
             top_k: int = 15) -> List[FusedResult]:
        scores: Dict[str, float] = {}
        id_map: Dict[str, FusedResult] = {}

        for rank, r in enumerate(dense_results, start=1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (self._k + rank)
            if r.chunk_id not in id_map:
                id_map[r.chunk_id] = FusedResult(
                    chunk_id=r.chunk_id, content=r.content,
                    fusion_score=0.0, metadata=r.metadata,
                )

        for rank, r in enumerate(sparse_results, start=1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (self._k + rank)
            if r.chunk_id not in id_map:
                id_map[r.chunk_id] = FusedResult(
                    chunk_id=r.chunk_id, content=r.content,
                    fusion_score=0.0, metadata=r.metadata,
                )

        for cid in scores:
            id_map[cid].fusion_score = scores[cid]

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [id_map[cid] for cid in sorted_ids]
