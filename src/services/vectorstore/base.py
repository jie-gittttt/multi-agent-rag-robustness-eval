from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VectorSearchResult:
    chunk_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, ids: List[str], embeddings: List[List[float]],
            documents: List[str], metadatas: List[dict]) -> None:
        ...

    @abstractmethod
    def query(self, embedding: List[float], top_k: int = 20,
              where: Optional[dict] = None) -> List[VectorSearchResult]:
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def get_collection_stats(self) -> dict:
        ...
