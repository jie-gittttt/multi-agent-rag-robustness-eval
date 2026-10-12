from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化，返回嵌套列表"""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """单条查询向量化（可能使用不同的 prompt）"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        ...

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """分批向量化，避免超长列表"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self.embed(batch))
        return all_embeddings
