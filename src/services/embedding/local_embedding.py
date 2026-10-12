import os
from typing import List
from .base import BaseEmbeddingProvider


class LocalBGEZhEmbeddingProvider(BaseEmbeddingProvider):
    """BAAI/bge-small-zh-v1.5 本地嵌入模型，~100MB，512 维，适配 4GB VRAM"""

    QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        cache = os.environ.get("HF_HUB_CACHE") or "F:/huggingface_cache"
        os.environ["HF_HUB_CACHE"] = cache
        os.environ["HF_HOME"] = cache
        os.environ["TRANSFORMERS_CACHE"] = cache
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, cache_folder=cache, local_files_only=True)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self._model.encode(
            self.QUERY_INSTRUCTION + query, normalize_embeddings=True
        ).tolist()

    @property
    def dimension(self) -> int:
        return 512
