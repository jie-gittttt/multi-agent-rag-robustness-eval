from typing import List
from .base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small",
                 api_key: str = None, base_url: str = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return 1536 if "small" in self._model else 3072
