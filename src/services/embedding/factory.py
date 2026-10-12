from .base import BaseEmbeddingProvider
from .openai_embedding import OpenAIEmbeddingProvider
from .local_embedding import LocalBGEZhEmbeddingProvider


class EmbeddingProviderFactory:
    @staticmethod
    def create(provider_type: str, settings) -> BaseEmbeddingProvider:
        if provider_type == "openai":
            return OpenAIEmbeddingProvider(
                model=settings.OPENAI_EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
            )
        elif provider_type == "local":
            return LocalBGEZhEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL,
            )
        raise ValueError(f"未知的嵌入服务商: {provider_type}")
