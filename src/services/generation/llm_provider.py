from abc import ABC, abstractmethod
from typing import Generator, List


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list, stream: bool = False,
                 **kwargs) -> str | Generator:
        ...

    @abstractmethod
    def generate_sync(self, prompt: str, **kwargs) -> str:
        ...


class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = None):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "OPENAI_API_KEY 未设置。请在 .env 文件中配置 API Key。"
                )
            from openai import OpenAI
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)

    def generate(self, messages: list, stream: bool = False, **kwargs):
        self._ensure_client()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=stream,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        if stream:
            return self._stream_response(response)
        return response.choices[0].message.content

    def _stream_response(self, response) -> Generator:
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_sync(self, prompt: str, **kwargs) -> str:
        return self.generate(
            [{"role": "user", "content": prompt}],
            stream=False, **kwargs,
        )


class AnthropicLLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY 未设置。请在 .env 文件中配置 API Key。"
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)

    def generate(self, messages: list, stream: bool = False, **kwargs):
        self._ensure_client()
        system = None
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)

        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=user_messages,
            stream=stream,
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.1),
        )
        if stream:
            return self._stream_response(response)
        return response.content[0].text

    def _stream_response(self, response) -> Generator:
        for event in response:
            if event.type == "content_block_delta":
                yield event.delta.text

    def generate_sync(self, prompt: str, **kwargs) -> str:
        return self.generate(
            [{"role": "user", "content": prompt}],
            stream=False, **kwargs,
        )


class LLMProviderFactory:
    @staticmethod
    def create(provider_type: str, settings) -> BaseLLMProvider:
        if provider_type == "openai":
            return OpenAILLMProvider(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL,
                base_url=settings.OPENAI_BASE_URL,
            )
        elif provider_type == "anthropic":
            return AnthropicLLMProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.ANTHROPIC_MODEL,
            )
        raise ValueError(f"未知的 LLM 服务商: {provider_type}")
