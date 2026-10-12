import json
import uuid
from typing import Generator, List

from .llm_provider import BaseLLMProvider
from .prompt_builder import PromptBuilder


class GenerationResult:
    def __init__(self, answer: str, sources: list):
        self.answer = answer
        self.sources = sources


class GenerationService:
    def __init__(self, llm_provider: BaseLLMProvider,
                 prompt_builder: PromptBuilder):
        self._llm = llm_provider
        self._prompt_builder = prompt_builder

    def generate_sync(self, question: str, chunks: list,
                      history: List[dict] = None) -> GenerationResult:
        messages = self._prompt_builder.build(question, chunks, history)
        answer = self._llm.generate(messages, stream=False)
        return GenerationResult(
            answer=answer,
            sources=self._build_source_list(chunks),
        )

    def generate_stream(self, question: str, chunks: list,
                        history: List[dict] = None) -> Generator:
        messages = self._prompt_builder.build(question, chunks, history)
        request_id = str(uuid.uuid4())[:8]
        sources = self._build_source_list(chunks)

        try:
            yield self._sse_event({"type": "start", "request_id": request_id})

            token_stream = self._llm.generate(messages, stream=True)
            full_answer = ""
            for token in token_stream:
                full_answer += token
                yield self._sse_event({"type": "token", "content": token})

            yield self._sse_event({
                "type": "done",
                "answer": full_answer,
                "sources": sources,
                "request_id": request_id,
            })
        except Exception as e:
            yield self._sse_event({"type": "error", "message": str(e)})

    def _sse_event(self, data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _build_source_list(self, chunks: list) -> list:
        return [
            {
                "index": i + 1,
                "filename": c.metadata.get("filename", "unknown"),
                "chunk_id": c.chunk_id if hasattr(c, "chunk_id") else c.id,
                "content_preview": c.content[:200] + "...",
                "score": round(getattr(c, "score", 0), 4),
            }
            for i, c in enumerate(chunks)
        ]
