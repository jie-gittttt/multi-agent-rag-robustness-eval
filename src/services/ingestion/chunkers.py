import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from .loaders import Document


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        ...


class FixedSizeChunker(BaseChunker):
    """固定窗口分块"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(
                    id=f"{document.id}_c{idx}",
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "doc_id": document.id,
                        "chunk_index": idx,
                    },
                ))
                idx += 1
            start = end - self._overlap if end < len(text) else end
        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """按自然分隔符递归分割：段落 → 句子 → 短语 → 字符，适合中文技术文本"""

    _SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        splits = self._split_text(document.content, self._SEPARATORS)
        chunks = []
        current = ""
        self._idx = 0
        for split in splits:
            if len(current) + len(split) <= self._chunk_size:
                current += split
            else:
                if current.strip():
                    chunks.append(self._make_chunk(document, current))
                    current = current[-self._overlap:] + split \
                        if self._overlap > 0 and len(current) > self._overlap \
                        else split
                else:
                    current = split
        if current.strip():
            chunks.append(self._make_chunk(document, current))
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        sep = separators[0]
        if sep == "":
            return list(text)
        parts = re.split(f"({re.escape(sep)})", text)
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                result.append(part)
            elif result:
                result[-1] += part
            else:
                result.append(part)
        final = []
        for r in result:
            if len(r) > self._chunk_size:
                final.extend(self._split_text(r, separators[1:]))
            else:
                final.append(r)
        return [f for f in final if f.strip()]

    def _make_chunk(self, doc: Document, text: str) -> Chunk:
        chunk = Chunk(
            id=f"{doc.id}_c{self._idx}",
            content=text,
            metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": self._idx},
        )
        self._idx += 1
        return chunk


class ChunkerFactory:
    @staticmethod
    def create(strategy: str, **kwargs) -> BaseChunker:
        if strategy == "fixed":
            return FixedSizeChunker(
                chunk_size=kwargs.get("chunk_size", 512),
                overlap=kwargs.get("overlap", 64),
            )
        elif strategy == "recursive":
            return RecursiveCharacterChunker(
                chunk_size=kwargs.get("chunk_size", 512),
                overlap=kwargs.get("overlap", 64),
            )
        raise ValueError(f"未知分块策略: {strategy}")
