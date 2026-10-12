from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import hashlib


@dataclass
class Document:
    id: str
    filename: str
    content: str
    metadata: dict = field(default_factory=dict)


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> Document:
        ...


class PDFLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        import pdfplumber
        content_parts = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        content_parts.append(text)
            content = "\n\n".join(content_parts)
        except Exception:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            content = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        file_hash = self._hash_file(file_path)
        return Document(
            id=file_hash[:16],
            filename=file_path.name,
            content=content,
            metadata={
                "filename": file_path.name,
                "file_hash": file_hash,
                "source_type": "pdf",
            },
        )

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class MarkdownLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        content = file_path.read_text(encoding="utf-8")
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        return Document(
            id=file_hash[:16],
            filename=file_path.name,
            content=content,
            metadata={
                "filename": file_path.name,
                "file_hash": file_hash,
                "source_type": "markdown",
            },
        )


class TextLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        content = file_path.read_text(encoding="utf-8")
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        return Document(
            id=file_hash[:16],
            filename=file_path.name,
            content=content,
            metadata={
                "filename": file_path.name,
                "file_hash": file_hash,
                "source_type": "text",
            },
        )


class LoaderFactory:
    _registry = {
        ".pdf": PDFLoader,
        ".md": MarkdownLoader,
        ".txt": TextLoader,
    }

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseLoader:
        ext = file_path.suffix.lower()
        loader_cls = cls._registry.get(ext)
        if not loader_cls:
            raise ValueError(f"不支持的文件类型: {ext}")
        return loader_cls()

    @classmethod
    def supported_extensions(cls) -> List[str]:
        return list(cls._registry.keys())
