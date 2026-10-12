"""基于 LlamaIndex 构建的 RAG 适配器

从 data/documents/*.md 读取文档，首次调用时自动建立向量索引并持久化。
LLM: DeepSeek（兼容 OpenAI API）  Embedding: BAAI/bge-small-zh-v1.5（本地）
"""

from __future__ import annotations
from pathlib import Path
from .base import BaseRAGAdapter, RetrievedChunk, GenerationResult


def _load_documents() -> list[str]:
    """从 data/documents/ 目录加载所有 markdown 文档"""
    doc_dir = Path(__file__).resolve().parent.parent.parent / "data" / "documents"
    docs = []
    for fpath in sorted(doc_dir.glob("*.md")):
        text = fpath.read_text(encoding="utf-8")
        if len(text) > 50:
            docs.append(text)
    return docs


_INDEX_CACHE: dict = {}  # {persist_dir: VectorStoreIndex}


class LlamaIndexRAGAdapter(BaseRAGAdapter):
    """基于 LlamaIndex 的 RAG 适配器，使用内置中文知识库自动建立向量索引。

    若 query_engine 已提供则直接使用；否则自动初始化内置知识库并建索引。
    """

    PERSIST_DIR = "data/llamaindex_index"

    def __init__(self, query_engine=None):
        self._engine = query_engine   # 外部传入时直接使用
        self._retriever = None
        self._llm = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "LlamaIndex RAG"

    # ── 懒初始化 ──────────────────────────────────────────────────────────────
    def _ensure_init(self):
        if self._initialized:
            return
        if self._engine is not None:
            self._initialized = True
            return
        self._build_index()

    def _build_index(self):
        from llama_index.core import (
            VectorStoreIndex, Document, Settings,
            StorageContext, load_index_from_storage,
        )
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        import llama_index.llms.openai.utils as _llm_utils

        from ..config import Settings as AppSettings
        app_cfg = AppSettings()

        # 配置全局 Embedding（BGE 本地模型，强制离线用本地缓存）
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=app_cfg.EMBEDDING_MODEL,
            query_instruction="为这个句子生成表示以用于检索相关文章：",
            cache_folder="F:/huggingface_cache",
            model_kwargs={"local_files_only": True},
        )

        # 运行时兼容补丁：把 DeepSeek 模型注入 LlamaIndex 白名单（不修改 .venv 源码）
        _llm_utils.ALL_AVAILABLE_MODELS.setdefault(app_cfg.OPENAI_MODEL, 128000)
        _llm_utils.CHAT_MODELS.setdefault(app_cfg.OPENAI_MODEL, 128000)

        # 统一给 LlamaIndex 注入真实 LLM，避免内部回退到 MockLLM 污染运行行为。
        self._llm = LlamaOpenAI(
            model=app_cfg.OPENAI_MODEL,
            api_key=app_cfg.OPENAI_API_KEY,
            api_base=app_cfg.OPENAI_BASE_URL or "https://api.openai.com/v1",
            temperature=0.0,
            max_tokens=1024,
            logprobs=None,
            default_headers={},
        )
        Settings.llm = self._llm

        persist_dir = self.PERSIST_DIR

        # 若索引已缓存（内存中），直接用
        if persist_dir in _INDEX_CACHE:
            index = _INDEX_CACHE[persist_dir]
        elif Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            # 从磁盘恢复
            storage_ctx = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_ctx)
        else:
            # 首次构建：从 data/documents/ 加载文档
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            texts = _load_documents()
            docs = [
                Document(text=t, doc_id=f"doc_{i}", metadata={"source": f"doc_{i}"})
                for i, t in enumerate(texts)
            ]
            index = VectorStoreIndex.from_documents(docs, show_progress=False)
            index.storage_context.persist(persist_dir=persist_dir)

        _INDEX_CACHE[persist_dir] = index
        self._retriever = index.as_retriever(similarity_top_k=8)

        self._initialized = True

    # ── retrieve ──────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self._ensure_init()
        try:
            if self._retriever is None:
                return []
            nodes = self._retriever.retrieve(query)
            return [
                RetrievedChunk(
                    content=node.node.text,
                    metadata=node.node.metadata or {},
                    score=float(node.score) if node.score is not None else 0.0,
                )
                for node in nodes[:top_k]
            ]
        except Exception as exc:
            return [RetrievedChunk(
                content=f"[检索异常: {exc}]", metadata={}, score=0.0
            )]

    # ── generate ──────────────────────────────────────────────────────────────
    def generate(self, query: str, contexts: list[RetrievedChunk]) -> GenerationResult:
        self._ensure_init()
        ctx_text = "\n\n".join(
            f"[文档 {i+1}] {c.content}" for i, c in enumerate(contexts)
        )
        prompt = (
            "你是一个知识助手，请仅根据以下上下文内容回答问题，不要引入上下文之外的信息。\n\n"
            f"上下文：\n{ctx_text}\n\n"
            f"问题：{query}\n\n"
            "请给出简洁准确的回答："
        )
        try:
            if self._llm is not None:
                from llama_index.core.llms import ChatMessage, MessageRole
                response = self._llm.chat(
                    [ChatMessage(role=MessageRole.USER, content=prompt)]
                )
                answer = response.message.content.strip()
            else:
                raise RuntimeError("LlamaIndex generation backend is not configured")
            return GenerationResult(answer=answer, sources=[])
        except Exception as exc:
            raise RuntimeError(f"LlamaIndex 生成失败: {exc}") from exc
