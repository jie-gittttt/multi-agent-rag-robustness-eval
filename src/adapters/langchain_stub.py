"""基于 LangChain 构建的 RAG 适配器

从 data/documents/*.md 读取文档，通过 Chroma + HuggingFace BGE embedding
构建向量索引，LLM 使用 DeepSeek（兼容 OpenAI API）。
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

_VECTORSTORE_CACHE: dict = {}  # {persist_dir: Chroma}


class LangChainRAGAdapter(BaseRAGAdapter):
    """基于 LangChain LCEL 的 RAG 适配器，使用内置中文知识库自动建立 Chroma 向量索引。

    若 retriever + llm_chain 已提供则直接使用；否则自动初始化内置知识库。
    """

    PERSIST_DIR = "data/langchain_chroma"

    def __init__(self, retriever=None, llm_chain=None):
        self._retriever = retriever
        self._chain = llm_chain
        self._vectorstore = None
        self._llm = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "LangChain RAG"

    # ── 懒初始化 ──────────────────────────────────────────────────────────────
    def _ensure_init(self):
        if self._initialized:
            return
        if self._retriever is not None:
            self._initialized = True
            return
        self._build_index()

    def _build_index(self):
        from langchain_openai import ChatOpenAI
        from langchain_core.embeddings import Embeddings
        from langchain_community.vectorstores import Chroma

        from ..config import Settings as AppSettings
        from ..services.embedding.local_embedding import LocalBGEZhEmbeddingProvider

        app_cfg = AppSettings()

        # ── 用项目已有的 BGE 模型包装为 LangChain Embeddings 接口 ──
        class _BGEWrapper(Embeddings):
            def __init__(self):
                self._provider = LocalBGEZhEmbeddingProvider(app_cfg.EMBEDDING_MODEL)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self._provider.embed(texts)

            def embed_query(self, text: str) -> list[float]:
                return self._provider.embed_query(text)

        embeddings = _BGEWrapper()
        persist_dir = self.PERSIST_DIR

        # 若向量库已在内存缓存，直接用
        if persist_dir in _VECTORSTORE_CACHE:
            vs = _VECTORSTORE_CACHE[persist_dir]
        elif Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            # 从磁盘恢复（Chroma 持久化目录存在）
            vs = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name="langchain_knowledge",
            )
        else:
            # 首次构建：从 data/documents/ 加载文档
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            texts = _load_documents()
            from langchain_core.documents import Document as LCDocument
            docs = [
                LCDocument(
                    page_content=t,
                    metadata={"source": f"doc_{i}"},
                )
                for i, t in enumerate(texts)
            ]
            vs = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name="langchain_knowledge",
            )

        _VECTORSTORE_CACHE[persist_dir] = vs
        self._vectorstore = vs
        self._retriever = vs.as_retriever(search_kwargs={"k": 8})

        # ── DeepSeek LLM ──
        kwargs: dict = dict(
            model=app_cfg.OPENAI_MODEL,
            api_key=app_cfg.OPENAI_API_KEY,
            temperature=0.0,
            max_tokens=1024,
        )
        if app_cfg.OPENAI_BASE_URL:
            kwargs["base_url"] = app_cfg.OPENAI_BASE_URL
        self._llm = ChatOpenAI(**kwargs)
        self._initialized = True

    # ── retrieve ──────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self._ensure_init()
        try:
            if self._vectorstore is not None:
                pairs = self._vectorstore.similarity_search_with_relevance_scores(
                    query, k=top_k
                )
                return [
                    RetrievedChunk(
                        content=doc.page_content,
                        metadata=doc.metadata or {},
                        score=float(score),
                    )
                    for doc, score in pairs
                ]
            if self._retriever is not None:
                docs = self._retriever.invoke(query)
                return [
                    RetrievedChunk(
                        content=doc.page_content,
                        metadata=doc.metadata or {},
                        score=0.0,
                    )
                    for doc in docs[:top_k]
                ]
            return []
        except Exception as exc:
            return [RetrievedChunk(content=f"[检索异常: {exc}]", metadata={}, score=0.0)]

    # ── generate ──────────────────────────────────────────────────────────────
    def generate(self, query: str, contexts: list[RetrievedChunk]) -> GenerationResult:
        self._ensure_init()
        ctx_text = "\n\n".join(
            f"[文档 {i+1}] {c.content}" for i, c in enumerate(contexts)
        )
        try:
            if self._llm is not None:
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser

                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "你是一个知识助手，请仅根据以下上下文内容回答问题，不要引入上下文之外的信息。\n\n"
                     "上下文：\n{context}"),
                    ("human", "{question}"),
                ])
                chain = prompt | self._llm | StrOutputParser()
                answer = chain.invoke({"context": ctx_text, "question": query})
                return GenerationResult(answer=answer.strip(), sources=[])

            if self._chain is not None:
                answer = self._chain.invoke({"query": query, "context": ctx_text})
                return GenerationResult(
                    answer=answer if isinstance(answer, str) else str(answer),
                    sources=[],
                )
        except Exception as exc:
            raise RuntimeError(f"LangChain 生成失败: {exc}") from exc

        raise RuntimeError("LangChain generation backend is not configured")
