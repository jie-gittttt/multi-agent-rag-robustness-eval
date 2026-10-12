"""RAGFlow 适配器 — 用 RAGFlow 检索 + 用自己的 LLM 生成

检索交给 RAGFlow（DeepDoc 分块 + 混合搜索 + rerank），
生成用自己的 DeepSeek（保证和 LlamaIndex/LangChain 同一个 LLM，对比公平）。
"""

from __future__ import annotations
from .base import BaseRAGAdapter, RetrievedChunk, GenerationResult

# ── 与另外两个适配器保持完全一致的 prompt 模板 ──
PROMPT_TEMPLATE = (
    "你是一个知识助手，请仅根据以下上下文内容回答问题，"
    "不要引入上下文之外的信息。如果上下文不足以回答，请明确说'无法回答'。\n\n"
    "上下文：\n{context}\n\n"
    "问题：{question}\n\n"
    "请给出简洁准确的回答："
)


class RAGFlowAdapter(BaseRAGAdapter):
    """RAGFlow 成品 RAG 系统的适配器。

    前置条件:
    1. pip install ragflow-sdk
    2. 注册 RAGFlow Cloud (ragflow.io) 或本地部署
    3. 在 RAGFlow 中创建知识库、上传文档
    4. 获取 API Key 和知识库 ID，写入 .env
    """

    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None, kb_id: str | None = None):
        from ..config import Settings

        cfg = Settings()
        self._base_url = (base_url or cfg.RAGFLOW_BASE_URL or "http://localhost:9380").rstrip("/")
        self._api_key = api_key if api_key is not None else (cfg.RAGFLOW_API_KEY or "")
        self._kb_id = kb_id if kb_id is not None else (cfg.RAGFLOW_KB_ID or "")
        self._llm = None
        self._rag = None
        self._ready = False

    @property
    def name(self) -> str:
        return "RAGFlow"

    # ── 懒初始化 ──────────────────────────────────────────────────────────────
    def _ensure_init(self):
        if self._ready:
            return

        if not self._kb_id:
            raise RuntimeError("RAGFlow 未配置知识库 ID。请在 .env 中设置 RAGFLOW_KB_ID。")

        if not self._api_key:
            raise RuntimeError("RAGFlow 未配置 API Key。请在 .env 中设置 RAGFLOW_API_KEY。")

        try:
            from ragflow_sdk import RAGFlow
        except Exception as exc:
            raise RuntimeError(
                "缺少 ragflow-sdk 依赖，请先执行: pip install ragflow-sdk"
            ) from exc

        self._rag = RAGFlow(api_key=self._api_key, base_url=self._base_url)

        # 验证连接
        try:
            datasets = self._rag.list_datasets(id=self._kb_id)
            if not datasets:
                raise RuntimeError(f"知识库 {self._kb_id} 不存在或无权访问")
            self._ready = True
        except Exception as e:
            raise RuntimeError(
                f"RAGFlow 连接失败 ({self._base_url}): {e}\n"
                f"请确认 API Key 和知识库 ID 正确。"
            )

    # ── retrieve（用 RAGFlow 的检索管线）──────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self._ensure_init()
        try:
            result = self._rag.retrieve(
                dataset_ids=[self._kb_id],
                question=query,
                similarity_threshold=0.2,
                top_k=top_k,
            )
            chunks = []
            for item in result[:top_k]:
                content = getattr(item, "content", "") or str(item)
                doc_name = getattr(item, "document_name", "")
                similarity = getattr(item, "similarity", 0.0)
                chunks.append(RetrievedChunk(
                    content=content,
                    metadata={"source": doc_name},
                    score=float(similarity) if similarity else 0.0,
                ))
            return chunks
        except Exception as exc:
            return [RetrievedChunk(
                content=f"[RAGFlow 检索异常: {exc}]", metadata={}, score=0.0
            )]

    # ── generate（用自己的 DeepSeek LLM，与另外两个适配器一致）────────────
    def generate(self, query: str, contexts: list[RetrievedChunk]) -> GenerationResult:
        self._ensure_init()

        ctx_text = "\n\n".join(
            f"[文档 {i+1}] {c.content}" for i, c in enumerate(contexts)
        )
        prompt = PROMPT_TEMPLATE.format(context=ctx_text, question=query)

        try:
            if self._llm is None:
                from ..config import Settings
                from ..services.generation.llm_provider import LLMProviderFactory
                settings = Settings()
                self._llm = LLMProviderFactory.create(
                    settings.LLM_PROVIDER, settings
                )
            response = self._llm.generate(
                [{"role": "user", "content": prompt}], stream=False
            )
            return GenerationResult(answer=response.strip(), sources=[])
        except Exception as exc:
            raise RuntimeError(f"RAGFlow 生成失败: {exc}") from exc
