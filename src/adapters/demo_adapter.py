"""Demo 适配器 — 使用假数据模拟 RAG 系统，无需外部依赖即可运行"""

import hashlib
from .base import BaseRAGAdapter, RetrievedChunk, GenerationResult

_DEMO_KNOWLEDGE = [
    "生命周期评价（LCA）包含四个核心阶段：目标与范围的确定、生命周期清单分析（LCI）、生命周期影响评价（LCIA）和结果解释。ISO 14040 和 ISO 14044 是 LCA 的国际标准。",
    "碳中和是指通过植树造林、节能减排、使用可再生能源、碳捕获与封存（CCS）等方式，抵消自身产生的二氧化碳排放量，实现净零排放。实现碳中和的路径包括能源结构转型、提高能源效率、发展可再生能源和自然碳汇。",
    "可再生能源包括太阳能、风能、水能、生物质能、地热能和海洋能。太阳能光伏发电和风力发电是增长最快的可再生能源技术。",
    "水泥生产过程中碳排放主要来源于两个方面：一是碳酸钙在高温下分解为氧化钙和二氧化碳（过程排放），二是窑炉燃料燃烧产生的二氧化碳（燃烧排放）。熟料煅烧是碳排放最集中的环节。",
    "监督学习使用带有标签的训练数据来预测输出，典型任务包括分类和回归。无监督学习不使用标签，从数据中自动发现模式和结构，典型任务包括聚类和降维。",
    "数据库索引是一种数据结构，用于加速数据库表的查询操作。常见的索引类型包括 B 树索引、哈希索引、全文索引和空间索引。索引可以提高查询性能但会降低写入性能。",
    "HTTP 是明文传输协议，HTTPS 在 HTTP 基础上增加了 SSL/TLS 加密层。HTTPS 使用数字证书验证服务器身份，确保数据传输的机密性和完整性。",
    "列表推导式一次性在内存中创建完整列表。生成器表达式使用惰性求值，逐个产生元素，不预先占用全部内存。对于大数据集，生成器表达式的内存效率更高。",
    # 新增 4 条，对应扩充的种子查询
    "检索增强生成（RAG）是一种结合信息检索和大语言模型的技术框架。RAG的核心流程：将外部知识分块并建立向量索引；用户提问时通过向量嵌入相似度检索相关上下文；将检索到的上下文与问题一起送入大语言模型（LLM）生成答案。RAG有效解决了LLM知识截止和幻觉问题。",
    "Transformer模型的注意力机制（Attention）是其核心组件。自注意力（Self-Attention）通过Query、Key、Value三个矩阵计算序列中各位置的关联权重。Multi-Head Attention并行运行多个注意力头，从不同子空间捕获语义关系。BERT、GPT等大语言模型均基于Transformer架构构建。",
    "气候变化是指由人类活动导致大气温室气体浓度上升，进而引起全球变暖的现象。主要原因是化石燃料燃烧和土地利用变化排放CO2等温室气体。主要后果包括极端天气事件频发（热浪、洪涝、干旱）、海平面上升、冰川融化和生态系统破坏。",
    "向量数据库是专门存储和检索高维嵌入向量的数据库系统，支持基于余弦相似度等指标的近似最近邻（ANN）搜索。向量数据库是RAG系统的核心组件，负责高效存储文档嵌入并快速检索相关内容。常见向量数据库包括ChromaDB、Faiss、Pinecone、Milvus和Weaviate。",
]


class DemoRAGAdapter(BaseRAGAdapter):
    """模拟 RAG 系统：检索用简单关键词匹配，生成用 LLM 或预设回答。

    队友可以直接在 Gradio 里选这个适配器跑通全流程，不需要接入任何真实 RAG 系统。
    """

    @property
    def name(self) -> str:
        return "Demo RAG（模拟系统）"

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """简单 TF 匹配检索，返回最相关的 demo 知识片段"""
        scored = []
        query_terms = set(query)
        for text in _DEMO_KNOWLEDGE:
            # 简单字符级 Jaccard 相似度
            text_terms = set(text)
            intersection = query_terms & text_terms
            union = query_terms | text_terms
            score = len(intersection) / max(len(union), 1)
            scored.append((text, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        return [
            RetrievedChunk(
                content=text,
                metadata={"source": f"demo_doc_{i}", "score": score},
                score=score,
            )
            for i, (text, score) in enumerate(top)
        ]

    def generate(self, query: str, contexts: list[RetrievedChunk]) -> GenerationResult:
        """用 LLM 生成回答，或 fallback 到预设回答"""
        try:
            from ..services.generation.llm_provider import LLMProviderFactory
            from ..config import Settings

            settings = Settings()
            llm = LLMProviderFactory.create(settings.LLM_PROVIDER, settings)
            ctx_text = "\n".join(c.content[:300] for c in contexts)
            prompt = (
                "你是一个知识助手。仅根据以下上下文回答问题。\n\n"
                f"上下文：\n{ctx_text}\n\n"
                f"问题：{query}"
            )
            answer = llm.generate(
                [{"role": "user", "content": prompt}], stream=False
            )
            return GenerationResult(answer=answer.strip(), sources=[])
        except Exception:
            # 无 LLM 时返回第一个上下文作为假回答
            fallback = contexts[0].content if contexts else "无法回答（Demo 适配器无 LLM）"
            return GenerationResult(answer=fallback, sources=[])
