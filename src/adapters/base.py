from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """适配器返回的检索结果块，统一格式"""
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class GenerationResult:
    """适配器返回的生成结果"""
    answer: str
    sources: list = field(default_factory=list)


class BaseRAGAdapter(ABC):
    """目标 RAG 系统的统一接口。

    评估工具只依赖此抽象层——攻击模块注入对抗上下文、
    评估模块跑 RAGAS 评分，均通过 retrieve() / generate() 调用。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """系统标识名，用于报告标题和日志"""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """检索接口：输入 query，返回 top_k 个相关块"""
        ...

    @abstractmethod
    def generate(self, query: str, contexts: list[RetrievedChunk]) -> GenerationResult:
        """生成接口：基于检索上下文生成回答"""
        ...

    def get_context_contents(self, chunks: list[RetrievedChunk]) -> list[str]:
        """从 RetrievedChunk 列表提取纯文本上下文，供 RAGAS 评分使用"""
        return [c.content for c in chunks]
