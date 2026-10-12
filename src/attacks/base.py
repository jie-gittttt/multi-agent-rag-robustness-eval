"""攻击策略抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class AttackCase:
    """单条攻击样本"""
    original_query: str
    perturbed_query: str                          # 攻击后 query（或原 query）
    original_contexts: list[str] = field(default_factory=list)   # 原始检索上下文
    perturbed_contexts: list[str] = field(default_factory=list)   # 攻击后上下文
    attack_type: str = ""
    attack_target: Literal["query", "context", "ranking", "mixed"] = "query"


class BaseAttack(ABC):
    """攻击策略基类。每个子类实现一种对抗攻击。"""

    name: str = ""
    description: str = ""
    target: Literal["query", "context", "ranking", "mixed"] = "query"

    @abstractmethod
    def apply(self, query: str, contexts: Optional[list[str]] = None) -> AttackCase:
        """对输入 query/contexts 施加攻击，返回攻击样本"""
        ...

    def __repr__(self) -> str:
        return f"{self.name}({self.target})"
