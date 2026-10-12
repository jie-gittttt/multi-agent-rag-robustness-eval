"""排序污染攻击 — 颠倒检索结果的排序，测试 RAG 对上下文排列顺序的敏感性"""

from typing import Optional
from .base import BaseAttack, AttackCase


class RankingPollutionAttack(BaseAttack):
    name = "排序污染"
    description = "将检索结果按相关度逆序排列，测试 RAG 对上下文顺序的依赖程度"
    target = "ranking"

    def apply(self, query: str, contexts: Optional[list[str]] = None) -> AttackCase:
        ctx_list = contexts or []
        # 逆序：最相关的排最后，最不相关的排第一
        perturbed = list(reversed(ctx_list))
        return AttackCase(
            original_query=query,
            perturbed_query=query,
            original_contexts=ctx_list,
            perturbed_contexts=perturbed,
            attack_type=self.name,
            attack_target="ranking",
        )
