"""语义干扰攻击 — 改写 query 使其表面相关但实际指向不同概念"""

from typing import Optional
from .base import BaseAttack, AttackCase


class SemanticPerturbAttack(BaseAttack):
    name = "语义干扰"
    description = "改写 query 使其表面词汇相关但语义方向偏移，测试检索对语义漂移的鲁棒性"
    target = "query"

    def __init__(self, llm_provider=None):
        self._llm = llm_provider

    def apply(self, query: str, contexts: Optional[list[str]] = None) -> AttackCase:
        perturbed = self._perturb(query)
        return AttackCase(
            original_query=query,
            perturbed_query=perturbed,
            original_contexts=contexts or [],
            perturbed_contexts=contexts or [],
            attack_type=self.name,
            attack_target="query",
        )

    def _perturb(self, query: str) -> str:
        if self._llm is None:
            return f"请从不同角度谈谈：{query}"

        prompt = (
            "你是一个对抗攻击生成器。改写以下问题，使其从表面词汇看像在问一个"
            "常见概念，但实际指向的是完全不同的问题。\n"
            "具体方法：加入限定词改变问题焦点，或加入否定词制造双重否定，"
            "或添加无关的时间/地点修饰使检索方向偏移。\n"
            "示例：'太阳能电池板的转换效率是多少？'→'非晶硅太阳能电池板与单晶硅相比，"
            "其光电转换效率的差异是否反映了半导体材料带隙的物理限制？'\n"
            "禁止只做同义词替换——必须改变检索的语义方向。\n"
            "只输出改写后的问题。\n\n"
            f"原问题：{query}"
        )
        try:
            response = self._llm.generate(
                [{"role": "user", "content": prompt}], stream=False
            )
            return response.strip() or query
        except Exception:
            return query
