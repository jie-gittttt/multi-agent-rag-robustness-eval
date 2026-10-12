"""关键词替换攻击 — 将领域术语替换为相似但语义不同的词，误导检索系统"""

from typing import Optional
from .base import BaseAttack, AttackCase


class KeywordReplaceAttack(BaseAttack):
    name = "关键词替换"
    description = "将 query 中领域术语替换为形近/音近但语义不同的术语，测试检索的术语鲁棒性"
    target = "query"

    def __init__(self, llm_provider=None):
        self._llm = llm_provider

    def apply(self, query: str, contexts: Optional[list[str]] = None) -> AttackCase:
        perturbed = self._replace_with_llm(query)
        return AttackCase(
            original_query=query,
            perturbed_query=perturbed,
            original_contexts=contexts or [],
            perturbed_contexts=contexts or [],
            attack_type=self.name,
            attack_target="query",
        )

    def _replace_with_llm(self, query: str) -> str:
        if self._llm is None:
            return self._fallback_replace(query)

        prompt = (
            "你是一个对抗攻击生成器。改写以下问题，使其措辞发生变化，"
            "可能导致检索系统检索到不同文档。\n"
            "用更具体或更宽泛的术语替换关键词，或改变句式结构。\n"
            "示例：'碳交易如何运作'→'碳排放权配额的市场化交易机制是怎样的'。\n"
            "必须改变至少一个实词，不能原样输出。\n"
            "只输出改写后的问题。\n\n"
            f"原问题：{query}"
        )
        try:
            response = self._llm.generate(
                [{"role": "user", "content": prompt}], stream=False
            )
            return response.strip() or query
        except Exception:
            return self._fallback_replace(query)

    def _fallback_replace(self, query: str) -> str:
        replacements = {
            "碳交易": "碳排放权期货交易",
            "温室效应": "大气保温效应",
            "可再生能源": "可持续能源",
            "碳中和": "净零碳排",
            "全球变暖": "地表增温",
            "光伏": "太阳能光电",
            "风能": "气流动力",
            "碳捕获": "碳收集",
            "京都议定书": "全球温室气体减排协议",
            "巴黎协定": "联合国气候治理框架公约",
            "退出": "终止参与",
            "宣布": "正式声明",
        }
        result = query
        replaced = 0
        for old, new in replacements.items():
            if old in result and replaced < 2:  # 最多替换两个词
                result = result.replace(old, new, 1)
                replaced += 1
        return result
