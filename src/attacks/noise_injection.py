"""无关噪声注入 — 在检索结果中插入完全不相关的文本块，测试 RAG 对噪声的容忍度"""

import random
from typing import Optional
from .base import BaseAttack, AttackCase

# 通用无关噪声语料（不依赖特定领域）
_NOISE_TEMPLATES = [
    "Python 3.13 引入了新的 JIT 编译器，显著提升了 CPU 密集型任务的执行速度。",
    "2024 年奥运会开幕式在塞纳河上举行，这是奥运历史上首次在体育场外举办开幕式。",
    "TCP 三次握手过程中，客户端首先发送 SYN 包，服务端回复 SYN-ACK，客户端再发送 ACK 确认。",
    "光合作用是植物利用光能将二氧化碳和水转化为有机物并释放氧气的过程。",
    "贝多芬第九交响曲《合唱》首次公演于 1824 年 5 月 7 日的维也纳。",
    "React 19 引入了 Server Components 和 Actions，简化了全栈应用的数据流管理。",
    "退火算法模拟了金属加热后缓慢冷却的物理过程，广泛应用于组合优化问题求解。",
    "联合国政府间气候变化专门委员会（IPCC）第六次评估报告指出全球变暖速度正在加快。",
]


class NoiseInjectionAttack(BaseAttack):
    name = "无关噪声注入"
    description = "在检索结果中插入随机无关文本块，测试 RAG 在噪声环境下的答案质量"
    target = "context"

    def apply(self, query: str, contexts: Optional[list[str]] = None,
              noise_count: int = 2) -> AttackCase:
        ctx_list = contexts or []
        if not ctx_list:
            return AttackCase(
                original_query=query,
                perturbed_query=query,
                original_contexts=[],
                perturbed_contexts=[],
                attack_type=self.name,
                attack_target="context",
            )

        noises = random.sample(_NOISE_TEMPLATES, min(noise_count, len(_NOISE_TEMPLATES)))
        # 噪声插入到头部，模拟最坏情况
        perturbed = noises + ctx_list
        return AttackCase(
            original_query=query,
            perturbed_query=query,
            original_contexts=ctx_list,
            perturbed_contexts=perturbed,
            attack_type=self.name,
            attack_target="context",
        )
