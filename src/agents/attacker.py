"""Agent 1: 攻击生成 — 阶段0: 从文档生成种子问题 → 阶段1: 生成对抗测试用例集"""

import json
import itertools
import re
from pathlib import Path
from ..attacks.keyword_replace import KeywordReplaceAttack
from ..attacks.semantic_perturb import SemanticPerturbAttack
from ..attacks.context_hijack import ContextHijackAttack
from ..attacks.prior_conflict import PriorConflictAttack
from ..attacks.noise_injection import NoiseInjectionAttack
from ..attacks.ranking_pollution import RankingPollutionAttack


class AttackerAgent:
    """攻击生成 Agent。输入种子问题 + 目标 RAG 适配器，输出 100+ 对抗测试用例。"""

    def __init__(self, llm_provider=None):
        self._llm = llm_provider
        self._attacks = [
            KeywordReplaceAttack(llm_provider),
            SemanticPerturbAttack(llm_provider),
            ContextHijackAttack(llm_provider),
            PriorConflictAttack(llm_provider),
            NoiseInjectionAttack(),
            RankingPollutionAttack(),
        ]

    # ── 阶段 0: 从文档生成种子问题 ──────────────────────────────────────────

    def generate_seeds(self, doc_dir: str, n_answerable: int = 15,
                       n_unanswerable: int = 5,
                       dev_mode: bool = False) -> list[dict]:
        """从文档生成种子问题。

        Args:
            doc_dir: 文档目录路径
            n_answerable: 可回答问题数（滑块传入，dev_mode 默认 1）
            n_unanswerable: 无答案问题数（dev_mode 默认 0）
            dev_mode: 开发模式，n_answerable/n_unanswerable 未显式指定时使用默认值

        Returns:
            [{"question": "...", "answer": "...", "evidence": "...",
              "has_answer": True/False, "doc_source": "..."}, ...]
        """
        if self._llm is None:
            return _FALLBACK_SEEDS

        docs = []
        for fpath in Path(doc_dir).glob("*.md"):
            docs.append({
                "title": fpath.stem,
                "content": fpath.read_text(encoding="utf-8")[:3000],
            })
        if not docs:
            return _FALLBACK_SEEDS

        seeds = []

        # ── 可回答型 ──
        prompt_template = (
            "你是一个评测数据生成器。请根据以下文档生成 1 条问答对。\n\n"
            "要求:\n"
            "- 问题的答案必须能从文档中找到原文依据\n"
            "- 问题需涉及具体数据/年份/专业术语/结论，不能是仅凭常识就能回答的通用问题\n"
            "- 同时给出: 正确答案 + 答案在原文中的关键句子\n\n"
            "输出严格 JSON（不要额外文本）:\n"
            '{{"question": "...", "answer": "...", "evidence": "..."}}\n\n'
            "文档:\n{doc_text}"
        )

        for doc in docs[:n_answerable]:
            prompt = prompt_template.format(doc_text=doc["content"])
            for attempt in range(3):
                try:
                    response = self._llm.generate(
                        [{"role": "user", "content": prompt}], stream=False
                    )
                    data = json.loads(response.strip())
                    data["has_answer"] = True
                    data["doc_source"] = doc["title"]
                    seeds.append(data)
                    break
                except Exception:
                    if attempt == 2:
                        pass
                    continue

        # ── 无答案型 ──
        titles = [d["title"] for d in docs]
        # 5 种不同角度 + 强约束：每个角度绑定具体的文档领域方向，避免 LLM 收敛到同一答案
        _UNANSWERABLE_SPECS = [
            {
                "angle": "某具体国家或地区的统计数据，该数据在文档中未提及",
                "hint": "例如：中国 2023 年地热发电装机容量、印度 2024 年风电弃风率、巴西乙醇汽油强制掺混比例等",
                "forbid": "",
            },
            {
                "angle": "特定技术参数或工程规格的精确数值，文档只有概念介绍",
                "hint": "例如：某种新型钙钛矿太阳能电池的实验室转换效率、某型号储能电池的循环寿命次数、某种碳捕集材料的吸附容量",
                "forbid": "",
            },
            {
                "angle": "某特定时间节点（某季度、某月份、某日）的价格、交易量或政策生效日期",
                "hint": "例如：2024 年某月欧盟碳配额日成交均价、某国光伏上网电价补贴截止日期、国际 LNG 现货价格峰值日期",
                "forbid": "",
            },
            {
                "angle": "跨领域、跨技术路线的精确量化对比数据",
                "hint": "例如：海上风电与陆上风电的全生命周期度电成本对比（含具体数值）、氢燃料电池与锂电池在重卡领域的 TCO 对比",
                "forbid": "",
            },
            {
                "angle": "特定企业、机构或示范项目的内部运营指标（商业/非公开信息）",
                "hint": "例如：特斯拉上海工厂 2024 年单车电耗、某碳捕集示范项目的年捕集量、台山核电站年发电量",
                "forbid": "",
            },
        ]

        generated_unans_questions: list[str] = []
        for i in range(n_unanswerable):
            spec = _UNANSWERABLE_SPECS[i % len(_UNANSWERABLE_SPECS)]
            prev_qs_text = ""
            forbid_text = ""
            if generated_unans_questions:
                forbid_text = (
                    "【强制约束】严禁生成与以下已生成问题相同或高度相似的问题（包括仅换一词的变体）。"
                    "你必须选择一个完全不同的主题、不同国家/地区、不同技术类型：\n"
                    + "\n".join(f"  ❌ 禁止: {q}" for q in generated_unans_questions)
                    + "\n\n"
                )
            _prompt_head = (
                "你是一个评测数据生成器。给定以下文档列表的标题，请识别文档未覆盖的主题，"
                "生成 1 条该领域相关但文档内容无法回答的问题。\n\n"
                f"本次要求的角度：{spec['angle']}\n"
                f"可参考的方向（仅作启发，请自选具体主题）：{spec['hint']}\n\n"
            )
            _prompt_tail = (
                "问题必须涉及具体数据（如某年某地区的具体数值），"
                "答案不可能在文档中找到。\n\n"
                "输出严格 JSON（不要额外文本）:\n"
                '{"question": "...", "answer": null, "evidence": null}\n\n'
                f"已有文档标题: {', '.join(titles)}"
            )
            for attempt in range(3):
                if attempt < 2 and forbid_text:
                    _prompt = _prompt_head + forbid_text + _prompt_tail
                else:
                    _prompt = _prompt_head + _prompt_tail
                try:
                    response = self._llm.generate(
                        [{"role": "user", "content": _prompt}], stream=False
                    )
                    data = json.loads(response.strip())
                    data["has_answer"] = False
                    data["doc_source"] = None
                    seeds.append(data)
                    generated_unans_questions.append(data.get("question", ""))
                    break
                except Exception:
                    continue

        return seeds if seeds else _FALLBACK_SEEDS

    # ── 阶段 1: 生成对抗样本 ────────────────────────────────────────────────

    def generate_cases(self, seed_queries: list,
                       adapter=None, top_k: int = 5,
                       dev_mode: bool = False) -> list[dict]:
        """生成对抗测试用例集。

        Args:
            seed_queries: 种子问题列表 [{"question": "...", "answer": "...", ...}, ...]
            adapter: BaseRAGAdapter 实例（用于先检索上下文再施加 context 攻击）
            top_k: 检索时取 top_k

        Returns:
            [{
                "original_query": "...",
                "perturbed_query": "...",
                "original_contexts": [...],
                "perturbed_contexts": [...],
                "attack_type": "关键词替换",
                "attack_target": "query",
                "ground_truth": "...",
                "has_answer": True/False,
            }, ...]
        """
        if dev_mode:
            active_attacks = self._attacks[:3]  # 开发模式只用前 3 种攻击
        else:
            active_attacks = self._attacks

        cases = []

        for sq in seed_queries:
            query = sq.get("question", "")
            gt = sq.get("answer", "") or " ".join(sq.get("expected_concepts", []))
            has_answer = sq.get("has_answer", True)

            # 检索原始上下文
            original_ctxs = []
            if adapter is not None:
                try:
                    chunks = adapter.retrieve(query, top_k=top_k)
                    original_ctxs = [c.content for c in chunks]
                except Exception:
                    original_ctxs = []

            # 对照组（has_answer 以种子生成为准，RAGAS 评分自然反映检索质量）
            cases.append({
                "original_query": query,
                "perturbed_query": query,
                "original_contexts": original_ctxs,
                "perturbed_contexts": original_ctxs,
                "attack_type": "无攻击（对照组）",
                "attack_target": "none",
                "ground_truth": gt,
                "has_answer": has_answer,
            })

            # 每种攻击 × 1 策略
            for attack in active_attacks:
                try:
                    ac = attack.apply(query, original_ctxs if original_ctxs else None)
                    cases.append({
                        "original_query": ac.original_query,
                        "perturbed_query": ac.perturbed_query,
                        "original_contexts": ac.original_contexts,
                        "perturbed_contexts": ac.perturbed_contexts,
                        "attack_type": ac.attack_type,
                        "attack_target": ac.attack_target,
                        "ground_truth": gt,
                        "has_answer": has_answer,
                    })
                except Exception:
                    continue

        return cases

    def _is_ground_truth_supported(self, ground_truth: str, contexts: list[str]) -> bool:
        if not ground_truth or not contexts:
            return False

        text = "\n".join(contexts)
        if not text.strip():
            return False

        # 先做完整短语命中（适配中文长句答案）
        phrases = [p.strip() for p in re.split(r"[，。；、,.!?！？\s]+", ground_truth) if len(p.strip()) >= 3]
        for p in phrases:
            if p in text:
                return True

        # 再做关键词覆盖率（适配“关键词拼接”的 ground_truth）
        tokens = [t.strip() for t in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", ground_truth)]
        if not tokens:
            return False
        hits = sum(1 for t in tokens if t in text)
        return (hits / len(tokens)) >= 0.4

    @property
    def attack_names(self) -> list[str]:
        return [a.name for a in self._attacks]


# 当 LLM 不可用时使用的兜底种子
_FALLBACK_SEEDS = [
    {
        "question": "根据IPCC评估报告，全球变暖的主要人为驱动因素是什么？",
        "answer": "化石燃料燃烧和土地利用变化排放的温室气体",
        "evidence": "IPCC报告指出人类活动导致的温室气体排放是变暖主因",
        "has_answer": True,
        "doc_source": "全球变暖.md",
    },
    {
        "question": "碳中和与净零排放的核心区别是什么？",
        "answer": "碳中和侧重碳抵消，净零排放要求源头减排为主",
        "evidence": "碳中和允许碳信用抵消，净零排放要求价值链深度脱碳",
        "has_answer": True,
        "doc_source": "碳中和.md",
    },
    {
        "question": "中国2023年光伏新增装机容量具体是多少GW？",
        "answer": None,
        "evidence": None,
        "has_answer": False,
        "doc_source": None,
    },
]
