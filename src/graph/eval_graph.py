"""LangGraph StateGraph — 编排三个 Agent 的评估流水线"""

import json
from pathlib import Path
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from ..agents.attacker import AttackerAgent
from ..agents.evaluator import EvaluatorAgent
from ..agents.diagnostician import DiagnosticianAgent


class EvalState(TypedDict):
    """LangGraph 状态定义 — 在三个 Agent 间传递的数据"""

    # 输入
    seed_queries: list[dict]
    adapter: object           # BaseRAGAdapter
    metrics: Optional[list[str]]
    settings: object | None
    embedding_provider: object | None

    # Attacker 输出
    attack_cases: list[dict]

    # Evaluator 输出
    eval_results: list[dict]

    # Diagnostician 输出
    diagnosis: dict

    # 进度
    error: str


def _run_attacker(state: EvalState) -> dict:
    """Attacker 节点：生成对抗测试用例"""
    try:
        agent = AttackerAgent()
        cases = agent.generate_cases(
            seed_queries=state["seed_queries"],
            adapter=state.get("adapter"),
        )
        return {"attack_cases": cases, "error": ""}
    except Exception as exc:
        return {"attack_cases": [], "error": str(exc)}


def _run_evaluator(state: EvalState) -> dict:
    """Evaluator 节点：跑正常+对抗配对评分"""
    try:
        from ..adapters.base import BaseRAGAdapter
        from ..metrics.ragas_runner import RagasRunner

        runner = RagasRunner(
            settings=state.get("settings"),
            embedding_provider=state.get("embedding_provider"),
        )
        eval_agent = EvaluatorAgent(runner)
        results = eval_agent.run_batch(
            cases=state["attack_cases"],
            adapter=state["adapter"],
            metrics=state.get("metrics"),
        )
        return {"eval_results": results}
    except Exception as exc:
        return {"eval_results": [], "error": str(exc)}


def _run_diagnostician(state: EvalState) -> dict:
    """Diagnostician 节点：生成诊断报告"""
    try:
        agent = DiagnosticianAgent()
        diagnosis = agent.diagnose(state["eval_results"])
        return {"diagnosis": diagnosis}
    except Exception as exc:
        return {"diagnosis": {}, "error": str(exc)}


def build_graph() -> StateGraph:
    """构建评估流水线 StateGraph"""
    graph = StateGraph(EvalState)

    graph.add_node("attacker", _run_attacker)
    graph.add_node("evaluator", _run_evaluator)
    graph.add_node("diagnostician", _run_diagnostician)

    graph.set_entry_point("attacker")
    graph.add_edge("attacker", "evaluator")
    graph.add_edge("evaluator", "diagnostician")
    graph.add_edge("diagnostician", END)

    return graph


def run_eval_pipeline(seed_queries: list[dict], adapter,
                      metrics: Optional[list[str]] = None,
                      settings=None, embedding_provider=None) -> dict:
    """一键运行完整评估流水线。"""
    graph = build_graph()
    app = graph.compile()

    initial_state: EvalState = {
        "seed_queries": seed_queries,
        "adapter": adapter,
        "metrics": metrics,
        "settings": settings,
        "embedding_provider": embedding_provider,
        "attack_cases": [],
        "eval_results": [],
        "diagnosis": {},
        "error": "",
    }

    result = app.invoke(initial_state)
    return result


def run_eval_stream(seed_queries: list[dict], adapter,
                    metrics: Optional[list[str]] = None,
                    dev_mode: bool = False,
                    doc_dir: str = None,
                    n_seeds_ans: int = 15,
                    settings=None, embedding_provider=None):
    """流式运行评估流水线。

    dev_mode=True: 默认 1 种子 + 3 攻击 → 约 4 case，快速验证。
    doc_dir: 文档目录路径，若 seed_queries 为空且 doc_dir 非空，则自动从文档生成种子。
    n_seeds_ans: 可回答种子数（dev 模式默认 1，full 模式默认 15）。
    """

    # ── Attacker（含阶段 0: 种子生成）──
    yield {"attacker": {"status": "running"}}

    _llm_provider = None
    if settings is not None:
        try:
            from ..services.generation.llm_provider import LLMProviderFactory
            _llm_provider = LLMProviderFactory.create(settings.LLM_PROVIDER, settings)
        except Exception:
            pass

    attacker = AttackerAgent(_llm_provider)

    # 阶段 0: 逐条生成种子（循环覆盖显示进度）
    if (not seed_queries) and doc_dir:
        n_unans = 0  # 无答案种子从 seed_queries.json 手动提供，不自动生成
        seed_queries = []
        # 可回答种子
        for i in range(n_seeds_ans):
            batch = attacker.generate_seeds(doc_dir, n_answerable=1, n_unanswerable=0)
            if batch:
                seed_queries.append(batch[0])
            yield {"attacker": {"status": "seed_progress", "seed_index": len(seed_queries), "seed_total": n_seeds_ans, "seed_type": "可回答"}}
        # 无答案种子
        for i in range(n_unans):
            batch = attacker.generate_seeds(doc_dir, n_answerable=0, n_unanswerable=1)
            if batch:
                seed_queries.append(batch[0])
            yield {"attacker": {"status": "seed_progress", "seed_index": len(seed_queries), "seed_total": n_seeds_ans + n_unans, "seed_type": "无答案"}}
        yield {"attacker": {
            "status": "seeds_done",
            "seed_count": len(seed_queries),
            "seeds": seed_queries,
        }}

    # 阶段 0.5: 自动种子健壮性校验
    # 若本批“可回答”种子在当前检索器上全部不成立，则回退到本地种子文件，
    # 避免出现整批 has_answer=False 导致的全 0 / 假 FAIL。
    if seed_queries and adapter is not None:
        def _seed_gt(sq: dict) -> str:
            return (
                sq.get("answer", "")
                or " ".join(sq.get("expected_concepts", []))
                or sq.get("evidence", "")
            )

        supported_answerable = 0
        answerable_total = 0
        for sq in seed_queries:
            if sq.get("has_answer", True) is False:
                continue
            answerable_total += 1
            q = sq.get("question", "")
            gt = _seed_gt(sq)
            if not q or not gt:
                continue
            try:
                chunks = adapter.retrieve(q, top_k=5)
                ctxs = [c.content for c in chunks]
            except Exception:
                ctxs = []
            if attacker._is_ground_truth_supported(gt, ctxs):
                supported_answerable += 1

        if answerable_total == 0 or (answerable_total > 0 and supported_answerable == 0):
            fallback_path = Path("data/seed_queries.json")
            if fallback_path.exists():
                try:
                    fallback = json.loads(fallback_path.read_text(encoding="utf-8"))
                    if isinstance(fallback, list) and fallback:
                        use_n = 1 if dev_mode else max(n_seeds_ans, 1)
                        seed_queries = fallback[:use_n]
                        yield {"attacker": {
                            "status": "seeds_fallback",
                            "reason": "auto_seeds_unanswerable",
                            "seed_count": len(seed_queries),
                        }}
                except Exception:
                    pass

    # 阶段 1: 生成对抗样本（逐攻击类型 yield 进度）
    attack_cases = []
    n_attacks = 3 if dev_mode else 6
    active_attacks = attacker._attacks[:n_attacks]
    for si, sq in enumerate(seed_queries):
        for atk in active_attacks:
            yield {"attacker": {"status": "attack_progress", "seed_index": si + 1, "seed_total": len(seed_queries), "attack_name": atk.name}}
    attack_cases = attacker.generate_cases(seed_queries, adapter, dev_mode=dev_mode)
    yield {"attacker": {
        "status": "done",
        "attack_cases": attack_cases,
        "case_count": len(attack_cases),
    }}

    # ── Evaluator（按种子分组，每个种子的正常侧只算一次，对抗侧复用基准）──
    from ..metrics.ragas_runner import RagasRunner
    runner = RagasRunner(settings=settings, embedding_provider=embedding_provider)
    eval_agent = EvaluatorAgent(runner)
    eval_results = []
    total = len(attack_cases)

    # 按种子分组预计算基准答案 + 评分（每个种子只算一次正常侧）
    from ..adapters.base import RetrievedChunk
    from ..metrics.rejection import evaluate_rejection
    seed_groups: dict[str, tuple] = {}
    seeds_total = len(set(c.get("original_query", "") for c in attack_cases))
    seeds_done = 0
    for case in attack_cases:
        q = case.get("original_query", "")
        if q not in seed_groups:
            chunks = [RetrievedChunk(content=t, metadata={}, score=0.0)
                      for t in case.get("original_contexts", [])]
            answer = adapter.generate(q, chunks)
            if case.get("has_answer", True):
                result = runner.score(
                    question=q, answer=answer.answer,
                    contexts=[c.content for c in chunks],
                    ground_truth=case.get("ground_truth", ""), metrics=metrics,
                )
            else:
                result = evaluate_rejection(answer.answer, has_answer=False)
            seed_groups[q] = (answer, result)
            seeds_done += 1
            yield {"evaluator": {"status": "preparing", "done": seeds_done, "total": seeds_total}}

    import time as _time
    def _save_intermediate(evals, cases, label):
        from pathlib import Path
        Path("data").mkdir(parents=True, exist_ok=True)
        import json as _json
        payload = {
            "system": label,
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "partial": True,
            "overall_robustness": 0,
            "eval_results": evals,
            "attack_cases": cases,
        }
        out = Path(f"data/results_{label}_{_time.strftime('%Y%m%d_%H%M%S')}_partial.json")
        out.write_text(_json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    processed = 0
    last_seed = None
    for case in attack_cases:
        q = case.get("original_query", "")
        cached = seed_groups.get(q)
        try:
            result = eval_agent.evaluate_pair(case, adapter, metrics=metrics, cached_normal=cached)
            eval_results.append(result)
        except Exception as exc:
            eval_results.append({"case": case, "error": str(exc)})
        processed += 1
        # 每完成一个种子就存盘（中断不丢数据）
        if q != last_seed and processed > 1:
            _save_intermediate(eval_results, attack_cases, adapter.name)
        last_seed = q

        yield {"evaluator": {
            "status": "running",
            "done": processed,
            "total": total,
            "seed_id": case.get("original_query", "")[:50],
            "attack_name": case.get("attack_type", ""),
        }}

    yield {"evaluator": {
        "status": "done",
        "eval_results": eval_results,
        "result_count": len(eval_results),
    }}

    # ── Diagnostician ──
    yield {"diagnostician": {"status": "running"}}
    diagnostician = DiagnosticianAgent()
    diagnosis = diagnostician.diagnose(eval_results)
    yield {"diagnostician": {
        "status": "done",
        "diagnosis": diagnosis,
    }}


def get_graph_mermaid() -> str:
    """返回 LangGraph 的 mermaid 图（用于前端展示 Agent 协作流程）"""
    graph = build_graph()
    return graph.compile().get_graph().draw_mermaid()
