"""Agent 2: 评估执行 — 在目标 RAG 上跑正常+对抗配对，输出差分报告"""

from ..adapters.base import BaseRAGAdapter, RetrievedChunk
from ..metrics.ragas_runner import RagasRunner
from ..metrics.robustness import compute_robustness
from ..metrics.rejection import evaluate_rejection


class EvaluatorAgent:
    """评估执行 Agent。对每条对抗测试用例跑正常+对抗配对评分。"""

    def __init__(self, ragas_runner: RagasRunner):
        self._runner = ragas_runner

    def evaluate_pair(self, case: dict, adapter: BaseRAGAdapter,
                      metrics: list[str] | None = None,
                      cached_normal=None) -> dict:
        """对一条测试用例跑正常 vs 对抗配对。

        可回答 case → RAGAS 四维评分
        无答案 case → 拒答率/幻觉率评估
        """
        original_query = case["original_query"]
        original_ctxs = self._to_chunks(case["original_contexts"])
        perturbed_query = case["perturbed_query"]
        perturbed_ctxs = self._to_chunks(case["perturbed_contexts"])
        ground_truth = case.get("ground_truth", "")
        has_answer = case.get("has_answer", True)
        is_control = case.get("attack_type", "") == "无攻击（对照组）"

        # ── 对照组 ──
        if is_control:
            if cached_normal:
                answer, result = cached_normal
            else:
                answer = adapter.generate(original_query, original_ctxs)
                ctx_contents = [c.content for c in original_ctxs]
                if has_answer:
                    result = self._runner.score(
                        question=original_query, answer=answer.answer,
                        contexts=ctx_contents, ground_truth=ground_truth,
                        metrics=metrics,
                    )
                else:
                    result = evaluate_rejection(answer.answer, has_answer=False)

            if has_answer:
                return {
                    "case": case,
                    "normal": {"answer": answer.answer,
                               "scores": result["scores"] if isinstance(result, dict) else {},
                               "details": result["details"] if isinstance(result, dict) and "details" in result else {}},
                    "attack": {"answer": "", "scores": {}, "details": {}},
                    "robustness": None,
                }
            else:
                return {
                    "case": case,
                    "normal": {"answer": answer.answer, "rejection": result},
                    "attack": {"answer": "", "rejection": {}},
                    "robustness": None,
                }

        # ── 对抗模式 ──
        # 优先使用外部传入的缓存基准（同一种子只算一次正常侧）
        if cached_normal:
            normal_answer, normal_result = cached_normal
        else:
            normal_answer = adapter.generate(original_query, original_ctxs)
            normal_ctx_contents = [c.content for c in original_ctxs]
            if has_answer:
                normal_result = self._runner.score(
                    question=original_query, answer=normal_answer.answer,
                    contexts=normal_ctx_contents, ground_truth=ground_truth,
                    metrics=metrics,
                )
            else:
                normal_result = evaluate_rejection(normal_answer.answer, has_answer=False)

        at = case.get("attack_target", "query")
        if at in ("context", "ranking"):
            attack_answer = adapter.generate(original_query, perturbed_ctxs)
            attack_ctx_contents = [c.content for c in perturbed_ctxs]
        else:
            attack_answer = adapter.generate(perturbed_query, original_ctxs)
            attack_ctx_contents = [c.content for c in original_ctxs]

        if has_answer:
            attack_result = self._runner.score(
                question=perturbed_query if at == "query" else original_query,
                answer=attack_answer.answer,
                contexts=attack_ctx_contents,
                ground_truth=ground_truth,
                metrics=metrics,
            )
            norm_scores = normal_result["scores"] if isinstance(normal_result, dict) else {}
            robustness = compute_robustness(norm_scores, attack_result["scores"])
            return {
                "case": case,
                "normal": {"answer": normal_answer.answer, "scores": norm_scores,
                           "details": normal_result["details"] if isinstance(normal_result, dict) and "details" in normal_result else {}},
                "attack": {"answer": attack_answer.answer, "scores": attack_result["scores"], "details": attack_result["details"]},
                "robustness": robustness,
            }
        else:
            attack_rej = evaluate_rejection(attack_answer.answer, has_answer=False)
            return {
                "case": case,
                "normal": {"answer": normal_answer.answer, "rejection": normal_result},
                "attack": {"answer": attack_answer.answer, "rejection": attack_rej},
                "robustness": None,
            }

    def run_batch(self, cases: list[dict], adapter: BaseRAGAdapter,
                  metrics: list[str] | None = None,
                  progress_callback=None) -> list[dict]:
        """批量跑配对评估。progress_callback(i, total) 每条完成时回调。"""
        results = []
        total = len(cases)
        for i, case in enumerate(cases):
            try:
                result = self.evaluate_pair(case, adapter, metrics=metrics)
                results.append(result)
            except Exception as exc:
                results.append({"case": case, "error": str(exc)})
            if progress_callback:
                progress_callback(i + 1, total)
        return results

    def _to_chunks(self, ctx_list: list[str]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(content=t, metadata={"index": i}, score=0.0)
            for i, t in enumerate(ctx_list)
        ]
