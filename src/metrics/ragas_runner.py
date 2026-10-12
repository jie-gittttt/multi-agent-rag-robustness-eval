"""RAGAS 四维评分运行器 — 对给定的 (query, answer, contexts, ground_truth) 评分

复用现有 ragas_evaluator.py 的手动编排方法，但不做检索和生成——由外部适配器提供数据。
"""

import asyncio
from typing import Optional


class RagasRunner:
    """轻量 RAGAS 评分器：只做评分，不做检索/生成"""

    def __init__(self, settings=None, embedding_provider=None):
        self._settings = settings
        self._embedding_provider = embedding_provider

        # 延迟创建 RAGASEvaluator，仅用其评分方法
        from ..services.evaluation.ragas_evaluator import RAGASEvaluator
        self._evaluator = RAGASEvaluator(
            retrieval_service=None,   # 不用于评分
            generation_service=None,  # 不用于评分
            settings=settings,
            embedding_provider=embedding_provider,
        )

    def score(self, question: str, answer: str, contexts: list[str],
              ground_truth: str = "",
              metrics: Optional[list[str]] = None) -> dict:
        """对单条数据跑 RAGAS 评分，返回四维分数 + 中间数据。

        Returns:
            {
                "scores": {"faithfulness": 0.855, ...},
                "details": {
                    "faithfulness": {"statements": [...], "verdicts": [...]},
                    "answer_relevancy": {"generated_questions": [...]},
                    "context_precision": {"verifications": [...]},
                    "context_recall": {"classifications": [...]},
                }
            }
        """
        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy",
                       "context_precision", "context_recall"]

        metric_pairs = self._evaluator._setup_metrics(metrics)
        row = self._evaluator._build_row(question, answer, contexts, ground_truth)

        dispatch = {
            "faithfulness": self._evaluator._run_faithfulness,
            "answer_relevancy": self._evaluator._run_answer_relevancy,
            "context_precision": self._evaluator._run_context_precision,
            "context_recall": self._evaluator._run_context_recall,
        }

        scores = {}
        details = {}

        for m_name, metric in metric_pairs:
            fn = dispatch.get(m_name)
            if fn:
                try:
                    details[m_name] = asyncio.run(fn(metric, row))
                except Exception as exc:
                    details[m_name] = {"score": 0.0, "error": str(exc)}
            else:
                details[m_name] = {"score": 0.0}
            scores[m_name] = details[m_name].get("score", 0.0)

        return {"scores": scores, "details": details}

    def score_batch(self, items: list[dict],
                    metrics: Optional[list[str]] = None) -> list[dict]:
        """批量评分。

        items: [{"question": ..., "answer": ..., "contexts": [...], "ground_truth": ""}, ...]
        返回: 每项附加 scores + details 字段
        """
        results = []
        for item in items:
            result = self.score(
                question=item["question"],
                answer=item["answer"],
                contexts=item.get("contexts", []),
                ground_truth=item.get("ground_truth", ""),
                metrics=metrics,
            )
            results.append({**item, **result})
        return results
