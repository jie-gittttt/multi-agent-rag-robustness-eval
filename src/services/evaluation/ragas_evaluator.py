import json
from typing import Generator, List
import pandas as pd


def _build_ragas_llm(settings):
    """根据应用配置构建 RAGAS 评分用的 LLM"""
    from ragas.llms import LangchainLLMWrapper

    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        from langchain_core.language_models.chat_models import BaseChatModel

        # DeepSeek 兼容性包装器：强制 n=1，处理 API 差异
        class DeepSeekCompatChatOpenAI(ChatOpenAI):
            @property
            def _default_params(self) -> dict:
                """剔除 n 参数，防止 ragas 异步路径直接修改 self.n 后传入 API。

                ragas 的 agenerate_text 会直接设置 self.langchain_llm.n = n，
                绕过 _generate/_agenerate 的 kwargs 拦截，所以必须在 payload 构建层防御。
                """
                params = super()._default_params
                params.pop("n", None)
                return params

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                kwargs.pop("n", None)  # 同步路径：generate_prompt(n=n) 传入
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                kwargs.pop("n", None)
                return await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )

        kwargs = {
            "model": settings.OPENAI_MODEL,
            "n": 1,
            "request_timeout": 120,
        }
        if settings.OPENAI_BASE_URL:
            kwargs["base_url"] = settings.OPENAI_BASE_URL
        lc_llm = DeepSeekCompatChatOpenAI(
            api_key=settings.OPENAI_API_KEY, **kwargs
        )
    elif settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        lc_llm = ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
        )
    else:
        raise ValueError(f"RAGAS 不支持的服务商: {settings.LLM_PROVIDER}")
    return LangchainLLMWrapper(lc_llm)


class LangchainEmbeddingsAdapter:
    """将项目的 BaseEmbeddingProvider 包装为 LangChain Embeddings 接口，
    供 ragas 的 MetricWithEmbeddings（如 answer_relevancy）使用。

    不传则会触发 ragas 默认创建 OpenAIEmbeddings → 走 /embeddings 端点；
    DeepSeek 无此端点，404。"""

    def __init__(self, provider):
        self._provider = provider

    def embed_documents(self, texts):
        return self._provider.embed(texts)

    def embed_query(self, text):
        return self._provider.embed_query(text)


class RAGASEvaluator:
    """
    RAGAS 四维评估：
    - Faithfulness（忠实度）：回答是否基于检索到的上下文
    - Answer Relevancy（回答相关性）：回答是否切题
    - Context Precision（上下文精度）：相关块是否排在前面
    - Context Recall（上下文召回率）：是否检索到所有相关块
    """

    def __init__(self, retrieval_service, generation_service, settings=None,
                 embedding_provider=None):
        self._retrieval = retrieval_service
        self._generation = generation_service
        self._settings = settings
        self._embedding_provider = embedding_provider

    def _get_ragas_embeddings(self):
        """将项目内 embedding provider 转为 ragas 可用的 embeddings 对象"""
        if self._embedding_provider is None:
            return None
        lc_embeddings = LangchainEmbeddingsAdapter(self._embedding_provider)
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        return LangchainEmbeddingsWrapper(lc_embeddings)

    def test_metric(self, metric_name: str) -> dict:
        """用最小数据测试单个指标能否正常评分，排查连通性问题。"""
        from ragas import evaluate
        from datasets import Dataset

        test_data = {
            "question": "测试问题：水泥熟料煅烧的碳排放来源是什么？",
            "answer": "水泥熟料煅烧过程中碳排放主要来源于碳酸钙分解和燃料燃烧。",
            "contexts": [
                "碳酸钙在高温下分解为氧化钙和二氧化碳，是水泥生产的主要碳排放源。",
                "水泥窑炉使用的煤粉燃烧也会产生大量二氧化碳排放。",
            ],
            "ground_truth": "碳酸钙分解和燃料燃烧",
        }

        dataset = Dataset.from_list([test_data])
        ragas_llm = _build_ragas_llm(self._settings) if self._settings else None
        ragas_embeddings = self._get_ragas_embeddings()
        metric = self._get_metric(metric_name)

        import time
        start = time.time()
        try:
            scores = evaluate(dataset, metrics=[metric], llm=ragas_llm,
                            embeddings=ragas_embeddings)
            df = scores.to_pandas()
            elapsed = round(time.time() - start, 1)
            val = float(df[metric_name].iloc[0])
            return {
                "metric": metric_name,
                "status": "ok",
                "score": round(val, 4) if not (val != val) else 0.0,
                "elapsed_seconds": elapsed,
            }
        except Exception as exc:
            elapsed = round(time.time() - start, 1)
            return {
                "metric": metric_name,
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": elapsed,
            }

    def evaluate(self, test_queries: List[dict]) -> pd.DataFrame:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        )
        from datasets import Dataset

        eval_data = []
        for tq in test_queries:
            question = tq["question"]
            chunks = self._retrieval.retrieve(question)
            result = self._generation.generate_sync(question, chunks)
            eval_data.append({
                "question": question,
                "answer": result.answer,
                "contexts": [c.content for c in chunks],
                "ground_truth": tq.get(
                    "expected_answer",
                    " ".join(tq.get("expected_concepts", [])),
                ),
            })

        dataset = Dataset.from_list(eval_data)
        ragas_llm = _build_ragas_llm(self._settings) if self._settings else None
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy,
                     context_precision, context_recall],
            llm=ragas_llm,
        )
        return scores.to_pandas()

    def benchmark(self, test_queries: List[dict]) -> dict:
        df = self.evaluate(test_queries)
        return {
            "avg_faithfulness": float(df["faithfulness"].mean()),
            "avg_answer_relevancy": float(df["answer_relevancy"].mean()),
            "avg_context_precision": float(df["context_precision"].mean()),
            "avg_context_recall": float(df["context_recall"].mean()),
            "per_query": df.to_dict(orient="records"),
        }

    METRIC_MAP = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }

    @staticmethod
    def _get_metric(name: str):
        """延迟导入 RAGAS 指标，避免启动时加载"""
        if RAGASEvaluator.METRIC_MAP[name] is not None:
            return RAGASEvaluator.METRIC_MAP[name]
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        )
        mapping = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        RAGASEvaluator.METRIC_MAP.update(mapping)
        return mapping[name]

    def _setup_metrics(self, metrics: List[str]) -> list:
        """初始化指标：设置 LLM、embeddings 并调用 init()。返回 [(name, metric), ...]"""
        from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
        from ragas.run_config import RunConfig

        ragas_llm = _build_ragas_llm(self._settings) if self._settings else None
        ragas_embeddings = self._get_ragas_embeddings()
        run_config = RunConfig()

        # 禁止无声回退到 MockLLM，避免产出“可运行但不可信”的评分结果。
        if ragas_llm is None:
            raise RuntimeError(
                "RAGAS 评分 LLM 未注入。请检查设置与环境变量（如 LLM_PROVIDER、OPENAI_API_KEY/ANTHROPIC_API_KEY）。"
            )

        pairs = []
        for m_name in metrics:
            metric = self._get_metric(m_name)
            if isinstance(metric, MetricWithLLM):
                # 指标可能预置 MockLLM；这里强制覆盖，避免静默回退导致结果失真。
                metric.llm = ragas_llm
            if isinstance(metric, MetricWithEmbeddings) and ragas_embeddings is not None:
                metric.embeddings = ragas_embeddings
            # DeepSeek 不支持 n>1，将 strictness 限制为 1 避免无效多次请求
            if m_name == "answer_relevancy" and hasattr(metric, "strictness"):
                metric.strictness = 1
            metric.init(run_config)
            pairs.append((m_name, metric))
        return pairs

    def _build_row(self, question: str, answer: str, contexts: list,
                   ground_truth: str) -> dict:
        """构建 ragas 指标所需的统一数据行"""
        return {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": ground_truth,
        }

    # ═══════════════════════════════════════════
    # 逐指标详细评估（替代 ragas 黑盒 evaluate）
    # ═══════════════════════════════════════════

    async def _run_faithfulness(self, metric, row) -> dict:
        """运行忠实度评估，返回 {score, statements, verdicts}"""
        calls = []
        st_output = await metric._create_statements(row, calls)
        statements = st_output.statements
        if not statements:
            return {"score": 0.0, "statements": [], "verdicts": []}
        v_output = await metric._create_verdicts(row, statements, calls)
        score = metric._compute_score(v_output)
        return {
            "score": round(float(score), 4) if score == score else 0.0,
            "statements": statements,
            "verdicts": [
                {"statement": v.statement, "reason": v.reason, "verdict": v.verdict}
                for v in v_output.statements
            ],
        }

    async def _run_answer_relevancy(self, metric, row) -> dict:
        """运行回答相关性评估，返回 {score, generated_questions}"""
        from ragas.metrics._answer_relevance import ResponseRelevanceInput

        prompt_input = ResponseRelevanceInput(response=row["response"])
        responses = await metric.question_generation.generate_multiple(
            data=prompt_input, llm=metric.llm, callbacks=[], n=metric.strictness
        )
        score = metric._calculate_score(responses, row)
        gen_questions = [r.question for r in responses]
        question = row["user_input"]
        cosine_sim = metric.calculate_similarity(question, gen_questions).tolist()
        return {
            "score": round(float(score), 4) if score == score else 0.0,
            "generated_questions": [
                {
                    "question": r.question,
                    "noncommittal": r.noncommittal,
                    "similarity": round(float(cosine_sim[i]), 4),
                }
                for i, r in enumerate(responses)
            ],
        }

    async def _run_context_precision(self, metric, row) -> dict:
        """运行上下文精度评估，返回 {score, verifications}"""
        from ragas.metrics._context_precision import QAC, Verification
        from ragas.metrics.base import ensembler

        user_input, contexts, reference = metric._get_row_attributes(row)
        per_ctx = []
        for ctx in contexts:
            data = QAC(question=user_input, context=ctx, answer=reference)
            results = await metric.context_precision_prompt.generate_multiple(
                data=data, llm=metric.llm, callbacks=[], n=1,
            )
            per_ctx.append([r.model_dump() for r in results])

        answers = []
        for response in per_ctx:
            agg = ensembler.from_discrete([response], "verdict")
            answers.append(Verification(**agg[0]))
        score = metric._calculate_average_precision(answers)
        return {
            "score": round(float(score), 4) if score == score else 0.0,
            "verifications": [
                {"context_preview": ctx[:200], "verdict": a.verdict, "reason": a.reason}
                for ctx, a in zip(contexts, answers)
            ],
        }

    async def _run_context_recall(self, metric, row) -> dict:
        """运行上下文召回率评估，返回 {score, classifications}"""
        from ragas.metrics._context_recall import QCA, ContextRecallClassification
        from ragas.metrics.base import ensembler

        data = QCA(
            question=row["user_input"],
            context="\n".join(row["retrieved_contexts"]),
            answer=row["reference"],
        )
        classifications_list = await metric.context_recall_prompt.generate_multiple(
            data=data, llm=metric.llm, callbacks=[], n=1,
        )
        classification_dicts = []
        for c in classifications_list:
            classification_dicts.append(
                [cl.model_dump() for cl in c.classifications]
            )
        ensembled = ensembler.from_discrete(classification_dicts, "attributed")
        result = [ContextRecallClassification(**c) for c in ensembled]
        score = metric._compute_score(result)
        return {
            "score": round(float(score), 4) if score == score else 0.0,
            "classifications": [
                {"statement": c.statement, "attributed": c.attributed, "reason": c.reason}
                for c in result
            ],
        }

    # ═══════════════════════════════════════════
    # 主评估流
    # ═══════════════════════════════════════════

    def evaluate_stream(self, test_queries: List[dict],
                        metrics: List[str] = None,
                        dense_enabled: bool = True,
                        sparse_enabled: bool = True) -> Generator:
        """逐条评估并以 SSE 事件流输出，包含详细中间过程。

        对每条问题：检索 → 生成 → 逐指标评分 → 立即推送 query_done 事件。
        前端可逐条看到评估进展及每个指标的中间输出（原子声明、追问等）。

        dense_enabled / sparse_enabled 控制检索消融实验：
        - (True, True)  = 混合检索（全量 RAG）
        - (True, False) = 仅稠密检索
        - (False, True) = 仅稀疏检索
        """
        import asyncio
        import math

        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy",
                       "context_precision", "context_recall"]

        metric_pairs = self._setup_metrics(metrics)
        _dispatch = {
            "faithfulness": self._run_faithfulness,
            "answer_relevancy": self._run_answer_relevancy,
            "context_precision": self._run_context_precision,
            "context_recall": self._run_context_recall,
        }

        total = len(test_queries)
        yield _sse({"type": "eval_start", "total": total,
                    "dense_enabled": dense_enabled,
                    "sparse_enabled": sparse_enabled})

        per_query_results = []

        for i, tq in enumerate(test_queries):
            question = tq["question"]
            yield _sse({
                "type": "query_progress", "current": i + 1, "total": total,
                "question": question, "status": "retrieving",
            })

            try:
                chunks = self._retrieval.retrieve(
                    question,
                    dense_enabled=dense_enabled,
                    sparse_enabled=sparse_enabled)
                result = self._generation.generate_sync(question, chunks)
                answer = result.answer
                contexts = [c.content for c in chunks]
            except Exception as exc:
                yield _sse({
                    "type": "query_error", "current": i + 1, "total": total,
                    "question": question, "error": str(exc),
                })
                continue

            gt = tq.get("expected_answer",
                        " ".join(tq.get("expected_concepts", [])))
            sources = []
            for j, c in enumerate(chunks):
                sources.append({
                    "index": j + 1,
                    "filename": c.metadata.get("filename", ""),
                    "content_preview": c.content[:200],
                })

            yield _sse({
                "type": "query_progress", "current": i + 1, "total": total,
                "question": question, "status": "scoring",
            })

            row = self._build_row(question, answer, contexts, gt)
            detail = {}
            scores = {}

            # 对该问题的每个指标逐一评分
            for m_name, metric in metric_pairs:
                try:
                    fn = _dispatch.get(m_name)
                    if fn:
                        detail[m_name] = asyncio.run(fn(metric, row))
                    else:
                        detail[m_name] = {"score": 0.0}
                except Exception as exc:
                    detail[m_name] = {"score": 0.0, "error": str(exc)}
                scores[m_name] = detail[m_name].get("score", 0.0)

            yield _sse({
                "type": "query_done",
                "current": i + 1,
                "total": total,
                "question": question,
                "answer": answer,
                "sources": sources,
                "scores": scores,
                "details": detail,
            })

            per_query_results.append({
                "question": question,
                "answer": answer,
                "sources": sources,
                **scores,
            })

        # 汇总
        summary = {}
        for m in metrics:
            vals = [r[m] for r in per_query_results if r.get(m, 0) == r.get(m, 0)]
            summary[m] = round(float(sum(vals)) / len(vals), 4) if vals else 0.0

        yield _sse({
            "type": "eval_done",
            "summary": summary,
            "per_query": per_query_results,
            "metrics": metrics,
        })


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
