"""Agent 3: 诊断报告 — 读差分报告 → 脆弱点归因 → 生成修复建议"""

from collections import defaultdict


class DiagnosticianAgent:
    """诊断分析 Agent。读取评估差分结果，按攻击类型聚类，输出脆弱点诊断和修复建议。"""

    # 脆弱点决策规则
    _RULES = [
        {
            "condition": lambda rd: rd.get("faithfulness", {}).get("delta", 0) > 0.2
                       and rd.get("context_precision", {}).get("delta", 0) < 0.1,
            "vulnerability": "生成层脆弱",
            "cause": "检索质量稳定但 LLM 被对抗内容误导，可能是 System Prompt 约束不足",
            "fix": "强化 System Prompt 约束（增加\"仅根据上下文回答\"指令）；考虑对检索上下文做真实性预筛选",
        },
        {
            "condition": lambda rd: rd.get("context_recall", {}).get("delta", 0) > 0.15,
            "vulnerability": "检索召回脆弱",
            "cause": "检索对对抗性改写、关键词替换敏感，召回不足",
            "fix": "启用混合检索（Dense + Sparse）；注入领域词典提升关键词命中率",
        },
        {
            "condition": lambda rd: rd.get("context_precision", {}).get("delta", 0) > 0.15,
            "vulnerability": "排序脆弱",
            "cause": "噪声注入或排序污染使无关内容排名上升",
            "fix": "接入 Cross-encoder Re-ranker 对 Top-K 做精确重排",
        },
        {
            "condition": lambda rd: rd.get("answer_relevancy", {}).get("delta", 0) > 0.2
                       and all(rd.get(m, {}).get("delta", 0) < 0.1
                              for m in ("faithfulness", "context_precision", "context_recall")),
            "vulnerability": "意图理解脆弱",
            "cause": "Query 被改写后语义模糊，LLM 意图理解偏离",
            "fix": "增加 Query Rewrite 模块，在检索前规范化用户输入",
        },
        {
            "condition": lambda rd: all(rd.get(m, {}).get("delta", 0) > 0.2
                                       for m in ("faithfulness", "answer_relevancy",
                                                  "context_precision", "context_recall")),
            "vulnerability": "全链路脆弱",
            "cause": "RAG 系统整体鲁棒性不足，所有指标同时大幅衰减",
            "fix": "综合加固：混合检索 + Re-ranker + Prompt 约束 + 查询改写",
        },
    ]

    def diagnose(self, eval_results: list[dict]) -> dict:
        """输入 EvaluatorAgent 的输出，生成诊断报告。

        Returns:
            {
                "per_attack_type": {"关键词替换": {"vulnerabilities": [...], "avg_robustness": 0.6}},
                "vulnerability_ranking": [("检索召回脆弱", 0.35), ...],
                "overall_robustness": 0.62,
                "overall_rating": "[WARN] 需要加固",
                "suggestions": ["启用混合检索", ...],
                "summary_text": "该 RAG 系统在关键词替换攻击下...",
            }
        """
        # 按攻击类型分组（排除对照组）
        by_attack = defaultdict(list)
        metric_error_count = 0
        for r in eval_results:
            at = r["case"].get("attack_type", "")
            if at == "无攻击（对照组）":
                continue
            if self._has_metric_error(r):
                metric_error_count += 1
            by_attack[at].append(r)

        # 逐攻击类型分析
        per_attack = {}
        all_robustness_scores = []
        valid_pair_count = 0

        for at, results in by_attack.items():
            # 聚合该攻击类型下的平均鲁棒性
            valid_results = [r for r in results if not self._has_metric_error(r)]
            avg_scores = self._avg_delta_by_metric(valid_results)
            robustness_vals = [r["robustness"]["robustness_score"]
                              for r in valid_results if r.get("robustness")]
            avg_robustness = sum(robustness_vals) / len(robustness_vals) if robustness_vals else 0.0
            all_robustness_scores.extend(robustness_vals)
            valid_pair_count += len(robustness_vals)

            vulnerabilities = []
            for rule in self._RULES:
                if rule["condition"](avg_scores):
                    vulnerabilities.append({
                        "vulnerability": rule["vulnerability"],
                        "cause": rule["cause"],
                        "fix": rule["fix"],
                    })

            per_attack[at] = {
                "avg_robustness": round(avg_robustness, 4),
                "avg_deltas": {k: round(v["delta"], 4) for k, v in avg_scores.items()},
                "vulnerabilities": vulnerabilities,
                "count": len(results),
            }

        # 脆弱点按严重度排序
        vulnerability_ranking = sorted(
            [(at, info["avg_robustness"]) for at, info in per_attack.items()],
            key=lambda x: x[1],
        )

        # 综合评分
        overall = (sum(all_robustness_scores) / len(all_robustness_scores)
                   if all_robustness_scores else 0.0)
        overall = round(overall, 4)

        if valid_pair_count == 0 and metric_error_count > 0:
            overall_rating = "[FAIL] 评分异常"
        elif overall >= 0.8:
            overall_rating = "[PASS] 鲁棒性良好"
        elif overall >= 0.5:
            overall_rating = "[WARN] 需要加固"
        else:
            overall_rating = "[FAIL] 严重脆弱"
        # overall_rating 已在上方分支赋值

        # 收集所有修复建议（去重）
        suggestions = []
        seen = set()
        for info in per_attack.values():
            for vuln in info["vulnerabilities"]:
                if vuln["fix"] not in seen:
                    suggestions.append(vuln["fix"])
                    seen.add(vuln["fix"])
        if metric_error_count > 0:
            suggestions.insert(0, "评分接口调用异常：请检查 API Key、余额或服务可用性后重试")

        # LLM 诊断摘要（如果有 LLM 可调用）
        if "overall_rating" not in locals():
            overall_rating = "[PASS] 鲁棒性良好"
        summary_text = self._build_summary(per_attack, overall_rating)

        return {
            "per_attack_type": per_attack,
            "vulnerability_ranking": vulnerability_ranking,
            "overall_robustness": overall,
            "overall_rating": overall_rating,
            "suggestions": suggestions,
            "summary_text": summary_text,
            "metric_error_count": metric_error_count,
            "valid_pair_count": valid_pair_count,
        }

    def _has_metric_error(self, result: dict) -> bool:
        """判断该条评估是否因指标评分异常而失真。"""
        for side in ("normal", "attack"):
            details = result.get(side, {}).get("details", {}) or {}
            for metric_detail in details.values():
                if isinstance(metric_detail, dict) and metric_detail.get("error"):
                    return True
        return False

    def _avg_delta_by_metric(self, results: list) -> dict:
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        sums = {m: 0.0 for m in metrics}
        counts = {m: 0 for m in metrics}
        for r in results:
            rd = r.get("robustness", {})
            if not rd:
                continue
            per = rd.get("per_metric", {})
            for m in metrics:
                if m in per:
                    sums[m] += per[m]["delta"]
                    counts[m] += 1
        return {
            m: {
                "delta": sums[m] / counts[m] if counts[m] > 0 else 0.0,
                "count": counts[m],
            }
            for m in metrics
        }

    def _build_summary(self, per_attack: dict, rating: str) -> str:
        parts = ["综合评级: " + rating]
        for at, info in per_attack.items():
            count = info["count"]
            rb = info["avg_robustness"]
            if info["vulnerabilities"]:
                names = [v["vulnerability"] for v in info["vulnerabilities"]]
                vul_str = "发现漏洞: " + ", ".join(names)
            else:
                vul_str = "未发现明显脆弱点"
            parts.append(f"[{at}] ({count}条) 鲁棒性 {rb:.3f}  {vul_str}")
        return "  ".join(parts)
