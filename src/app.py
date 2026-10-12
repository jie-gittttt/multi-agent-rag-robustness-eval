"""Gradio Web UI - RAG robust evaluation tool."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Generator

# ── 必须在任何模型加载前设置，阻止 HuggingFace 联网 ──
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import gradio as gr

from .adapters.cement_lca import CementLCARAGAdapter
from .graph.eval_graph import get_graph_mermaid, run_eval_stream

_METRIC_CHOICES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def load_seed_queries(seed_json_text: str | None = None) -> list[dict]:
    default = [
        {"query": "什么是生命周期评价（LCA）？"},
        {"query": "水泥生产碳排放主要来自哪些环节？"},
        {"query": "RAG 的核心流程是什么？"},
    ]
    if seed_json_text:
        data = json.loads(seed_json_text)
        if not isinstance(data, list):
            raise ValueError("种子问题 JSON 必须是数组")
        return data

    path = Path("data/seed_queries.json")
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    return default


_DEFAULT_SEED = json.dumps(load_seed_queries(), ensure_ascii=False, indent=2)


def _build_cement_adapter():
    """构建水泥 LCA 适配器，失败时回退 Demo。"""
    try:
        return CementLCARAGAdapter()
    except Exception:
        from .adapters.demo_adapter import DemoRAGAdapter

        return DemoRAGAdapter()


def _select_adapter(target: str):
    if "RAGFlow" in target:
        from .adapters.ragflow_adapter import RAGFlowAdapter
        from .config import Settings
        cfg = Settings()
        return RAGFlowAdapter(
            base_url=getattr(cfg, 'RAGFLOW_BASE_URL', 'http://localhost:9380'),
            api_key=getattr(cfg, 'RAGFLOW_API_KEY', ''),
            kb_id=getattr(cfg, 'RAGFLOW_KB_ID', ''),
        )
    if "LlamaIndex" in target:
        from .adapters.llamaindex_stub import LlamaIndexRAGAdapter
        return LlamaIndexRAGAdapter()
    if "LangChain" in target:
        from .adapters.langchain_stub import LangChainRAGAdapter
        return LangChainRAGAdapter()
    if "水泥" in target or "cement" in target.lower() or "Cement" in target:
        return _build_cement_adapter()

    # 默认返回 LlamaIndex
    from .adapters.llamaindex_stub import LlamaIndexRAGAdapter
    return LlamaIndexRAGAdapter()


def _display_system_name(raw: str) -> str:
    s = str(raw or "").strip()
    s_low = s.lower()
    if "ragflow" in s_low:
        return "RAGFlow"
    if "llamaindex" in s_low:
        return "LlamaIndex"
    if "langchain" in s_low:
        return "LangChain"
    if "水泥" in s or "cement" in s_low:
        return "CementLCA"
    return s or "Unknown"


def _list_reports() -> list[str]:
    """列出所有已保存的评估报告（按文件名倒序）。"""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    files = sorted(data_dir.glob("results_*.json"), reverse=True)
    return [f.name for f in files]


def _load_report_by_name(filename: str) -> dict | None:
    """根据文件名加载评估报告。"""
    fpath = Path("data") / filename
    if not fpath.exists():
        return None
    return json.loads(fpath.read_text(encoding="utf-8"))


def _read_saved_system_results() -> list[dict]:
    data_dir = Path("data")
    if not data_dir.exists():
        return []

    payloads = []
    for p in sorted(data_dir.glob("results_*.json")):
        # Skip .bak backup files
        if p.name.endswith(".bak"):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj["_file"] = p.name
                payloads.append(obj)
        except Exception:
            continue
    return payloads


def _summarize_system_payload(payload: dict) -> dict:
    eval_results = payload.get("eval_results", []) or []
    pair_count = 0
    err_count = 0
    for r in eval_results:
        if r.get("case", {}).get("attack_type") == "无攻击（对照组）":
            continue
        pair_count += 1
        if r.get("error") or r.get("normal", {}).get("error") or r.get("attack", {}).get("error"):
            err_count += 1

    overall = float(payload.get("overall_robustness", 0) or 0)
    if overall >= 0.8:
        rating = "稳健"
    elif overall >= 0.5:
        rating = "中等"
    else:
        rating = "脆弱"

    has_valid = pair_count > 0 and err_count < pair_count
    return {
        "system": _display_system_name(payload.get("system", "")),
        "overall": overall,
        "rating": payload.get("overall_rating") or rating,
        "pair_count": pair_count,
        "error_rows": err_count,
        "per_attack_type": payload.get("per_attack_type", {}) or {},
        "has_valid_data": has_valid,
    }


def _build_experiment_dashboard_html(current_result: dict | None = None, current_target: str | None = None, single: bool = False) -> str:
    """实验结果可视化看板。single=True: 仅展示一份报告；single=False: 多系统对比。"""

    def _to_float(v, d=0.0):
        try:
            return float(v)
        except Exception:
            return d

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        xs = sorted(values)
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        frac = pos - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    if single and current_result:
        payloads = [current_result]
    else:
        payloads = _read_saved_system_results()
    if current_result and current_result.get("diagnosis"):
        diagnosis = current_result.get("diagnosis", {})
        payloads.append(
            {
                "system": current_target or "当前运行",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "overall_robustness": diagnosis.get("overall_robustness", 0),
                "overall_rating": diagnosis.get("overall_rating", "N/A"),
                "per_attack_type": diagnosis.get("per_attack_type", {}),
                "eval_results": current_result.get("eval_results", []),
                "attack_cases": current_result.get("attack_cases", []),
            }
        )

    if not payloads:
        return "<div style='padding:16px;border:1px solid #dbe4f0;border-radius:12px;background:#f8fbff;'>暂无评估结果，请先运行评估。</div>"

    latest_by_system: dict[str, dict] = {}
    for p in payloads:
        name = _display_system_name(p.get("system", ""))
        ts = str(p.get("timestamp", "") or "")
        if name not in latest_by_system:
            latest_by_system[name] = p
            continue
        prev = latest_by_system[name]
        prev_ts = str(prev.get("timestamp", "") or "")
        use_new = ts > prev_ts
        if ts == prev_ts:
            use_new = ".bak" in str(prev.get("_file", "")) and ".bak" not in str(p.get("_file", ""))
        if use_new:
            latest_by_system[name] = p

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summaries = []
    history_points = []

    for p in payloads:
        history_points.append(
            {
                "system": _display_system_name(p.get("system", "")),
                "timestamp": str(p.get("timestamp", "") or ""),
                "overall": round(_to_float(p.get("overall_robustness", 0)), 3),
            }
        )

    for payload in latest_by_system.values():
        summary = _summarize_system_payload(payload)
        rows = payload.get("eval_results", []) or []
        attack_rows = [r for r in rows if r.get("case", {}).get("attack_type") != "无攻击（对照组）"]

        metric_values = {m: [] for m in metrics}
        rb_values = []
        failure = {"检索不足": 0, "生成偏题": 0, "评分异常": 0, "调用失败": 0}

        for row in attack_rows:
            normal = row.get("normal", {}) or {}
            attack = row.get("attack", {}) or {}
            robust = row.get("robustness", {}) or {}

            rb = _to_float(robust.get("robustness_score"), None)
            if rb is not None:
                rb_values.append(rb)

            n_scores = normal.get("scores", {}) or {}
            a_scores = attack.get("scores", {}) or {}
            for m in metrics:
                v = _to_float(a_scores.get(m), None)
                if v is None:
                    v = _to_float(n_scores.get(m), None)
                if v is not None:
                    metric_values[m].append(v)

            pool = (row.get("case", {}).get("original_contexts", []) or []) + (
                row.get("case", {}).get("perturbed_contexts", []) or []
            )
            retrieval_insuff = (not pool) or any("检索异常" in str(x) for x in pool)

            text_pool = [
                str(normal.get("answer", "")),
                str(attack.get("answer", "")),
                str(normal.get("error", "")),
                str(attack.get("error", "")),
            ]
            call_failure = any(k in t for t in text_pool for k in ["生成异常", "Unknown model", "rate limit", "timeout", "API"])
            rel = _to_float(a_scores.get("answer_relevancy"), _to_float(n_scores.get("answer_relevancy"), 1.0))
            fai = _to_float(a_scores.get("faithfulness"), _to_float(n_scores.get("faithfulness"), 1.0))
            generation_offtopic = rel < 0.5 or fai < 0.5
            scoring_exception = (not n_scores and not a_scores) or (row.get("robustness") is None)

            if scoring_exception:
                failure["评分异常"] += 1
            elif call_failure:
                failure["调用失败"] += 1
            elif retrieval_insuff:
                failure["检索不足"] += 1
            elif generation_offtopic:
                failure["生成偏题"] += 1

        summary["metric_means"] = {m: round(_mean(metric_values[m]), 3) for m in metrics}
        summary["box"] = {
            "min": round(min(rb_values), 3) if rb_values else 0.0,
            "q1": round(_quantile(rb_values, 0.25), 3),
            "median": round(_quantile(rb_values, 0.5), 3),
            "q3": round(_quantile(rb_values, 0.75), 3),
            "max": round(max(rb_values), 3) if rb_values else 0.0,
        }
        summary["failure"] = failure
        summaries.append(summary)

    summaries.sort(key=lambda x: x["overall"], reverse=True)
    valid_summaries = [s for s in summaries if s.get("has_valid_data", True)]
    best = valid_summaries[0] if valid_summaries else summaries[0]
    avg_overall = _mean([s["overall"] for s in valid_summaries]) if valid_summaries else 0.0
    total_pairs = sum(s["pair_count"] for s in summaries)
    total_errors = sum(s["error_rows"] for s in summaries)
    target_threshold = 0.85

    # P0-1: Chart data uses valid systems only; KPI cards show all (with warnings for invalid)
    systems = [s["system"] for s in valid_summaries]
    overall_scores = [round(s["overall"], 3) for s in valid_summaries]
    attack_types = sorted({at for s in valid_summaries for at in s["per_attack_type"].keys() if at != "无攻击（对照组）"})

    # ── NEW: Additional data aggregations for upgraded charts ──
    # Std dev per system
    system_stddevs = []
    for s in valid_summaries:
        payload = latest_by_system.get(s["system"], {})
        rows = payload.get("eval_results", []) or []
        rb_vals = []
        for r in rows:
            if r.get("case", {}).get("attack_type") == "无攻击（对照组）":
                continue
            rb = _to_float((r.get("robustness", {}) or {}).get("robustness_score"), None)
            if rb is not None:
                rb_vals.append(rb)
        if rb_vals:
            mean_rb = _mean(rb_vals)
            variance = sum((v - mean_rb) ** 2 for v in rb_vals) / len(rb_vals)
            system_stddevs.append(round(variance ** 0.5, 4))
        else:
            system_stddevs.append(0.0)

    # Per-attack-type impact ranking (avg drop across all systems)
    attack_impacts = []
    for at in attack_types:
        drops = []
        for s in valid_summaries:
            info = s["per_attack_type"].get(at, {})
            robust_score = _to_float(info.get("avg_robustness", 0))
            deltas = info.get("avg_deltas", {}) or {}
            delta_vals = [_to_float(v) for v in deltas.values()]
            drop = _mean([max(v, 0.0) for v in delta_vals]) if delta_vals else max(0.0, 1 - robust_score)
            drops.append(drop)
        avg_drop = _mean(drops)
        max_drop_sys = systems[drops.index(max(drops))] if drops else "?"
        attack_impacts.append({
            "attack_type": at,
            "avg_drop": round(avg_drop, 3),
            "max_drop": round(max(drops), 3),
            "worst_system": max_drop_sys,
            "risk": "高风险" if avg_drop >= 0.2 else ("中风险" if avg_drop >= 0.1 else "稳健"),
        })
    attack_impacts.sort(key=lambda x: x["avg_drop"], reverse=True)

    # Per-metric degradation (avg delta across all attack types per system per metric)
    metric_degradation = {}  # {system: {metric: avg_delta}}
    for s in summaries:
        sys_name = s["system"]
        metric_degradation[sys_name] = {}
        payload = latest_by_system.get(sys_name, {})
        rows = payload.get("eval_results", []) or []
        for m in metrics:
            deltas_m = []
            for r in rows:
                case_at = r.get("case", {}).get("attack_type", "")
                if case_at == "无攻击（对照组）":
                    continue
                robust = r.get("robustness", {}) or {}
                per_metric = robust.get("per_metric", {}) or {}
                pm = per_metric.get(m, {}) or {}
                delta = _to_float(pm.get("delta"), None)
                if delta is not None:
                    deltas_m.append(max(delta, 0.0))  # positive delta = degradation
            metric_degradation[sys_name][m] = round(_mean(deltas_m), 3) if deltas_m else 0.0

    # Vulnerability details (extracted from per_attack_type)
    vulnerability_cards = []
    for s in summaries:
        sys_name = s["system"]
        for at in attack_types:
            info = s["per_attack_type"].get(at, {})
            vulns = info.get("vulnerabilities", []) or []
            for v in vulns:
                vulnerability_cards.append({
                    "system": sys_name,
                    "attack_type": at,
                    "vulnerability": v.get("vulnerability", "未知漏洞"),
                    "cause": v.get("cause", ""),
                    "fix": v.get("fix", ""),
                    "avg_robustness": round(_to_float(info.get("avg_robustness", 0)), 3),
                })

    # Distribution bins for enhanced box plot (per system)
    distribution_bins = {}
    for s in summaries:
        sys_name = s["system"]
        payload = latest_by_system.get(sys_name, {})
        rows = payload.get("eval_results", []) or []
        bins = {"<0.5": 0, "0.5-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-0.95": 0, "0.95-1.0": 0, "1.0+": 0}
        for r in rows:
            if r.get("case", {}).get("attack_type") == "无攻击（对照组）":
                continue
            rb = _to_float((r.get("robustness", {}) or {}).get("robustness_score"), None)
            if rb is None:
                continue
            if rb < 0.5:
                bins["<0.5"] += 1
            elif rb < 0.7:
                bins["0.5-0.7"] += 1
            elif rb < 0.8:
                bins["0.7-0.8"] += 1
            elif rb < 0.9:
                bins["0.8-0.9"] += 1
            elif rb < 0.95:
                bins["0.9-0.95"] += 1
            elif rb < 1.0:
                bins["0.95-1.0"] += 1
            else:
                bins["1.0+"] += 1
        distribution_bins[sys_name] = bins

    # P0-3: Per-attack-type failure mechanism breakdown (for diagnostic drill-down)
    attack_mechanisms = {}  # {attack_type: {mechanism: count}}
    for at in attack_types:
        attack_mechanisms[at] = {"context_precision降": 0, "answer_relevancy降": 0, "faithfulness降": 0, "context_recall降": 0}
    for s in valid_summaries:
        sys_name = s["system"]
        payload = latest_by_system.get(sys_name, {})
        for r in (payload.get("eval_results", []) or []):
            case_at = r.get("case", {}).get("attack_type", "")
            if case_at == "无攻击（对照组）" or case_at not in attack_mechanisms:
                continue
            per_metric = (r.get("robustness", {}) or {}).get("per_metric", {}) or {}
            for m in metrics:
                delta = _to_float(per_metric.get(m, {}).get("delta", 0), 0)
                if delta > 0.05:  # Significant degradation
                    attack_mechanisms[case_at][f"{m}降"] += 1

    # P1-1: Extract per-case drill-down data (sample up to 5 cases per system-attack combo)
    drilldown_cases = []  # [{system, attack_type, original_query, perturbed_query, normal_scores, attack_scores, robustness_score, answer_excerpt}]
    for s in valid_summaries:
        sys_name = s["system"]
        payload = latest_by_system.get(sys_name, {})
        seen_combo = {}
        for r in (payload.get("eval_results", []) or []):
            case = r.get("case", {}) or {}
            case_at = case.get("attack_type", "")
            if case_at == "无攻击（对照组）":
                continue
            combo_key = f"{sys_name}|{case_at}"
            seen_combo[combo_key] = seen_combo.get(combo_key, 0) + 1
            if seen_combo[combo_key] > 5:  # Max 5 cases per combo
                continue
            normal = r.get("normal", {}) or {}
            attack = r.get("attack", {}) or {}
            robust = r.get("robustness", {}) or {}
            drilldown_cases.append({
                "system": sys_name,
                "attack_type": case_at,
                "original_query": str(case.get("original_query", ""))[:200],
                "perturbed_query": str(case.get("perturbed_query", ""))[:200],
                "normal_scores": {m: round(_to_float((normal.get("scores", {}) or {}).get(m)), 3) for m in metrics},
                "attack_scores": {m: round(_to_float((attack.get("scores", {}) or {}).get(m)), 3) for m in metrics},
                "robustness_score": round(_to_float(robust.get("robustness_score", 0)), 3),
                "answer_excerpt": str(attack.get("answer", normal.get("answer", "")))[:300],
                "per_metric": {m: {
                    "normal": round(_to_float((robust.get("per_metric", {}) or {}).get(m, {}).get("normal", 0)), 3),
                    "attack": round(_to_float((robust.get("per_metric", {}) or {}).get(m, {}).get("attack", 0)), 3),
                    "delta": round(_to_float((robust.get("per_metric", {}) or {}).get(m, {}).get("delta", 0)), 3),
                } for m in metrics},
            })

    # P0-2: Sample count per system (for trend credibility)
    sample_counts = {}
    for s in valid_summaries:
        sys_name = s["system"]
        payload = latest_by_system.get(sys_name, {})
        rows = (payload.get("eval_results", []) or [])
        attack_rows = [r for r in rows if r.get("case", {}).get("attack_type") != "无攻击（对照组）"]
        scorable = [r for r in attack_rows if not r.get("error")]
        sample_counts[sys_name] = {"total_cases": len(rows), "attack_pairs": len(attack_rows), "scored": len(scorable)}

    detail_rows = []
    heat_rows = []
    for at in attack_types:
        item = {"attack_type": at}
        for s in valid_summaries:
            info = s["per_attack_type"].get(at, {})
            robust_score = _to_float(info.get("avg_robustness", 0))
            deltas = info.get("avg_deltas", {}) or {}
            delta_vals = [_to_float(v) for v in deltas.values()]
            drop = _mean([max(v, 0.0) for v in delta_vals]) if delta_vals else max(0.0, 1 - robust_score)
            item[s["system"]] = round(drop, 3)
            detail_rows.append(
                {
                    "system": s["system"],
                    "attack_type": at,
                    "score": round(drop, 3),
                    "robustness": round(robust_score, 3),
                    "rating": "高风险" if drop >= 0.2 else "中风险" if drop >= 0.1 else "稳健",
                }
            )
        heat_rows.append(item)

    cards_html = ""
    for s in summaries:
        score = s["overall"]
        valid = s.get("has_valid_data", True)
        if not valid:
            cards_html += (
                f"<div class='sys-card' style='border-color:#fecaca;background:#fff5f5'><div class='sys-name'>{s['system']}</div>"
                f"<div class='sys-score' style='color:#dc2626'>N/A</div>"
                f"<div class='sys-meta'>⚠️ 数据不可用（全部评分出错）</div>"
                f"<div class='sys-meta'>错误行：{s['error_rows']} / 总数：{s['pair_count'] + s['error_rows']}</div>"
                f"<div class='bullet-wrap' style='background:#fee2e2'>"
                f"<div class='bullet-target' style='left:{target_threshold * 100:.1f}%'></div>"
                f"<div class='bullet-fill' style='width:0%;background:#fca5a5'></div></div>"
                f"<div class='sys-meta' style='color:#dc2626'>目标阈值 {target_threshold:.2f} — 请检查 LLM 配置后重新评估</div></div>"
            )
        else:
            cards_html += (
                f"<div class='sys-card'><div class='sys-name'>{s['system']}</div><div class='sys-score'>{score:.3f}</div>"
                f"<div class='sys-meta'>评级：{s['rating']}</div><div class='sys-meta'>评分对数：{s['pair_count']} | 错误行：{s['error_rows']}</div>"
                f"<div class='bullet-wrap'><div class='bullet-target' style='left:{target_threshold * 100:.1f}%'></div>"
                f"<div class='bullet-fill' style='width:{min(max(score, 0), 1) * 100:.1f}%'></div></div><div class='sys-meta'>目标阈值 {target_threshold:.2f}</div></div>"
            )

    matrix_html = "<table class='heat-table'><thead><tr><th>攻击类型</th>"
    for sys in systems:
        matrix_html += f"<th>{sys}</th>"
    matrix_html += "</tr></thead><tbody>"
    for row in heat_rows:
        matrix_html += f"<tr><td>{row['attack_type']}</td>"
        for sys in systems:
            v = row[sys]
            if v >= 0.2:
                bg, fg = "#fef0f0", "#b91c1c"
            elif v >= 0.1:
                bg, fg = "#fef9e7", "#92400e"
            else:
                bg, fg = "#d1fae5", "#065f46"
            matrix_html += (
                f"<td class='heat-cell' data-system='{sys}' data-attack='{row['attack_type']}' "
                f"style='background:{bg};color:{fg};font-weight:700;cursor:pointer'>{v:.3f}</td>"
            )
        matrix_html += "</tr>"
    matrix_html += "</tbody></table>"

    # ── Python helpers: pre-render all charts server-side ──
    import math as _math

    # Chart color palette
    COLORS = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]
    METRIC_LABELS = {"faithfulness": "忠实度", "answer_relevancy": "答案相关性", "context_precision": "上下文精确度", "context_recall": "上下文召回率"}

    def _color_for(v, thresholds=(0.8, 0.5), colors=("#10b981", "#f59e0b", "#ef4444")):
        return colors[0] if v >= thresholds[0] else (colors[1] if v >= thresholds[1] else colors[2])

    # ── Panel 2: Overall robustness bar chart with std-dev whiskers ──
    def _svg_overall_bar() -> str:
        w, h = 640, 260
        m = {"l": 56, "r": 20, "t": 12, "b": 42}
        pw, ph = w - m["l"] - m["r"], h - m["t"] - m["b"]
        step = pw / max(len(systems), 1)
        parts = []
        for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            y = m["t"] + (1 - t) * ph
            parts.append(f'<line x1="{m["l"]}" y1="{y:.1f}" x2="{w - m["r"]}" y2="{y:.1f}" stroke="#e2e8f0" stroke-dasharray="4,4"/>')
            parts.append(f'<text x="{m["l"] - 4}" y="{y + 4:.1f}" text-anchor="end" font-size="9" fill="#94a3b8">{t:.1f}</text>')
        # Target threshold line
        ty = m["t"] + (1 - target_threshold) * ph
        parts.append(f'<line x1="{m["l"]}" y1="{ty:.1f}" x2="{w - m["r"]}" y2="{ty:.1f}" stroke="#0f172a" stroke-dasharray="8,4" stroke-width="1.5" opacity="0.6"/>')
        parts.append(f'<text x="{w - m["r"] + 2}" y="{ty + 4:.1f}" font-size="9" fill="#0f172a" font-weight="600">阈值{target_threshold}</text>')
        for i, (s, v) in enumerate(zip(systems, overall_scores)):
            bw = max(36, step * 0.56)
            x = m["l"] + i * step + (step - bw) / 2
            vc = max(0, min(1, v))
            y = m["t"] + (1 - vc) * ph
            hb = max(2, m["t"] + ph - y)
            color = _color_for(v)
            # Bar with CSS animation
            # Bar with actual height (CSS animation provides entrance effect)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{hb:.1f}" rx="6" fill="{color}" class="bar-grow"/>')
            # Std-dev whisker
            sd = system_stddevs[i] if i < len(system_stddevs) else 0
            if sd > 0:
                y_upper = m["t"] + (1 - max(0, min(1, v + sd))) * ph
                y_lower = m["t"] + (1 - max(0, min(1, v - sd))) * ph
                cx = x + bw / 2
                parts.append(f'<line x1="{cx:.1f}" y1="{y_upper:.1f}" x2="{cx:.1f}" y2="{y_lower:.1f}" stroke="#334155" stroke-width="1.5" opacity="0.5"/>')
                parts.append(f'<line x1="{cx - 6:.1f}" y1="{y_upper:.1f}" x2="{cx + 6:.1f}" y2="{y_upper:.1f}" stroke="#334155" stroke-width="1" opacity="0.5"/>')
                parts.append(f'<line x1="{cx - 6:.1f}" y1="{y_lower:.1f}" x2="{cx + 6:.1f}" y2="{y_lower:.1f}" stroke="#334155" stroke-width="1" opacity="0.5"/>')
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{h - 18}" text-anchor="middle" font-size="12" fill="#334155" font-weight="600">{s}</text>')
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{max(y - 8, 14):.1f}" text-anchor="middle" font-size="13" fill="#0f172a" font-weight="700">{v:.3f}</text>')
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="260" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

    # ── Panel 3: Radar chart ──
    def _svg_radar() -> str:
        w, h = 500, 320
        cx, cy, R = 240, 160, 110
        parts = []
        for ring in range(1, 6):
            pts = []
            for i in range(len(metrics)):
                ang = -_math.pi / 2 + i * 2 * _math.pi / len(metrics)
                r = (ring / 5) * R
                pts.append(f"{cx + _math.cos(ang) * r:.1f},{cy + _math.sin(ang) * r:.1f}")
            parts.append(f'<polygon points="{" ".join(pts)}" fill="none" stroke="#dbe4f0"/>')
            sx = cx + _math.cos(-_math.pi / 2 + 0.01) * R * 0.2
            sy = cy + _math.sin(-_math.pi / 2 + 0.01) * R * 0.2 - 4
            parts.append(f'<text x="{sx:.1f}" y="{sy:.1f}" font-size="9" fill="#94a3b8" text-anchor="middle">{ring * 0.2:.1f}</text>')
        for i, metric in enumerate(metrics):
            ang = -_math.pi / 2 + i * 2 * _math.pi / len(metrics)
            ex = cx + _math.cos(ang) * R
            ey = cy + _math.sin(ang) * R
            parts.append(f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#cbd5e1" stroke-dasharray="3,3"/>')
            lx = cx + _math.cos(ang) * (R + 22)
            ly = cy + _math.sin(ang) * (R + 22)
            anchor = "start" if lx > cx else ("end" if lx < cx else "middle")
            lbl = METRIC_LABELS.get(metric, metric)
            parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" dominant-baseline="middle" font-size="11" fill="#334155" font-weight="600">{lbl}</text>')
        palette = ["rgba(14,165,233,0.22)", "rgba(16,185,129,0.22)", "rgba(245,158,11,0.22)", "rgba(239,68,68,0.22)"]
        line_palette = ["#0284c7", "#059669", "#d97706", "#dc2626"]
        gmv = {s["system"]: s["metric_means"] for s in valid_summaries}
        for si, s in enumerate(systems):
            vals = {m: gmv.get(s, {}).get(m, 0) for m in metrics}
            pts = []
            for i, metric in enumerate(metrics):
                v = max(0, min(1, float(vals.get(metric, 0) or 0)))
                ang = -_math.pi / 2 + i * 2 * _math.pi / len(metrics)
                pts.append(f"{cx + _math.cos(ang) * R * v:.1f},{cy + _math.sin(ang) * R * v:.1f}")
            parts.append(f'<polygon points="{" ".join(pts)}" fill="{palette[si % len(palette)]}" stroke="{line_palette[si % len(line_palette)]}" stroke-width="2" class="radar-shape" style="animation: fadeIn 0.6s ease-out {si * 0.2}s both;"/>')
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="320" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

    # ── Panel 4: Attack Impact Ranking (NEW - replaces failure stack) ──
    def _svg_attack_impact() -> str:
        w, h = 640, 320
        m = {"l": 160, "r": 60, "t": 10, "b": 10}
        pw, ph = w - m["l"] - m["r"], h - m["t"] - m["b"]
        bar_h = min(32, (ph - 10) / max(len(attack_impacts), 1))
        gap = 8
        parts = []
        max_drop = max((x["avg_drop"] for x in attack_impacts), default=0.01)
        for i, imp in enumerate(attack_impacts):
            y = m["t"] + i * (bar_h + gap)
            drop = imp["avg_drop"]
            bar_w = (drop / max(max_drop, 0.01)) * pw * 0.85
            color = "#ef4444" if drop >= 0.2 else ("#f59e0b" if drop >= 0.1 else "#10b981")
            # Label
            parts.append(f'<text x="{m["l"] - 8}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end" font-size="12" fill="#334155" font-weight="500">{imp["attack_type"]}</text>')
            # Bar (width set to actual value; animation provides entrance effect)
            parts.append(f'<rect x="{m["l"]}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}" class="bar-grow-h"/>')
            # Value
            parts.append(f'<text x="{m["l"] + bar_w + 12:.1f}" y="{y + bar_h / 2 + 4:.1f}" font-size="13" fill="#0f172a" font-weight="700">{drop:.3f}</text>')
            # Worst system hint
            parts.append(f'<text x="{m["l"] + bar_w + 12:.1f}" y="{y + bar_h / 2 + 18:.1f}" font-size="9" fill="#64748b">最差:{imp["worst_system"]}</text>')
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="320" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

    # ── Panel 5: Metric Degradation Heatmap (NEW - replaces old grouped bar) ──
    def _svg_metric_degradation() -> str:
        w, h = 700, 260
        m = {"l": 30, "r": 20, "t": 30, "b": 30}
        if not systems or not metrics:
            return (
                f'<svg viewBox="0 0 {w} {h}" width="100%" height="260" xmlns="http://www.w3.org/2000/svg">'
                f'<rect x="0" y="0" width="{w}" height="{h}" rx="18" fill="#f8fafc"/>'
                f'<text x="{w / 2:.1f}" y="{h / 2 - 6:.1f}" text-anchor="middle" font-size="16" fill="#334155" font-weight="700">暂无可视化对比数据</text>'
                f'<text x="{w / 2:.1f}" y="{h / 2 + 18:.1f}" text-anchor="middle" font-size="12" fill="#64748b">当前报告不包含系统对比矩阵，已跳过热力图渲染</text>'
                '</svg>'
            )
        cell_w = (w - m["l"] - m["r"]) / len(systems)
        cell_h = (h - m["t"] - m["b"]) / len(metrics)
        # Color scale: degradation 0 -> green, 0.10 -> yellow, 0.25+ -> red
        def heat_color(v):
            if v <= 0.01: return "#d1fae5", "#065f46"
            if v <= 0.05: return "#a7f3d0", "#064e3b"
            if v <= 0.10: return "#fef3c7", "#92400e"
            if v <= 0.20: return "#fed7aa", "#9a3412"
            return "#fecaca", "#991b1b"
        parts = []
        # Column headers (systems)
        for si, s in enumerate(systems):
            cx = m["l"] + si * cell_w + cell_w / 2
            parts.append(f'<text x="{cx:.1f}" y="{m["t"] - 10}" text-anchor="middle" font-size="12" fill="#334155" font-weight="700">{s}</text>')
        # Row headers (metrics)
        for mi, metric in enumerate(metrics):
            lbl = METRIC_LABELS.get(metric, metric)
            cy = m["t"] + mi * cell_h + cell_h / 2 + 4
            parts.append(f'<text x="{m["l"] - 6}" y="{cy:.1f}" text-anchor="end" font-size="10" fill="#475569" font-weight="500">{lbl}</text>')
        # Cells
        for si, s in enumerate(systems):
            for mi, metric in enumerate(metrics):
                v = metric_degradation.get(s, {}).get(metric, 0)
                bg, fg = heat_color(v)
                x = m["l"] + si * cell_w + 2
                y = m["t"] + mi * cell_h + 2
                cw = cell_w - 4
                ch = cell_h - 4
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" rx="6" fill="{bg}" class="heat-cell-svg" style="animation: fadeIn 0.4s ease-out {(si * 4 + mi) * 0.07}s both;">')
                parts.append(f'<title>{s} / {METRIC_LABELS.get(metric, metric)}: 平均退化 {v:.3f}</title>')
                parts.append(f'</rect>')
                parts.append(f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" text-anchor="middle" font-size="13" fill="{fg}" font-weight="700">{v:.3f}</text>')
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="260" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

    # ── Panel 7: Enhanced Box Plot with distribution bins ──
    def _svg_box_plot() -> str:
        w, h = 640, 280
        m = {"l": 56, "r": 20, "t": 14, "b": 60}
        pw, ph = w - m["l"] - m["r"], h - m["t"] - m["b"]
        step = pw / max(len(systems), 1)
        box_data = {s["system"]: s["box"] for s in valid_summaries}
        def y_of(v):
            return m["t"] + (1 - max(0, min(1, float(v or 0)))) * ph
        parts = []
        for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            y = y_of(t)
            parts.append(f'<line x1="{m["l"]}" y1="{y:.1f}" x2="{w - m["r"]}" y2="{y:.1f}" stroke="#e2e8f0" stroke-dasharray="4,4"/>')
            parts.append(f'<text x="{m["l"] - 4}" y="{y + 4:.1f}" text-anchor="end" font-size="9" fill="#94a3b8">{t:.1f}</text>')
        for i, s in enumerate(systems):
            b = box_data.get(s, {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0})
            cx = m["l"] + i * step + step / 2
            bw = max(26, step * 0.28)
            ymin, yq1, ymed, yq3, ymax = y_of(b["min"]), y_of(b["q1"]), y_of(b["median"]), y_of(b["q3"]), y_of(b["max"])
            # Whiskers
            parts.append(f'<line x1="{cx:.1f}" y1="{ymin:.1f}" x2="{cx:.1f}" y2="{ymax:.1f}" stroke="#334155" stroke-width="1.2"/>')
            # Box
            parts.append(f'<rect x="{cx - bw / 2:.1f}" y="{yq3:.1f}" width="{bw:.1f}" height="{max(1, yq1 - yq3):.1f}" fill="#bfdbfe" stroke="#1d4ed8" stroke-width="1.5" rx="2"/>')
            # Median line
            parts.append(f'<line x1="{cx - bw / 2:.1f}" y1="{ymed:.1f}" x2="{cx + bw / 2:.1f}" y2="{ymed:.1f}" stroke="#1e3a8a" stroke-width="2.5"/>')
            # Labels
            parts.append(f'<text x="{cx:.1f}" y="{h - 42}" text-anchor="middle" font-size="12" fill="#334155" font-weight="600">{s}</text>')
            parts.append(f'<text x="{cx:.1f}" y="{ymed - 6:.1f}" text-anchor="middle" font-size="10" fill="#1e3a8a" font-weight="600">{float(b["median"]):.3f}</text>')
            # Distribution bins as mini bars
            bins = distribution_bins.get(s, {})
            max_bin = max(bins.values()) if bins else 1
            bin_w = bw * 0.9
            bin_h_max = 20
            bin_y = h - 28
            bin_labels = ["<.5", ".5-.7", ".7-.8", ".8-.9", ".9-.95", ".95-1", "1+"]
            bin_keys = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1.0", "1.0+"]
            sub_step = bin_w / len(bin_keys)
            for bi, bk in enumerate(bin_keys):
                count = bins.get(bk, 0)
                bh = (count / max(max_bin, 1)) * bin_h_max if count > 0 else 1
                bx = cx - bin_w / 2 + bi * sub_step
                by = bin_y - bh
                parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{sub_step - 1:.1f}" height="{max(1, bh):.1f}" fill="#93c5fd" rx="1" opacity="0.7"><title>{bin_labels[bi]}: {count}</title></rect>')
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="280" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'

    # ── Panel 8: Vulnerability Cards (NEW - replaces old failure stack bottom) ──
    def _render_vulnerability_cards() -> str:
        if not vulnerability_cards:
            return '<div style="padding:12px;color:#64748b;font-size:13px;">✅ 所有攻击类型均未检测到显著漏洞。</div>'
        # Deduplicate by cause
        seen = set()
        unique_cards = []
        for vc in vulnerability_cards:
            key = (vc["cause"], vc["fix"])
            if key not in seen:
                seen.add(key)
                unique_cards.append(vc)
        parts = []
        for vc in unique_cards[:6]:  # Show top 6
            risk_color = "#ef4444" if vc["avg_robustness"] < 0.8 else ("#f59e0b" if vc["avg_robustness"] < 0.9 else "#10b981")
            parts.append(f"""<details class="vuln-card" style="margin-bottom:6px;border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;background:#f8fbff;">
<summary style="cursor:pointer;font-weight:600;font-size:13px;color:#0f172a;display:flex;align-items:center;gap:8px;">
<span style="background:{risk_color};color:#fff;border-radius:4px;padding:1px 6px;font-size:10px;">{vc['attack_type']}</span>
<span>{vc['vulnerability']}</span>
<span style="margin-left:auto;font-size:11px;color:#64748b;">鲁棒性: {vc['avg_robustness']}</span>
</summary>
<div style="margin-top:8px;font-size:12px;line-height:1.6;">
<div style="color:#991b1b;margin-bottom:4px;">🔍 <strong>原因：</strong>{vc['cause']}</div>
<div style="color:#065f46;">🔧 <strong>修复建议：</strong>{vc['fix']}</div>
<div style="color:#64748b;margin-top:4px;">📋 影响系统：{vc['system']}</div>
</div>
</details>""")
        return "".join(parts)

    # ── Panel 9: Trend SVG ──
    def _svg_trend() -> tuple:
        w, h = 920, 280
        m = {"l": 56, "r": 20, "t": 12, "b": 50}
        pw, ph = w - m["l"] - m["r"], h - m["t"] - m["b"]
        points = sorted([pt for pt in history_points if pt.get("timestamp")], key=lambda x: str(x.get("timestamp", "")))
        hint = ""
        parts = []
        for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            y = m["t"] + (1 - t) * ph
            parts.append(f'<line x1="{m["l"]}" y1="{y:.1f}" x2="{w - m["r"]}" y2="{y:.1f}" stroke="#e2e8f0" stroke-dasharray="4,4"/>')
            parts.append(f'<text x="{m["l"] - 4}" y="{y + 4:.1f}" text-anchor="end" font-size="9" fill="#94a3b8">{t:.1f}</text>')
        if not points:
            hint = "暂无历史数据，请先完成评估。"
        else:
            def x_of(i):
                return m["l"] + (i / max(len(points) - 1, 1)) * pw
            def y_of(v):
                return m["t"] + (1 - max(0, min(1, float(v)))) * ph
            grouped = {}
            for i, pt in enumerate(points):
                sys_name = pt.get("system", "?")
                if sys_name not in grouped:
                    grouped[sys_name] = []
                grouped[sys_name].append((i, float(pt.get("overall", 0))))
            colors = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
            for idx, (sys_name, arr) in enumerate(grouped.items()):
                if len(arr) >= 2:
                    pts_str = " ".join(f"{x_of(p[0]):.1f},{y_of(p[1]):.1f}" for p in arr)
                    parts.append(f'<polyline points="{pts_str}" fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="2"/>')
                for p in arr:
                    px, py = x_of(p[0]), y_of(p[1])
                    parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{colors[idx % len(colors)]}" stroke="#fff" stroke-width="1.5"/>')
                    parts.append(f'<text x="{px:.1f}" y="{py - 10:.1f}" text-anchor="middle" font-size="9" fill="{colors[idx % len(colors)]}" font-weight="600">{p[1]:.3f}</text>')
            max_per_system = max(len(arr) for arr in grouped.values()) if grouped else 0
            if max_per_system < 2:
                hint = "⚠️ 各系统当前仅单次评估快照，暂无时间序列趋势。对同一系统进行多次评估后才会自动连线，敬请期待。"
            elif len(points) <= 3:
                hint = f"当前每系统最多 {max_per_system} 个时间点，趋势仅供参考。建议积累 ≥5 轮数据后再做趋势解读。"
            else:
                hint = "连线表示同一系统在不同时间点的鲁棒性变化趋势。"
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="280" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>', hint

    # ── Detail table rows ──
    # ── P0: Pure-HTML detail tree (System -> Attack -> Cases, no JS) ──
    def _render_detail_tree() -> str:
        color_map = {"高风险": "#ef4444", "中风险": "#f59e0b", "稳健": "#10b981"}
        parts = []
        for sys_name in systems:
            sys_rows = [r for r in detail_rows if r["system"] == sys_name]
            if not sys_rows:
                continue
            parts.append(f'<details class="dt-sys" id="dt-{sys_name}" open>')
            parts.append(f'<summary class="dt-summary-sys"><strong>{sys_name}</strong> - {len(sys_rows)} 条攻击记录</summary>')
            parts.append('<div class="dt-content">')
            for at in attack_types:
                at_rows = [r for r in sys_rows if r["attack_type"] == at]
                if not at_rows:
                    continue
                r = at_rows[0]
                risk_color = color_map.get(r["rating"], "#64748b")
                anchor_id = f"dt-{sys_name}-{at}"
                parts.append(f'<details class="dt-atk" id="{anchor_id}">')
                parts.append(f'<summary class="dt-summary-atk">')
                parts.append(f'<span class="atk-badge" style="background:{risk_color};">{at}</span>')
                parts.append(f'<span class="atk-stat">降分: <b>{r["score"]:.3f}</b></span>')
                parts.append(f'<span class="atk-stat">鲁棒性: <b>{r["robustness"]:.3f}</b></span>')
                parts.append(f'<span class="atk-stat">评级: <b style="color:{risk_color}">{r["rating"]}</b></span>')
                parts.append(f'</summary>')
                parts.append('<table class="mini-table"><thead><tr><th>系统</th><th>攻击类型</th><th>降分</th><th>鲁棒性</th><th>评级</th></tr></thead><tbody>')
                for row in at_rows:
                    parts.append(f'<tr><td>{row["system"]}</td><td>{row["attack_type"]}</td><td>{row["score"]:.3f}</td><td>{row["robustness"]:.3f}</td><td style="color:{color_map.get(row["rating"],"#64748b")}">{row["rating"]}</td></tr>')
                parts.append('</tbody></table>')
                # Case drilldown
                combo_cases = [dc for dc in drilldown_cases if dc["system"] == sys_name and dc["attack_type"] == at]
                if combo_cases:
                    parts.append(f'<details class="dt-cases"><summary>展开 Case 样本（{len(combo_cases)} 条）</summary>')
                    for dc in combo_cases:
                        pm = dc["per_metric"]
                        delta_rows = ""
                        for m in metrics:
                            d = pm.get(m, {})
                            dv = d.get("delta", 0)
                            dc_color = "#ef4444" if dv > 0.1 else ("#f59e0b" if dv > 0.02 else "#10b981")
                            delta_rows += f'<tr><td>{METRIC_LABELS.get(m,m)}</td><td>{d.get("normal",0):.3f}</td><td>{d.get("attack",0):.3f}</td><td style="color:{dc_color};font-weight:600">{dv:+.3f}</td></tr>'
                        parts.append(f'<div class="case-card">')
                        parts.append(f'<div class="case-header">鲁棒性: <b style="color:{_color_for(dc["robustness_score"])}">{dc["robustness_score"]:.3f}</b></div>')
                        parts.append(f'<div class="case-query"><span>原始问题:</span> {dc["original_query"][:150]}</div>')
                        parts.append(f'<div class="case-query"><span>扰动问题:</span> {dc["perturbed_query"][:150]}</div>')
                        parts.append(f'<div class="case-answer"><span>回答:</span> {dc["answer_excerpt"][:150]}</div>')
                        parts.append(f'<table class="mini-table" style="margin-top:6px;"><thead><tr><th>指标</th><th>正常分</th><th>攻击分</th><th>变化</th></tr></thead><tbody>{delta_rows}</tbody></table>')
                        parts.append(f'</div>')
                    parts.append('</details>')
                parts.append('</details>')
            parts.append('</div></details>')
        return "".join(parts)

    # ── P0: Heatmap with anchor links (click jumps to detail tree) ──
    def _render_heatmap_anchored() -> str:
        parts = ['<table class="heat-table"><thead><tr><th>攻击类型</th>']
        for sys_name in systems:
            parts.append(f'<th>{sys_name}</th>')
        parts.append('</tr></thead><tbody>')
        for row in heat_rows:
            parts.append(f'<tr><td>{row["attack_type"]}</td>')
            for sys_name in systems:
                v = row.get(sys_name, 0)
                if v >= 0.2:
                    bg, fg = "#fef0f0", "#b91c1c"
                elif v >= 0.1:
                    bg, fg = "#fef9e7", "#92400e"
                else:
                    bg, fg = "#d1fae5", "#065f46"
                anchor = f"dt-{sys_name}-{row['attack_type']}"
                parts.append(f'<td class="heat-cell" style="background:{bg};color:{fg};font-weight:700;"><a href="#{anchor}" style="color:{fg};text-decoration:none;display:block;">{v:.3f}</a></td>')
            parts.append('</tr>')
        parts.append('</tbody></table>')
        return "".join(parts)

    # ── P0-3: Per-attack mechanism breakdown mini-bars ──
        if not attack_mechanisms:
            return ""
        mechanism_labels = {"context_precision降": "上下文精确度", "answer_relevancy降": "答案相关性", "faithfulness降": "忠实度", "context_recall降": "上下文召回率"}
        mechanism_colors = {"context_precision降": "#ef4444", "answer_relevancy降": "#f59e0b", "faithfulness降": "#6366f1", "context_recall降": "#10b981"}
        parts = []
        for at in attack_types:
            mech = attack_mechanisms.get(at, {})
            total = sum(mech.values()) or 1
            parts.append(f'<div style="margin-bottom:6px;font-size:11px;"><strong>{at}</strong> (共{sum(mech.values())}次显著退化)</div>')
            parts.append('<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:2px;background:#ecf2f8;">')
            for mk, ml in mechanism_labels.items():
                count = mech.get(mk, 0)
                if count > 0:
                    pct = (count / total) * 100
                    parts.append(f'<div style="width:{pct:.1f}%;background:{mechanism_colors[mk]};" title="{ml}: {count}次"></div>')
            parts.append('</div>')
            parts.append('<div style="display:flex;gap:12px;font-size:9px;color:#64748b;margin-bottom:10px;">')
            for mk, ml in mechanism_labels.items():
                count = mech.get(mk, 0)
                if count > 0:
                    parts.append(f'<span style="display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;border-radius:2px;background:{mechanism_colors[mk]};display:inline-block;"></span>{ml}:{count}</span>')
            parts.append('</div>')
        return "".join(parts)

    # ── P0-3: Per-attack mechanism breakdown mini-bars ──
    def _render_mechanism_breakdown() -> str:
        if not attack_mechanisms:
            return ""
        mechanism_labels = {"context_precision降": "上下文精确度", "answer_relevancy降": "答案相关性", "faithfulness降": "忠实度", "context_recall降": "上下文召回率"}
        mechanism_colors = {"context_precision降": "#ef4444", "answer_relevancy降": "#f59e0b", "faithfulness降": "#6366f1", "context_recall降": "#10b981"}
        parts = []
        for at in attack_types:
            mech = attack_mechanisms.get(at, {})
            total = sum(mech.values()) or 1
            parts.append(f'<div style="margin-bottom:6px;font-size:11px;"><strong>{at}</strong> ({sum(mech.values())} 次显著退化)</div>')
            parts.append('<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:2px;background:#ecf2f8;">')
            for mk, ml in mechanism_labels.items():
                count = mech.get(mk, 0)
                if count > 0:
                    pct = (count / total) * 100
                    parts.append(f'<div style="width:{pct:.1f}%;background:{mechanism_colors[mk]};" title="{ml}: {count}"></div>')
            parts.append('</div>')
            parts.append('<div style="display:flex;gap:12px;font-size:9px;color:#64748b;margin-bottom:10px;">')
            for mk, ml in mechanism_labels.items():
                count = mech.get(mk, 0)
                if count > 0:
                    parts.append(f'<span style="display:flex;align-items:center;gap:3px;"><span style="width:8px;height:8px;border-radius:2px;background:{mechanism_colors[mk]};display:inline-block;"></span>{ml}:{count}</span>')
            parts.append('</div>')
        return "".join(parts)

    # ── Build all chart components ──
    overall_bar_svg = _svg_overall_bar()
    radar_svg = _svg_radar()
    attack_impact_svg = _svg_attack_impact()
    metric_degrad_svg = _svg_metric_degradation()
    box_svg = _svg_box_plot()
    # 优先展示 LLM 诊断，其次展示规则诊断
    llm_diags = [(p.get("system", ""), p.get("llm_diagnosis", "")) for p in payloads if p.get("llm_diagnosis")]
    if llm_diags:
        import re as _re
        def _md_to_html(text):
            # 简单 markdown → HTML
            text = _re.sub(r'^### (.+)', r'<h4 style="margin:8px 0 4px;color:#0f172a;">\1</h4>', text, flags=_re.M)
            text = _re.sub(r'^## (.+)', r'<h3 style="margin:10px 0 6px;color:#1e3a5f;">\1</h3>', text, flags=_re.M)
            text = _re.sub(r'^# (.+)', r'<h2 style="margin:12px 0 6px;color:#0f172a;">\1</h2>', text, flags=_re.M)
            text = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            text = _re.sub(r'^\- (.+)', r'<li>\1</li>', text, flags=_re.M)
            text = _re.sub(r'(?<!\n)\n(?!\n)', r'<br>', text)
            return text

        parts = []
        for sys_name, diag in llm_diags:
            html_body = _md_to_html(diag)
            parts.append(f"""
<details class='llm-diag' style='margin-bottom:8px;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;'>
<summary style='cursor:pointer;padding:10px 14px;background:linear-gradient(135deg,#f0f9ff,#e0f2fe);font-size:14px;font-weight:700;color:#0369a1;display:flex;align-items:center;gap:8px;'>
<span>🤖</span><span>{sys_name} — LLM 深度诊断</span>
</summary>
<div style='padding:12px 16px;font-size:13px;line-height:1.8;color:#334155;max-height:500px;overflow-y:auto;background:#fafcfd;'>
{html_body}
</div>
</details>""")
        vulnerability_html = "".join(parts)
    else:
        vulnerability_html = _render_vulnerability_cards()
    mechanism_html = _render_mechanism_breakdown()
    detail_tree_html = _render_detail_tree()
    heatmap_anchored_html = _render_heatmap_anchored()
    trend_svg, trend_hint = _svg_trend()

    sample_counts_text = " | ".join(f"{k}: n={v['scored']}/{v['attack_pairs']}" for k, v in sample_counts.items())

    invalid_systems = [s["system"] for s in summaries if not s.get("has_valid_data", True)]
    quality_warning = ""
    if invalid_systems:
        names = "、".join(invalid_systems)
        quality_warning = f"<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:10px 16px;margin-bottom:12px;color:#991b1b;font-size:13px;'>⚠️ <strong>数据质量提醒：</strong>以下系统所有评分均出错（{names}），可能由于 LLM 未正确配置。请检查 API Key 和模型配置后重新评估。</div>"

    return f"""<!DOCTYPE html>
<html translate="no">
<head>
<meta charset="utf-8">
<meta name="google" content="notranslate">
<style>
:root{{--bg:#f4f8fb;--panel:#fff;--ink:#0f172a;--muted:#475569;--line:#dbe4f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:'Microsoft YaHei',sans-serif}}
.dash-wrap{{padding:14px}}
.head{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px;min-width:160px;transition:transform 0.2s,box-shadow 0.2s}}
.kpi:hover{{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,0.08)}}
.kpi .label{{color:var(--muted);font-size:12px}}
.kpi .value{{font-size:20px;font-weight:800;margin-top:4px}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;transition:box-shadow 0.3s}}
.panel:hover{{box-shadow:0 2px 8px rgba(0,0,0,0.04)}}
.span-12{{grid-column:span 12}}
.span-6{{grid-column:span 6}}
.panel h3{{margin:0 0 10px;font-size:16px;color:#134e4a;display:flex;align-items:center;gap:8px}}
.panel h3::before{{content:'';display:inline-block;width:4px;height:18px;background:#14b8a6;border-radius:2px}}
.sys-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}}
.sys-card{{border:1px solid #d7e5ef;border-radius:10px;padding:10px;background:#f9fcff;transition:transform 0.2s,box-shadow 0.2s}}
.sys-card:hover{{transform:translateY(-2px);box-shadow:0 3px 10px rgba(0,0,0,0.06)}}
.sys-name{{font-weight:700}}
.sys-score{{font-size:24px;font-weight:900;color:#0f766e}}
.sys-meta{{color:var(--muted);font-size:12px}}
.bullet-wrap{{margin:8px 0;height:10px;background:#e6edf5;border-radius:999px;position:relative;overflow:hidden}}
.bullet-fill{{position:absolute;left:0;top:0;height:100%;background:linear-gradient(90deg,#34d399,#14b8a6);transition:width 0.8s ease-out}}
.bullet-target{{position:absolute;top:-2px;width:2px;height:14px;background:#0f172a;opacity:.75}}
.legend{{font-size:12px;color:#64748b;margin-top:6px}}
.svg-wrap{{width:100%;overflow-x:auto}}
.svg-wrap svg{{display:block;max-width:100%}}
.heat-table,.mini-table{{width:100%;border-collapse:collapse;font-size:13px}}
.heat-table th,.heat-table td,.mini-table th,.mini-table td{{border:1px solid #e2e8f0;padding:7px 8px;text-align:center;transition:background 0.2s}}
.heat-table th,.mini-table th{{background:#eff6ff}}
.heat-table td:hover{{filter:brightness(0.92)}}
.heat-cell{{cursor:pointer;transition:transform 0.15s,outline 0.15s}}
.heat-cell:hover{{transform:scale(1.08);z-index:1;position:relative}}
.filter-row{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;align-items:center}}
.filter-row select{{border:1px solid #cbd5e1;border-radius:8px;padding:6px 8px;background:#fff;transition:border-color 0.2s}}
.filter-row select:focus{{border-color:#0ea5e9;outline:none}}
.vuln-card{{transition:border-color 0.2s}}
.vuln-card:hover{{border-color:#93c5fd}}
.vuln-card summary{{list-style:none}}
.vuln-card summary::-webkit-details-marker{{display:none}}
/* Collapsible more-section (checkbox toggle, always in DOM) */
.more-toggle{{position:absolute;opacity:0;width:0;height:0;}}
.more-section{{display:block;width:100%;box-sizing:border-box;border:1px dashed #cbd5e1;border-radius:12px;padding:0;}}
.more-summary{{padding:12px 16px;cursor:pointer;font-size:14px;font-weight:600;color:#475569;text-align:center;background:#f8fafc;border-radius:12px;transition:background 0.2s;display:block;width:100%;box-sizing:border-box;}}
.more-summary:hover{{background:#f0f9ff;color:#0ea5e9;}}
.more-toggle:checked~.more-summary{{border-bottom:1px solid #e2e8f0;border-radius:12px 12px 0 0;}}
.more-grid{{display:none;grid-template-columns:repeat(12,1fr);gap:12px;align-items:start;width:100%;box-sizing:border-box;padding:12px;}}
.more-toggle:checked~.more-grid{{display:grid;}}
.more-grid>.panel{{min-width:0;overflow:visible;}}
.more-grid>.panel.span-12{{grid-column:1/-1;}}
.more-grid>.panel.span-6{{grid-column:span 6;}}
/* Detail tree styles */
.dt-sys{{margin-bottom:8px;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;}}
.dt-sys[open]{{border-color:#0ea5e9;}}
.dt-summary-sys{{padding:10px 14px;background:#f0f9ff;cursor:pointer;font-size:14px;list-style:none;display:flex;align-items:center;gap:8px;}}
.dt-summary-sys::-webkit-details-marker{{display:none;}}
.dt-summary-sys::before{{content:">";display:inline-block;margin-right:8px;transition:transform 0.2s;font-size:10px;}}
.dt-sys[open]>.dt-summary-sys::before{{transform:rotate(90deg);}}
.dt-content{{padding:8px 10px 10px 10px;background:#fff;}}
.dt-atk{{margin:6px 0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;background:#fff;}}
.dt-atk[open]{{border-color:#93c5fd;}}
.dt-summary-atk{{padding:8px 12px;cursor:pointer;display:flex;align-items:center;gap:12px;font-size:13px;list-style:none;background:#fafbfc;flex-wrap:wrap;}}
.dt-summary-atk::-webkit-details-marker{{display:none;}}
.dt-summary-atk::before{{content:">";display:inline-block;margin-right:4px;transition:transform 0.2s;font-size:9px;color:#94a3b8;}}
.dt-atk[open]>.dt-summary-atk::before{{transform:rotate(90deg);}}
.atk-badge{{color:#fff;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;}}
.atk-stat{{color:#64748b;font-size:12px;}}
.dt-cases{{margin:8px 10px 10px 10px;border:1px dashed #e2e8f0;border-radius:6px;padding:8px 10px;background:#fcfdff;}}
.dt-cases summary{{cursor:pointer;font-size:12px;color:#475569;list-style:none;}}
.dt-cases summary::-webkit-details-marker{{display:none;}}
.case-card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px;margin:8px 0;font-size:12px;}}
.case-header{{margin-bottom:6px;font-size:13px;font-weight:600;}}
.case-query{{margin:4px 0;color:#334155;line-height:1.55;}}
.case-query span{{color:#64748b;font-size:11px;}}
.case-answer{{margin:4px 0;color:#475569;max-height:72px;overflow-y:auto;line-height:1.55;}}
/* Target highlight animation (热力图跳转高亮) */
.dt-atk:target{{border-color:#0ea5e9 !important;box-shadow:0 0 0 4px rgba(14,165,233,0.25);animation:targetPulse 1.5s ease-out;}}
.dt-atk:target>.dt-summary-atk{{background:#f0f9ff !important;}}
@keyframes targetPulse{{0%{{box-shadow:0 0 0 0 rgba(14,165,233,0.5)}}50%{{box-shadow:0 0 0 8px rgba(14,165,233,0.15)}}100%{{box-shadow:0 0 0 4px rgba(14,165,233,0.25)}}}}
/* Trend note bar (趋势说明条) */
.trend-note{{background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:8px 12px;margin-bottom:8px;font-size:12px;color:#0c4a6e;line-height:1.6;display:flex;align-items:flex-start;gap:8px;}}
.trend-note-icon{{font-size:16px;flex-shrink:0;}}
/* Spacing rhythm (统一间距) */
.panel{{padding:14px;}}
.panel h3{{margin:0 0 10px;font-size:16px;}}
.legend{{font-size:12px;color:#64748b;margin-top:6px;line-height:1.5;}}
@media (max-width:980px){{.span-6{{grid-column:span 12}}}}
/* P1-2: Global color threshold legend */
.color-legend{{display:flex;gap:16px;align-items:center;font-size:11px;margin-bottom:8px;}}
.color-legend-item{{display:flex;align-items:center;gap:4px;}}
.color-swatch{{width:12px;height:12px;border-radius:3px;display:inline-block;}}
/* P2: Mobile responsive */
@media (max-width:768px){{
  .grid{{grid-template-columns:1fr;}}
  .span-6,.span-12{{grid-column:span 1;}}
  .head{{flex-direction:column;}}
  .kpi{{min-width:auto;}}
  .sys-cards{{grid-template-columns:1fr;}}
}}
.more-grid{{grid-template-columns:repeat(12,minmax(0,1fr));}}
@media (max-width:768px){{
    .more-grid{{grid-template-columns:1fr;}}
    .more-grid > .panel.span-6,.more-grid > .panel.span-12{{grid-column:1 / -1;}}
}}
/* P2: Texture patterns for colorblind accessibility */
.risk-high{{background:repeating-linear-gradient(45deg,#fee2e2,#fee2e2 3px,#fecaca 3px,#fecaca 6px);}}
.risk-medium{{background:repeating-linear-gradient(45deg,#fef3c7,#fef3c7 3px,#fde68a 3px,#fde68a 6px);}}
.risk-low{{background:#d1fae5;}}
/* CSS Animations */
@keyframes growUp{{from{{transform:scaleY(0)}}to{{transform:scaleY(1)}}}}
@keyframes growRight{{from{{transform:scaleX(0)}}to{{transform:scaleX(1)}}}}
@keyframes fadeIn{{from{{opacity:0}}to{{opacity:1}}}}
@keyframes slideIn{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
.bar-grow{{transform-origin:bottom}}
.bar-grow-h{{transform-origin:left}}
.panel{{animation:slideIn 0.5s ease-out both}}
.panel:nth-child(1){{animation-delay:0s}}.panel:nth-child(2){{animation-delay:0.05s}}.panel:nth-child(3){{animation-delay:0.1s}}
.panel:nth-child(4){{animation-delay:0.15s}}.panel:nth-child(5){{animation-delay:0.2s}}.panel:nth-child(6){{animation-delay:0.25s}}
.panel:nth-child(7){{animation-delay:0.3s}}.panel:nth-child(8){{animation-delay:0.35s}}.panel:nth-child(9){{animation-delay:0.4s}}
.panel:nth-child(10){{animation-delay:0.45s}}
</style>
</head>
<body>
<div class="dash-wrap" translate="no">
<div class="head">
<div class="kpi"><div class="label">最佳系统</div><div class="value">{best['system']}</div></div>
<div class="kpi"><div class="label">最佳鲁棒性</div><div class="value">{best['overall']:.3f}</div></div>
<div class="kpi"><div class="label">系统平均鲁棒性</div><div class="value">{avg_overall:.3f}</div></div>
<div class="kpi"><div class="label">评分总对数</div><div class="value">{total_pairs}</div></div>
<div class="kpi"><div class="label">有效系统数</div><div class="value">{len(valid_summaries)}</div></div>
</div>
{quality_warning}
<div class="grid">
<div class="panel span-12"><h3>KPI 卡片 + 子弹图（目标阈值 {target_threshold}）</h3><div class="sys-cards">{cards_html}</div></div>
<div class="panel span-6"><h3>系统综合鲁棒性对比（须线=±1σ）</h3><div class="svg-wrap">{overall_bar_svg}</div></div>
<div class="panel span-6"><h3>系统能力雷达图</h3><div class="svg-wrap">{radar_svg}</div></div>
<div class="panel span-12"><h3>攻击影响排名（按平均降幅排序）</h3><div class="legend">越长的条 = 该攻击类型对系统造成的平均降分越大。红色≥0.2 黄色≥0.1 绿色&lt;0.1</div><div class="svg-wrap">{attack_impact_svg}</div></div>
<div class="panel span-12"><h3>指标退化热力图（系统 × 指标）</h3><div class="legend">单元格值 = 该指标在所有攻击类型下的平均退化幅度。context_precision 是通用弱点。</div><div class="svg-wrap">{metric_degrad_svg}</div>
</div>
<div class="panel span-12"><h3>攻击脆弱性热力图（点击跳转到下方明细）</h3><div class="legend">全局阈值：<span style="color:#991b1b;">红色>=0.2</span> <span style="color:#92400e;">黄色>=0.1</span> <span style="color:#065f46;">绿色&lt;0.1</span>。点击数字跳转到对应系统+攻击类型的明细折叠区。</div>{heatmap_anchored_html}</div>
<div class="panel span-12"><h3>退化机制诊断（攻击类型 x 指标降因拆解）</h3><div class="legend">每个攻击类型下，哪些指标贡献了退化？颜色=指标，宽度=受影响case占比。</div>{mechanism_html}</div>
<div class="panel span-6"><h3>鲁棒性分布（箱线图 + 密度分布）</h3><div class="svg-wrap">{box_svg}</div><div class="legend">箱体=IQR，须线=min/max，底部小条=各区间样本数分布（含&lt;0.5 极端失败桶）</div></div>
<div class="panel span-6"><h3>漏洞诊断与修复建议</h3><div style="max-height:280px;overflow-y:auto;">{vulnerability_html}</div></div>
<div class="panel span-12"><h3>趋势与回归（按时间）</h3><div class="trend-note"><span class="trend-note-icon">[i]</span><span>鲁棒性得分可能超过 1.0（当攻击后某指标得分反升时）。这是评分算法的正常行为，不影响结论解读。</span></div><div class="svg-wrap">{trend_svg}</div><div class="legend">{trend_hint}</div><div class="legend" style="margin-top:4px;">样本量 | {sample_counts_text}</div></div>
<div class="panel span-12"><h3>明细数据（折叠树：系统 > 攻击类型 > Case 样本）</h3><div class="legend">点击展开各级折叠。热力图数字为锚点链接，点击可跳转到对应条目。纯 HTML 原生交互，无需 JavaScript。</div>{detail_tree_html}</div>
</div></div>
</body></html>"""



def _refresh_experiment_dashboard() -> str:
    return _build_experiment_dashboard_html()


def _save_attack_cases(attack_cases: list[dict]) -> None:
    Path("data").mkdir(parents=True, exist_ok=True)
    out = Path("data/attack_cases.json")
    out.write_text(json.dumps(attack_cases, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _save_system_result(target: str, diagnosis: dict, eval_results: list[dict], attack_cases: list[dict]) -> str:
    """保存评估结果，文件名带时间戳。返回保存的路径。"""
    Path("data").mkdir(parents=True, exist_ok=True)
    name = _display_system_name(target).lower()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"data/results_{name}_{ts}.json")
    payload = {
        "system": target,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_executable": sys.executable,
        "overall_robustness": diagnosis.get("overall_robustness", 0),
        "overall_rating": diagnosis.get("overall_rating", "N/A"),
        "per_attack_type": diagnosis.get("per_attack_type", {}),
        "eval_results": eval_results,
        "attack_cases": attack_cases,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(out)


def _build_summary_md(final_result: dict, diagnosis: dict) -> str:
    overall = float(diagnosis.get("overall_robustness", 0) or 0)
    rating = diagnosis.get("overall_rating", "N/A")
    eval_results = final_result.get("eval_results", []) or []
    count = len(eval_results)

    metric_labels = {"faithfulness": "忠实度", "answer_relevancy": "答案相关性",
                     "context_precision": "上下文精确度", "context_recall": "上下文召回率"}
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # ── 总体四维均值 ──
    metric_scores = {}
    for key in metric_keys:
        vals_n, vals_a = [], []
        for r in eval_results:
            if r.get("case", {}).get("attack_type") == "无攻击（对照组）":
                continue
            n_s = (r.get("normal", {}) or {}).get("scores", {}) or {}
            a_s = (r.get("attack", {}) or {}).get("scores", {}) or {}
            if key in n_s: vals_n.append(n_s[key])
            if key in a_s: vals_a.append(a_s[key])
        if vals_n:
            metric_scores[key] = {
                "normal": round(sum(vals_n)/len(vals_n), 3),
                "attack": round(sum(vals_a)/len(vals_a), 3),
                "delta": round(sum(vals_n)/len(vals_n) - sum(vals_a)/len(vals_a), 3) if vals_a else 0,
            }

    lines = [
        "### 评估摘要",
        "#### 总体四维均值",
        f"| 指标 | 正常 | 对抗 | Δ |",
        f"|------|:---:|:---:|:---:|",
    ]
    for key in metric_keys:
        ms = metric_scores.get(key)
        if ms:
            d = ms["delta"]
            flag = "🔴" if d > 0.15 else ("🟡" if d > 0.05 else "🟢")
            lines.append(f"| {metric_labels[key]} | {ms['normal']:.3f} | {ms['attack']:.3f} | {flag} {d:+.3f} |")
    lines.append(f"\n综合鲁棒性: **{overall:.3f}** | 评级: **{rating}** | 评分条数: {count}")

    # ── 按攻击类型分项均值 ──
    per_attack = diagnosis.get("per_attack_type", {})
    if per_attack:
        lines.append("\n#### 按攻击类型评分矩阵")
        lines.append(f"| | { ' | '.join(metric_labels.values()) } | 鲁棒性 |")
        lines.append(f"|------|{':---:|' * 4}:---:|")

        # 对照组（只跑正常侧，无对抗）
        ctrl_cells = []
        for m in metric_keys:
            vals = []
            for r in eval_results:
                if r.get("case", {}).get("attack_type") != "无攻击（对照组）": continue
                s = (r.get("normal", {}) or {}).get("scores", {}) or {}
                if m in s: vals.append(s[m])
            ctrl_cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "-")
        lines.append(f"| 正常（对照组） | {' | '.join(ctrl_cells)} | 1.000 |")

        for at, info in per_attack.items():
            cells = []
            for m in metric_keys:
                vals = []
                for r in eval_results:
                    if r.get("case", {}).get("attack_type") != at: continue
                    s = (r.get("attack", {}) or {}).get("scores", {}) or {}
                    if m in s: vals.append(s[m])
                cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "-")
            rb = info.get("avg_robustness", 0)
            lines.append(f"| {at} | {' | '.join(cells)} | {rb:.3f} |")
    return "\n".join(lines)


def _build_html_report(final_result: dict, diagnosis: dict) -> str:
    summary = _build_summary_md(final_result, diagnosis).replace("\n", "<br>")
    return f"<div style='padding:12px;border:1px solid #dbe4f0;border-radius:8px;background:#fff'>{summary}</div>"


def _save_report_file(report_html: str) -> str:
    fd, path = tempfile.mkstemp(prefix="rag_eval_report_", suffix=".html")
    Path(path).write_text(report_html, encoding="utf-8")
    return path


def _render_seed_detail(eval_results: list) -> str:
    """按种子问题分组，展示每个种子的对抗变体、回答和评分。"""
    if not eval_results:
        return "<div style='padding:16px;color:#64748b;'>暂无评估结果，请先运行评估。</div>"

    from collections import defaultdict
    metric_labels = {"faithfulness": "忠实度", "answer_relevancy": "答案相关性",
                     "context_precision": "上下文精确度", "context_recall": "上下文召回率"}

    # 按原始问题分组
    groups = defaultdict(list)
    for r in eval_results:
        case = r.get("case", {})
        q = case.get("original_query", "?")
        groups[q].append(r)

    parts = []
    for qi, (query, items) in enumerate(groups.items()):
        # 排除对照组获取攻击结果
        attacks = [it for it in items
                   if it.get("case", {}).get("attack_type") != "无攻击（对照组）"]
        controls = [it for it in items
                    if it.get("case", {}).get("attack_type") == "无攻击（对照组）"]

        score_color = lambda v: "#ef4444" if v < 0.5 else ("#f59e0b" if v < 0.8 else "#10b981")
        ctrl = controls[0] if controls else None
        ctrl_score = ""
        if ctrl:
            scores = ctrl.get("normal", {}).get("scores", {}) or {}
            ctrl_score = " ".join(f"{metric_labels.get(k,k)}:{v:.3f}" for k, v in scores.items())

        parts.append(f"""
<details class='sd-seed' style='border:1px solid #e2e8f0;border-radius:10px;padding:0;margin-bottom:8px;overflow:hidden;'>
<summary style='cursor:pointer;padding:10px 14px;background:#f8fafc;font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px;'>
<span style='color:#0ea5e9;'>种子 #{qi+1}</span>
<span style='flex:1;'>{query[:80]}{'...' if len(query)>80 else ''}</span>
<span style='font-size:12px;color:#64748b;font-weight:400;'>{len(attacks)} 条对抗变体</span>
<span style='font-size:12px;color:#64748b;'>{ctrl_score}</span>
</summary>
<div style='padding:10px 14px;background:#fff;'>""")
        if ctrl:
            parts.append(f"""
<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;margin-bottom:10px;'>
<span style='color:#166534;font-weight:600;'>对照组</span>
<span style='color:#64748b;margin-left:8px;'>回答: {ctrl.get('normal',{}).get('answer','')[:200]}</span>
</div>""")

        for it in items:
            case = it.get("case", {})
            at_type = case.get("attack_type", "?")
            if at_type == "无攻击（对照组）":
                continue
            rob = it.get("robustness", {}) or {}
            rob_score = rob.get("robustness_score", "N/A")
            if isinstance(rob_score, (int, float)):
                rob_score = f"{rob_score:.3f}"
            per_m = rob.get("per_metric", {}) or {}
            delta_parts = ""
            for m in metric_labels:
                pm = per_m.get(m, {})
                if pm:
                    d = pm.get("delta", 0)
                    dc = "#ef4444" if abs(d) > 0.2 else ("#f59e0b" if abs(d) > 0.05 else "#10b981")
                    delta_parts += f"<span style='font-size:11px;color:{dc};margin-left:8px;'>{metric_labels.get(m,m)}: {d:+.3f}</span>"

            pq = case.get("perturbed_query", "")
            atk_ctx_changed = (case.get("original_contexts") != case.get("perturbed_contexts"))
            a_score = it.get("attack", {}).get("scores", {}) or {}
            a_score_str = " ".join(f"{metric_labels.get(k,k)}:{v:.3f}" for k, v in a_score.items())

            # 构建四维分数表格
            metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            score_rows = ""
            n_scores = (it.get("normal", {}).get("scores", {}) or {})
            a_scores = (it.get("attack", {}).get("scores", {}) or {})
            for m in metric_names:
                nv = n_scores.get(m, None)
                av = a_scores.get(m, None)
                if nv is None and av is None: continue
                d = round((nv or 0) - (av or 0), 3)
                dc = "#ef4444" if d > 0.15 else ("#f59e0b" if d > 0.05 else "#10b981")
                score_rows += f"<tr><td>{metric_labels.get(m,m)}</td><td>{nv:.3f}</td><td>{av:.3f}</td><td style='color:{dc};font-weight:600;'>{d:+.3f}</td></tr>"

            parts.append(f"""
<div style='border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;margin-bottom:6px;'>
<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
<span style='background:#dbeafe;color:#1e40af;font-size:11px;font-weight:600;padding:1px 8px;border-radius:4px;'>{at_type}</span>
<span style='font-size:13px;font-weight:600;color:{score_color(rob.get("robustness_score",0) if isinstance(rob.get("robustness_score"),(int,float)) else 0)};'>鲁棒性 {rob_score}</span>
{delta_parts}
</div>
<div style='font-size:12px;color:#334155;margin:4px 0;'><b>对抗问题:</b> {pq[:120]}{'...' if len(pq)>120 else ''}</div>
{f'<div style="font-size:11px;color:#94a3b8;margin:4px 0;">⚠ 上下文已修改</div>' if atk_ctx_changed else ''}
<div style='font-size:12px;color:#475569;margin:4px 0;'><b>正常回答:</b> {it.get("normal",{}).get("answer","")[:120]}</div>
<div style='font-size:12px;color:#475569;margin:4px 0;'><b>对抗回答:</b> {it.get("attack",{}).get("answer","")[:120]}</div>
<details style='margin-top:6px;'><summary style='cursor:pointer;font-size:12px;color:#475569;font-weight:500;'>📊 展开四维评分详情</summary>
<table style='width:100%;border-collapse:collapse;font-size:12px;margin-top:6px;'>
<thead><tr style='background:#f1f5f9;'><th>指标</th><th>正常分</th><th>对抗分</th><th>Δ</th></tr></thead>
<tbody>{score_rows}</tbody>
</table>
</details>
</div>""")
        parts.append("</div></details>")

    return f"""
<div style='padding:8px;font-family:Microsoft YaHei,sans-serif;'>
<h3 style='margin:0 0 10px;color:#0f172a;'>按种子问题查看 ({len(groups)} 个种子)</h3>
{"".join(parts)}
</div>"""


def _render_agent_panel(status: dict, attacker_info: str, evaluator_info: str, diagnostic_info: str,
                         progress_pct: dict | None = None) -> str:
    """三卡片 Agent 状态面板，依次点亮展示流水线进度。"""
    if progress_pct is None:
        progress_pct = {"attacker": 0, "evaluator": 0, "diagnostician": 0}
    cards = [
        ("🎯 Attacker", "攻击生成", status.get("attacker", ""), attacker_info,
         progress_pct.get("attacker", 0), "#f59e0b"),
        ("📊 Evaluator", "配对评分", status.get("evaluator", ""), evaluator_info,
         progress_pct.get("evaluator", 0), "#0ea5e9"),
        ("🩺 Diagnostician", "诊断报告", status.get("diagnostician", ""), diagnostic_info,
         progress_pct.get("diagnostician", 0), "#10b981"),
    ]
    parts = []
    for icon_label, role, state, info, pct, active_color in cards:
        is_active = any(kw in state for kw in ["🔄", "🌱", "🔫", "🔍", "✅", "🔄"])
        is_done = state.startswith("✅")
        if is_done:
            border_color = "#10b981"
            bg = "#f0fdf4"
            bar_color = "#10b981"
        elif is_active:
            border_color = active_color
            bg = "#f8fafc"
            bar_color = active_color
        else:
            border_color = "#e2e8f0"
            bg = "#f1f5f9"
            bar_color = "#cbd5e1"
        info_text = info if info else ("✅ 完成" if is_done else "⏳ 等待上游...")
        parts.append(f"""
<div style='flex:1;min-width:180px;background:{bg};border:2px solid {border_color};border-radius:10px;padding:10px 12px;transition:border 0.3s;'>
  <div style='font-size:13px;font-weight:700;margin-bottom:4px;color:#0f172a;'>{icon_label}</div>
  <div style='font-size:11px;color:#64748b;margin-bottom:6px;'>{role}</div>
  <div style='height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;margin-bottom:6px;'>
    <div style='height:100%;width:{pct}%;background:{bar_color};border-radius:3px;transition:width 0.4s;'></div>
  </div>
  <div style='font-size:11px;color:#475569;font-weight:500;'>{info_text}</div>
</div>""")
    return f"""
<div style='display:flex;gap:10px;flex-wrap:wrap;padding:8px 0;'>
  {''.join(parts)}
</div>
"""


def run_eval_with_viz(seed_json_text: str, metrics: list[str], mode_text: str, target: str, auto_generate: bool = True, n_seeds_ans: int = 1) -> Generator:
    status = {
        "attacker": "⏳ 等待运行",
        "evaluator": "⏳ 等待运行",
        "diagnostician": "⏳ 等待运行",
    }
    attacker_info = ""
    evaluator_info = ""
    diagnostic_info = ""
    seed_queries = []
    doc_dir = None

    try:
        if auto_generate or not seed_json_text.strip():
            # 尝试多条路径找到 data/documents
            possible = [
                Path(__file__).resolve().parent / "data" / "documents",
                Path(__file__).resolve().parent.parent / "data" / "documents",
            ]
            doc_dir = ""
            for p in possible:
                if p.exists() and list(p.glob("*.md")):
                    doc_dir = str(p)
                    break
            if not doc_dir:
                err = "文档目录为空，找不到 data/documents/，请确认文件路径"
                panel = _render_agent_panel(status, attacker_info, evaluator_info, diagnostic_info)
                yield (panel, err, err, "", "{}", None, _build_experiment_dashboard_html(), "*预估 LLM 调用: —*", _render_seed_detail([]), gr.update())
                return
        else:
            seed_queries = load_seed_queries(seed_json_text)
    except Exception as exc:
        err = f"种子问题解析失败: {exc}"
        panel = _render_agent_panel(status, attacker_info, evaluator_info, diagnostic_info)
        yield (panel, err, err, "", "{}", None, _build_experiment_dashboard_html(), "*预估 LLM 调用: —*", _render_seed_detail([]), gr.update())
        return

    selected_metrics = metrics or _METRIC_CHOICES
    is_dev = "开发模式" in str(mode_text)
    adapter = _select_adapter(target)

    # 构建 settings + embedding（供 RAGAS 评分和 Attacker 使用）
    from .config import Settings
    from .services.embedding.factory import EmbeddingProviderFactory
    settings_obj = Settings()

    # 防止 RAGAS 静默回退到 MockLLM，避免“能跑完但结果不可信”。
    if settings_obj.LLM_PROVIDER == "openai" and not settings_obj.OPENAI_API_KEY:
        err = "配置错误：OPENAI_API_KEY 未设置，已阻止运行以避免 MockLLM 污染评测结果。"
        panel = _render_agent_panel(status, attacker_info, evaluator_info, diagnostic_info)
        yield (panel, err, err, "", "{}", None, _build_experiment_dashboard_html(), "*预估 LLM 调用: —*", _render_seed_detail([]), gr.update())
        return
    if settings_obj.LLM_PROVIDER == "anthropic" and not settings_obj.ANTHROPIC_API_KEY:
        err = "配置错误：ANTHROPIC_API_KEY 未设置，已阻止运行以避免 MockLLM 污染评测结果。"
        panel = _render_agent_panel(status, attacker_info, evaluator_info, diagnostic_info)
        yield (panel, err, err, "", "{}", None, _build_experiment_dashboard_html(), "*预估 LLM 调用: —*", _render_seed_detail([]), gr.update())
        return

    embedding_provider = EmbeddingProviderFactory.create(
        settings_obj.EMBEDDING_PROVIDER, settings_obj
    )

    panel = _render_agent_panel(status, attacker_info, evaluator_info, diagnostic_info)
    yield (panel, "评估开始...", "初始化", gr.update(), gr.update(), gr.update(), _build_experiment_dashboard_html(), "*预估 LLM 调用: 运行中...*", _render_seed_detail([]), gr.update())

    attack_cases = []
    eval_results = []
    diagnosis = {}
    progress_pct = {"attacker": 0, "evaluator": 0, "diagnostician": 0}
    # 预估 attacker 总步数: 种子数 + 种子×攻击数 (用于进度条)
    n_atks = 3 if is_dev else 6
    total_attacker_steps = n_seeds_ans * (1 + n_atks)  # seeds + attacks
    attacker_step = 0
    evaluator_total = 0

    try:
        for step in run_eval_stream(seed_queries, adapter, metrics=selected_metrics, dev_mode=is_dev, doc_dir=doc_dir,
                                     n_seeds_ans=n_seeds_ans, settings=settings_obj, embedding_provider=embedding_provider):
            if "attacker" in step:
                info = step["attacker"] or {}
                st = info.get("status", "")
                if st == "seed_progress":
                    attacker_step += 1
                    pct = min(100, int(attacker_step / max(total_attacker_steps, 1) * 100))
                    progress_pct["attacker"] = pct
                    si, stot = info.get("seed_index", 0), info.get("seed_total", 0)
                    status["attacker"] = "🔄 生成种子"
                    attacker_info = f"第 {si}/{stot} 条"
                elif st == "attack_progress":
                    attacker_step += 1
                    pct = min(100, int(attacker_step / max(total_attacker_steps, 1) * 100))
                    progress_pct["attacker"] = pct
                    si, stot = info.get("seed_index", 0), info.get("seed_total", 0)
                    aname = info.get("attack_name", "")
                    status["attacker"] = "🔄 生成攻击"
                    attacker_info = f"种子{si}/{stot} | {aname}"
                elif st == "done":
                    progress_pct["attacker"] = 100
                    status["attacker"] = "✅ 已完成"
                    attack_cases = info.get("attack_cases", []) or []
                    attacker_info = f"共 {len(attack_cases)} 条样本"
                    evaluator_total = len(attack_cases)
                elif st == "seeds_done":
                    status["attacker"] = "🔄 生成攻击变体"
                    attacker_info = f"共 {info.get('seed_count', 0)} 条种子"
                elif st == "running":
                    status["attacker"] = "🔄 准备中..."

            if "evaluator" in step:
                info = step["evaluator"] or {}
                if info.get("status") == "preparing":
                    d, t = info.get("done", 0), info.get("total", 0)
                    progress_pct["evaluator"] = int(d / max(t, 1) * 50)
                    status["evaluator"] = "📋 预计算基准"
                    evaluator_info = f"种子 {d}/{t} 基准评分中..."
                elif info.get("status") == "running":
                    d, t = info.get("done", 0), info.get("total", 0)
                    progress_pct["evaluator"] = 50 + int(d / max(t, 1) * 50) if t > 0 else 50
                    an = info.get("attack_name", "")
                    status["evaluator"] = "🔄 评分中"
                    evaluator_info = f"{d}/{t} | {an}"
                elif info.get("status") == "done":
                    progress_pct["evaluator"] = 100
                    status["evaluator"] = "✅ 已完成"
                    eval_results = info.get("eval_results", []) or []
                    evaluator_info = f"共 {len(eval_results)} 条评分"

            if "diagnostician" in step:
                info = step["diagnostician"] or {}
                if info.get("status") == "running":
                    progress_pct["diagnostician"] = 50
                    status["diagnostician"] = "🔄 诊断中"
                    diagnostic_info = "分析脆弱点..."
                elif info.get("status") == "done":
                    progress_pct["diagnostician"] = 100
                    status["diagnostician"] = "✅ 已完成"
                    diagnosis = info.get("diagnosis", {}) or {}
                    diagnostic_info = f"鲁棒性 {diagnosis.get('overall_robustness', 0):.3f}"

            panel = _render_agent_panel(status, attacker_info, evaluator_info, diagnostic_info, progress_pct)
            status_text_str = f"🌱 {attacker_info}  |  🔍 {evaluator_info}  |  📊 {diagnostic_info}"
            yield (panel, gr.update(), status_text_str, gr.update(), gr.update(), gr.update(), _build_experiment_dashboard_html(), "*预估 LLM 调用: 运行中...*", _render_seed_detail(eval_results), gr.update())

        final_result = {
            "target": target,
            "mode": mode_text,
            "metrics": selected_metrics,
            "attack_cases": attack_cases,
            "eval_results": eval_results,
            "diagnosis": diagnosis,
        }

        _save_attack_cases(attack_cases)
        _save_system_result(target, diagnosis, eval_results, attack_cases)
        summary_md = _build_summary_md(final_result, diagnosis)
        report_html = _build_experiment_dashboard_html(final_result, target, single=True)
        report_file = _save_report_file(report_html)

        # 计算 LLM 调用量
        n_cases = len(attack_cases)
        n_control = sum(1 for c in attack_cases if c.get("attack_type") == "无攻击（对照组）")
        n_attack = n_cases - n_control
        has_answer = sum(1 for c in attack_cases if c.get("has_answer", True))
        no_answer = n_cases - has_answer
        # 种子 gen: n_seeds 次；攻击 gen: n_attack × ~2/3；评估: 对照=~12/条，攻击=~24/条，无答案=~2/条
        n_seeds = 20 if auto_generate else len(seed_queries)
        llm_seeds = n_seeds
        llm_attack_gen = n_attack * 2 // 3  # 6种攻击中4种用LLM
        llm_eval = n_control * 13 + (n_attack - no_answer // 7 * 7) * 26 + no_answer * 2
        total_llm = llm_seeds + llm_attack_gen + llm_eval
        llm_text = f"*预估 LLM 调用: ~{total_llm} 次 (种子{llm_seeds} + 攻击生成{llm_attack_gen} + 评估{llm_eval})*"

        saved_file = _save_system_result(target, diagnosis, eval_results, attack_cases)
        panel = _render_agent_panel(status, attacker_info, evaluator_info, diagnostic_info)
        new_list = ["（未选择）"] + _list_reports()
        yield (
            panel,
            summary_md,
            "评估完成",
            report_html,
            json.dumps(final_result, ensure_ascii=False, indent=2, default=str),
            report_file,
            _build_experiment_dashboard_html(final_result, target),
            llm_text,
            _render_seed_detail(eval_results),
            gr.update(choices=new_list, value=Path(saved_file).name if saved_file else "（未选择）"),
        )
    except Exception as exc:
        status_msg = f"评估失败: {exc}"
        panel = _render_agent_panel(status, attacker_info, evaluator_info, diagnostic_info)
        yield (panel, status_msg, status_msg, "", "{}", None, _build_experiment_dashboard_html(), "*预估 LLM 调用: 评估失败*", _render_seed_detail([]), gr.update())


def _refresh_seed_detail() -> str:
    """从保存的 results_*.json 读取最新评估，生成种子详细视图"""
    payloads = _read_saved_system_results()
    if not payloads:
        return "<div style='padding:16px;color:#64748b;'>暂无保存的评估结果，请先运行评估。</div>"
    latest = payloads[-1]
    return _render_seed_detail(latest.get("eval_results", []))


def _gen_save_seeds(n_ans: int):
    """生成种子问题并保存到 data/seed_queries.json，供三系统统一复用。
    
    生成规则: n_ans 条可回答 + max(0, 20-n_ans) 条无答案 = 共 20 条。
    """
    import json
    from pathlib import Path

    n_ans = int(n_ans)
    n_unans = max(0, 20 - n_ans)

    yield ("⏳ 正在初始化 LLM...", gr.update(), gr.update())

    try:
        from .config import Settings
        from .services.generation.llm_provider import LLMProviderFactory
        from .agents.attacker import AttackerAgent

        settings_obj = Settings()
        if settings_obj.LLM_PROVIDER == "openai" and not settings_obj.OPENAI_API_KEY:
            yield ("❌ 未配置 OPENAI_API_KEY，无法生成种子。请检查 .env 文件。", gr.update(), gr.update())
            return
        if settings_obj.LLM_PROVIDER == "anthropic" and not settings_obj.ANTHROPIC_API_KEY:
            yield ("❌ 未配置 ANTHROPIC_API_KEY，无法生成种子。请检查 .env 文件。", gr.update(), gr.update())
            return

        llm = LLMProviderFactory.create(settings_obj.LLM_PROVIDER, settings_obj)

        # 定位文档目录
        possible = [
            Path(__file__).resolve().parent / "data" / "documents",
            Path(__file__).resolve().parent.parent / "data" / "documents",
        ]
        doc_dir = ""
        for p in possible:
            if p.exists() and list(p.glob("*.md")):
                doc_dir = str(p)
                break
        if not doc_dir:
            yield ("❌ 未找到 data/documents/ 目录或目录内无 .md 文件，请先准备文档。", gr.update(), gr.update())
            return

        yield (f"⏳ 正在生成 {n_ans} 条可回答 + {n_unans} 条无答案种子（共 {n_ans + n_unans} 条）...", gr.update(), gr.update())

        attacker = AttackerAgent(llm)
        seeds = attacker.generate_seeds(doc_dir, n_answerable=n_ans, n_unanswerable=n_unans)

        # 保存
        Path("data").mkdir(parents=True, exist_ok=True)
        Path("data/seed_queries.json").write_text(
            json.dumps(seeds, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        json_text = json.dumps(seeds, ensure_ascii=False, indent=2)
        n_ans_count = sum(1 for s in seeds if s.get('has_answer', True))
        n_unans_count = sum(1 for s in seeds if not s.get('has_answer', True))
        msg = (
            f"✅ 已生成 **{len(seeds)}** 条种子并保存至 `data/seed_queries.json`。"
            f"（{n_ans_count} 条可回答 + {n_unans_count} 条无答案）\n\n"
            "**下一步**：自动取消勾选已完成 → 分别选择 LlamaIndex / LangChain / RAGFlow 点击开始评估，三系统复用此 JSON。"
        )
        # 同时取消 auto_gen_checkbox，让三系统直接用这份种子 JSON
        yield (msg, gr.update(value=json_text), gr.update(value=False))

    except Exception as exc:
        yield (f"❌ 种子生成失败: {exc}", gr.update(), gr.update())


def create_ui() -> gr.Blocks:
    app_css = """
    .gradio-container {
        max-width: none !important;
        width: 100% !important;
    }
    .gradio-container .main,
    .gradio-container .app,
    .gradio-container .wrap {
        max-width: none !important;
        width: 100% !important;
    }
    """
    with gr.Blocks(title="RAG 鲁棒性评估工具") as app:
        gr.Markdown("## RAG 鲁棒性评估工具")
        gr.Markdown("支持 Demo / LlamaIndex / LangChain 评估与可视化。")

        with gr.Accordion("工作流图", open=False):
            try:
                mermaid = get_graph_mermaid()
                gr.Code(value=mermaid, language="markdown", lines=12, label="Mermaid")
            except Exception:
                gr.Markdown("工作流图加载失败")

        with gr.Row():
            target_selector = gr.Dropdown(
                choices=["LlamaIndex RAG", "LangChain RAG", "RAGFlow"],
                value="LlamaIndex RAG",
                label="目标系统",
            )
            mode_radio = gr.Radio(
                choices=["开发模式（快速验证，约 8 条用例）", "正式模式（完整评估，100+ 条用例）"],
                value="开发模式（快速验证，约 8 条用例）",
                label="运行模式",
            )

        metric_checkboxes = gr.CheckboxGroup(
            choices=_METRIC_CHOICES,
            value=_METRIC_CHOICES,
            label="评估指标",
        )
        auto_gen_checkbox = gr.Checkbox(
            value=True, label="自动从 data/documents/ 生成种子问题（勾选时忽略下方 JSON）",
        )
        seed_count_slider = gr.Slider(
            minimum=1, maximum=20, value=15, step=1,
            label="可回答种子数量（推荐 15，配合 5 条无答案 = 共 20 条）",
        )
        gr.Markdown(
            "**三系统统一种子流程**：\n"
            "1. 设置可回答数量（推荐 15） \n"
            "2. 点击下方按钮生成并保存 \n"
            "3. 自动取消勾选上方自动生成复选框 \n"
            "4. 分别选择 LlamaIndex / LangChain / RAGFlow 依次评估"
        )
        with gr.Row():
            gen_seed_btn = gr.Button("🌱 生成并保存 20 条共享种子 → data/seed_queries.json", variant="secondary", scale=3)
        gen_seed_status = gr.Markdown("*尚未生成。点击上方按钮，用 LLM 从 data/documents/ 自动生成种子并保存，三系统统一复用。*")
        seed_input = gr.Code(value=_DEFAULT_SEED, language="json", lines=10, label="种子问题 JSON（取消勾选上方复选框后生效）")

        def _estimate_llm(seeds_ans, n_metrics, is_dev):
            """估算 LLM 调用量"""
            n_seeds_noans = 0 if is_dev else 5
            n_seeds = seeds_ans + n_seeds_noans
            n_attacks = 3 if is_dev else 6
            n_cases = n_seeds * (1 + n_attacks)
            n_attack_cases = n_seeds * n_attacks
            # 种子生成
            seed_llm = n_seeds
            # 攻击生成（约 4/6 用 LLM）
            attack_gen_llm = n_attack_cases * 2 // 3
            # 可回答 case 评估: 对照 ~(1 + n_metrics*4), 攻击 ~(2 + 2*n_metrics*4)
            eval_per_ctrl = 1 + n_metrics * 4
            eval_per_atk = 2 + 2 * n_metrics * 4
            eval_llm = seeds_ans * eval_per_ctrl + seeds_ans * n_attacks * eval_per_atk
            # 无答案 case: 对照~(1), 攻击~(2)
            eval_llm += n_seeds_noans * 1 + n_seeds_noans * n_attacks * 2
            total = seed_llm + attack_gen_llm + eval_llm
            return (
                f"*预估 LLM 调用: ~{total} 次"
                f"（种子生成 {seed_llm} + 攻击生成 {attack_gen_llm} + 评估 {eval_llm}）*"
                f"\n*{n_seeds} 种子 × {n_attacks} 攻击 = {n_cases} case*"
            )

        llm_estimate = gr.Markdown(value=_estimate_llm(1, len(_METRIC_CHOICES), True))

        # 实时更新 LLM 预估
        def _update_estimate(seeds, metrics, is_dev):
            return _estimate_llm(int(seeds), len(metrics), is_dev)
        seed_count_slider.change(_update_estimate, [seed_count_slider, metric_checkboxes, mode_radio], [llm_estimate])
        metric_checkboxes.change(_update_estimate, [seed_count_slider, metric_checkboxes, mode_radio], [llm_estimate])
        mode_radio.change(_update_estimate, [seed_count_slider, metric_checkboxes, mode_radio], [llm_estimate])

        run_btn = gr.Button("开始评估", variant="primary")

        with gr.Accordion("Agent 状态", open=True):
            agent_panel = gr.HTML()
            status_text = gr.Textbox(label="状态", interactive=False)

        with gr.Accordion("摘要", open=True):
            summary_md = gr.Markdown()

        with gr.Accordion("HTML 报告", open=False):
            report_html = gr.HTML()

        # ── 历史报告分析 ──
        gr.Markdown("---\n### 历史报告分析")
        with gr.Row():
            report_selector = gr.Dropdown(
                choices=["（未选择）"] + _list_reports(),
                value="（未选择）",
                label="选择已保存的评估报告",
                scale=3,
            )
            refresh_list_btn = gr.Button("刷新列表", scale=1)
            download_dashboard_btn = gr.Button("📥 下载看板 HTML", scale=1)

        with gr.Accordion("实验结果可视化看板", open=False):
            dashboard_html = gr.HTML(value=_build_experiment_dashboard_html())

        with gr.Accordion("种子问题详细结果", open=False):
            seed_detail_html = gr.HTML()

        def _on_select_report(filename):
            if not filename or filename == "（未选择）":
                return _build_experiment_dashboard_html(), "", "{}", "### 评估摘要\n*请选择报告*"
            payload = _load_report_by_name(filename)
            if payload is None:
                return _build_experiment_dashboard_html(), "", "{}", "### 评估摘要\n*报告加载失败*"
            summary = _build_summary_md({"eval_results": payload.get("eval_results", [])}, payload)
            llm_diag = payload.get("llm_diagnosis", "")
            if llm_diag:
                summary += f"\n\n---\n### 🤖 LLM 深度诊断\n{llm_diag}"
            return (_build_experiment_dashboard_html(payload, payload.get("system", ""), single=True),
                    _render_seed_detail(payload.get("eval_results", [])),
                    json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    summary)

        def _download_dashboard(filename):
            if filename and filename != "（未选择）":
                payload = _load_report_by_name(filename)
                if payload:
                    html = _build_experiment_dashboard_html(payload, payload.get("system", ""), single=True)
                else:
                    html = _build_experiment_dashboard_html()
            else:
                html = _build_experiment_dashboard_html()
            return _save_report_file(html)

        def _on_refresh_list():
            choices = ["（未选择）"] + _list_reports()
            return gr.update(choices=choices, value="（未选择）")

        refresh_list_btn.click(_on_refresh_list, [], [report_selector])

        with gr.Accordion("原始 JSON 输出", open=False):
            detail_json = gr.Code(language="json", lines=20)

        report_selector.change(_on_select_report, [report_selector], [dashboard_html, seed_detail_html, detail_json, summary_md])

        download_report = gr.File(label="下载 HTML 报告", visible=True, interactive=False)

        download_dashboard_btn.click(_download_dashboard, [report_selector], [download_report])

        run_btn.click(
            fn=run_eval_with_viz,
            inputs=[seed_input, metric_checkboxes, mode_radio, target_selector, auto_gen_checkbox, seed_count_slider],
            outputs=[agent_panel, summary_md, status_text, report_html, detail_json, download_report, dashboard_html, llm_estimate, seed_detail_html, report_selector],
        )

        gen_seed_btn.click(
            fn=_gen_save_seeds,
            inputs=[seed_count_slider],
            outputs=[gen_seed_status, seed_input, auto_gen_checkbox],
        )

    app.launch(server_name="127.0.0.1", server_port=7860, css=app_css)


def main():
    create_ui()


if __name__ == "__main__":
    main()
