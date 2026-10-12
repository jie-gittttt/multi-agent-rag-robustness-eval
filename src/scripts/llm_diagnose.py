"""独立脚本：读取已有 results_*.json，用 LLM 生成深度诊断报告。"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import Settings
from services.generation.llm_provider import LLMProviderFactory

RESULT_FILE = sys.argv[1] if len(sys.argv) > 1 else "data/results_llamaindex_20260614_183445.json"


def main():
    with open(RESULT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    eval_results = data.get("eval_results", [])
    per_attack = data.get("per_attack_type", {})
    system_name = data.get("system", "Unknown")

    # ── 构造 prompt ──
    # 1. 聚合表格
    table_lines = []
    for at, info in per_attack.items():
        ds = info["avg_deltas"]
        table_lines.append(
            f"| {at} | rb={info['avg_robustness']:.3f} | "
            f"faith={ds['faithfulness']:+.3f} ans={ds['answer_relevancy']:+.3f} "
            f"prec={ds['context_precision']:+.3f} rec={ds['context_recall']:+.3f} |"
        )

    # 2. 每种攻击取 2 条代表性 case（delta 最大的）
    sample_text = ""
    for at in per_attack:
        cases = [r for r in eval_results if r["case"].get("attack_type") == at]
        cases.sort(
            key=lambda r: (r.get("robustness") or {}).get("robustness_score", 1.0)
        )
        for c in cases[:2]:
            n = c.get("normal", {})
            a = c.get("attack", {})
            sample_text += (
                f"\n[{at}]\n"
                f"  原始问题: {c['case']['original_query'][:100]}\n"
                f"  对抗问题: {c['case']['perturbed_query'][:100]}\n"
                f"  正常回答: {n.get('answer', '')[:150]}\n"
                f"  对抗回答: {a.get('answer', '')[:150]}\n"
                f"  正常分: {n.get('scores', {})}\n"
                f"  对抗分: {a.get('scores', {})}\n"
            )

    prompt = f"""你是一个 RAG 系统安全评估专家。以下是针对"{system_name}"的对抗攻击评估数据。

## 按攻击类型聚合的指标退化（平均 delta）

{chr(10).join(table_lines)}

## 每种攻击的典型失败案例

{sample_text}

## 任务

请基于以上数据，完成以下分析。**必须覆盖全部 6 种攻击类型，不得省略任何一种。**

1. **逐攻击类型诊断**：对下面列出的每种攻击类型单独分析其脆弱根因和受影响指标。必须覆盖：关键词替换、语义干扰、上下文劫持、先验冲突、无关噪声注入、排序污染。每种攻击至少写 2-3 句话，说明其具体影响机制。

2. **跨攻击模式归纳**：哪些攻击类型有相同的根因？哪些根因是独立的？

3. **按优先级排序的修复建议**（必须包含）：给出 3-5 条具体可操作的修复建议，按优先级从高到低排列。每条必须包含：
   - 建议标题
   - 针对的攻击类型
   - 具体措施
   - 预期效果

4. **系统整体评估**：用 2-3 句话总结该 RAG 系统的鲁棒性状况。

请用中文输出，结构化格式，不要截断任何内容。"""

    # ── 调用 LLM ──
    settings = Settings()
    llm = LLMProviderFactory.create(settings.LLM_PROVIDER, settings)
    print(f"正在调用 {settings.OPENAI_MODEL} 进行诊断分析...\n")
    response = llm.generate([{"role": "user", "content": prompt}], stream=False, max_tokens=4096)

    # ── 写回 JSON ──
    data["llm_diagnosis"] = response.strip()
    out_json = Path(RESULT_FILE)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(response.strip())
    print(f"\n\n[OK] LLM diagnosis written to: {out_json}")


if __name__ == "__main__":
    main()
