"""抗干扰性评分 — 计算正常 vs 对抗配对评估的鲁棒性指标"""


def compute_robustness(normal_scores: dict, attack_scores: dict) -> dict:
    """输入正常和对抗模式下的四维分数，输出鲁棒性指标。

    Args:
        normal_scores: {"faithfulness": 0.85, "answer_relevancy": 0.72, ...}
        attack_scores: {"faithfulness": 0.42, ...}

    Returns:
        {
            "per_metric": {"faithfulness": {"normal": 0.85, "attack": 0.42, "delta": 0.43}},
            "robustness_score": 0.57,   # 1 - mean(delta)
            "rating": "⚠ 需要加固",
        }
    """
    deltas = {}
    for key in normal_scores:
        n = normal_scores.get(key, 0.0)
        a = attack_scores.get(key, 0.0)
        deltas[key] = {
            "normal": round(n, 4),
            "attack": round(a, 4),
            "delta": round(n - a, 4),
        }

    # 退化强度：仅统计“攻击后下降”的部分
    mean_delta = sum(max(d["delta"], 0.0) for d in deltas.values()) / max(len(deltas), 1)

    # 基线质量：正常模式下四维平均分。
    # 仅看 1-mean_delta 会导致“正常全 0 但攻击也不更差”得到高鲁棒分，
    # 因此引入基线因子，避免失真 PASS。
    baseline = sum(max(float(v), 0.0) for v in normal_scores.values()) / max(len(normal_scores), 1)
    stability = max(0.0, 1.0 - mean_delta)
    robustness = round(stability * baseline, 4)

    if robustness >= 0.8:
        rating = "[PASS] 鲁棒"
    elif robustness >= 0.5:
        rating = "[WARN] 需要加固"
    else:
        rating = "[FAIL] 严重脆弱"

    return {
        "per_metric": deltas,
        "robustness_score": robustness,
        "baseline_score": round(baseline, 4),
        "stability_score": round(stability, 4),
        "mean_delta": round(mean_delta, 4),
        "rating": rating,
    }
