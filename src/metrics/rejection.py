"""拒答率与幻觉率评估 — 用于无答案 case 的独立评估路径"""

import re

_REJECTION_PATTERNS = [
    r"无法回答", r"没有.{0,5}信息", r"无法提供", r"对不起",
    r"不.{0,3}清楚", r"未提及", r"信息不足", r"无法确定",
    r"找不到", r"无法.{0,3}判断", r"文档中没有", r"据.{0,5}所知.{0,3}没有",
    r"抱歉", r"无相关", r"不能确定", r"资料不足",
]


def is_rejection(answer: str) -> bool:
    """判断回答是否为拒答"""
    for pattern in _REJECTION_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return True
    return False


def evaluate_rejection(answer: str, has_answer: bool) -> dict:
    """评估单条回答的拒答/幻觉表现。

    Returns:
        {
            "is_rejection": bool,
            "correct_behavior": bool,
            "hallucination_risk": bool,
            "verdict": str,
        }
    """
    rejected = is_rejection(answer)

    if has_answer:
        correct = not rejected
        hallucination = False
        verdict = "正确回答" if correct else "错误拒答"
    else:
        correct = rejected
        hallucination = not rejected
        verdict = "正确拒答" if correct else "幻觉——编造了答案"

    return {
        "is_rejection": rejected,
        "correct_behavior": correct,
        "hallucination_risk": hallucination,
        "verdict": verdict,
    }


def compute_rejection_rate(results: list[dict]) -> float:
    """计算拒答率（仅适用于无答案 case）"""
    no_answer_results = [
        r for r in results
        if r.get("case", {}).get("has_answer") is False
    ]
    if not no_answer_results:
        return 1.0
    correct = sum(
        1 for r in no_answer_results
        if r.get("normal_rejection", {}).get("correct_behavior")
    )
    return correct / len(no_answer_results)


def compute_hallucination_rate(results: list[dict]) -> float:
    """计算幻觉率（仅适用于无答案 case）"""
    no_answer_results = [
        r for r in results
        if r.get("case", {}).get("has_answer") is False
    ]
    if not no_answer_results:
        return 0.0
    hallucinated = sum(
        1 for r in no_answer_results
        if r.get("normal_rejection", {}).get("hallucination_risk")
    )
    return hallucinated / len(no_answer_results)
