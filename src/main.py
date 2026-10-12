"""CLI 入口 — rag-eval 命令行工具"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 必须在所有模型加载前设置，阻止 HuggingFace 联网
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _persist_system_result(target: str, result: dict) -> Path:
    """Write a dashboard-compatible system result file under data/results_*.json."""
    diagnosis = result.get("diagnosis", {}) or {}
    eval_results = result.get("eval_results", []) or []
    attack_cases = result.get("attack_cases", []) or []

    Path("data").mkdir(parents=True, exist_ok=True)
    name = str(target or "unknown").strip().lower()
    out = Path(f"data/results_{name}.json")
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

    attack_out = Path("data/attack_cases.json")
    attack_out.write_text(json.dumps(attack_cases, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="RAG 鲁棒性评估工具 — Multi-Agent 对抗攻击 + RAGAS 评分 + 诊断",
    )
    sub = parser.add_subparsers(dest="command")

    # rag-eval run
    run_parser = sub.add_parser("run", help="运行评估流水线")
    run_parser.add_argument("--seed", default="data/seed_queries.json",
                           help="种子问题集 JSON 路径")
    run_parser.add_argument("--target", default="demo",
                           choices=["demo", "cement_lca", "llamaindex", "langchain", "ragflow"],
                           help="目标 RAG 系统")
    run_parser.add_argument("--output", default="-",
                           help="输出 JSON 路径，默认 stdout")

    # rag-eval ui
    sub.add_parser("ui", help="启动 Gradio Web UI")

    args = parser.parse_args()

    if args.command == "ui":
        from .app import create_ui
        create_ui()
        return

    if args.command == "run":
        from .config import Settings
        from .services.embedding.factory import EmbeddingProviderFactory

        seed_path = Path(args.seed)
        if not seed_path.exists():
            print(f"种子问题集不存在: {args.seed}", file=sys.stderr)
            sys.exit(1)

        seed_queries = json.loads(seed_path.read_text(encoding="utf-8"))

        if args.target == "demo":
            from .adapters.demo_adapter import DemoRAGAdapter
            adapter = DemoRAGAdapter()
        elif args.target == "cement_lca":
            from .app import _build_cement_adapter
            adapter = _build_cement_adapter()
        elif args.target == "llamaindex":
            from .adapters.llamaindex_stub import LlamaIndexRAGAdapter
            adapter = LlamaIndexRAGAdapter()
        elif args.target == "langchain":
            from .adapters.langchain_stub import LangChainRAGAdapter
            adapter = LangChainRAGAdapter()
        elif args.target == "ragflow":
            from .adapters.ragflow_adapter import RAGFlowAdapter
            adapter = RAGFlowAdapter()
        else:
            print(f"未知目标系统: {args.target}", file=sys.stderr)
            sys.exit(1)

        settings_obj = Settings()
        embedding_provider = EmbeddingProviderFactory.create(
            settings_obj.EMBEDDING_PROVIDER, settings_obj
        )

        from .graph.eval_graph import run_eval_pipeline
        print("正在运行评估流水线...")
        result = run_eval_pipeline(
            seed_queries,
            adapter,
            settings=settings_obj,
            embedding_provider=embedding_provider,
        )

        output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        if args.output == "-":
            print(output)
        else:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"报告已写入 {args.output}")

        system_file = _persist_system_result(args.target, result)
        print(f"系统结果已写入 {system_file}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
