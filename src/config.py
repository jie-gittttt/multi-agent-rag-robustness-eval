from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Literal, Optional

_env_path = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # 应用
    APP_NAME: str = "RAG Robustness Eval"
    DEBUG: bool = True
    DATA_DIR: str = "./data"

    # LLM — openai / anthropic 均可，DeepSeek 通过 OPENAI_BASE_URL 接入
    LLM_PROVIDER: Literal["openai", "anthropic"] = "openai"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # 嵌入 — 默认本地 BGE，评估工具通常不需要向量库（由被评估 RAG 提供检索）
    EMBEDDING_PROVIDER: Literal["openai", "local"] = "local"
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # RAGFlow (HTTP API)
    RAGFLOW_BASE_URL: Optional[str] = None
    RAGFLOW_API_KEY: Optional[str] = None
    RAGFLOW_KB_ID: Optional[str] = None
    RAGFLOW_RETRIEVAL_PATH: str = "/api/v1/retrieval"
    RAGFLOW_CHAT_PATH: str = "/api/v1/chat/completions"
    RAGFLOW_MODEL: str = "deepseek-chat"
    RAGFLOW_TIMEOUT_SEC: float = 60.0

    # 检索参数（仅当评估工具本身需要检索时使用）
    RETRIEVAL_TOP_K: int = 5
    RRF_K: int = 60

    model_config = {"env_file": str(_env_path), "env_file_encoding": "utf-8", "extra": "ignore"}
