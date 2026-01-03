from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    log_level: str = "INFO"


class TelegramConfig(BaseModel):
    bot_username: str = ""
    token: str = Field(default="", repr=False)


class MCPConfig(BaseModel):
    url: str = "https://mcp001.vkusvill.ru/mcp"


class DBConfig(BaseModel):
    path: str = "../vkusvill.db"


class VectorConfig(BaseModel):
    index_path: str = "data/products.faiss"
    embedding_model: str = "qwen/qwen3-embedding-8b"
    candidate_pool: int = 200
    fts_boost: bool = True


class LLMConfig(BaseModel):
    provider: str = "openrouter"
    model: str = "qwen/qwen3-235b-a22b:free"
    http_referer: str = "https://openrouter.ai"
    title: str = "VkusVill Bot"
    provider_order: str = ""
    api_key: str = Field(default="", repr=False)
    proxy_url: str | None = Field(default=None, repr=False)


class SgrConfig(BaseModel):
    max_steps: int = 8
    max_items_per_search: int = 10
    temperature: float = 0.4
    history_messages: int = 8
    local_fresh_hours: int = 24
    use_mcp_refresh: bool = True


class Settings(BaseModel):
    app: AppConfig = AppConfig()
    telegram: TelegramConfig = TelegramConfig()
    mcp: MCPConfig = MCPConfig()
    db: DBConfig = DBConfig()
    vector: VectorConfig = VectorConfig()
    llm: LLMConfig = LLMConfig()
    sgr: SgrConfig = SgrConfig()

    @classmethod
    def load(cls, path: str | None = None) -> "Settings":
        env_path = os.getenv("ENV_PATH", ".env")
        load_dotenv(env_path)
        config_path = Path(path or os.getenv("CONFIG_PATH", "config.yaml"))
        data: dict[str, Any] = {}
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        telegram = data.get("telegram", {})
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            telegram["token"] = token
        data["telegram"] = telegram

        db = data.get("db", {})
        db_path = os.getenv("DB_PATH")
        if db_path:
            db["path"] = db_path
        data["db"] = db

        vector = data.get("vector", {})
        index_path = os.getenv("VECTOR_INDEX_PATH")
        if index_path:
            vector["index_path"] = index_path
        embedding_model = os.getenv("OPENROUTER_EMBEDDING_MODEL")
        if embedding_model:
            vector["embedding_model"] = embedding_model
        data["vector"] = vector

        llm = data.get("llm", {})
        llm_api_key = os.getenv("OPENROUTER_API_KEY")
        if llm_api_key:
            llm["api_key"] = llm_api_key
        llm_proxy = os.getenv("OPENROUTER_PROXY_URL")
        if llm_proxy:
            llm["proxy_url"] = llm_proxy
        data["llm"] = llm

        return cls(**data)
