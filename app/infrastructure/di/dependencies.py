from functools import lru_cache

from app.config.settings import get_settings
from app.infrastructure.adapters.openai_llm_adapter import OpenAiLlmAdapter
from app.infrastructure.adapters.sqlite_adapter import SqliteAdapter


@lru_cache
def get_llm_adapter() -> OpenAiLlmAdapter:
    return OpenAiLlmAdapter()


@lru_cache
def get_sql_executor() -> SqliteAdapter:
    return SqliteAdapter(db_path=get_settings().CHINOOK_DB_PATH)
