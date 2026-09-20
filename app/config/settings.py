from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_PROJECT_ROOT / ".env", extra="ignore")

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    CHINOOK_DB_PATH: str = str(_PROJECT_ROOT / "data" / "chinook.db")


@lru_cache
def get_settings() -> Settings:
    return Settings()
