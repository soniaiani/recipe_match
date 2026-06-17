from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    supabase_url: str
    supabase_key: str
    supabase_service_key: str
    secret_key: str
    environment: str = "development"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash-001"
    gemini_fallback_models: str = "gemini-2.5-flash-lite"
    gemini_timeout_seconds: float = 30.0

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
