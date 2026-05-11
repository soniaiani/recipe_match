from functools import lru_cache
from supabase import create_client, Client
from app.config import get_settings


@lru_cache
def get_supabase() -> Client:
    """Anon client — used for auth and RLS-protected operations."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)


@lru_cache
def get_supabase_admin() -> Client:
    """Service-role client — bypasses RLS, used for admin operations."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_key)
