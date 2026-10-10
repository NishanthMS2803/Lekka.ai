"""PostgreSQL tool cache with fact-based TTL policies and tool decorator."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from lekka_ai.db.models import ToolCache
from lekka_ai.db.session import get_session_factory
from lekka_ai.logging import get_logger

logger = get_logger(__name__)

TTL_POLICIES: dict[str, timedelta | None] = {
    "price": timedelta(minutes=5),
    "news": timedelta(hours=6),
    "estimates": timedelta(hours=24),
    "financials": timedelta(days=7),
    "filing_doc": None,  # never expires
}


def normalize_input(data: Any) -> Any:
    """Recursively normalize inputs into a canonical, sort-stable structure."""
    if isinstance(data, BaseModel):
        return normalize_input(data.model_dump(mode="json"))
    if isinstance(data, dict):
        return {str(k): normalize_input(v) for k, v in sorted(data.items())}
    if isinstance(data, (list, tuple)):
        return [normalize_input(v) for v in data]
    return data


def build_cache_key(tool_name: str, input_data: Any) -> str:
    """Build a deterministic cache key from tool name and normalized input."""
    normalized = normalize_input(input_data)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return f"{tool_name}:{serialized}"


class Cache:
    """PostgreSQL-backed tool cache supporting fact-based TTL policies."""

    def __init__(
        self,
        session: Session | None = None,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self.session = session
        self.session_factory = session_factory

    def _resolve_session(
        self, explicit_session: Session | None
    ) -> tuple[Session, bool]:
        """Resolve database session, returning (session, should_close_flag)."""
        if explicit_session is not None:
            return explicit_session, False
        if self.session is not None:
            return self.session, False
        factory = self.session_factory or get_session_factory()
        return factory(), True

    def calculate_expiration(
        self,
        fact_type: str,
        fetched_at: datetime | None = None,
    ) -> datetime | None:
        """Calculate expires_at timestamp based on fact_type TTL policy."""
        base_time = fetched_at or datetime.now(UTC)
        if fact_type not in TTL_POLICIES:
            valid = list(TTL_POLICIES.keys())
            raise ValueError(
                f"Unknown fact_type '{fact_type}'. Expected one of: {valid}"
            )
        ttl = TTL_POLICIES[fact_type]
        if ttl is None:
            return None
        return base_time + ttl

    def get(
        self,
        key: str,
        session: Session | None = None,
    ) -> Any | None:
        """Retrieve value by key if present and not expired."""
        sess, should_close = self._resolve_session(session)
        try:
            record = sess.get(ToolCache, key)
            if record is None:
                return None

            now = datetime.now(UTC)
            if record.expires_at is not None and record.expires_at <= now:
                logger.debug(
                    "Cache entry expired",
                    key=key,
                    expires_at=record.expires_at,
                )
                return None

            return record.value_json
        finally:
            if should_close:
                sess.close()

    def set(
        self,
        key: str,
        value: Any,
        fact_type: str,
        fetched_at: datetime | None = None,
        expires_at: datetime | None = None,
        session: Session | None = None,
    ) -> ToolCache:
        """Store or update value in cache using fact_type TTL policy."""
        sess, should_close = self._resolve_session(session)
        try:
            now = fetched_at or datetime.now(UTC)
            calc_expires_at = (
                expires_at
                if expires_at is not None
                else self.calculate_expiration(fact_type, now)
            )

            val_json = (
                value.model_dump(mode="json") if isinstance(value, BaseModel) else value
            )

            existing = sess.get(ToolCache, key)
            if existing is not None:
                existing.value_json = val_json
                existing.fact_type = fact_type
                existing.fetched_at = now
                existing.expires_at = calc_expires_at
                record = existing
            else:
                record = ToolCache(
                    key=key,
                    value_json=val_json,
                    fact_type=fact_type,
                    fetched_at=now,
                    expires_at=calc_expires_at,
                )
                sess.add(record)

            sess.commit()
            sess.refresh(record)
            return record
        finally:
            if should_close:
                sess.close()

    def delete(self, key: str, session: Session | None = None) -> bool:
        """Delete a cache entry by key, returning True if deleted."""
        sess, should_close = self._resolve_session(session)
        try:
            record = sess.get(ToolCache, key)
            if record is not None:
                sess.delete(record)
                sess.commit()
                return True
            return False
        finally:
            if should_close:
                sess.close()


def cached_tool(
    cache: Cache,
    fact_type: str,
    key_builder: Callable[[str, Any], str] | None = None,
) -> Callable[[Any], Any]:
    """Decorator for caching Tool executions based on tool name and normalized input."""

    def decorator(target: Any) -> Any:
        target_any: Any = target
        if isinstance(target, type) and hasattr(target, "run"):
            original_run = target_any.run

            @wraps(original_run)
            async def wrapped_class_run(
                self: Any, input_data: Any, *args: Any, **kwargs: Any
            ) -> Any:
                tool_name = getattr(self, "name", self.__class__.__name__)
                output_model = getattr(self, "output_model", None)

                cache_key = (
                    key_builder(tool_name, input_data)
                    if key_builder
                    else build_cache_key(tool_name, input_data)
                )

                cached_val = cache.get(cache_key)
                if cached_val is not None:
                    if (
                        output_model
                        and isinstance(cached_val, dict)
                        and issubclass(output_model, BaseModel)
                    ):
                        return output_model.model_validate(cached_val)
                    return cached_val

                result = await original_run(self, input_data, *args, **kwargs)
                cache.set(cache_key, result, fact_type=fact_type)
                return result

            target_any.run = wrapped_class_run
            return target

        original_fn = getattr(target, "run", target)

        @wraps(original_fn)
        async def wrapped_method(*args: Any, **kwargs: Any) -> Any:
            if len(args) >= 2:
                self_obj, input_data = args[0], args[1]
                tool_name = getattr(self_obj, "name", self_obj.__class__.__name__)
                output_model = getattr(self_obj, "output_model", None)
            elif len(args) == 1:
                self_obj = None
                input_data = args[0]
                tool_name = getattr(target, "name", original_fn.__name__)
                output_model = getattr(target, "output_model", None)
            else:
                return await original_fn(*args, **kwargs)

            cache_key = (
                key_builder(tool_name, input_data)
                if key_builder
                else build_cache_key(tool_name, input_data)
            )

            cached_val = cache.get(cache_key)
            if cached_val is not None:
                if (
                    output_model
                    and isinstance(cached_val, dict)
                    and issubclass(output_model, BaseModel)
                ):
                    return output_model.model_validate(cached_val)
                return cached_val

            result = await original_fn(*args, **kwargs)
            cache.set(cache_key, result, fact_type=fact_type)
            return result

        if hasattr(target, "run") and not isinstance(target, type):
            target_any.run = wrapped_method
            return target

        return wrapped_method

    return decorator
