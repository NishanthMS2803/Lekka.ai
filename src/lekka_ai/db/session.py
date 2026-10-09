"""Database engine and session factory."""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from lekka_ai.config import get_settings


@lru_cache
def get_engine(database_url: str | None = None) -> Engine:
    """Return cached SQLAlchemy engine."""
    url = database_url if database_url is not None else get_settings().DATABASE_URL
    return create_engine(
        url,
        pool_pre_ping=True,
        echo=False,
    )


def get_session_factory(
    engine: Engine | None = None,
) -> sessionmaker[Session]:
    """Return a sessionmaker instance bound to an engine."""
    target_engine = engine if engine is not None else get_engine()
    return sessionmaker(
        bind=target_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session scope."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
