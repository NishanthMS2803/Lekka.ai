FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
ARG UID=1001
ARG GID=1001
RUN groupadd -g "${GID}" appuser && \
    useradd -u "${UID}" -g appuser -m -s /bin/bash appuser

WORKDIR /app

# Configure uv and python environment
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source, migrations, and evaluation assets
COPY alembic.ini alembic.ini
COPY alembic alembic
COPY src src
COPY eval eval
COPY prompts prompts
COPY README.md README.md
COPY docker-entrypoint.sh /docker-entrypoint.sh

# Install project into virtual environment
RUN uv sync --frozen --no-dev

# Set permissions for non-root user
RUN chmod +x /docker-entrypoint.sh && \
    chown -R appuser:appuser /app /docker-entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "lekka_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
