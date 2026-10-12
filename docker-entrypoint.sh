#!/usr/bin/env bash
set -e

echo "==> Running database migrations via Alembic..."
MAX_RETRIES=15
RETRY_COUNT=0

until alembic upgrade head || [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; do
  echo "Database migration attempt failed or database not ready. Retrying in 2s... ($((RETRY_COUNT + 1))/$MAX_RETRIES)"
  RETRY_COUNT=$((RETRY_COUNT + 1))
  sleep 2
done

if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
  echo "Error: Failed to apply database migrations after $MAX_RETRIES retries."
  exit 1
fi

echo "==> Database migrations applied successfully."
echo "==> Starting application: $*"
exec "$@"
