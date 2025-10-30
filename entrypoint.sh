#!/bin/bash
set -e

# Wait for database to be ready
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "db" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - executing commands"

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Wait for migrations to settle
echo "Waiting for migrations to settle..."
sleep 2

# Start Celery worker in background if enabled
if [ "$ENABLE_CELERY" = "1" ]; then
  echo "Starting Celery worker"
  celery -A app.task.celery_app worker --loglevel=info &
fi

# Start uvicorn
echo "Starting Uvicorn server"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1