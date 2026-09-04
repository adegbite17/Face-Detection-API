import json
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


class Cache:
    """
    Unified Cache class for async and sync PostgreSQL access.

    - use_async=True: use async engine (for FastAPI async endpoints)
    - use_async=False: use sync engine (for threadpool or sync workers)
    """

    def __init__(self, database_url: str, use_async: bool = True):
        self.database_url = database_url
        self.use_async = use_async

        if use_async:
            # Async engine for async routes
            self.async_engine = create_async_engine(database_url, echo=False)
            self.async_session = sessionmaker(
                self.async_engine, expire_on_commit=False, class_=AsyncSession
            )
            self.sync_engine = None
        else:
            # Sync engine for threadpool/worker
            self.async_engine = None
            self.async_session = None
            self.sync_engine = create_engine(
                database_url.replace("+asyncpg", ""),  # e.g., "postgresql://..."
                future=True,
                echo=False
            )

    # ---------- Database Initialization ----------

    async def init_db(self):
        """Async DB table creation (async usage only)."""
        if not self.use_async:
            raise RuntimeError("init_db() called on sync Cache instance")
        async with self.async_engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS mask_cache (
                    phash TEXT PRIMARY KEY,
                    mask_json JSONB
                );
            """))
            await conn.commit()

    def init_db_sync(self):
        """Sync DB table creation (sync usage only)."""
        if self.use_async:
            raise RuntimeError("init_db_sync() called on async Cache instance")
        with self.sync_engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS mask_cache (
                    phash TEXT PRIMARY KEY,
                    mask_json JSONB
                );
            """))

    # ---------- Async cache operations ----------

    async def get(self, phash_str: str):
        """Retrieve cached mask JSON (async)."""
        if not self.use_async:
            raise RuntimeError("get() called on sync Cache instance")
        async with self.async_session() as s:
            query = text("SELECT mask_json FROM mask_cache WHERE phash = :p LIMIT 1")
            result = await s.execute(query, {"p": phash_str})
            row = result.fetchone()
            if row:
                return json.loads(row[0])
            return None

    async def set(self, phash_str: str, result: dict):
        """Store or update cached mask JSON (async)."""
        if not self.use_async:
            raise RuntimeError("set() called on sync Cache instance")
        async with self.async_session() as s:
            mask_json = json.dumps(result)
            query = text("""
                INSERT INTO mask_cache (phash, mask_json)
                VALUES (:p, :m)
                ON CONFLICT(phash)
                DO UPDATE SET mask_json = excluded.mask_json
            """)
            await s.execute(query, {"p": phash_str, "m": mask_json})
            await s.commit()

    # ---------- Sync cache operations ----------

    def get_sync(self, phash_str: str):
        """Retrieve cached mask JSON (sync)."""
        if self.use_async:
            raise RuntimeError("get_sync() called on async Cache instance")
        with self.sync_engine.begin() as conn:
            result = conn.execute(
                text("SELECT mask_json FROM mask_cache WHERE phash = :p LIMIT 1"),
                {"p": phash_str}
            )
            row = result.fetchone()
            if row:
                # Check if the result is already a dict
                if isinstance(row[0], dict):
                    return row[0]
                # Otherwise parse JSON string
                return json.loads(row[0])
            return None

    def set_sync(self, phash_str: str, result: dict):
        """Store or update cached mask JSON (sync)."""
        if self.use_async:
            raise RuntimeError("set_sync() called on async Cache instance")
        with self.sync_engine.begin() as conn:
            # Ensure result is JSON serialized if it's not already a string
            mask_json = result if isinstance(result, str) else json.dumps(result)
            conn.execute(text("""
                INSERT INTO mask_cache (phash, mask_json)
                VALUES (:p, :m)
                ON CONFLICT(phash)
                DO UPDATE SET mask_json = excluded.mask_json
            """), {"p": phash_str, "m": mask_json})
