import os
import aiosqlite
from pathlib import Path
from contextlib import asynccontextmanager

# Define DB path in the .shipai directory (or local dir for dev)
DB_DIR = Path(os.getenv("SHIPAI_DATA_DIR", "."))
DB_PATH = DB_DIR / "shipai.db"

async def init_db():
    """Initialize the SQLite database with required tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                display_name TEXT,
                tier TEXT DEFAULT 'free',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                template TEXT NOT NULL,
                config_json TEXT,
                status TEXT DEFAULT 'created',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        await db.commit()

@asynccontextmanager
async def get_db():
    """Async context manager for DB connections."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()
