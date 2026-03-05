"""Database setup utilities for the Chinook SQLite dataset."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
import zipfile

import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

DEFAULT_CHINOOK_ZIP_URL = "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip"


def get_engine_for_chinook_db(zip_url: str = DEFAULT_CHINOOK_ZIP_URL) -> Engine:
    """Download Chinook SQLite DB and load it into an in-memory engine."""
    response = requests.get(zip_url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        db_members = [name for name in archive.namelist() if name.lower().endswith(".db")]
        if not db_members:
            raise ValueError("No SQLite database file found in Chinook archive.")
        db_member = db_members[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive.extract(db_member, path=tmp_dir)
            disk_db_path = Path(tmp_dir) / db_member

            disk_connection = sqlite3.connect(str(disk_db_path), check_same_thread=False)
            memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            disk_connection.backup(memory_connection)
            disk_connection.close()

    return create_engine(
        "sqlite://",
        creator=lambda: memory_connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def build_sql_database(engine: Engine | None = None) -> SQLDatabase:
    """Create a LangChain SQLDatabase wrapper around the Chinook engine."""
    resolved_engine = engine or get_engine_for_chinook_db()
    return SQLDatabase(resolved_engine)
