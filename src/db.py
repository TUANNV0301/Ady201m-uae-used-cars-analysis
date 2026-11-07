"""
Database Access and Execution Utilities
UAE Used-Car Market Analysis (ADY201m)
"""

import sqlite3
from pathlib import Path
from typing import Optional, Union, List, Tuple
import pandas as pd
from src.config import DB_PATH, SQL_DIR


def get_connection(
    db_path: Union[str, Path] = DB_PATH, 
    read_only: bool = False
) -> sqlite3.Connection:
    """
    Creates and returns a SQLite connection with foreign keys enabled.
    read_only: if True, connects using URI mode=ro.
    """
    path = Path(db_path).resolve()
    if read_only:
        uri = f"file:{path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
    
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def run_sql_file(
    sql_file_path: Union[str, Path], 
    conn: Optional[sqlite3.Connection] = None
) -> None:
    """Reads and executes all SQL statements in a script file."""
    path = Path(sql_file_path)
    if not path.is_absolute():
        path = SQL_DIR / path
        
    with open(path, "r", encoding="utf-8") as f:
        script = f.read()
        
    close_on_finish = False
    if conn is None:
        conn = get_connection()
        close_on_finish = True
        
    try:
        conn.executescript(script)
        conn.commit()
    finally:
        if close_on_finish:
            conn.close()


def query_to_dataframe(
    query: str, 
    params: Optional[Union[List, Tuple, dict]] = None, 
    conn: Optional[sqlite3.Connection] = None
) -> pd.DataFrame:
    """Executes a SQL query and returns results as a pandas DataFrame."""
    close_on_finish = False
    if conn is None:
        conn = get_connection(read_only=True)
        close_on_finish = True
        
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    finally:
        if close_on_finish:
            conn.close()


def init_db(db_path: Union[str, Path] = DB_PATH) -> None:
    """Initializes the database schema from sql/01_schema.sql."""
    schema_file = SQL_DIR / "01_schema.sql"
    conn = get_connection(db_path, read_only=False)
    try:
        run_sql_file(schema_file, conn)
    finally:
        conn.close()
