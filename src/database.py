"""
ResearchLens AI - Database Management Layer.
Provides SQLite database initialization, connection handling, and user CRUD operations
using parameterized queries to ensure security against SQL injection.
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

DEFAULT_DB_NAME = "researchlens.db"


def get_default_db_path() -> str:
    """
    Resolves the default SQLite database path.
    Prioritizes RESEARCHLENS_DB_PATH environment variable if set;
    otherwise places researchlens.db in the project root.
    """
    env_path = os.environ.get("RESEARCHLENS_DB_PATH")
    if env_path:
        return env_path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, DEFAULT_DB_NAME)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Creates and returns an active SQLite connection with row factory enabled
    and foreign key enforcement turned on.
    """
    target_path = db_path if db_path is not None else get_default_db_path()
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes database tables and indexes if they do not already exist.
    Prepares schema architecture for user accounts and future user-specific artifacts
    (papers, analyses, comparisons, reports).
    """
    target_path = db_path if db_path is not None else get_default_db_path()
    
    # Ensure directory exists if saving to a non-current directory
    dir_name = os.path.dirname(target_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with get_connection(target_path) as conn:
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)

        # Fast lookup index on email
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        """)

        # Extensible Architecture:
        # The schema is structured to seamlessly support future tables:
        # - user_papers (id, user_id, filename, file_type, file_content, uploaded_at)
        # - user_analyses (id, user_id, paper_id, analysis_json, created_at)
        # - user_comparisons (id, user_id, comparison_json, created_at)
        # - user_reports (id, user_id, report_type, content, generated_at)
        conn.commit()


def create_user(
    name: str,
    email: str,
    password_hash: str,
    db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Inserts a new user into the users table using parameterized queries.
    Returns the created user record dict, or None if insertion failed (e.g. duplicate email).
    """
    target_path = db_path if db_path is not None else get_default_db_path()
    clean_name = name.strip()
    clean_email = email.strip().lower()
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with get_connection(target_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (clean_name, clean_email, password_hash, created_at)
            )
            conn.commit()
            user_id = cursor.lastrowid

            cursor.execute("SELECT id, name, email, created_at FROM users WHERE id = ?;", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    except sqlite3.IntegrityError:
        # Duplicate email or constraint violation
        return None
    except sqlite3.Error:
        return None


def get_user_by_email(email: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user record by email (case-insensitive) using parameterized queries.
    Returns a dictionary including id, name, email, password_hash, and created_at, or None.
    """
    target_path = db_path if db_path is not None else get_default_db_path()
    clean_email = email.strip().lower()

    with get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, name, email, password_hash, created_at
            FROM users
            WHERE email = ?;
            """,
            (clean_email,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_user_by_id(user_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user record by user ID using parameterized queries.
    Returns a dictionary or None if not found.
    """
    target_path = db_path if db_path is not None else get_default_db_path()

    with get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, name, email, created_at
            FROM users
            WHERE id = ?;
            """,
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_user_with_hash_by_id(user_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves full user record including password_hash for authenticated password management.
    """
    target_path = db_path if db_path is not None else get_default_db_path()

    with get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, name, email, password_hash, created_at
            FROM users
            WHERE id = ?;
            """,
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def update_user_name(user_id: int, new_name: str, db_path: Optional[str] = None) -> bool:
    """
    Updates the display name of a user.
    """
    target_path = db_path if db_path is not None else get_default_db_path()
    clean_name = new_name.strip()
    if not clean_name:
        return False

    with get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users
            SET name = ?
            WHERE id = ?;
            """,
            (clean_name, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0


def update_user_password(
    email: str,
    new_password_hash: str,
    db_path: Optional[str] = None
) -> bool:
    """
    Updates the password_hash for the specified user email.
    Returns True if a row was updated, False otherwise.
    """
    target_path = db_path if db_path is not None else get_default_db_path()
    clean_email = email.strip().lower()

    with get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE email = ?;
            """,
            (new_password_hash, clean_email)
        )
        conn.commit()
        return cursor.rowcount > 0


def list_users(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns a list of all registered users (excluding password hashes) for administrative / diagnostic use.
    """
    target_path = db_path if db_path is not None else get_default_db_path()

    with get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, created_at FROM users ORDER BY id ASC;")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
