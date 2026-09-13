"""
Unit tests for src/database.py (SQLite CRUD, initialization, uniqueness constraints).
"""

import os
import pytest
import sqlite3

from src.database import (
    init_db,
    get_connection,
    create_user,
    get_user_by_email,
    get_user_by_id,
    get_user_with_hash_by_id,
    update_user_name,
    update_user_password,
    list_users
)


@pytest.fixture
def test_db(tmp_path):
    """Provides a fresh isolated SQLite database file for each test."""
    db_file = str(tmp_path / "test_researchlens.db")
    init_db(db_file)
    return db_file


class TestDatabaseInitialization:
    def test_init_db_creates_tables_and_indexes(self, test_db):
        conn = get_connection(test_db)
        cursor = conn.cursor()

        # Check users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
        assert cursor.fetchone() is not None

        # Check email index exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_users_email';")
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_db_is_idempotent(self, test_db):
        # Calling init_db again should not raise errors
        init_db(test_db)
        init_db(test_db)


class TestUserCRUD:
    def test_create_user_success(self, test_db):
        user = create_user("Alice Researcher", "alice@university.edu", "$2b$12$fakehash123", db_path=test_db)
        assert user is not None
        assert user["id"] is not None
        assert user["name"] == "Alice Researcher"
        assert user["email"] == "alice@university.edu"
        assert "created_at" in user

    def test_create_user_duplicate_email_fails(self, test_db):
        create_user("Alice Researcher", "alice@university.edu", "hash1", db_path=test_db)
        dup = create_user("Alice Clone", "alice@university.edu", "hash2", db_path=test_db)
        assert dup is None

    def test_create_user_case_insensitive_duplicate(self, test_db):
        create_user("Bob", "bob@lab.org", "hash1", db_path=test_db)
        dup = create_user("Bob 2", "BOB@LAB.ORG", "hash2", db_path=test_db)
        assert dup is None

    def test_get_user_by_email(self, test_db):
        create_user("Charlie", "charlie@mit.edu", "secret_hash", db_path=test_db)
        user = get_user_by_email("CHARLIE@MIT.EDU", db_path=test_db)
        assert user is not None
        assert user["name"] == "Charlie"
        assert user["password_hash"] == "secret_hash"

    def test_get_user_by_email_nonexistent(self, test_db):
        user = get_user_by_email("ghost@unknown.com", db_path=test_db)
        assert user is None

    def test_get_user_by_id(self, test_db):
        created = create_user("Dana", "dana@stanford.edu", "hash4", db_path=test_db)
        user = get_user_by_id(created["id"], db_path=test_db)
        assert user is not None
        assert user["email"] == "dana@stanford.edu"

    def test_get_user_by_id_nonexistent(self, test_db):
        assert get_user_by_id(99999, db_path=test_db) is None

    def test_update_user_password(self, test_db):
        create_user("Eve", "eve@secure.org", "old_hash", db_path=test_db)
        updated = update_user_password("eve@secure.org", "new_hash_456", db_path=test_db)
        assert updated is True

        user = get_user_by_email("eve@secure.org", db_path=test_db)
        assert user["password_hash"] == "new_hash_456"

    def test_update_user_password_nonexistent(self, test_db):
        updated = update_user_password("nonexistent@domain.com", "new_hash", db_path=test_db)
        assert updated is False

    def test_list_users(self, test_db):
        create_user("User 1", "u1@test.com", "h1", db_path=test_db)
        create_user("User 2", "u2@test.com", "h2", db_path=test_db)

        users = list_users(db_path=test_db)
        assert len(users) == 2
        assert "password_hash" not in users[0]

    def test_get_user_with_hash_by_id(self, test_db):
        created = create_user("Frank", "frank@research.org", "frank_secure_hash", db_path=test_db)
        user = get_user_with_hash_by_id(created["id"], db_path=test_db)
        assert user is not None
        assert user["name"] == "Frank"
        assert user["email"] == "frank@research.org"
        assert user["password_hash"] == "frank_secure_hash"

        # Nonexistent
        assert get_user_with_hash_by_id(8888, db_path=test_db) is None

    def test_update_user_name(self, test_db):
        created = create_user("Grace Hopper", "grace@navy.mil", "hash", db_path=test_db)
        updated = update_user_name(created["id"], "Rear Admiral Grace Hopper", db_path=test_db)
        assert updated is True

        user = get_user_by_id(created["id"], db_path=test_db)
        assert user["name"] == "Rear Admiral Grace Hopper"

        # Empty name should fail
        assert update_user_name(created["id"], "   ", db_path=test_db) is False

        # Nonexistent user
        assert update_user_name(99999, "Nobody", db_path=test_db) is False

