"""
Unit tests for src/auth.py (Password hashing, registration, authentication, validation, session state).
"""

import pytest
from src.database import init_db, get_user_by_email
from src.auth import (
    validate_email,
    validate_password,
    hash_password,
    verify_password,
    register_user,
    authenticate_user,
    reset_password,
    change_password,
    update_profile_name
)


@pytest.fixture
def auth_db(tmp_path):
    """Provides an isolated test database for authentication tests."""
    db_file = str(tmp_path / "test_auth.db")
    init_db(db_file)
    return db_file


class TestPasswordHashing:
    def test_hash_password_produces_salted_bcrypt(self):
        pwd = "AcademicResearch2026!"
        h1 = hash_password(pwd)
        h2 = hash_password(pwd)

        assert h1.startswith("$2b$")
        # Salted: two hashes of the same password should not be equal
        assert h1 != h2
        # Original password should never appear in the hash
        assert pwd not in h1

    def test_verify_password_correct(self):
        pwd = "SecurePassword123"
        pwd_hash = hash_password(pwd)
        assert verify_password(pwd, pwd_hash) is True

    def test_verify_password_incorrect(self):
        pwd = "SecurePassword123"
        pwd_hash = hash_password(pwd)
        assert verify_password("WrongPassword!", pwd_hash) is False

    def test_verify_password_edge_cases(self):
        assert verify_password("", "$2b$12$fake") is False
        assert verify_password("pwd", "") is False
        assert verify_password("pwd", "not_a_valid_hash") is False


class TestInputValidation:
    def test_validate_email_valid(self):
        valid_emails = [
            "researcher@university.edu",
            "vishesh.k@lab.ai",
            "dr.smith+papers@sub.institute.ac.uk",
            "student_123@college.org"
        ]
        for email in valid_emails:
            assert validate_email(email) is True

    def test_validate_email_invalid(self):
        invalid_emails = [
            "",
            "plainaddress",
            "@missingusername.com",
            "user@.com",
            "user@domain",
            "user name@domain.com",
            None
        ]
        for email in invalid_emails:
            assert validate_email(email) is False

    def test_validate_password_policies(self):
        # Valid password (>= 8 chars)
        ok, msg = validate_password("12345678")
        assert ok is True
        assert msg == ""

        # Short password (< 8 chars)
        ok, msg = validate_password("short")
        assert ok is False
        assert "8 characters" in msg

        # Empty password
        ok, msg = validate_password("")
        assert ok is False


class TestUserRegistration:
    def test_register_user_success(self, auth_db):
        success, msg = register_user(
            name="Dr. Jane Doe",
            email="jane.doe@oxford.ac.uk",
            password="securePassword456",
            confirm_password="securePassword456",
            db_path=auth_db
        )
        assert success is True
        assert "Account created successfully" in msg

        user = get_user_by_email("jane.doe@oxford.ac.uk", db_path=auth_db)
        assert user is not None
        assert user["name"] == "Dr. Jane Doe"
        assert verify_password("securePassword456", user["password_hash"]) is True

    def test_register_user_duplicate_email(self, auth_db):
        register_user("Alice", "alice@mit.edu", "pass12345", "pass12345", db_path=auth_db)
        success, msg = register_user("Alice 2", "alice@mit.edu", "pass67890", "pass67890", db_path=auth_db)
        assert success is False
        assert "already exists" in msg

    def test_register_user_password_mismatch(self, auth_db):
        success, msg = register_user(
            name="Bob",
            email="bob@mit.edu",
            password="Password123",
            confirm_password="PasswordMismatch",
            db_path=auth_db
        )
        assert success is False
        assert "Passwords do not match" in msg

    def test_register_user_short_password(self, auth_db):
        success, msg = register_user(
            name="Bob",
            email="bob@mit.edu",
            password="short",
            confirm_password="short",
            db_path=auth_db
        )
        assert success is False
        assert "8 characters" in msg

    def test_register_user_invalid_email(self, auth_db):
        success, msg = register_user(
            name="Bob",
            email="not-an-email",
            password="validPassword88",
            confirm_password="validPassword88",
            db_path=auth_db
        )
        assert success is False
        assert "valid email" in msg

    def test_register_user_empty_name(self, auth_db):
        success, msg = register_user(
            name="",
            email="bob@test.com",
            password="validPassword88",
            confirm_password="validPassword88",
            db_path=auth_db
        )
        assert success is False
        assert "Full Name is required" in msg


class TestUserAuthentication:
    def test_authenticate_user_success(self, auth_db):
        register_user("Vishesh", "vishesh@researchlens.ai", "MasterKey2026", "MasterKey2026", db_path=auth_db)

        success, msg, user_data = authenticate_user(
            "vishesh@researchlens.ai",
            "MasterKey2026",
            db_path=auth_db
        )
        assert success is True
        assert "Login successful" in msg
        assert user_data is not None
        assert user_data["name"] == "Vishesh"
        assert user_data["email"] == "vishesh@researchlens.ai"
        assert "password_hash" not in user_data

    def test_authenticate_user_incorrect_password(self, auth_db):
        register_user("Vishesh", "vishesh@researchlens.ai", "MasterKey2026", "MasterKey2026", db_path=auth_db)

        success, msg, user_data = authenticate_user(
            "vishesh@researchlens.ai",
            "WrongPassword",
            db_path=auth_db
        )
        assert success is False
        assert "Incorrect email or password" in msg
        assert user_data is None

    def test_authenticate_user_nonexistent_email(self, auth_db):
        success, msg, user_data = authenticate_user(
            "nonexistent@researchlens.ai",
            "SomePassword123",
            db_path=auth_db
        )
        assert success is False
        assert "Incorrect email or password" in msg
        assert user_data is None

    def test_authenticate_user_empty_fields(self, auth_db):
        success, msg, _ = authenticate_user("", "pass12345", db_path=auth_db)
        assert success is False
        assert "enter your email" in msg

        success, msg, _ = authenticate_user("vishesh@test.com", "", db_path=auth_db)
        assert success is False
        assert "enter your password" in msg


class TestPasswordResetFlow:
    def test_reset_password_success(self, auth_db):
        register_user("Researcher", "res@lab.edu", "initialPass123", "initialPass123", db_path=auth_db)

        success, msg = reset_password(
            email="res@lab.edu",
            new_password="NewSecretPassword2026",
            confirm_password="NewSecretPassword2026",
            db_path=auth_db
        )
        assert success is True
        assert "password has been reset" in msg

        # Old password no longer works
        old_auth, _, _ = authenticate_user("res@lab.edu", "initialPass123", db_path=auth_db)
        assert old_auth is False

        # New password works
        new_auth, _, user = authenticate_user("res@lab.edu", "NewSecretPassword2026", db_path=auth_db)
        assert new_auth is True
        assert user["email"] == "res@lab.edu"

    def test_reset_password_nonexistent_email_safe(self, auth_db):
        # Should return generic success without revealing email does not exist
        success, msg = reset_password(
            email="nobody@nowhere.edu",
            new_password="BrandNewPassword123",
            confirm_password="BrandNewPassword123",
            db_path=auth_db
        )
        assert success is True
        assert "If an account exists" in msg

    def test_reset_password_mismatch(self, auth_db):
        success, msg = reset_password(
            email="res@lab.edu",
            new_password="BrandNewPassword123",
            confirm_password="DifferentPassword456",
            db_path=auth_db
        )
        assert success is False
        assert "Passwords do not match" in msg


class TestSessionStateFlow:
    def test_session_state_lifecycle(self):
        """Simulates session state transitions during Login, Dashboard usage, and Logout."""
        session = {
            "authenticated": False,
            "user_id": None,
            "user_name": None,
            "user_email": None
        }

        # Simulated Login
        user_info = {"id": 42, "name": "Vishesh", "email": "vishesh@example.com"}
        session["authenticated"] = True
        session["user_id"] = user_info["id"]
        session["user_name"] = user_info["name"]
        session["user_email"] = user_info["email"]

        assert session["authenticated"] is True
        assert session["user_name"] == "Vishesh"

        # Simulated Logout
        session["authenticated"] = False
        session["user_id"] = None
        session["user_name"] = None
        session["user_email"] = None

        assert session["authenticated"] is False
        assert session["user_id"] is None
        assert session["user_name"] is None


class TestProfileManagement:
    def test_change_password_success(self, auth_db):
        register_user("Prof. Turing", "alan@cambridge.edu", "EnigmaBreak1940", "EnigmaBreak1940", db_path=auth_db)
        ok, _, user = authenticate_user("alan@cambridge.edu", "EnigmaBreak1940", db_path=auth_db)
        assert ok is True
        user_id = user["id"]

        # Successful password change
        changed, msg = change_password(
            user_id=user_id,
            current_password="EnigmaBreak1940",
            new_password="NewUltraSecure2026!",
            confirm_password="NewUltraSecure2026!",
            db_path=auth_db
        )
        assert changed is True
        assert "successfully" in msg

        # Old password must fail
        old_ok, _, _ = authenticate_user("alan@cambridge.edu", "EnigmaBreak1940", db_path=auth_db)
        assert old_ok is False

        # New password succeeds
        new_ok, _, _ = authenticate_user("alan@cambridge.edu", "NewUltraSecure2026!", db_path=auth_db)
        assert new_ok is True

    def test_change_password_incorrect_current_password(self, auth_db):
        register_user("Prof. Turing", "alan@cambridge.edu", "EnigmaBreak1940", "EnigmaBreak1940", db_path=auth_db)
        _, _, user = authenticate_user("alan@cambridge.edu", "EnigmaBreak1940", db_path=auth_db)

        changed, msg = change_password(
            user_id=user["id"],
            current_password="WrongCurrentPassword",
            new_password="NewUltraSecure2026!",
            confirm_password="NewUltraSecure2026!",
            db_path=auth_db
        )
        assert changed is False
        assert "Current password is incorrect" in msg

    def test_change_password_mismatch(self, auth_db):
        register_user("Prof. Turing", "alan@cambridge.edu", "EnigmaBreak1940", "EnigmaBreak1940", db_path=auth_db)
        _, _, user = authenticate_user("alan@cambridge.edu", "EnigmaBreak1940", db_path=auth_db)

        changed, msg = change_password(
            user_id=user["id"],
            current_password="EnigmaBreak1940",
            new_password="NewUltraSecure2026!",
            confirm_password="DifferentConfirmation123",
            db_path=auth_db
        )
        assert changed is False
        assert "do not match" in msg

    def test_change_password_too_short(self, auth_db):
        register_user("Prof. Turing", "alan@cambridge.edu", "EnigmaBreak1940", "EnigmaBreak1940", db_path=auth_db)
        _, _, user = authenticate_user("alan@cambridge.edu", "EnigmaBreak1940", db_path=auth_db)

        changed, msg = change_password(
            user_id=user["id"],
            current_password="EnigmaBreak1940",
            new_password="short",
            confirm_password="short",
            db_path=auth_db
        )
        assert changed is False
        assert "8 characters" in msg

    def test_change_password_nonexistent_user(self, auth_db):
        changed, msg = change_password(
            user_id=9999,
            current_password="AnyPassword",
            new_password="ValidNewPassword123",
            confirm_password="ValidNewPassword123",
            db_path=auth_db
        )
        assert changed is False
        assert "not found" in msg

    def test_update_profile_name(self, auth_db):
        register_user("Ada Lovelace", "ada@analytical.org", "FirstProgrammer1843", "FirstProgrammer1843", db_path=auth_db)
        _, _, user = authenticate_user("ada@analytical.org", "FirstProgrammer1843", db_path=auth_db)

        # Valid name update
        ok, msg = update_profile_name(user["id"], "Countess Ada Lovelace", db_path=auth_db)
        assert ok is True
        assert "successfully" in msg

        # Empty name should fail
        ok, msg = update_profile_name(user["id"], "   ", db_path=auth_db)
        assert ok is False
        assert "cannot be empty" in msg

