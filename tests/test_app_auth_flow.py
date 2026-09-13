"""
Integration tests for Streamlit App authentication flow using Streamlit AppTest.
Tests Login, Sign Up, and Forgot Password UI states, plus Authenticated dashboard and Logout.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest
from src.database import init_db, create_user
from src.auth import hash_password


APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture
def isolated_app_db(tmp_path, monkeypatch):
    """Configures an isolated SQLite database path for AppTest execution."""
    test_db = str(tmp_path / "app_test_db.db")
    monkeypatch.setenv("RESEARCHLENS_DB_PATH", test_db)
    init_db(test_db)
    return test_db


class TestStreamlitAuthFlow:
    def test_unauthenticated_user_sees_login_view(self, isolated_app_db):
        """Unauthenticated user lands on Login view and cannot see dashboard navigation."""
        at = AppTest.from_file(APP_PATH, default_timeout=15).run()
        assert not at.exception

        # Verify login form elements are present
        assert len(at.text_input) >= 2
        # Inputs: email and password
        labels = [ti.label for ti in at.text_input]
        assert "Email Address" in labels
        assert "Password" in labels

        # Verify dashboard navigation radio is NOT present (protected behind authentication gate)
        assert len(at.sidebar.radio) == 0

    def test_switch_to_signup_view(self, isolated_app_db):
        """User can toggle to Sign Up view and see registration form fields."""
        at = AppTest.from_file(APP_PATH, default_timeout=15).run()
        assert not at.exception

        # Find and click "Don't have an account? Sign Up" button
        signup_btn = [b for b in at.button if "Sign Up" in b.label]
        assert len(signup_btn) > 0
        signup_btn[0].click().run()

        assert not at.exception
        labels = [ti.label for ti in at.text_input]
        assert "Full Name" in labels
        assert "Email Address" in labels
        assert "Password" in labels
        assert "Confirm Password" in labels

    def test_switch_to_forgot_password_view(self, isolated_app_db):
        """User can toggle to Forgot Password view and see reset fields."""
        at = AppTest.from_file(APP_PATH, default_timeout=15).run()
        assert not at.exception

        # Find and click "Forgot Password?" button
        fp_btn = [b for b in at.button if "Forgot Password" in b.label]
        assert len(fp_btn) > 0
        fp_btn[0].click().run()

        assert not at.exception
        labels = [ti.label for ti in at.text_input]
        assert "Registered Email Address" in labels
        assert "New Password" in labels
        assert "Confirm New Password" in labels

    def test_authenticated_user_sees_dashboard_and_logout(self, isolated_app_db):
        """When session state is authenticated, dashboard and sidebar profile/logout are visible."""
        at = AppTest.from_file(APP_PATH, default_timeout=15)
        # Pre-set session state as authenticated
        at.session_state["authenticated"] = True
        at.session_state["user_id"] = 1
        at.session_state["user_name"] = "Vishesh"
        at.session_state["user_email"] = "vishesh@researchlens.ai"
        at.run()

        assert not at.exception
        # Sidebar now contains navigation radio
        assert len(at.sidebar.radio) > 0
        assert at.sidebar.radio[0].label == "Navigation"

        # Sidebar contains Logout button
        logout_btns = [b for b in at.sidebar.button if "Logout" in b.label]
        assert len(logout_btns) == 1

        # Click Logout
        logout_btns[0].click().run()
        assert not at.exception
        # After logout, user is unauthenticated
        assert at.session_state["authenticated"] is False
        assert at.session_state["user_id"] is None
        # Navigation radio is gone from sidebar
        assert len(at.sidebar.radio) == 0

    def test_signup_and_login_e2e_flow(self, isolated_app_db):
        """End-to-end test registering a user via the UI, then logging in."""
        at = AppTest.from_file(APP_PATH, default_timeout=15).run()
        assert not at.exception

        # Switch to Sign Up
        signup_btn = [b for b in at.button if "Sign Up" in b.label]
        signup_btn[0].click().run()

        # Fill Sign Up form
        at.text_input(key="su_name_input").input("Vishesh Kumar")
        at.text_input(key="su_email_input").input("vishesh.kumar@ai.edu")
        at.text_input(key="su_pwd_input").input("AcademicPassword123")
        at.text_input(key="su_confirm_input").input("AcademicPassword123")

        # Submit signup
        at.button[0].click().run()
        assert not at.exception

        # User is redirected to login view with success flash message
        assert at.session_state["auth_view"] == "login"
        assert at.session_state["authenticated"] is False

        # Fill Login form
        at.text_input(key="login_email_input").input("vishesh.kumar@ai.edu")
        at.text_input(key="login_password_input").input("AcademicPassword123")

        # Submit login
        at.button[0].click().run()
        assert not at.exception

        # User is now authenticated
        assert at.session_state["authenticated"] is True
        assert at.session_state["user_name"] == "Vishesh Kumar"
        assert at.session_state["user_email"] == "vishesh.kumar@ai.edu"
        assert len(at.sidebar.radio) > 0

    def test_authenticated_user_can_view_profile_section(self, isolated_app_db):
        """Authenticated user navigates to Profile section and can see profile details & change password form."""
        created_user = create_user("Dr. Marie Curie", "marie@sorbonne.fr", hash_password("RadiumPassword123"), db_path=isolated_app_db)
        assert created_user is not None

        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.session_state["authenticated"] = True
        at.session_state["user_id"] = created_user["id"]
        at.session_state["user_name"] = created_user["name"]
        at.session_state["user_email"] = created_user["email"]
        at.run()

        assert not at.exception
        # Navigation options include Profile
        nav_options = at.sidebar.radio[0].options
        assert "👤 Profile" in nav_options

        # Select Profile page
        at.sidebar.radio[0].set_value("👤 Profile").run()
        assert not at.exception

        # Verify profile details and display name edit input are present
        ti_keys = [ti.key for ti in at.text_input if ti.key]
        assert "new_display_name_input" in ti_keys
        # Verify password change inputs are removed
        assert "profile_curr_pwd" not in ti_keys
        assert "profile_new_pwd" not in ti_keys
        assert "profile_confirm_pwd" not in ti_keys

