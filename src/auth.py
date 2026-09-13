"""
ResearchLens AI - Authentication & Password Security Layer.
Handles registration, authentication, bcrypt password hashing/verification,
input validation, and safe MVP password reset operations.
"""

import re
import bcrypt
from typing import Optional, Tuple, Dict, Any

from src.database import (
    create_user,
    get_user_by_email,
    get_user_with_hash_by_id,
    update_user_name,
    update_user_password
)

# Standard RFC-compliant email pattern
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
MIN_PASSWORD_LENGTH = 8


def validate_email(email: str) -> bool:
    """
    Validates email format using regex.
    """
    if not email or not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Enforces minimum password security policies.
    Requires at least 8 characters.
    """
    if not password or not isinstance(password, str):
        return False, "Password cannot be empty."
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
    return True, ""


def hash_password(password: str) -> str:
    """
    Generates a secure, salted bcrypt password hash.
    NEVER stores plain-text passwords.
    """
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_bytes.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verifies a plain-text password against a stored bcrypt hash.
    Returns True if matches, False otherwise.
    """
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def register_user(
    name: str,
    email: str,
    password: str,
    confirm_password: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Registers a new user with full validation.
    Checks:
    - All fields required
    - Valid email format
    - Minimum 8-character password
    - Password and confirm password match
    - Email uniqueness
    """
    # 1. Required fields check
    if not name or not name.strip():
        return False, "Full Name is required."
    if not email or not email.strip():
        return False, "Email address is required."
    if not password:
        return False, "Password is required."
    if not confirm_password:
        return False, "Please confirm your password."

    clean_name = name.strip()
    clean_email = email.strip().lower()

    # 2. Email format validation
    if not validate_email(clean_email):
        return False, "Please enter a valid email address."

    # 3. Password length check
    is_valid_pw, pw_msg = validate_password(password)
    if not is_valid_pw:
        return False, pw_msg

    # 4. Password confirmation match
    if password != confirm_password:
        return False, "Passwords do not match."

    # 5. Email uniqueness check
    existing_user = get_user_by_email(clean_email, db_path=db_path)
    if existing_user is not None:
        return False, "An account with this email address already exists."

    # 6. Hash and store
    pwd_hash = hash_password(password)
    created_user = create_user(
        name=clean_name,
        email=clean_email,
        password_hash=pwd_hash,
        db_path=db_path
    )

    if created_user is None:
        return False, "An error occurred while creating your account. Please try again."

    return True, "Account created successfully. Please login."


def authenticate_user(
    email: str,
    password: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticates a user by email and password.
    Validates input fields, verifies bcrypt hash, and prevents information leakage.
    Returns (success, message, user_dict_or_None).
    """
    if not email or not email.strip():
        return False, "Please enter your email.", None

    clean_email = email.strip().lower()
    if not validate_email(clean_email):
        return False, "Please enter a valid email address.", None

    if not password:
        return False, "Please enter your password.", None

    user = get_user_by_email(clean_email, db_path=db_path)
    if not user:
        return False, "Incorrect email or password.", None

    if not verify_password(password, user["password_hash"]):
        return False, "Incorrect email or password.", None

    safe_user_data = {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "created_at": user.get("created_at", "")
    }
    return True, "Login successful.", safe_user_data


def reset_password(
    email: str,
    new_password: str,
    confirm_password: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Safe MVP password reset flow.
    Enforces validation on password rules and updates the stored hash if the user exists.
    Always returns a safe, consistent confirmation message to prevent user email enumeration.
    """
    if not email or not email.strip():
        return False, "Please enter your email address."

    clean_email = email.strip().lower()
    if not validate_email(clean_email):
        return False, "Please enter a valid email address."

    if not new_password:
        return False, "Please enter a new password."

    is_valid_pw, pw_msg = validate_password(new_password)
    if not is_valid_pw:
        return False, pw_msg

    if new_password != confirm_password:
        return False, "Passwords do not match."

    # If user exists, update password hash
    user = get_user_by_email(clean_email, db_path=db_path)
    if user:
        new_hash = hash_password(new_password)
        update_user_password(clean_email, new_hash, db_path=db_path)

    # Safe confirmation response that avoids user enumeration
    return True, "If an account exists with this email address, the password has been reset. You may now log in."


def change_password(
    user_id: int,
    current_password: str,
    new_password: str,
    confirm_password: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Allows an authenticated user to change their password from their profile.
    Requires verifying the current password before applying the new password.
    """
    if not current_password:
        return False, "Please enter your current password."
    if not new_password:
        return False, "Please enter a new password."

    is_valid_pw, pw_msg = validate_password(new_password)
    if not is_valid_pw:
        return False, pw_msg

    if new_password != confirm_password:
        return False, "New passwords do not match."

    user = get_user_with_hash_by_id(user_id, db_path=db_path)
    if not user:
        return False, "User account not found."

    if not verify_password(current_password, user["password_hash"]):
        return False, "Current password is incorrect."

    new_hash = hash_password(new_password)
    updated = update_user_password(user["email"], new_hash, db_path=db_path)
    if updated:
        return True, "Password updated successfully."
    return False, "Failed to update password. Please try again."


def update_profile_name(
    user_id: int,
    new_name: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Updates the authenticated user's display name.
    """
    if not new_name or not new_name.strip():
        return False, "Name cannot be empty."

    clean_name = new_name.strip()
    updated = update_user_name(user_id, clean_name, db_path=db_path)
    if updated:
        return True, "Profile name updated successfully."
    return False, "Failed to update profile name."
