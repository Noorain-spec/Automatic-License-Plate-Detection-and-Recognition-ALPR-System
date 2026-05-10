"""
database.py — SQLite database for vehicle owner records.

Table: vehicle_owners
  id               INTEGER  PRIMARY KEY AUTOINCREMENT
  plate_number     TEXT     UNIQUE (stored uppercase, spaces stripped)
  name             TEXT
  contact_number   TEXT     (digits only, 10-digit Indian mobile)
  email            TEXT     (owner's email address for notifications)
  rc_expiry        TEXT     (YYYY-MM-DD, Registration Certificate expiry)
  puc_expiry       TEXT     (YYYY-MM-DD, Pollution Under Control expiry)
  insurance_expiry TEXT     (YYYY-MM-DD, Insurance expiry)
"""

import sqlite3
import os
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "vehicle_owners.db")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_plate(plate: str) -> str:
    """Upper-case and strip whitespace from a plate number."""
    return plate.upper().strip()


def _validate_contact(contact: str) -> bool:
    """Accept 10-digit Indian mobile numbers (digits only)."""
    return bool(re.fullmatch(r"\d{10}", contact.strip()))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize_db() -> None:
    """Create the database and table if they do not already exist."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicle_owners (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number     TEXT    UNIQUE NOT NULL COLLATE NOCASE,
                name             TEXT    NOT NULL,
                contact_number   TEXT    NOT NULL,
                email            TEXT    NOT NULL DEFAULT '',
                rc_expiry        TEXT    NOT NULL DEFAULT '',
                puc_expiry       TEXT    NOT NULL DEFAULT '',
                insurance_expiry TEXT    NOT NULL DEFAULT ''
            )
            """
        )
        # Migrate existing DBs that are missing newer columns
        for col_def in (
            "email TEXT NOT NULL DEFAULT ''",
            "rc_expiry TEXT NOT NULL DEFAULT ''",
            "puc_expiry TEXT NOT NULL DEFAULT ''",
            "insurance_expiry TEXT NOT NULL DEFAULT ''",
        ):
            col_name = col_def.split()[0]
            try:
                conn.execute(f"ALTER TABLE vehicle_owners ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()


def _validate_email(email: str) -> bool:
    """Basic email format check."""
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email.strip()))


def _validate_date(val: str) -> bool:
    """Accept blank (no date set) or a valid YYYY-MM-DD string."""
    if not val:
        return True
    try:
        date.fromisoformat(val)
        return True
    except ValueError:
        return False


def add_owner(
    plate_number: str,
    name: str,
    contact_number: str,
    email: str = "",
    rc_expiry: str = "",
    puc_expiry: str = "",
    insurance_expiry: str = "",
) -> Tuple[bool, str]:
    """
    Insert or replace a vehicle owner record.
    Returns (success, message).
    """
    contact = contact_number.strip()
    if not _validate_contact(contact):
        return False, "Contact number must be exactly 10 digits."

    plate = _normalize_plate(plate_number)
    if not plate:
        return False, "Plate number cannot be empty."

    name = name.strip()
    if not name:
        return False, "Name cannot be empty."

    email = email.strip()
    if email and not _validate_email(email):
        return False, "Invalid email address format."

    rc_expiry = rc_expiry.strip()
    puc_expiry = puc_expiry.strip()
    insurance_expiry = insurance_expiry.strip()
    for field_val, field_name in [
        (rc_expiry, "RC expiry"),
        (puc_expiry, "PUC expiry"),
        (insurance_expiry, "Insurance expiry"),
    ]:
        if not _validate_date(field_val):
            return False, f"{field_name} must be a valid date (YYYY-MM-DD)."

    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vehicle_owners "
                "(plate_number, name, contact_number, email, "
                "rc_expiry, puc_expiry, insurance_expiry) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (plate, name, contact, email, rc_expiry, puc_expiry, insurance_expiry),
            )
            conn.commit()
        return True, "Record for '{}' saved successfully.".format(plate)
    except sqlite3.Error as exc:
        return False, "Database error: {}".format(exc)


def lookup_plate(plate_number: str) -> Optional[Dict]:
    """
    Return the owner record for the given plate, or None if not found.
    """
    plate = _normalize_plate(plate_number)
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM vehicle_owners WHERE plate_number = ?", (plate,)
        ).fetchone()
    return dict(row) if row else None


def get_all_owners() -> List[Dict]:
    """Return all records ordered by plate number."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM vehicle_owners ORDER BY plate_number"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_owner(plate_number: str) -> Tuple[bool, str]:
    """
    Delete the record for the given plate number.
    Returns (success, message).
    """
    plate = _normalize_plate(plate_number)
    try:
        with _get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM vehicle_owners WHERE plate_number = ?", (plate,)
            )
            conn.commit()
        if cursor.rowcount:
            return True, f"Record for '{plate}' deleted."
        return False, f"No record found for '{plate}'."
    except sqlite3.Error as exc:
        return False, f"Database error: {exc}"


# ---------------------------------------------------------------------------
# Violation checking
# ---------------------------------------------------------------------------

_DOCUMENT_FIELDS = [
    ("rc_expiry",        "RC (Registration Certificate)"),
    ("puc_expiry",       "PUC (Pollution Under Control)"),
    ("insurance_expiry", "Insurance"),
]


def check_document_violations(owner: Dict) -> List[str]:
    """
    Compare each document expiry date against today.
    Returns a list of human-readable violation strings for expired documents.
    An empty list means all documents are valid (or not set).
    """
    today = date.today()
    violations = []
    for field, label in _DOCUMENT_FIELDS:
        val = (owner.get(field) or "").strip()
        if not val:
            continue  # date not recorded — skip
        try:
            expiry = date.fromisoformat(val)
            if expiry < today:
                violations.append(
                    f"{label} expired on {expiry.strftime('%d %b %Y')}"
                )
        except ValueError:
            pass  # malformed date — ignore
    return violations
