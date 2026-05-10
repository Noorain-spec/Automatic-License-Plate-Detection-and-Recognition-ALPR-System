"""
test_email.py — Quick test to verify Gmail SMTP credentials work.

Reads credentials from the same environment variables used by the app:
    export GMAIL_ADDRESS='yourname@gmail.com'
    export GMAIL_APP_PASSWORD='xxxx xxxx xxxx xxxx'

Run from the project root:
    python3 models/test_email.py
"""

import sys
import os
import getpass

sys.path.insert(0, os.path.dirname(__file__))

from messenger import send_email, build_alert_message, build_alert_subject

# Read from env vars (same ones the Streamlit app uses)
SMTP_USER     = os.environ.get("GMAIL_ADDRESS", "").strip()
SMTP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "").strip()

# Prompt interactively for anything missing
if not SMTP_USER:
    SMTP_USER = input("Gmail address: ").strip()
if not SMTP_PASSWORD:
    SMTP_PASSWORD = getpass.getpass("Gmail App Password: ").strip()

RECIPIENT = input("Send test email to: ").strip()

plate   = "MH14EH5819"
name    = "Test User"
subject = build_alert_subject(plate)
body    = build_alert_message(name, plate)

print("\nSending test email...")
print("  From    :", SMTP_USER)
print("  To      :", RECIPIENT)
print("  Subject :", subject)
print()

ok, detail = send_email(SMTP_USER, SMTP_PASSWORD, RECIPIENT, subject, body)

if ok:
    print("SUCCESS:", detail)
    print("Check your inbox at", RECIPIENT)
else:
    print("FAILED :", detail)
