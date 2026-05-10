"""
messenger.py — Send email notifications via Gmail SMTP (or any SMTP server).

No third-party libraries needed — uses Python's built-in smtplib.

Gmail setup (one-time):
  1. Enable 2-Step Verification on your Google account.
  2. Go to https://myaccount.google.com/apppasswords
  3. Create an App Password (select "Mail" + "Other").
  4. Use that 16-character password as smtp_password, NOT your real Gmail password.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Tuple


SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465  # SSL — port 587 (STARTTLS) is blocked on many servers


def send_email(
    smtp_user: str,
    smtp_password: str,
    recipient_email: str,
    subject: str,
    message: str,
) -> Tuple[bool, str]:
    """
    Send an email via Gmail SMTP (TLS on port 587).

    Args:
        smtp_user:       Your Gmail address (e.g. yourname@gmail.com).
        smtp_password:   Gmail App Password (16 chars, no spaces).
        recipient_email: Destination email address.
        subject:         Email subject line.
        message:         Plain-text body.

    Returns:
        (success: bool, detail: str)
    """
    if not smtp_user or not smtp_user.strip():
        return False, "Sender Gmail address is not configured."
    if not smtp_password or not smtp_password.strip():
        return False, "Gmail App Password is not configured."
    if not recipient_email or not recipient_email.strip():
        return False, "Recipient email address is missing."
    if not message or not message.strip():
        return False, "Message cannot be empty."

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject.strip()
    msg["From"] = smtp_user.strip()
    msg["To"] = recipient_email.strip()
    msg.attach(MIMEText(message.strip(), "plain"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            server.ehlo()
            server.login(smtp_user.strip(), smtp_password.strip())
            server.sendmail(smtp_user.strip(), recipient_email.strip(), msg.as_string())
        return True, "Email sent successfully to {}.".format(recipient_email.strip())

    except smtplib.SMTPAuthenticationError:
        return False, (
            "Gmail authentication failed. "
            "Make sure you are using an App Password, not your Gmail login password."
        )
    except smtplib.SMTPException as exc:
        return False, "SMTP error: {}".format(exc)
    except Exception as exc:
        return False, "Unexpected error: {}".format(exc)


# Keep the name send_sms as an alias so app.py doesn't need changes.
# It now sends an email instead of an SMS.
def send_sms(api_key: str, contact_number: str, message: str) -> Tuple[bool, str]:
    """
    Compatibility shim — not used directly.
    Call send_email() from app.py instead.
    """
    return False, "Use send_email() directly."


def build_alert_message(name: str, plate_number: str, violations: list = None) -> str:
    """Build a challan / alert email body, including any document violations."""
    lines = [
        f"Dear {name},",
        "",
        f"Your vehicle with license plate {plate_number} has been detected by our "
        "ALPR (Automatic License Plate Recognition) system.",
    ]
    if violations:
        lines += [
            "",
            "The following violation(s) have been recorded for this vehicle:",
        ]
        for v in violations:
            lines.append(f"  • {v}")
        lines += [
            "",
            "Please renew the expired documents immediately to avoid further penalties.",
            "Failure to comply may result in a formal challan being issued.",
        ]
    else:
        lines += [
            "",
            "No document violations were found at this time.",
            "If this detection was unexpected, please contact us immediately.",
        ]
    lines += [
        "",
        "-- ALPR Challan System",
    ]
    return "\n".join(lines)


def build_alert_subject(plate_number: str, violations: list = None) -> str:
    """Build the email subject line; uses 'Challan' wording if violations exist."""
    if violations:
        return f"Traffic Challan Notice: Vehicle {plate_number} — {len(violations)} Violation(s)"
    return f"ALPR Alert: Vehicle {plate_number} Detected"
