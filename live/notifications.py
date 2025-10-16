import os
import smtplib
from email.message import EmailMessage
from typing import Optional


def send_email(subject: str, body: str, to_addr: Optional[str] = None) -> bool:
    to_addr = to_addr or os.getenv("NOTIFY_EMAIL_TO")
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    if not (to_addr and host and user and pwd):
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, port) as s:
            s.starttls()
            s.login(user, pwd)
            s.send_message(msg)
        return True
    except Exception:
        return False


def send_sms_via_email(subject: str, body: str) -> bool:
    gateway = os.getenv("SMS_EMAIL_GATEWAY")
    if not gateway:
        return False
    return send_email(subject, body, to_addr=gateway)


