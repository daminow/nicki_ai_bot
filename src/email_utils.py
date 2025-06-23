import smtplib
import ssl
import random
from email.message import EmailMessage
from pydantic import EmailStr
from .config import settings

CODE_CACHE: dict[str, str] = {}


def generate_code() -> str:
    return str(random.randint(100000, 999999))


def send_verification_email(address: EmailStr) -> str:
    code = generate_code()
    context = ssl.create_default_context()
    msg = EmailMessage()
    msg["From"] = settings.smtp_user
    msg["To"] = address
    msg["Subject"] = "Nicki AI verification code"
    msg.set_content(
        f"Hello!\n\nYour verification code for Nicki AI is: {code}\n"
        "Enter this code in Telegram to complete your registration.\n\n"
        "If you did not request this, simply ignore the email."
    )

    with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=context) as s:
        s.login(settings.smtp_user, settings.smtp_password)
        s.send_message(msg)

    CODE_CACHE[str(address)] = code
    return code


def verify_code(address: EmailStr, code: str) -> bool:
    return CODE_CACHE.get(str(address)) == code.strip()