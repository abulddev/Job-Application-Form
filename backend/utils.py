from passlib.context import CryptContext
import smtplib
from email.mime.text import MIMEText

# Use Argon2 for hashing passwords (no 72-byte limit)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password: str) -> str:
    """
    Hash a password using Argon2.
    """
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a password against a hashed value using Argon2.
    """
    return pwd_context.verify(plain, hashed)

# Email function
def send_reset_email(to_email: str, reset_link: str):
    """
    Send a password reset email.
    """
    msg = MIMEText(f"Click this link to reset your password: {reset_link}")
    msg['Subject'] = "Password Reset"
    msg['From'] = "no-reply@example.com"
    msg['To'] = to_email
    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login('smtp_user', 'smtp_password')
        server.send_message(msg)
