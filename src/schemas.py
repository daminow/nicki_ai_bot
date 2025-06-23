"""Pydantic DTOs that decouple raw Telegram input from DB models."""
from pydantic import BaseModel, EmailStr, constr

# --- registration flow -------------------------------------------------------
class Registration(BaseModel):
    name: constr(min_length=2, max_length=64)
    email: EmailStr
    phone: constr(regex=r"^[\d\+\-\(\) ]{7,20}$")  # lenient but safe


class EmailCode(BaseModel):
    email: EmailStr
    code: constr(min_length=4, max_length=6)