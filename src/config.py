import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, EmailStr

load_dotenv(".env")


class Settings(BaseModel):
    bot_token: str = Field(..., env="BOT_TOKEN")
    novita_api_key: str = Field(..., env="NOVITA_API_KEY")
    novita_model: str = Field(default="deepseek/deepseek-v3-0324", env="NOVITA_MODEL")
    bot_base_url: str = Field(default="https://api.novita.ai/v3/openai", env="BOT_BASE_URL")
    postgres_user: str = Field(..., env="POSTGRES_USER")
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD")
    postgres_db: str = Field(..., env="POSTGRES_DB")
    postgres_host: str = Field(default="db", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    smtp_host: str = Field(..., env="SMTP_HOST")
    smtp_port: int = Field(..., env="SMTP_PORT")
    smtp_user: EmailStr = Field(..., env="SMTP_USER")
    smtp_password: str = Field(..., env="SMTP_PASSWORD")

    @property
    def database_uri(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:"
            f"{self.postgres_password}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()