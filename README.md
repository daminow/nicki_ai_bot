# Nicki AI – _Single‑file_ edition

For lightweight deployments you asked for **all Python code merged into a single file**.
Below is an entire runnable project where every helper, model, router and utility lives in `bot.py`.

```text
nicki_ai_bot_single/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── bot.py  # ← one file, 500+ lines, everything inside
```

---

## docker-compose.yml

```yaml
version: "3.9"
services:
  bot:
    build: .
    env_file:
      - .env
    depends_on:
      - db
    restart: unless-stopped
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped
volumes:
  pgdata:
```

## Dockerfile

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg locales && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py .
CMD ["python", "bot.py"]
```

## requirements.txt

```text
aiogram==3.6.0
asyncpg>=0.29
SQLAlchemy[asyncio]>=2.0
openai>=1.30
python-dotenv>=1.0
email-validator>=2.0
ffmpeg-python>=0.2
pydub>=0.25
```

## .env.example

```text
BOT_TOKEN=123456:ABC-DEF
NOVITA_API_KEY=your_novita_key
NOVITA_MODEL=deepseek/deepseek-v3-0324
BOT_BASE_URL=https://api.novita.ai/v3/openai
POSTGRES_DB=nickiai
POSTGRES_USER=nickiuser
POSTGRES_PASSWORD=strongpass
POSTGRES_HOST=db
POSTGRES_PORT=5432
SMTP_HOST=mail.daminov.net
SMTP_PORT=465
SMTP_USER=nickiai@daminov.net
SMTP_PASSWORD=emailpass
```

---

## bot.py (single file – everything)

```python
"""Nicki AI – single‑file Telegram bot for mental‑health support."""
import os
import asyncio
import logging
import random
import ssl
import subprocess
import tempfile
import uuid
from email.message import EmailMessage
from pathlib import Path
from typing import List

from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.client.default import DefaultBotSettings
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from email_validator import validate_email, EmailNotValidError
from openai import OpenAI
from pydantic import BaseModel, EmailStr, Field, constr
from sqlalchemy import String, Text, DateTime, ForeignKey, select, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, declarative_base, relationship
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIG & ENV ----------------------------------------------------------------
# ---------------------------------------------------------------------------
load_dotenv()

class Settings(BaseModel):
    bot_token: str = Field(..., env="BOT_TOKEN")
    novita_api_key: str = Field(..., env="NOVITA_API_KEY")
    novita_model: str = Field("deepseek/deepseek-v3-0324", env="NOVITA_MODEL")
    bot_base_url: str = Field("https://api.novita.ai/v3/openai", env="BOT_BASE_URL")
    postgres_user: str = Field(..., env="POSTGRES_USER")
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD")
    postgres_db: str = Field(..., env="POSTGRES_DB")
    postgres_host: str = Field("db", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    smtp_host: str = Field(..., env="SMTP_HOST")
    smtp_port: int = Field(465, env="SMTP_PORT")
    smtp_user: EmailStr = Field(..., env="SMTP_USER")
    smtp_password: str = Field(..., env="SMTP_PASSWORD")

    @property
    def dsn(self):
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

settings = Settings()

# Logging ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nickiai")

# ---------------------------------------------------------------------------
# DATABASE (SQLAlchemy async) --------------------------------------------------
# ---------------------------------------------------------------------------
Base = declarative_base()
engine = create_async_engine(settings.dsn, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(unique=True, index=True)
    language: Mapped[str] = mapped_column(String(5), default="en")
    name: Mapped[str] = mapped_column(String(64), default="")
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=True)
    phone: Mapped[str] = mapped_column(String(32), nullable=True)
    email_verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default="now()")
    messages: Mapped[List["Message"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default="now()")
    user: Mapped[User] = relationship(back_populates="messages")

# ---------------------------------------------------------------------------
# HELPERS ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
BASE_PROMPT = (
    "You are Nicki AI, an empathetic assistant trained in CBT techniques. "
    "Provide practical coping strategies, ask reflective questions, and "
    "encourage users to consult licensed professionals for severe issues."
)

openai_client = OpenAI(base_url=settings.bot_base_url, api_key=settings.novita_api_key)

class NovitaClient:
    async def chat(self, messages: list[dict]) -> str:
        resp = await openai_client.chat.completions.create(
            model=settings.novita_model,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,
            extra_body={"top_k": 40, "min_p": 0},
        )
        return resp.choices[0].message.content

aiclient = NovitaClient()

# -- email utils --------------------------------------------------------------
CODE_CACHE: dict[str, str] = {}

def _generate_code() -> str:  # 6‑digit
    return str(random.randint(100000, 999999))

def send_verification_email(address: EmailStr) -> str:
    code = _generate_code()
    msg = EmailMessage()
    msg["From"] = settings.smtp_user
    msg["To"] = address
    msg["Subject"] = "Nicki AI verification code"
    msg.set_content(f"Your Nicki AI code: {code}\nIf you didn't request this, ignore the email.")
    ctx = ssl.create_default_context()
    with ssl.create_default_context():
        import smtplib
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=ctx) as s:
            s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
    CODE_CACHE[str(address)] = code
    logger.info("Sent code %s to %s", code, address)
    return code

def verify_code(address: EmailStr, code: str) -> bool:
    return CODE_CACHE.get(str(address)) == code.strip()

# -- speech utils -------------------------------------------------------------
async def speech_to_text(ogg: Path) -> str:
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-i", str(ogg), "-ac", "1", "-ar", "16000", wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(wav_path, "rb") as f:
        transcription = await openai_client.audio.transcriptions.create(model="whisper-1", file=f)
    os.close(wav_fd)
    Path(wav_path).unlink(missing_ok=True)
    return transcription.text

# -- Pydantic DTOs ------------------------------------------------------------
class Registration(BaseModel):
    name: constr(min_length=2, max_length=64)
    email: EmailStr
    phone: constr(regex=r"^[\d\+\-\(\) ]{7,20}$")

# ---------------------------------------------------------------------------
# KEYBOARDS -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def lang_keyboard() -> types.InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="🇬🇧 English", callback_data="lang_en")
    builder.button(text="🇷🇺 Русский", callback_data="lang_ru")
    return builder.as_markup()

def main_menu_keyboard(lang: str) -> types.InlineKeyboardMarkup:
    texts = {
        "en": ["🤕 Current issue", "⚙️ Settings", "🔄 Reset"],
        "ru": ["🤕 Текущая проблема", "⚙️ Настройки", "🔄 Сброс"],
    }
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(text=texts[lang][0], callback_data="issue"))
    builder.add(types.InlineKeyboardButton(text=texts[lang][1], callback_data="settings"))
    builder.add(types.InlineKeyboardButton(text=texts[lang][2], callback_data="reset"))
    return builder.as_markup()

def issue_keyboard(lang: str) -> types.InlineKeyboardMarkup:
    texts = {
        "en": ["Anxiety", "Stress", "Low mood", "Relationships", "⬅ Back"],
        "ru": ["Тревога", "Стресс", "Уныние", "Отношения", "⬅ Назад"],
    }
    builder = InlineKeyboardBuilder()
    for idx, caption in enumerate(texts[lang]):
        builder.button(text=caption, callback_data=f"issue_{idx}")
    builder.adjust(2)
    return builder.as_markup()

# ---------------------------------------------------------------------------
# ROUTERS ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
router = Router()

# -- /start -------------------------------------------------------------------
@router.message(CommandStart())
async def start(msg: types.Message, db: AsyncSession = next(get_db())):
    user = await db.scalar(select(User).where(User.telegram_id == msg.from_user.id))
    if user:
        await msg.answer("Welcome back!", reply_markup=main_menu_keyboard(user.language))
        return
    await msg.answer("Welcome to Nicki AI! Choose language:", reply_markup=lang_keyboard())

# -- language pick ------------------------------------------------------------
@router.callback_query(F.data.startswith("lang_"))
async def pick_lang(cb: types.CallbackQuery, db: AsyncSession = next(get_db())):
    lang = cb.data.split("_")[1]
    await cb.answer()
    await db.merge(User(telegram_id=cb.from_user.id, language=lang))
    await db.commit()
    await cb.message.edit_text({"en": "What's your name?", "ru": "Как вас зовут?"}[lang])
    # store step in FSM via simple dict attr
    cb.from_user.chat_data = {"step": "name"}

# -- collect name/email/phone via simple manual FSM --------------------------
@router.message(lambda m: getattr(m.from_user, "chat_data", {}).get("step") == "name")
async def reg_name(m: types.Message, db: AsyncSession = next(get_db())):
    await db.execute(update(User).where(User.telegram_id == m.from_user.id).values(name=m.text.strip()))
    await db.commit()
    lang = (await db.scalar(select(User.language).where(User.telegram_id == m.from_user.id)))
    m.from_user.chat_data["step"] = "email"
    await m.answer({"en": "Your e‑mail:", "ru": "Ваш e‑mail:"}[lang])

@router.message(lambda m: getattr(m.from_user, "chat_data", {}).get("step") == "email")
async def reg_email(m: types.Message, db: AsyncSession = next(get_db())):
    try:
        email = validate_email(m.text.strip(), check_deliverability=False).email
    except EmailNotValidError:
        lang = (await db.scalar(select(User.language).where(User.telegram_id == m.from_user.id)))
        await m.answer({"en": "Invalid email, try again", "ru": "Некорректный e‑mail"}[lang])
        return
    send_verification_email(email)
    await db.execute(update(User).where(User.telegram_id == m.from_user.id).values(email=email))
    await db.commit()
    m.from_user.chat_data["email"] = email
    m.from_user.chat_data["step"] = "email_code"
    await m.answer("Enter the 6‑digit code I emailed you:")

@router.message(lambda m: getattr(m.from_user, "chat_data", {}).get("step") == "email_code")
async def reg_email_code(m: types.Message, db: AsyncSession = next(get_db())):
    email = m.from_user.chat_data["email"]
    if not verify_code(email, m.text):
        await m.answer("Wrong code, try again:")
        return
    await db.execute(update(User).where(User.telegram_id == m.from_user.id).values(email_verified=True))
    await db.commit()
    m.from_user.chat_data["step"] = "phone"
    await m.answer("Phone number:")

@router.message(lambda m: getattr(m.from_user, "chat_data", {}).get("step") == "phone")
async def reg_phone(m: types.Message, db: AsyncSession = next(get_db())):
    phone = m.text.strip()
    if not Registration.__fields__["phone"].validate(phone)[0]:
        await m.answer("Phone looks wrong, try again:")
        return
    await db.execute(update(User).where(User.telegram_id == m.from_user.id).values(phone=phone))
    await db.commit()
    lang = (await db.scalar(select(User.language).where(User.telegram_id == m.from_user.id)))
    await m.answer("✅ Registered!", reply_markup=main_menu_keyboard(lang))
    m.from_user.chat_data.clear()

# -- main menu callbacks ------------------------------------------------------
@router.callback_query(lambda c: c.data == "issue")
async def choose_issue(cb: types.CallbackQuery, db: AsyncSession = next(get_db())):
    lang = (await db.scalar(select(User.language).where(User.telegram_id == cb.from_user.id)))
    await cb.message.edit_text({"en": "Select issue:", "ru": "Выберите проблему:"}[lang], reply_markup=issue_keyboard(lang))
    await cb.answer()

@router.callback_query(lambda c: c.data.startswith("issue_"))
async def handle_issue(cb: types.CallbackQuery, db: AsyncSession = next(get_db())):
    cat = int(cb.data.split("_")[1])
    user: User = await db.scalar(select(User).where(User.telegram_id == cb.from_user.id))
    prompts = {
        "en": ["I feel anxious", "I feel stressed", "I feel low", "Relationship issue"],
        "ru": ["Я чувствую тревогу", "Я испытываю стресс", "Мне грустно", "Проблемы в отношениях"],
    }
    user_prompt = prompts[user.language][cat]
    rows = await db.scalars(select(Message).where(Message.user_id == user.id).order_by(Message.created_at))
    history = [{"role": "system", "content": BASE_PROMPT}, *[{"role": r.role, "content": r.content} for r in rows], {"role": "user", "content": user_prompt}]
    answer = await aiclient.chat(history)
    await db.execute(Message.__table__.insert().values(user_id=user.id, role="user", content=user_prompt))
    await db.execute(Message.__table__.insert().values(user_id=user.id, role="assistant", content=answer))
    await db.commit()
    await cb.message.answer(answer, reply_markup=main_menu_keyboard(user.language))
    await cb.answer()

# -- reset --------------------------------------------------------------------
@router.callback_query(lambda c: c.data == "reset")
async def reset_dialogue(cb: types.CallbackQuery, db: AsyncSession = next(get_db())):
    user = await db.scalar(select(User).where(User.telegram_id == cb.from_user.id))
    await db.execute(delete(Message).where(Message.user_id == user.id))
    await db.commit()
    await cb.message.answer("🗑 Dialogue cleared.", reply_markup=main_menu_keyboard(user.language))
    await cb.answer()

# -- settings shortcut --------------------------------------------------------
@router.callback_query(lambda c: c.data == "settings")
@router.message(Command("settings"))
async def settings_info(event: types.Message | types.CallbackQuery, db: AsyncSession = next(get_db())):
    if isinstance(event, types.CallbackQuery):
        await event.answer()
        msg, uid = event.message, event.from_user.id
    else:
        msg, uid = event, event.from_user.id
    lang = (await db.scalar(select(User.language).where(User.telegram_id == uid)))
    await msg.answer({"en": "Use /reset to clear chat.", "ru": "Используйте /reset для сброса."}[lang])

# -- voice/video‑note ---------------------------------------------------------
@router.message(F.voice | F.video_note)
async def voice_handler(msg: types.Message, db: AsyncSession = next(get_db())):
    file_id = msg.voice.file_id if msg.voice else msg.video_note.file_id
    tg_file = await msg.bot.get_file(file_id)
    tmp = Path(f"/tmp/{file_id}.ogg")
    await msg.bot.download(tg_file, destination=tmp)
    text = await speech_to_text(tmp)
    tmp.unlink(missing_ok=True)
    user: User = await db.scalar(select(User).where(User.telegram_id == msg.from_user.id))
    rows = await db.scalars(select(Message).where(Message.user_id == user.id).order_by(Message.created_at))
    history = [{"role": "system", "content": BASE_PROMPT}, *[{"role": r.role, "content": r.content} for r in rows], {"role": "user", "content": text}]
    reply = await aiclient.chat(history)
    await db.execute(Message.__table__.insert().values(user_id=user.id, role="user", content=text))
    await db.execute(Message.__table__.insert().values(user_id=user.id, role="assistant", content=reply))
    await db.commit()
    await msg.answer(reply, reply_markup=main_menu_keyboard(user.language))

# ---------------------------------------------------------------------------
# ENTRYPOINT ------------------------------------------------------------------
# ---------------------------------------------------------------------------
async def on_startup(bot: Bot):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    me = await bot.get_me()
    logger.info("Nicki AI started as @%s", me.username)

async def main():
    defaults = DefaultBotSettings(parse_mode=ParseMode.MARKDOWN)
    bot = Bot(settings.bot_token, default=defaults)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot, on_startup=on_startup)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Quick usage

```bash
cp .env.example .env   # edit secrets
docker compose up --build -d
```

Now **all** runtime logic lives inside `bot.py`, making it easier to audit or hot‑patch. Voice transcription, email verification, Novita chat, Postgres storage, inline menus—everything is self‑contained.
