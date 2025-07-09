# bot.py
"""Nicki AI – single-file Telegram bot for mental-health support."""
import os
import asyncio
import logging
import random
import ssl
import subprocess
import tempfile
import uuid
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import List

from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import BotCommand
from aiogram.utils.keyboard import InlineKeyboardBuilder
from email_validator import validate_email, EmailNotValidError
from openai import OpenAI
from pydantic import BaseModel, EmailStr, constr
from sqlalchemy import String, Text, DateTime, ForeignKey, select, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, declarative_base, relationship
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
load_dotenv()

class Settings:
    def __init__(self):
        self.bot_token         = os.getenv("BOT_TOKEN")         or self._fail("BOT_TOKEN")
        self.novita_api_key    = os.getenv("NOVITA_API_KEY")    or self._fail("NOVITA_API_KEY")
        self.novita_model      = os.getenv("NOVITA_MODEL", "deepseek/deepseek-v3-0324")
        self.bot_base_url      = os.getenv("BOT_BASE_URL", "https://api.novita.ai/v3/openai")
        self.postgres_user     = os.getenv("POSTGRES_USER")     or self._fail("POSTGRES_USER")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD") or self._fail("POSTGRES_PASSWORD")
        self.postgres_db       = os.getenv("POSTGRES_DB")       or self._fail("POSTGRES_DB")
        self.postgres_host     = os.getenv("POSTGRES_HOST", "db")
        self.postgres_port     = int(os.getenv("POSTGRES_PORT", "5432"))
        self.smtp_host         = os.getenv("SMTP_HOST")         or self._fail("SMTP_HOST")
        self.smtp_port         = int(os.getenv("SMTP_PORT", "465"))
        self.smtp_user         = os.getenv("SMTP_USER")         or self._fail("SMTP_USER")
        self.smtp_password     = os.getenv("SMTP_PASSWORD")     or self._fail("SMTP_PASSWORD")
        self.dsn = (
            f"postgresql+asyncpg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @staticmethod
    def _fail(name: str):
        raise RuntimeError(f"Environment variable {name} is required")

settings = Settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nickiai")

# ---------------------------------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------------------------------
Base = declarative_base()
engine = create_async_engine(settings.dsn, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

class User(Base):
    __tablename__ = "users"
    id: Mapped[int]              = mapped_column(primary_key=True, autoincrement=True)
    telegram_id: Mapped[int]     = mapped_column(unique=True, index=True)
    language: Mapped[str]        = mapped_column(String(5), default="en")
    name: Mapped[str]            = mapped_column(String(64), default="")
    email: Mapped[str]           = mapped_column(String(255), unique=True, nullable=True)
    phone: Mapped[str]           = mapped_column(String(32), nullable=True)
    email_verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default="now()")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID]        = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int]         = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[str]            = mapped_column(String(16))
    content: Mapped[str]         = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default="now()")
    user: Mapped[User]           = relationship(back_populates="messages")

# ---------------------------------------------------------------------------
# AI & UTILITIES
# ---------------------------------------------------------------------------
BASE_PROMPT = (
    "You are Nicki AI, an empathetic assistant trained in CBT techniques. "
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

CODE_CACHE: dict[str, str] = {}
def _generate_code() -> str:
    return str(random.randint(100000, 999999))

def send_verification_email(address: EmailStr) -> None:
    code = _generate_code()
    msg = EmailMessage()
    msg["From"] = settings.smtp_user
    msg["To"] = address
    msg["Subject"] = "Nicki AI verification code"
    msg.set_content(f"Your Nicki AI code: {code}\nIf you didn't request this, ignore this email.")
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=ctx) as s:
        s.login(settings.smtp_user, settings.smtp_password)
        s.send_message(msg)
    CODE_CACHE[address] = code
    logger.info("Sent code %s to %s", code, address)

def verify_code(address: EmailStr, code: str) -> bool:
    return CODE_CACHE.get(address) == code.strip()

async def speech_to_text(ogg: Path) -> str:
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-i", str(ogg), "-ac", "1", "-ar", "16000", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    with open(wav_path, "rb") as f:
        transcription = await openai_client.audio.transcriptions.create(model="whisper-1", file=f)
    os.close(wav_fd)
    Path(wav_path).unlink(missing_ok=True)
    return transcription.text

class Registration(BaseModel):
    name: constr(min_length=2, max_length=64)
    email: EmailStr
    phone: constr(pattern=r"^[\d\+\-\(\) ]{7,20}$")

# ---------------------------------------------------------------------------
# KEYBOARDS
# ---------------------------------------------------------------------------
def lang_keyboard() -> types.InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🇬🇧 English", callback_data="lang_en")
    b.button(text="🇷🇺 Русский", callback_data="lang_ru")
    return b.as_markup()

def main_menu_keyboard(lang: str) -> types.InlineKeyboardMarkup:
    opts = {"en": ["🤕 Current issue","⚙️ Settings","🔄 Reset"], "ru": ["🤕 Текущая проблема","⚙️ Настройки","🔄 Сброс"]}[lang]
    b = InlineKeyboardBuilder()
    for idx,label in enumerate(opts):
        b.button(text=label, callback_data=["issue","settings","reset"][idx])
    return b.as_markup()

def issue_keyboard(lang: str) -> types.InlineKeyboardMarkup:
    texts = {"en":["Anxiety","Stress","Low mood","Relationships","⬅ Back"],"ru":["Тревога","Стресс","Уныние","Отношения","⬅ Назад"]}[lang]
    b = InlineKeyboardBuilder()
    for i,t in enumerate(texts):
        b.button(text=t, callback_data=f"issue_{i}")
    b.adjust(2)
    return b.as_markup()

# ---------------------------------------------------------------------------
# ROUTERS
# ---------------------------------------------------------------------------
router = Router()

@router.message(CommandStart())
async def start(msg: types.Message):
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.telegram_id == msg.from_user.id))
        if user:
            await msg.answer("Welcome back!", reply_markup=main_menu_keyboard(user.language))
        else:
            await db.merge(User(telegram_id=msg.from_user.id))
            await db.commit()
            await msg.answer("Welcome to Nicki AI! Choose language:", reply_markup=lang_keyboard())

@router.callback_query(F.data.startswith("lang_"))
async def pick_lang(cb: types.CallbackQuery):
    lang = cb.data.split("_", 1)[1]
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(User)
            .where(User.telegram_id == cb.from_user.id)
            .values(language=lang)
        )
        await db.commit()

    await cb.message.edit_text(
        {"en": "What's your name?", "ru": "Как вас зовут?"}[lang],
        reply_markup=None
    )
    cb.from_user.chat_data = {"step": "name"}
    await cb.answer()


def settings_keyboard(lang: str) -> types.InlineKeyboardMarkup:
    texts = {
        "en": ["Change language", "Change e-mail", "Change phone", "⬅ Back"],
        "ru": ["Изменить язык", "Изменить e-mail", "Изменить телефон", "⬅ Назад"],
    }[lang]
    b = InlineKeyboardBuilder()
    for i, t in enumerate(texts):
        b.button(text=t, callback_data=f"settings_{i}")
    b.adjust(1)
    return b.as_markup()


@router.callback_query(F.data == "settings")
async def open_settings(cb: types.CallbackQuery):
    async with AsyncSessionLocal() as db:
        lang = await db.scalar(
            select(User.language)
            .where(User.telegram_id == cb.from_user.id)
        )
    await cb.message.edit_text(
        {"en": "Settings", "ru": "Настройки"}[lang],
        reply_markup=settings_keyboard(lang)
    )
    await cb.answer()


@router.message(lambda m: getattr(m.from_user, "chat_data", {}).get("step") == "name")
async def reg_name(m: types.Message):
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(User)
            .where(User.telegram_id == m.from_user.id)
            .values(name=m.text.strip())
        )
        await db.commit()
        lang = await db.scalar(
            select(User.language).where(User.telegram_id == m.from_user.id)
        )

    m.from_user.chat_data["step"] = "email"
    await m.answer(
        {"en": "Your e-mail:", "ru": "Ваш e-mail:"}[lang]
    )


@router.message(lambda m: getattr(m.from_user, "chat_data", {}).get("step") == "email")
async def reg_email(m: types.Message):
    try:
        email = validate_email(
            m.text.strip(),
            check_deliverability=False
        ).email
    except EmailNotValidError:
        async with AsyncSessionLocal() as db:
            lang = await db.scalar(
                select(User.language).where(User.telegram_id == m.from_user.id)
            )
        await m.answer(
            {"en": "Invalid email, try again", "ru": "Некорректный e-mail"}[lang]
        )
        return

    send_verification_email(email)
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(User)
            .where(User.telegram_id == m.from_user.id)
            .values(email=email)
        )
        await db.commit()

    m.from_user.chat_data.update({"step": "email_code", "email": email})
    await m.answer("Enter the 6-digit code I emailed you:")

@router.message(lambda m: getattr(m.from_user, "chat_data", {}).get("step") == "email_code")
async def reg_email_code(m: types.Message):
    email = m.from_user.chat_data["email"]
    if not verify_code(email, m.text):
        await m.answer("Wrong code, try again:")
        return
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(User)
            .where(User.telegram_id == m.from_user.id)
            .values(email_verified=True)
        )
        await db.commit()
    m.from_user.chat_data["step"] = "phone"
    await m.answer("Phone number:")

@router.message(lambda m: getattr(m.from_user, "chat_data", {}).get("step") == "phone")
async def reg_phone(m: types.Message):
    phone = m.text.strip()
    try:
        Registration(name="x", email="x@x.com", phone=phone)
    except Exception:
        await m.answer("Phone looks wrong, try again:")
        return
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(User)
            .where(User.telegram_id == m.from_user.id)
            .values(phone=phone)
        )
        await db.commit()
        lang = await db.scalar(
            select(User.language).where(User.telegram_id == m.from_user.id)
        )
    await m.answer("✅ Registered!", reply_markup=main_menu_keyboard(lang))
    m.from_user.chat_data.clear()

@router.callback_query(lambda c: c.data == "issue")
async def choose_issue(cb: types.CallbackQuery):
    async with AsyncSessionLocal() as db:
        lang = await db.scalar(
            select(User.language).where(User.telegram_id == cb.from_user.id)
        )
    await cb.message.edit_text("Select issue:", reply_markup=issue_keyboard(lang))
    await cb.answer()

@router.callback_query(lambda c: c.data.startswith("issue_"))
async def handle_issue(cb: types.CallbackQuery):
    cat = int(cb.data.split("_", 1)[1])
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.telegram_id == cb.from_user.id))
        prompts = {
            "en": ["I feel anxious", "I feel stressed", "I feel low", "Relationship issue"],
            "ru": ["Я чувствую тревогу", "Я испытываю стресс", "Мне грустно", "Проблемы в отношениях"],
        }
        user_prompt = prompts[user.language][cat]
        rows = await db.scalars(
            select(Message).where(Message.user_id == user.id).order_by(Message.created_at)
        )
        history = (
            [{"role": "system", "content": BASE_PROMPT}]
            + [{"role": r.role, "content": r.content} for r in rows]
            + [{"role": "user", "content": user_prompt}]
        )
        answer = await aiclient.chat(history)
        await db.execute(
            Message.__table__.insert().values(user_id=user.id, role="user", content=user_prompt)
        )
        await db.execute(
            Message.__table__.insert().values(user_id=user.id, role="assistant", content=answer)
        )
        await db.commit()
    await cb.message.answer(answer, reply_markup=main_menu_keyboard(user.language))
    await cb.answer()


@router.callback_query(F.data == "settings")
@router.message(Command("settings"))
async def settings_info(event: types.CallbackQuery | types.Message):
    if isinstance(event, types.CallbackQuery):
        await event.answer()               
        msg = event.message
        user_id = event.from_user.id
    else:
        msg = event
        user_id = event.from_user.id
    async with AsyncSessionLocal() as db:
        lang = await db.scalar(
            select(User.language).where(User.telegram_id == user_id)
        )
    text = {"en": "Settings", "ru": "Настройки"}[lang]
    kb   = settings_keyboard(lang)
    if isinstance(event, types.CallbackQuery):
        await msg.edit_text(text, reply_markup=kb)
    else:
        await msg.answer(text, reply_markup=kb)


@router.message(F.voice | F.video_note)
async def voice_handler(msg: types.Message):
    fid = msg.voice.file_id if msg.voice else msg.video_note.file_id
    tg_file = await msg.bot.get_file(fid)
    tmp = Path(f"/tmp/{fid}.ogg")
    await msg.bot.download(tg_file, destination=tmp)
    text = await speech_to_text(tmp)
    tmp.unlink(missing_ok=True)
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.telegram_id == msg.from_user.id))  
        rows = await db.scalars(
            select(Message).where(Message.user_id == user.id).order_by(Message.created_at)
        )
        history = (
            [{"role": "system", "content": BASE_PROMPT}]
            + [{"role": r.role, "content": r.content} for r in rows]
            + [{"role": "user", "content": text}]
        )
        reply = await aiclient.chat(history)
        await db.execute(
            Message.__table__.insert().values(user_id=user.id, role="user", content=text)
        )
        await db.execute(
            Message.__table__.insert().values(user_id=user.id, role="assistant", content=reply)
        )
        await db.commit()
    await msg.answer(reply, reply_markup=main_menu_keyboard(user.language))


@router.message(Command(commands=["reset"]))
async def reset_command(message: types.Message):
    async with AsyncSessionLocal() as db:
        user = await db.scalar(
            select(User).where(User.telegram_id == message.from_user.id)
        )
        lang = user.language if user else "en"
        if user:
            await db.execute(
                delete(Message).where(Message.user_id == user.id)
            )
            await db.commit()
    message.from_user.chat_data.clear()
    await message.answer(
        "🗑 Диалог очищен.",
        reply_markup=main_menu_keyboard(lang)
    )


@router.callback_query(F.data == "reset")
async def reset_callback(cb: types.CallbackQuery):
    async with AsyncSessionLocal() as db:
        user = await db.scalar(
            select(User).where(User.telegram_id == cb.from_user.id)
        )
        lang = user.language if user else "en"
        if user:
            await db.execute(
                delete(Message).where(Message.user_id == user.id)
            )
            await db.commit()
    cb.from_user.chat_data.clear()
    await cb.message.edit_text(
        {
            "en": "All set! Let's start over. How are you feeling today?",
            "ru": "Готово! Давайте начнём заново. Что у вас на душе?"
        }[lang],
        reply_markup=main_menu_keyboard(lang)
    )
    await cb.answer("Conversation reset ✔️")

# ---------------------------------------------------------------------------
# STARTUP & POLLING
# ---------------------------------------------------------------------------
async def on_startup(bot: Bot):
    await bot.delete_webhook(drop_pending_updates=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    me = await bot.get_me()
    logger.info("Nicki AI started as @%s", me.username)

async def main():
    defaults = DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    bot = Bot(settings.bot_token, default=defaults)
    await bot.delete_webhook(drop_pending_updates=True)
    dp = Dispatcher()
    dp.include_router(router)
    dp.startup.register(on_startup)
    await dp.start_polling(bot, skip_updates=True)
    defaults = DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    bot = Bot(settings.bot_token, default=defaults)
    dp = Dispatcher()
    dp.include_router(router)
    dp.startup.register(on_startup)
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
