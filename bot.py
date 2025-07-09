"""
Nicki AI 2.0 — дружелюбный психолог-бот на Telegram.
Inline-меню, голосовые, статистика, админ-панель, ротация логов.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import ssl
import tempfile
import uuid
from datetime import datetime, timedelta, date
from email.message import EmailMessage
from pathlib import Path
from typing import List

from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode, ContentType
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup,
    KeyboardButton, BotCommand, Contact
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv
from email_validator import validate_email, EmailNotValidError
from openai import OpenAI
from pydantic import BaseModel, EmailStr, constr
from sqlalchemy import (
    String, Text, DateTime, BigInteger, ForeignKey, func, select, update, delete
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession
)
from sqlalchemy.orm import (
    Mapped, mapped_column, declarative_base, relationship
)
from logging.handlers import TimedRotatingFileHandler

# --------------------------------------------------------------------------- #
# ────────────────────────────  CONFIG & LOGGING  ─────────────────────────── #
# --------------------------------------------------------------------------- #
load_dotenv()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

handler = TimedRotatingFileHandler(
    filename=LOG_DIR / "bot.log",
    when="midnight",
    backupCount=14,
    encoding="utf-8"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[handler, logging.StreamHandler()]
)
logger = logging.getLogger("nickiai")

class Settings:
    """Load environment once."""
    def __init__(self):
        self.bot_token         = os.environ["BOT_TOKEN"]
        self.novita_api_key    = os.environ["NOVITA_API_KEY"]
        self.novita_model      = os.getenv("NOVITA_MODEL", "deepseek/deepseek-v3-0324")
        self.bot_base_url      = os.getenv("BOT_BASE_URL", "https://api.novita.ai/v3/openai")

        self.postgres_user     = os.environ["POSTGRES_USER"]
        self.postgres_password = os.environ["POSTGRES_PASSWORD"]
        self.postgres_db       = os.environ["POSTGRES_DB"]
        self.postgres_host     = os.getenv("POSTGRES_HOST", "db")
        self.postgres_port     = int(os.getenv("POSTGRES_PORT", "5432"))

        self.smtp_host         = os.environ["SMTP_HOST"]
        self.smtp_port         = int(os.getenv("SMTP_PORT", "465"))
        self.smtp_user         = os.environ["SMTP_USER"]
        self.smtp_password     = os.environ["SMTP_PASSWORD"]

        self.master_password   = os.environ["MASTER_PASSWORD"]

        self.dsn = (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

settings = Settings()

# --------------------------------------------------------------------------- #
# ────────────────────────────────  DATABASE  ─────────────────────────────── #
# --------------------------------------------------------------------------- #
Base = declarative_base()
engine = create_async_engine(settings.dsn, pool_pre_ping=True)
Session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine, expire_on_commit=False
)

class Role(Base):
    __tablename__ = "roles"
    id: Mapped[int]      = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str]    = mapped_column(String(32), unique=True)

class User(Base):
    __tablename__ = "users"
    id: Mapped[int]          = mapped_column(primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    name: Mapped[str]        = mapped_column(String(64), default="")
    language: Mapped[str]    = mapped_column(String(5), default="ru")
    email: Mapped[str | None]      = mapped_column(String(255))
    email_verified: Mapped[bool]   = mapped_column(default=False)
    phone: Mapped[str | None]      = mapped_column(String(32))
    phone_verified: Mapped[bool]   = mapped_column(default=False)
    created_at: Mapped[DateTime]   = mapped_column(DateTime, server_default=func.now())
    messages: Mapped[List["Message"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    roles: Mapped[List["UserRole"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

class UserRole(Base):
    __tablename__ = "user_roles"
    user_id: Mapped[int]  = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id: Mapped[int]  = mapped_column(ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)

    user: Mapped[User] = relationship(back_populates="roles")
    role: Mapped[Role] = relationship()

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID]   = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int]    = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[str]       = mapped_column(String(16))
    content: Mapped[str]    = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    user: Mapped[User]      = relationship(back_populates="messages")

class AdminInvite(Base):
    __tablename__ = "admin_invites"
    code: Mapped[str]       = mapped_column(primary_key=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())
    expires_at: Mapped[DateTime]
    used: Mapped[bool]      = mapped_column(default=False)

# --------------------------------------------------------------------------- #
# ─────────────────────────────────  AI  ──────────────────────────────────── #
# --------------------------------------------------------------------------- #
BASE_PROMPT = (
    "Ты Nicki AI — дружелюбный собеседник, обученный методикам когнитивно-поведенческой "
    "терапии. Поддерживай пользователя, задавай уточняющие вопросы, предлагай простые "
    "практики. Если запрос выходит за рамки самопомощи, мягко порекомендуй обратиться "
    "к лицензированному специалисту."
)

openai_client = OpenAI(
    base_url=settings.bot_base_url,
    api_key=settings.novita_api_key,
)

async def ai_chat(history: list[dict]) -> str:
    resp = await openai_client.chat.completions.create(
        model=settings.novita_model,
        messages=history,
        max_tokens=2048,
        temperature=0.75,
        top_p=1,
        extra_body={"top_k": 40, "min_p": 0},
    )
    return resp.choices[0].message.content

# --------------------------------------------------------------------------- #
# ─────────────────────────────  E-MAIL UTILS  ───────────────────────────── #
# --------------------------------------------------------------------------- #
CODE_CACHE: dict[str, str] = {}

def _gen_code() -> str:
    return str(random.randint(100_000, 999_999))

def send_email_code(address: EmailStr) -> None:
    """Send 6-digit code."""
    code = _gen_code()
    msg = EmailMessage()
    msg["From"] = settings.smtp_user
    msg["To"] = address
    msg["Subject"] = "Ваш код подтверждения Nicki AI"
    msg.set_content(
        f"Код подтверждения: {code}\n\n"
        f"Если это были не вы, просто проигнорируйте письмо."
    )
    ctx = ssl.create_default_context()
    with ssl.create_default_context() as ctx:
        with ssl.SSLContext().wrap_socket:
            pass
    with ssl.create_default_context() as ctx:  # dummy for mypy
        pass
    with ssl.create_default_context() as ctx:
        pass
    with ssl.create_default_context() as ctx:
        pass
    with ssl.create_default_context() as ctx:
        pass
    # actual sending:
    with ssl.create_default_context() as ctx:
        pass
    import smtplib
    with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=ctx) as s:
        s.login(settings.smtp_user, settings.smtp_password)
        s.send_message(msg)
    CODE_CACHE[address] = code
    logger.info("Email code %s sent to %s", code, address)

def verify_email_code(address: str, code: str) -> bool:
    return CODE_CACHE.get(address) == code.strip()

# --------------------------------------------------------------------------- #
# ────────────────────────────  VALIDATION MODELS  ───────────────────────── #
# --------------------------------------------------------------------------- #
class _Registration(BaseModel):
    name: constr(min_length=2, max_length=64)
    # email & phone validated separately

# --------------------------------------------------------------------------- #
# ────────────────────────────────  STATES  ──────────────────────────────── #
# --------------------------------------------------------------------------- #
class Reg(StatesGroup):
    name        = State()
    verify_type = State()
    email       = State()
    email_code  = State()
    phone       = State()

class SettingsFSM(StatesGroup):
    awaiting    = State()
    new_name    = State()
    new_email   = State()
    email_code  = State()
    new_phone   = State()

# --------------------------------------------------------------------------- #
# ────────────────────────────  KEYBOARD HELPERS  ────────────────────────── #
# --------------------------------------------------------------------------- #
def menu_kb(lang: str, is_admin: bool = False) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="🤕 Текущая проблема", callback_data="issue")
    builder.button(text="⚙️ Настройки", callback_data="settings")
    builder.button(text="🗑 Сброс диалога", callback_data="reset")
    if is_admin:
        builder.button(text="🛠️ Админ-панель", callback_data="admin")
    builder.adjust(1)
    return builder.as_markup()

def issue_kb(lang: str) -> InlineKeyboardMarkup:
    ru = ["😰 Тревога", "💼 Стресс", "😞 Уныние", "❤️ Отношения"]
    en = ["😰 Anxiety", "💼 Stress", "😞 Low mood", "❤️ Relationships"]
    texts = ru if lang == "ru" else en
    builder = InlineKeyboardBuilder()
    for i, t in enumerate(texts):
        builder.button(text=t, callback_data=f"issue_{i}")
    builder.button(text="🏠 Главное меню", callback_data="home")
    builder.adjust(2)
    return builder.as_markup()

def settings_kb(lang: str) -> InlineKeyboardMarkup:
    ru = ["🌐 Сменить язык", "📧 Сменить e-mail", "📱 Сменить телефон",
          "📝 Сменить имя", "🗑 Удалить данные", "🏠 Главное меню"]
    en = ["🌐 Change language", "📧 Change e-mail", "📱 Change phone",
          "📝 Change name", "🗑 Delete data", "🏠 Home"]
    texts = ru if lang == "ru" else en
    builder = InlineKeyboardBuilder()
    for idx, t in enumerate(texts):
        builder.button(text=t, callback_data=f"set_{idx}")
    builder.adjust(1)
    return builder.as_markup()

def admin_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="📊 Статистика", callback_data="a_stats")
    builder.button(text="➕ Добавить админа", callback_data="a_add")
    builder.button(text="➖ Удалить админа", callback_data="a_del")
    builder.button(text="🏠 Главное меню", callback_data="home")
    builder.adjust(1)
    return builder.as_markup()

def verify_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="🔒 E-mail", callback_data="ver_email")
    builder.button(text="📞 Телефон", callback_data="ver_phone")
    builder.adjust(2)
    return builder.as_markup()

# --------------------------------------------------------------------------- #
# ────────────────────────────────  ROUTER  ──────────────────────────────── #
# --------------------------------------------------------------------------- #
router = Router()

async def _is_admin(tg_id: int, session: AsyncSession) -> bool:
    q = (
        select(Role.name)
        .join(UserRole, Role.id == UserRole.role_id)
        .join(User, User.id == UserRole.user_id)
        .where(User.telegram_id == tg_id)
    )
    roles = [r for (r,) in await session.execute(q)]
    return "admin" in roles

# -----------------------  /start  ---------------------------------------- #
@router.message(CommandStart())
async def cmd_start(m: Message, state: FSMContext):
    async with Session() as db:
        user = await db.scalar(select(User).where(User.telegram_id == m.from_user.id))
        if user:
            # Приветствие старого пользователя
            is_admin = await _is_admin(m.from_user.id, db)
            await m.answer(
                "👋 Снова привет! Чем могу помочь сегодня?",
                reply_markup=menu_kb(user.language, is_admin)
            )
        else:
            # Новый пользователь
            await m.answer(
                "Привет, я Nicki AI 🦋\n\n"
                "Я умею слушать, задавать вопросы и предлагать техники, "
                "которые помогают справляться со стрессом, тревогой и унынием. "
                "Давай познакомимся! Как тебя зовут?",
            )
            await state.set_state(Reg.name)
    await _delete_cmd(m)

# -----------------------  REGISTRATION  ----------------------------------- #
@router.message(Reg.name, F.text.len() > 1)
async def reg_name(m: Message, state: FSMContext):
    name = m.text.strip()
    try:
        _Registration(name=name)
    except Exception:
        await m.answer("Имя должно быть от 2 до 64 символов. Попробуй ещё раз:")
        return
    await state.update_data(name=name)
    await state.set_state(Reg.verify_type)
    await m.answer(
        f"Рада знакомству, {name}! Какой способ подтверждения выберешь?",
        reply_markup=verify_kb()
    )
    await _delete_cmd(m)

@router.callback_query(Reg.verify_type, F.data == "ver_email")
async def reg_pick_email(cb: CallbackQuery, state: FSMContext):
    await state.set_state(Reg.email)
    await cb.message.edit_text("Укажи свой e-mail:")
    await cb.answer()

@router.callback_query(Reg.verify_type, F.data == "ver_phone")
async def reg_pick_phone(cb: CallbackQuery, state: FSMContext):
    await state.set_state(Reg.phone)
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📲 Поделиться контактом", request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    await cb.message.answer("Нажми кнопку ниже, чтобы отправить свой контакт 👇", reply_markup=kb)
    await cb.answer()

@router.message(Reg.email, F.text)
async def reg_email(m: Message, state: FSMContext):
    try:
        email = validate_email(m.text.strip(), check_deliverability=False).email
    except EmailNotValidError:
        await m.answer("Хмм… кажется, это не похоже на настоящий e-mail. Попробуй ещё:")
        return
    send_email_code(email)
    await state.update_data(email=email)
    await state.set_state(Reg.email_code)
    await m.answer("Я отправила 6-значный код на твою почту. Введи его здесь:")
    await _delete_cmd(m)

@router.message(Reg.email_code, F.text)
async def reg_email_code(m: Message, state: FSMContext):
    data = await state.get_data()
    if not verify_email_code(data["email"], m.text):
        await m.answer("Код не подходит. Попробуй снова:")
        return
    await state.update_data(email_verified=True)
    # Финал регистрации
    await _finish_registration(m, state)

@router.message(Reg.phone, F.contact)
async def reg_phone_contact(m: Message, state: FSMContext):
    contact: Contact = m.contact
    if contact.user_id != m.from_user.id:
        await m.answer("Можно поделиться только собственным контактом 💡")
        return
    phone = contact.phone_number
    await state.update_data(phone=phone, phone_verified=True)
    await _finish_registration(m, state)
    # убрать reply-клаву
    await m.answer("👍 Готово!", reply_markup=types.ReplyKeyboardRemove())

async def _finish_registration(m: Message, state: FSMContext):
    data = await state.get_data()
    async with Session() as db:
        user = User(
            telegram_id=m.from_user.id,
            name=data["name"],
            email=data.get("email"),
            email_verified=data.get("email_verified", False),
            phone=data.get("phone"),
            phone_verified=data.get("phone_verified", False),
        )
        db.add(user)
        # роль "user"
        role_user = await db.scalar(select(Role).where(Role.name == "user"))
        if not role_user:
            role_user = Role(name="user")
            db.add(role_user)
            await db.flush()
        db.add(UserRole(user=user, role=role_user))
        await db.commit()

    is_admin = False
    await m.answer(
        "🎉 Регистрация завершена! Как ты себя сейчас чувствуешь?",
        reply_markup=menu_kb("ru", is_admin)
    )
    await state.clear()
    await _delete_cmd(m)

# -----------------------  ISSUES  ---------------------------------------- #
@router.callback_query(F.data == "issue")
async def cb_issue_root(cb: CallbackQuery):
    async with Session() as db:
        lang = await db.scalar(
            select(User.language).where(User.telegram_id == cb.from_user.id)
        )
    await cb.message.edit_text("Выбери, что беспокоит:", reply_markup=issue_kb(lang))
    await cb.answer()

@router.callback_query(F.data.startswith("issue_"))
async def cb_issue_selected(cb: CallbackQuery):
    cat = int(cb.data.split("_")[1])
    prompts_ru = ["Я чувствую тревогу", "Я испытываю стресс",
                  "Мне грустно", "У меня проблемы в отношениях"]
    prompts_en = ["I feel anxious", "I feel stressed",
                  "I feel low", "Relationship issue"]
    async with Session() as db:
        user = await db.scalar(select(User).where(User.telegram_id == cb.from_user.id))
        prompt = prompts_ru[cat] if user.language == "ru" else prompts_en[cat]

        history_rows = await db.scalars(
            select(Message).where(Message.user_id == user.id).order_by(Message.created_at)
        )
        history = (
            [{"role": "system", "content": BASE_PROMPT}] +
            [{"role": r.role, "content": r.content} for r in history_rows] +
            [{"role": "user", "content": prompt}]
        )
        answer = await ai_chat(history)
        await db.add_all([
            Message(user_id=user.id, role="user", content=prompt),
            Message(user_id=user.id, role="assistant", content=answer)
        ])
        await db.commit()

        is_admin = await _is_admin(cb.from_user.id, db)
    await cb.message.answer(answer, reply_markup=menu_kb(user.language, is_admin))
    await cb.answer()

# -----------------------  SETTINGS  -------------------------------------- #
@router.callback_query(F.data == "settings")
async def cb_settings(cb: CallbackQuery, state: FSMContext):
    async with Session() as db:
        lang = await db.scalar(
            select(User.language).where(User.telegram_id == cb.from_user.id)
        )
    await cb.message.edit_text("⚙️ Настройки:", reply_markup=settings_kb(lang))
    await state.set_state(SettingsFSM.awaiting)
    await cb.answer()

@router.callback_query(SettingsFSM.awaiting, F.data == "set_0")  # language
async def set_lang(cb: CallbackQuery, state: FSMContext):
    async with Session() as db:
        user = await db.scalar(select(User).where(User.telegram_id == cb.from_user.id))
        new_lang = "en" if user.language == "ru" else "ru"
        await db.execute(
            update(User).where(User.id == user.id).values(language=new_lang)
        )
        await db.commit()
    await cb.message.edit_text(
        ("✨ Language switched to English!" if new_lang == "en" else "✨ Язык переключён на русский!"),
        reply_markup=settings_kb(new_lang)
    )
    await cb.answer()

@router.callback_query(SettingsFSM.awaiting, F.data == "set_1")  # change email
async def set_email(cb: CallbackQuery, state: FSMContext):
    await cb.message.edit_text("Введи новый e-mail:")
    await state.set_state(SettingsFSM.new_email)
    await cb.answer()

@router.message(SettingsFSM.new_email, F.text)
async def process_new_email(m: Message, state: FSMContext):
    try:
        email = validate_email(m.text.strip(), check_deliverability=False).email
    except EmailNotValidError:
        await m.answer("Это не похоже на правильный адрес, попробуй снова:")
        return
    send_email_code(email)
    await state.update_data(new_email=email)
    await state.set_state(SettingsFSM.email_code)
    await m.answer("Код отправлен! Введи его сюда:")
    await _delete_cmd(m)

@router.message(SettingsFSM.email_code, F.text)
async def process_email_code(m: Message, state: FSMContext):
    data = await state.get_data()
    if not verify_email_code(data["new_email"], m.text):
        await m.answer("Код не совпадает. Попробуй ещё:")
        return
    async with Session() as db:
        await db.execute(
            update(User)
            .where(User.telegram_id == m.from_user.id)
            .values(email=data["new_email"], email_verified=True)
        )
        await db.commit()
    await m.answer("📧 E-mail обновлён!", reply_markup=settings_kb("ru"))
    await state.set_state(SettingsFSM.awaiting)
    await _delete_cmd(m)

@router.callback_query(SettingsFSM.awaiting, F.data == "set_2")  # change phone
async def set_phone(cb: CallbackQuery, state: FSMContext):
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📲 Поделиться контактом", request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    await cb.message.answer("Нажми кнопку, чтобы отправить новый контакт:", reply_markup=kb)
    await state.set_state(SettingsFSM.new_phone)
    await cb.answer()

@router.message(SettingsFSM.new_phone, F.contact)
async def process_new_phone(m: Message, state: FSMContext):
    phone = m.contact.phone_number
    async with Session() as db:
        await db.execute(
            update(User)
            .where(User.telegram_id == m.from_user.id)
            .values(phone=phone, phone_verified=True)
        )
        await db.commit()
    await m.answer("📱 Телефон обновлён!", reply_markup=settings_kb("ru"))
    await state.set_state(SettingsFSM.awaiting)

@router.callback_query(SettingsFSM.awaiting, F.data == "set_3")  # change name
async def set_name(cb: CallbackQuery, state: FSMContext):
    await cb.message.edit_text("Как мне теперь к тебе обращаться?")
    await state.set_state(SettingsFSM.new_name)
    await cb.answer()

@router.message(SettingsFSM.new_name, F.text)
async def process_new_name(m: Message, state: FSMContext):
    name = m.text.strip()
    try:
        _Registration(name=name)
    except Exception:
        await m.answer("Имя должно быть 2-64 символа. Попробуй ещё раз:")
        return
    async with Session() as db:
        await db.execute(
            update(User)
            .where(User.telegram_id == m.from_user.id)
            .values(name=name)
        )
        await db.commit()
    await m.answer(f"Имя обновлено на {name}!", reply_markup=settings_kb("ru"))
    await state.set_state(SettingsFSM.awaiting)
    await _delete_cmd(m)

@router.callback_query(SettingsFSM.awaiting, F.data == "set_4")  # delete data
async def delete_all_data(cb: CallbackQuery, state: FSMContext):
    async with Session() as db:
        user = await db.scalar(select(User).where(User.telegram_id == cb.from_user.id))
        await db.execute(delete(Message).where(Message.user_id == user.id))
        await db.execute(delete(UserRole).where(UserRole.user_id == user.id))
        await db.execute(delete(User).where(User.id == user.id))
        await db.commit()
    await cb.message.edit_text(
        "🗑 Данные удалены. Если захочешь вернуться, просто введи /start.",
        reply_markup=None
    )
    await state.clear()
    await cb.answer()

@router.callback_query(F.data == "home")
async def cb_home(cb: CallbackQuery):
    async with Session() as db:
        user = await db.scalar(select(User).where(User.telegram_id == cb.from_user.id))
        is_admin = await _is_admin(cb.from_user.id, db)
    await cb.message.edit_text("🏠 Главное меню:", reply_markup=menu_kb(user.language, is_admin))
    await cb.answer()

# -----------------------  RESET  ----------------------------------------- #
@router.callback_query(F.data == "reset")
@router.message(Command("reset"))
async def reset_dialog(ev: Message | CallbackQuery):
    if isinstance(ev, Message):
        user_id = ev.from_user.id
        msg = ev
    else:
        user_id = ev.from_user.id
        msg = ev.message
    async with Session() as db:
        user = await db.scalar(select(User).where(User.telegram_id == user_id))
        if user:
            await db.execute(delete(Message).where(Message.user_id == user.id))
            await db.commit()
        is_admin = await _is_admin(user_id, db)
    await msg.answer("🗑 История очищена!", reply_markup=menu_kb("ru", is_admin))
    if isinstance(ev, Message):
        await _delete_cmd(ev)
    else:
        await ev.answer()

# -----------------------  STATISTICS & ADMIN  ---------------------------- #
@router.callback_query(F.data == "admin")
@router.message(Command("admin"))
async def admin_panel(ev: Message | CallbackQuery):
    if isinstance(ev, Message):
        user_id = ev.from_user.id
        msg = ev
    else:
        user_id = ev.from_user.id
        msg = ev.message
    async with Session() as db:
        if not await _is_admin(user_id, db):
            await msg.answer("🚫 Панель администратора недоступна.")
            return
    await msg.answer("🛠️ Панель администратора:", reply_markup=admin_kb())
    if isinstance(ev, Message):
        await _delete_cmd(ev)

@router.callback_query(F.data == "a_stats")
async def admin_stats(cb: CallbackQuery):
    async with Session() as db:
        today = date.today()
        d1 = datetime.combine(today, datetime.min.time())
        d7 = d1 - timedelta(days=7)
        d30 = d1 - timedelta(days=30)

        # новые
        new_day = await db.scalar(select(func.count()).select_from(User).where(User.created_at >= d1))
        new_week = await db.scalar(select(func.count()).select_from(User).where(User.created_at >= d7))
        new_month = await db.scalar(select(func.count()).select_from(User).where(User.created_at >= d30))
        # активные
        act_day = await db.scalar(
            select(func.count(func.distinct(Message.user_id)))
            .where(Message.created_at >= d1)
        )
        act_week = await db.scalar(
            select(func.count(func.distinct(Message.user_id)))
            .where(Message.created_at >= d7)
        )
        act_month = await db.scalar(
            select(func.count(func.distinct(Message.user_id)))
            .where(Message.created_at >= d30)
        )
    text = (
        "📊 *Статистика*\n"
        f"Новых пользователей: 24ч — {new_day}, 7д — {new_week}, 30д — {new_month}\n"
        f"Активных старых: 24ч — {act_day}, 7д — {act_week}, 30д — {act_month}"
    )
    await cb.message.edit_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=admin_kb())
    await cb.answer()

@router.callback_query(F.data == "a_add")
async def admin_add(cb: CallbackQuery):
    code = str(uuid.uuid4())[:8]
    expires = datetime.utcnow() + timedelta(hours=6)
    async with Session() as db:
        inv = AdminInvite(code=code, created_by=None, expires_at=expires)  # created_by optional
        db.add(inv)
        await db.commit()
    await cb.message.edit_text(
        f"Скопируй команду и отправь будущему админу:\n\n"
        f"`/becomeadmin {code}`\n\n"
        "Код активен 6 часов.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=admin_kb()
    )
    await cb.answer()

@router.message(Command("becomeadmin"))
async def cmd_become_admin(m: Message):
    parts = m.text.strip().split(maxsplit=1)
    if len(parts) != 2:
        await m.answer("⚠️ Формат: /becomeadmin <код>")
        return
    code = parts[1]
    async with Session() as db:
        inv = await db.scalar(
            select(AdminInvite).where(
                AdminInvite.code == code,
                AdminInvite.expires_at > datetime.utcnow(),
                AdminInvite.used == False
            )
        )
        if not inv:
            await m.answer("⛔️ Код недействителен или устарел.")
            return
        user = await db.scalar(select(User).where(User.telegram_id == m.from_user.id))
        if not user:
            await m.answer("Сначала зарегистрируйся, затем повтори команду 😉")
            return
        role_admin = await db.scalar(select(Role).where(Role.name == "admin"))
        if not role_admin:
            role_admin = Role(name="admin")
            db.add(role_admin)
            await db.flush()
        db.add(UserRole(user_id=user.id, role_id=role_admin.id))
        inv.used = True
        await db.commit()
    await m.answer("🎉 Теперь ты администратор!", reply_markup=admin_kb())
    await _delete_cmd(m)

# --------------------------------------------------------------------------- #
# ─────────────────────────────  UTILITIES  ──────────────────────────────── #
# --------------------------------------------------------------------------- #
async def _delete_cmd(msg: Message):
    """Попытаться удалить команду пользователя для чистоты чата."""
    try:
        await msg.delete()
    except Exception:
        pass  # not critical

# --------------------------------------------------------------------------- #
# ─────────────────────────────  STARTUP  ────────────────────────────────── #
# --------------------------------------------------------------------------- #
async def on_startup(bot: Bot):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # ensure roles
        for role_name in ("user", "admin"):
            await conn.execute(
                Role.__table__.insert()
                .values(name=role_name)
                .on_conflict_do_nothing(index_elements=["name"])
            )
    me = await bot.get_me()
    logger.info("Nicki AI started as @%s", me.username)
    # set bot commands
    cmds = [
        BotCommand(command="start", description="Перезапустить бота 🏁"),
        BotCommand(command="issue", description="Открыть список проблем 🤕"),
        BotCommand(command="settings", description="Настройки ⚙️"),
        BotCommand(command="reset", description="Сброс диалога 🗑"),
        BotCommand(command="admin", description="Админ-панель 🛠️"),
    ]
    await bot.set_my_commands(cmds)

async def main() -> None:
    defaults = DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    bot = Bot(settings.bot_token, default=defaults)
    dp = Dispatcher()
    dp.include_router(router)
    dp.startup.register(on_startup)
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())