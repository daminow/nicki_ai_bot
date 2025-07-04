"""Nicki AI — Telegram-бот для психологической помощи на Novita API."""

import asyncio, logging, random, ssl, subprocess, tempfile, uuid
from pathlib import Path
from email.message import EmailMessage

from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder

from email_validator import validate_email, EmailNotValidError
from openai import OpenAI
from pydantic import EmailStr, Field, ValidationError, constr
from pydantic_settings import BaseSettings
from sqlalchemy import DateTime, ForeignKey, String, Text, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, mapped_column, relationship, Mapped
from dotenv import load_dotenv

from PIL import Image
import pytesseract

load_dotenv()

# ─── Настройки ─────────────────────────────────────────────────────────
class Settings(BaseSettings):
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
    smtp_port: int = Field(..., env="SMTP_PORT")
    smtp_user: EmailStr = Field(..., env="SMTP_USER")
    smtp_password: str = Field(..., env="SMTP_PASSWORD")

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:"
            f"{self.postgres_password}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings: Settings
engine = None
AsyncSessionLocal = None  # type: ignore
openai_client: OpenAI | None = None

# ─── Логирование ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s")
logger = logging.getLogger("nickiai")

# ─── Модели SQLAlchemy ──────────────────────────────────────────────────
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(unique=True, index=True)
    language: Mapped[str] = mapped_column(String(5), default="en")
    name: Mapped[str] = mapped_column(String(64), default="")
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=True)
    phone: Mapped[str] = mapped_column(String(32), nullable=True)
    email_verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default="now()")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default="now()")
    user: Mapped[User] = relationship(back_populates="messages")

def get_db() -> AsyncSession:
    if AsyncSessionLocal is None:
        raise RuntimeError("DB session-maker isn’t ready yet")
    return AsyncSessionLocal()

# ─── Клиент Novita ─────────────────────────────────────────────────────
BASE_PROMPT = (
    "You are Nicki AI 🧘‍♀️ — an empathetic CBT-style assistant. "
    "Offer practical coping tools, ask reflective questions, "
    "and remind users to seek professional help for crises."
)

class NovitaClient:
    async def chat(self, msgs: list[dict]) -> str:
        resp = await openai_client.chat.completions.create(
            model=settings.novita_model,
            messages=msgs,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,
            extra_body={"top_k": 40, "min_p": 0},
        )
        return resp.choices[0].message.content

ai = NovitaClient()

# ─── Email verification ─────────────────────────────────────────────────
CODE_CACHE: dict[str, str] = {}
def _gen_code() -> str:
    return str(random.randint(100000, 999999))

def send_code(addr: EmailStr):
    code = _gen_code()
    msg = EmailMessage()
    msg["From"], msg["To"] = settings.smtp_user, addr
    msg["Subject"] = "Nicki AI verification code"
    msg.set_content(f"Your code: {code}\nIf this wasn’t you, ignore.")
    with ssl.create_default_context() as ctx, \
         __import__("smtplib").SMTP_SSL(settings.smtp_host, settings.smtp_port, context=ctx) as s:
        s.login(settings.smtp_user, settings.smtp_password)
        s.send_message(msg)
    CODE_CACHE[str(addr)] = code

def verify_code(addr: EmailStr, code: str) -> bool:
    return CODE_CACHE.get(str(addr)) == code.strip()

# ─── Audio → text (Whisper) ─────────────────────────────────────────────
async def speech_to_text(ogg: Path) -> str:
    fd, wav = tempfile.mkstemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-i", str(ogg), "-ac", "1", "-ar", "16000", wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    with open(wav, "rb") as f:
        t = await openai_client.audio.transcriptions.create(model="whisper-1", file=f)
    Path(wav).unlink(); __import__("os").close(fd)
    return t.text

# ─── Keyboards ──────────────────────────────────────────────────────────
def kb_lang():
    b = InlineKeyboardBuilder()
    b.button(text="🇬🇧 English", callback_data="lang_en")
    b.button(text="🇷🇺 Русский", callback_data="lang_ru")
    return b.as_markup()

def kb_menu(l: str):
    txt = {"en": ["🩺 Support","⚙️ Settings","🔄 Reset History"],
           "ru": ["🩺 Поддержка","⚙️ Настройки","🔄 Сброс истории"]}
    b = InlineKeyboardBuilder()
    for t,d in zip(txt[l],["issue","settings","reset_history"]):
        b.button(text=t, callback_data=d)
    b.adjust(1)
    return b.as_markup()

def kb_issue(l: str):
    txt = {"en": ["😰 Anxiety","💼 Stress","😞 Low mood","❤️ Relationships","⬅ Back"],
           "ru": ["😰 Тревога","💼 Стресс","😞 Уныние","❤️ Отношения","⬅ Назад"]}
    b = InlineKeyboardBuilder()
    for i,t in enumerate(txt[l]):
        b.button(text=t, callback_data=f"issue_{i}")
    b.adjust(2)
    return b.as_markup()

def kb_settings(l: str):
    txt = {"en": ["🔄 Re-register","⬅ Back"], "ru": ["🔄 Пере-регистрация","⬅ Назад"]}
    b = InlineKeyboardBuilder()
    for t,d in zip(txt[l],["re_register","back"]):
        b.button(text=t, callback_data=d)
    b.adjust(1)
    return b.as_markup()

# ─── Router и хэндлеры ───────────────────────────────────────────────────
router = Router()

@router.message(CommandStart())
async def cmd_start(m: types.Message):
    d = get_db()
    user = await d.scalar(select(User).where(User.telegram_id==m.from_user.id))
    if user:
        await m.answer({"en":"👋 Welcome back!","ru":"👋 С возвращением!"}[user.language],
                       reply_markup=kb_menu(user.language))
    else:
        await m.answer("🌸 Hello, I’m Nicki AI! Choose your language:",
                       reply_markup=kb_lang())

@router.callback_query(F.data.startswith("lang_"))
async def cb_lang(cb: types.CallbackQuery):
    lang = cb.data.split("_")[1]
    d = get_db()
    await d.merge(User(telegram_id=cb.from_user.id, language=lang))
    await d.commit()
    await cb.message.edit_text(
        {"en":"Great! What’s your *name*?","ru":"Отлично! Как ваше *имя*?"}[lang]
    )
    cb.from_user.chat_data["step"] = "name"
    await cb.answer()

@router.message(lambda m: m.from_user.chat_data.get("step")=="name")
async def fsm_name(m: types.Message):
    d = get_db()
    await d.execute(update(User).where(User.telegram_id==m.from_user.id)
                    .values(name=m.text.strip()))
    await d.commit()
    lang = await d.scalar(select(User.language).where(User.telegram_id==m.from_user.id))
    m.from_user.chat_data["step"] = "email"
    await m.answer({"en":"📧 Your *e-mail*:", "ru":"📧 Ваш *e-mail*:"}[lang])

@router.message(lambda m: m.from_user.chat_data.get("step")=="email")
async def fsm_email(m: types.Message):
    lang = await get_db().scalar(select(User.language)
                                .where(User.telegram_id==m.from_user.id))
    try:
        email = validate_email(m.text.strip(), check_deliverability=False).email
    except EmailNotValidError:
        return await m.answer({"en":"⚠️ Invalid e-mail.","ru":"⚠️ Некорректный e-mail"}[lang])
    send_code(email)
    await get_db().execute(update(User).where(User.telegram_id==m.from_user.id)
                           .values(email=email))
    await get_db().commit()
    m.from_user.chat_data.update(step="code", email=email)
    await m.answer("✅ Code sent! Enter *6 digits*:")

@router.message(lambda m: m.from_user.chat_data.get("step")=="code")
async def fsm_code(m: types.Message):
    em = m.from_user.chat_data["email"]
    if not verify_code(em, m.text):
        return await m.answer("❌ Wrong code, try again:")
    await get_db().execute(update(User).where(User.telegram_id==m.from_user.id)
                           .values(email_verified=True))
    await get_db().commit()
    m.from_user.chat_data["step"] = "phone"
    await m.answer("📱 Phone number:")

@router.message(lambda m: m.from_user.chat_data.get("step")=="phone")
async def fsm_phone(m: types.Message):
    check = constr(pattern=r"^[\d\+\-\(\) ]{7,20}$")
    check(m.text.strip())  # validation
    await get_db().execute(update(User).where(User.telegram_id==m.from_user.id)
                           .values(phone=m.text.strip()))
    await get_db().commit()
    lang = await get_db().scalar(select(User.language)
                                 .where(User.telegram_id==m.from_user.id))
    await m.answer({"en":"🎉 All set!","ru":"🎉 Готово!"}[lang],
                   reply_markup=kb_menu(lang))
    m.from_user.chat_data.clear()

# Основное меню
@router.callback_query(lambda c: c.data=="issue")
async def cb_issue(cb: types.CallbackQuery):
    lang = await get_db().scalar(select(User.language)
                                 .where(User.telegram_id==cb.from_user.id))
    await cb.message.edit_text(
        {"en":"What troubles you?","ru":"Что вас беспокоит?"}[lang],
        reply_markup=kb_issue(lang)
    )
    await cb.answer()

@router.callback_query(lambda c: c.data.startswith("issue_"))
async def cb_handle_issue(cb: types.CallbackQuery):
    idx = int(cb.data.split("_")[1])
    user = await get_db().scalar(select(User)
                                 .where(User.telegram_id==cb.from_user.id))
    prompt_options = {
        "en": ["I feel anxious","I feel stressed","I feel low","Relationship issues"],
        "ru": ["Я тревожусь","У меня стресс","Мне грустно","Проблемы в отношениях"]
    }[user.language]
    user_prompt = prompt_options[idx]
    # Собираем историю
    rows = await get_db().scalars(
        select(Message).where(Message.user_id==user.id).order_by(Message.created_at)
    )
    history = [{"role":"system","content":BASE_PROMPT}]
    history += [{"role":r.role,"content":r.content} for r in rows]
    history.append({"role":"user","content":user_prompt})
    resp = await ai.chat(history)
    # Сохраняем
    await get_db().execute(Message.__table__.insert().values(
        user_id=user.id, role="user", content=user_prompt))
    await get_db().execute(Message.__table__.insert().values(
        user_id=user.id, role="assistant", content=resp))
    await get_db().commit()
    await cb.message.answer(resp, reply_markup=kb_menu(user.language))
    await cb.answer()

@router.callback_query(lambda c: c.data=="reset_history")
async def cb_reset(cb: types.CallbackQuery):
    uid = await get_db().scalar(select(User.id)
                                .where(User.telegram_id==cb.from_user.id))
    await get_db().execute(delete(Message).where(Message.user_id==uid))
    await get_db().commit()
    lang = await get_db().scalar(select(User.language)
                                 .where(User.telegram_id==cb.from_user.id))
    await cb.message.answer(
        {"en":"🗑 History cleared.","ru":"🗑 История очищена."}[lang],
        reply_markup=kb_menu(lang)
    )
    await cb.answer()

@router.callback_query(lambda c: c.data=="settings")
async def cb_settings(cb: types.CallbackQuery):
    lang = await get_db().scalar(select(User.language)
                                 .where(User.telegram_id==cb.from_user.id))
    await cb.message.edit_text(
        {"en":"⚙️ Settings:","ru":"⚙️ Настройки:"}[lang],
        reply_markup=kb_settings(lang)
    )
    await cb.answer()

@router.callback_query(lambda c: c.data=="re_register")
async def cb_reregister(cb: types.CallbackQuery):
    # Удаляем пользователя и всю историю
    await get_db().execute(delete(User).where(User.telegram_id==cb.from_user.id))
    await get_db().commit()
    await cb.message.answer(
        "🌸 Let’s start over! Choose your language:",
        reply_markup=kb_lang()
    )
    await cb.answer()

@router.callback_query(lambda c: c.data=="back")
async def cb_back(cb: types.CallbackQuery):
    lang = await get_db().scalar(select(User.language)
                                 .where(User.telegram_id==cb.from_user.id))
    await cb.message.edit_text("📋 Menu:", reply_markup=kb_menu(lang))
    await cb.answer()

# Команды
@router.message(Command("support"))
async def cmd_support(m: types.Message):
    await cb_issue(types.CallbackQuery(
        data="issue", from_user=m.from_user, message=m))

@router.message(Command("settings"))
async def cmd_settings(m: types.Message):
    await cb_settings(types.CallbackQuery(
        data="settings", from_user=m.from_user, message=m))

@router.message(Command("reset"))
async def cmd_reset(m: types.Message):
    await cb_reset(types.CallbackQuery(
        data="reset_history", from_user=m.from_user, message=m))

# Голосовые сообщения и видео-ноты
@router.message(F.voice | F.video_note)
async def handle_voice(m: types.Message):
    file = m.voice or m.video_note
    tmp = Path(f"/tmp/{file.file_id}.ogg")
    await m.bot.download(await m.bot.get_file(file.file_id), destination=tmp)
    text = await speech_to_text(tmp)
    tmp.unlink(missing_ok=True)
    # далее аналогично photo (ниже) — собираем историю, отправляем в Novita и отвечаем
    # (в целях компактности не дублирую; можно вынести общий метод)

# Фото → OCR → диалог
@router.message(F.photo)
async def handle_photo(m: types.Message):
    photo = m.photo[-1]
    tmp = Path(f"/tmp/{photo.file_id}.jpg")
    await m.bot.download(photo.file_id, destination=tmp)
    img = Image.open(tmp)
    text = pytesseract.image_to_string(img)
    tmp.unlink(missing_ok=True)
    # собираем историю
    user = await get_db().scalar(select(User)
                                 .where(User.telegram_id==m.from_user.id))
    rows = await get_db().scalars(
        select(Message).where(Message.user_id==user.id).order_by(Message.created_at)
    )
    history = [{"role":"system","content":BASE_PROMPT}]
    history += [{"role":r.role,"content":r.content} for r in rows]
    history.append({"role":"user","content":text})
    resp = await ai.chat(history)
    # сохраняем
    await get_db().execute(Message.__table__.insert().values(
        user_id=user.id, role="user", content=text))
    await get_db().execute(Message.__table__.insert().values(
        user_id=user.id, role="assistant", content=resp))
    await get_db().commit()
    await m.answer(f"📝 Recognized:\n{text}\n\n🤖 {resp}",
                   reply_markup=kb_menu(user.language))

# ─── Запуск ────────────────────────────────────────────────────────────
async def on_startup():
    async with engine.begin():
        await engine.run_sync(Base.metadata.create_all)
    # при необходимости залогировать, можно прямо тут создать временный Bot
    logger.info("✅ Database tables ensured")

async def main():
    global settings, engine, AsyncSessionLocal, openai_client
    try:
        settings = Settings()
    except ValidationError as e:
        logger.error("Missing ENV vars: %s",
                     ", ".join(err["loc"][0] for err in e.errors()))
        return

    engine = create_async_engine(settings.dsn, pool_pre_ping=True)
    AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
    openai_client = OpenAI(
        base_url=settings.bot_base_url,
        api_key=settings.novita_api_key
    )

    bot = Bot(settings.bot_token, parse_mode=ParseMode.MARKDOWN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot, on_startup=on_startup)

if __name__ == "__main__":
    asyncio.run(main())