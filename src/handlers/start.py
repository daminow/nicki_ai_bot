from aiogram import Router, F, types
from aiogram.filters import CommandStart
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import User
from ..keyboards.common import lang_keyboard, main_menu_keyboard
from ..database import get_db
from ..email_utils import send_verification_email, verify_code
from ..config import settings

router = Router()


@router.message(CommandStart())
async def start(message: types.Message, db: AsyncSession = get_db()):
    user = await db.scalar(select(User).where(User.telegram_id == message.from_user.id))
    if user:
        await message.answer(
            "Welcome back to Nicki AI!",
            reply_markup=main_menu_keyboard(user.language).as_markup(),
        )
        return

    await message.answer(
        "Welcome to Nicki AI! Please select your preferred language.",
        reply_markup=lang_keyboard().as_markup(),
    )


@router.callback_query(F.data.startswith("lang_"))
async def process_language(callback: types.CallbackQuery, db: AsyncSession = get_db()):
    lang = callback.data.split("_")[1]
    await callback.message.edit_text(
        {
            "en": "Great! What's your first name?",
            "ru": "Отлично! Как вас зовут?",
        }[lang]
    )
    await db.execute(
        User.__table__.insert().values(
            telegram_id=callback.from_user.id, language=lang, name=""
        )
    )
    await db.commit()
    await callback.answer()