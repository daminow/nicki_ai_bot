"""Clear the dialogue history on user request."""
from aiogram import Router, types
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select

from ..database import get_db
from ..models import User, Message
from ..keyboards.common import main_menu_keyboard

router = Router()


@router.callback_query(lambda c: c.data == "reset")
async def confirm_reset(c: types.CallbackQuery):
    await c.message.answer(
        "⚠️ Are you sure you want to delete the whole conversation?",
        reply_markup=types.InlineKeyboardMarkup(
            inline_keyboard=[
                [types.InlineKeyboardButton(text="✅ Yes", callback_data="confirm_reset")],
                [types.InlineKeyboardButton(text="❌ No", callback_data="cancel")],
            ]
        ),
    )
    await c.answer()


@router.callback_query(lambda c: c.data == "confirm_reset")
async def do_reset(c: types.CallbackQuery, db: AsyncSession = get_db()):
    user = await db.scalar(select(User).where(User.telegram_id == c.from_user.id))
    await db.execute(delete(Message).where(Message.user_id == user.id))
    await db.commit()
    await c.message.answer("🗑 Dialogue cleared.", reply_markup=main_menu_keyboard(user.language).as_markup())
    await c.answer()


@router.callback_query(lambda c: c.data == "cancel")
async def cancel(c: types.CallbackQuery):
    await c.message.delete()
    await c.answer()