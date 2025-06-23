"""Transcribe voice / video‑note messages and continue the dialogue."""
from pathlib import Path
import asyncio
from aiogram import Router, types
from aiogram.filters import CommandObject
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..speech_utils import speech_to_text
from ..models import User, Message
from ..prompts import BASE_PROMPT
from ..novita_client import NovitaClient
from ..database import get_db
from ..keyboards.common import main_menu_keyboard

router = Router()
ai = NovitaClient()


async def _download(file: types.File, path: Path, bot):
    await bot.download(file, destination=path)


@router.message(F.voice | F.video_note)
async def handle_voice(message: types.Message, db: AsyncSession = get_db()):
    """Save voice/video‑note, convert to text, feed to model."""
    file_id = message.voice.file_id if message.voice else message.video_note.file_id
    tg_file = await message.bot.get_file(file_id)

    tmp = Path(f"/tmp/{file_id}.ogg")
    await _download(tg_file, tmp, message.bot)

    try:
        user_text = await speech_to_text(tmp)
    finally:
        asyncio.create_task(asyncio.to_thread(tmp.unlink, missing_ok=True))

    user = await db.scalar(select(User).where(User.telegram_id == message.from_user.id))
    if not user:
        await message.answer("Please send /start first.")
        return

    # build chat history ------------------------------------------------------
    rows = await db.scalars(
        select(Message).where(Message.user_id == user.id).order_by(Message.created_at)
    )
    history = [
        {"role": "system", "content": BASE_PROMPT},
        *[{"role": m.role, "content": m.content} for m in rows],
        {"role": "user", "content": user_text},
    ]

    ai_response = await ai.chat(history)

    # persist both sides ------------------------------------------------------
    await db.execute(
        Message.__table__.insert().values(user_id=user.id, role="user", content=user_text)
    )
    await db.execute(
        Message.__table__.insert().values(user_id=user.id, role="assistant", content=ai_response)
    )
    await db.commit()

    await message.answer(ai_response, reply_markup=main_menu_keyboard(user.language).as_markup())