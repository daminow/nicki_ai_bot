from aiogram import Router, types
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models import User, Message
from ..novita_client import NovitaClient
from ..prompts import BASE_PROMPT
from ..database import get_db
from ..keyboards import main_menu_keyboard, issue_keyboard

router = Router()
ai = NovitaClient()


@router.callback_query(lambda c: c.data == "issue")
async def choose_issue(c: types.CallbackQuery, db: AsyncSession = get_db()):
    user = await db.scalar(select(User).where(User.telegram_id == c.from_user.id))
    await c.message.edit_text(
        {
            "en": "What are you struggling with right now?",
            "ru": "С чем вы сейчас сталкиваетесь?",
        }[user.language],
        reply_markup=issue_keyboard(user.language).as_markup(),
    )
    await c.answer()


@router.callback_query(lambda c: c.data.startswith("issue_"))
async def handle_issue(c: types.CallbackQuery, db: AsyncSession = get_db()):
    category = int(c.data.split("_")[1])
    user = await db.scalar(select(User).where(User.telegram_id == c.from_user.id))

    prompt_map = {
        "en": [
            "I feel anxious",
            "I feel stressed",
            "I feel low",
            "I have a relationship issue",
        ],
        "ru": [
            "Я чувствую тревогу",
            "Я испытываю стресс",
            "Мне грустно",
            "Проблема в отношениях",
        ],
    }
    user_prompt = prompt_map[user.language][category]

    # build history -----------------------------------------------------------
    rows = await db.scalars(
        select(Message).where(Message.user_id == user.id).order_by(Message.created_at)
    )
    history = [
        {"role": "system", "content": BASE_PROMPT},
        *[{"role": m.role, "content": m.content} for m in rows],
        {"role": "user", "content": user_prompt},
    ]

    ai_response = await ai.chat(history)

    await db.execute(
        Message.__table__.insert().values(user_id=user.id, role="user", content=user_prompt)
    )
    await db.execute(
        Message.__table__.insert().values(user_id=user.id, role="assistant", content=ai_response)
    )
    await db.commit()

    await c.message.answer(ai_response, reply_markup=main_menu_keyboard(user.language).as_markup())
    await c.answer()