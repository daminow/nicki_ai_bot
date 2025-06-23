"""Handles onboarding & user‑settings using an FSM."""
from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from email_validator import validate_email, EmailNotValidError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ..database import get_db
from ..models import User
from ..schemas import Registration, EmailCode
from ..email_utils import send_verification_email, verify_code
from ..keyboards.common import main_menu_keyboard

router = Router()

# ------------------------------------------------------------------------ FSM
class Reg(StatesGroup):
    name = State()
    email = State()
    email_code = State()
    phone = State()


# ----------------------------- /settings entry point ------------------------
@router.callback_query(lambda c: c.data == "settings")
@router.message(Command("settings"))
async def settings_entry(event: types.Message | types.CallbackQuery, state: FSMContext, db: AsyncSession = get_db()):
    """Ask what to change or launch registration if user incomplete."""
    if isinstance(event, types.CallbackQuery):
        message = event.message
        from_user = event.from_user
        await event.answer()
    else:
        message = event
        from_user = event.from_user

    user: User | None = await db.scalar(select(User).where(User.telegram_id == from_user.id))

    if not user or not user.email_verified:
        await message.answer("Please tell me your *first name*", parse_mode="Markdown")
        await state.set_state(Reg.name)
        return

    # show simple inline settings menu (could be extended)
    await message.answer(
        "*Settings*\n• Use /reset to clear chat history\n• Use this menu again to update personal data.",
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard(user.language).as_markup(),
    )


# ----------------------------- name step ------------------------------------
@router.message(Reg.name, F.text)
async def reg_name(message: types.Message, state: FSMContext):
    await state.update_data(name=message.text.strip())
    await message.answer("Nice! Now type your e‑mail address:")
    await state.set_state(Reg.email)


# ----------------------------- email step -----------------------------------
@router.message(Reg.email, F.text)
async def reg_email(message: types.Message, state: FSMContext):
    try:
        email = validate_email(message.text.strip(), check_deliverability=False).email
    except EmailNotValidError as err:
        await message.answer(f"❌ {err}. Try again:")
        return
    await state.update_data(email=email)
    send_verification_email(email)  # send code synchronously (fast)
    await message.answer("I emailed you a 6‑digit code. Please enter it:")
    await state.set_state(Reg.email_code)


# ----------------------------- verify code ----------------------------------
@router.message(Reg.email_code, F.text)
async def reg_email_code(message: types.Message, state: FSMContext):
    data = await state.get_data()
    if not verify_code(data["email"], message.text):
        await message.answer("❌ Invalid code. Try again:")
        return
    await message.answer("✅ E‑mail confirmed! Now send a phone number (digits, +, (), ‑)")
    await state.set_state(Reg.phone)


# ----------------------------- phone step -----------------------------------
@router.message(Reg.phone, F.text)
async def reg_phone(message: types.Message, state: FSMContext, db: AsyncSession = get_db()):
    phone = message.text.strip()
    if not Registration.__fields__["phone"].validate(phone)[0]:
        await message.answer("❌ That doesn’t look like a phone number. Try again:")
        return

    raw = await state.get_data()
    reg = Registration(name=raw["name"], email=raw["email"], phone=phone)

    # upsert user -------------------------------------------------------------
    await db.execute(
        update(User)
        .where(User.telegram_id == message.from_user.id)
        .values(
            name=reg.name,
            email=reg.email,
            phone=reg.phone,
            email_verified=True,
        )
    )
    await db.commit()

    await message.answer(
        "🎉 Registration complete!",
        reply_markup=main_menu_keyboard().as_markup(),
    )
    await state.clear()