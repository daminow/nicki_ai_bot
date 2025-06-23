"""Domain‑specific inline keyboards that don’t fit `common.py`."""
from aiogram.utils.keyboard import InlineKeyboardBuilder

def issue_keyboard(lang: str = "en") -> InlineKeyboardBuilder:
    """Return a keyboard asking the user to pick a broad problem area."""
    builder = InlineKeyboardBuilder()
    texts = {
        "en": ["Anxiety", "Stress", "Low mood", "Relationships", "⬅ Back"],
        "ru": ["Тревога", "Стресс", "Уныние", "Отношения", "⬅ Назад"],
    }
    for idx, caption in enumerate(texts[lang]):
        builder.button(text=caption, callback_data=f"issue_{idx}")
    builder.adjust(2)  # two buttons per row looks nicer
    return builder