from aiogram.utils.keyboard import InlineKeyboardBuilder


def lang_keyboard() -> InlineKeyboardBuilder:
    kb = InlineKeyboardBuilder()
    kb.button(text="🇬🇧 English", callback_data="lang_en")
    kb.button(text="🇷🇺 Русский", callback_data="lang_ru")
    return kb


def main_menu_keyboard(lang: str = "en") -> InlineKeyboardBuilder:
    builder = InlineKeyboardBuilder()
    texts = {
        "en": ["🤕 Current issue", "⚙️ Settings", "🔄 Reset dialogue"],
        "ru": ["🤕 Текущая проблема", "⚙️ Настройки", "🔄 Сбросить диалог"],
    }
    builder.button(text=texts[lang][0], callback_data="issue")
    builder.button(text=texts[lang][1], callback_data="settings")
    builder.button(text=texts[lang][2], callback_data="reset")
    return builder