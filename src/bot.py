"""Entrypoint – creates DB tables and starts polling."""
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotSettings
from aiogram.enums import ParseMode

from .config import settings
from .handlers import routers
from .database import engine, Base


async def on_startup(bot: Bot):
    # ensure tables exist ------------------------------------------------------
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    me = await bot.get_me()
    print(f"Nicki AI started as @{me.username}")


async def main():
    # global defaults ---------------------------------------------------------
    defaults = DefaultBotSettings(parse_mode=ParseMode.MARKDOWN)
    bot = Bot(settings.bot_token, default=defaults)
    dp = Dispatcher()

    for r in routers:
        dp.include_router(r)

    # run
    await dp.start_polling(bot, on_startup=on_startup)


if __name__ == "__main__":
    asyncio.run(main())