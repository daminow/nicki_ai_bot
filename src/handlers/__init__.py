"""Aggregate routers so `bot.py` can include them in one go."""
from .start import router as start_router
from .settings import router as settings_router
from .menu import router as menu_router
from .voice import router as voice_router
from .reset import router as reset_router

routers = (
    start_router,
    settings_router,
    menu_router,
    voice_router,
    reset_router,
)