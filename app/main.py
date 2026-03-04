"""
Ponto de entrada da aplicação — FastAPI + APScheduler
Serve o hub.html e os arquivos estáticos diretamente.
"""
from dotenv import load_dotenv

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from app.config import get_settings
from app.database.connection import init_db, health_check
from app.scheduler.jobs import setup_scheduler, scheduler
from app.api.routes import router
from app.api.callback_handler import router as callback_router
from app.telegram_bot.sender import TelegramSender
from app.telegram_bot.polling import start_polling, stop_polling

settings = get_settings()
load_dotenv()

# ─── Dirs ─────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ─── Logs ─────────────────────────────────────────────────────────────────────
logger.add(
    settings.log_file,
    rotation="1 day",
    retention="30 days",
    compression="gz",
    level=settings.log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Iniciando Sports EV System...")

    await init_db()
    if not await health_check():
        raise RuntimeError("Falha ao conectar ao banco de dados")

    setup_scheduler()
    scheduler.start()
    logger.info(f"⏱  Scheduler iniciado — análise a cada {settings.analysis_interval_minutes} minutos")

    start_polling()

    telegram = TelegramSender()
    await telegram.send_system_alert(
        "🟢 SISTEMA INICIADO",
        f"Betoven Hub online\n"
        f"Ligas: {settings.monitored_leagues}\n"
        f"Intervalo: {settings.analysis_interval_minutes}min\n"
        f"EV mínimo: {settings.min_ev_threshold:.0%}"
    )

    yield

    stop_polling()
    scheduler.shutdown(wait=False)
    logger.info("🔴 Sistema encerrado")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Betoven Hub",
    description="Sistema de análise estatística e detecção de value bets",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API routes ───────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")
app.include_router(callback_router)  # /telegram/callback — inline buttons do Telegram

# ─── Servir data/ (profiles.json, etc.) ──────────────────────────────────────
if Path("data").exists():
    app.mount("/data", StaticFiles(directory="data"), name="data")

# ─── Hub HTML ─────────────────────────────────────────────────────────────────
@app.get("/hub")
async def hub():
    """Abre o dashboard do Betoven Hub"""
    hub_path = Path("hub.html")
    if not hub_path.exists():
        return {"error": "hub.html não encontrado. Coloque o arquivo na raiz do projeto."}
    return FileResponse(hub_path, media_type="text/html")

@app.get("/")
async def root():
    """Redireciona para o hub"""
    hub_path = Path("hub.html")
    if hub_path.exists():
        return FileResponse(hub_path, media_type="text/html")
    return {
        "name": "Betoven Hub",
        "version": "1.0.0",
        "hub": "/hub",
        "api": "/api/v1",
        "docs": "/docs",
    }
