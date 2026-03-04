"""
app/api/callback_handler.py
Webhook do Telegram para callbacks de inline buttons.

Registre no main.py:
    from app.api.callback_handler import router as callback_router
    app.include_router(callback_router)

E configure o webhook no Telegram (uma vez):
    https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://SEU_DOMINIO/telegram/callback
"""
from fastapi import APIRouter, Request
from loguru import logger

from app.telegram_bot.sender import TelegramSender
from app.telegram_bot.callback_store import get_match_signals, get_all_signals

router = APIRouter(prefix="/telegram", tags=["telegram"])
telegram = TelegramSender()


@router.post("/callback")
async def telegram_callback(request: Request):
    """Recebe updates do Telegram (callback_query de inline buttons)."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": True}

    callback = body.get("callback_query")
    if not callback:
        return {"ok": True}

    callback_id   = callback.get("id", "")
    callback_data = callback.get("data", "")

    logger.info(f"[Callback] data={callback_data!r}")

    # Responde imediatamente para remover o loading do botão
    await telegram.answer_callback(callback_id, "⏳ Carregando sinais...")

    if callback_data == "sig:all":
        # Envia todos os sinais
        all_items = get_all_signals()
        if not all_items:
            await telegram.send_message("⚠️ Nenhum sinal disponível no momento.")
        else:
            for match_label, sigs in all_items:
                await telegram.send_match_signals(match_label, sigs)

    elif callback_data.startswith("sig:"):
        idx_str = callback_data.split(":")[1]
        try:
            idx = int(idx_str)
            result = get_match_signals(idx)
            if result:
                match_label, sigs = result
                await telegram.send_match_signals(match_label, sigs)
            else:
                await telegram.send_message("⚠️ Sinal não encontrado. Rode um novo ciclo.")
        except ValueError:
            pass

    return {"ok": True}
