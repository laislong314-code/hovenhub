"""
Telegram long-polling — substitui webhook quando não há domínio público.
Roda como task asyncio em background junto com o FastAPI.

Uso: chamado no lifespan do main.py
"""
import asyncio
from loguru import logger
from app.config import get_settings
from app.telegram_bot.callback_store import get_match_signals, get_all_signals
from app.telegram_bot.sender import TelegramSender

settings = get_settings()

_polling_task: asyncio.Task | None = None


async def _process_update(update: dict, telegram: TelegramSender):
    """Processa um update do Telegram (callback_query ou mensagem)."""
    callback = update.get("callback_query")
    if not callback:
        return

    callback_id   = callback.get("id", "")
    callback_data = callback.get("data", "")

    if not callback_data.startswith("sig:"):
        return

    logger.info(f"[Polling] callback_data={callback_data!r}")
    await telegram.answer_callback(callback_id, "⏳ Carregando sinais...")

    if callback_data == "sig:all":
        all_items = get_all_signals()
        if not all_items:
            await telegram.send_message("⚠️ Nenhum sinal disponível. Rode um novo ciclo.")
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


async def _get_updates(token: str, offset: int) -> list:
    """Faz long-poll com timeout HTTP adequado (35s para suportar timeout=30 do Telegram)."""
    import httpx
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {
        "offset": offset,
        "timeout": 25,                  # Telegram aguarda 25s por novos updates
        "allowed_updates": ["callback_query"],
    }
    async with httpx.AsyncClient(timeout=35) as c:   # HTTP timeout > Telegram timeout
        r = await c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("result") or []


async def _polling_loop():
    """Loop de long-polling. Roda indefinidamente até cancelamento."""
    telegram = TelegramSender()
    if not telegram._enabled:
        logger.warning("[Polling] Telegram não configurado — polling desativado.")
        return

    # Remove webhook existente para ativar polling
    await telegram._post("deleteWebhook", {"drop_pending_updates": False})
    logger.info("[Polling] Webhook removido. Iniciando long-polling do Telegram...")

    offset = 0
    consecutive_errors = 0

    while True:
        try:
            updates = await _get_updates(telegram.token, offset)
            consecutive_errors = 0

            for update in updates:
                uid = update.get("update_id", 0)
                offset = max(offset, uid + 1)
                try:
                    await _process_update(update, telegram)
                except Exception as e:
                    logger.error(f"[Polling] Erro ao processar update {uid}: {e}")

        except asyncio.CancelledError:
            logger.info("[Polling] Loop encerrado.")
            return
        except Exception as e:
            consecutive_errors += 1
            wait = min(2 ** consecutive_errors, 60)
            logger.warning(f"[Polling] Erro ({consecutive_errors}x): {e} — aguardando {wait}s")
            await asyncio.sleep(wait)


def start_polling():
    """Inicia o loop de polling como task asyncio em background."""
    global _polling_task
    if _polling_task and not _polling_task.done():
        logger.debug("[Polling] Já em execução.")
        return
    _polling_task = asyncio.create_task(_polling_loop(), name="telegram_polling")
    logger.info("[Polling] Task iniciada.")


def stop_polling():
    """Cancela o loop de polling."""
    global _polling_task
    if _polling_task and not _polling_task.done():
        _polling_task.cancel()
        logger.info("[Polling] Task cancelada.")
