"""
TelegramSender — envio de sinais com menu interativo.

Fluxo:
  1. Fim do ciclo → envia RESUMO com inline buttons (um botão por jogo)
  2. Usuário clica no botão → callback handler envia os sinais daquele jogo
  3. Botão "Ver todos" → envia todos de uma vez
"""
from typing import Optional
from loguru import logger
from app.config import get_settings

settings = get_settings()


class TelegramSender:
    BASE_URL = "https://api.telegram.org"

    def __init__(self):
        self.token    = settings.telegram_bot_token
        self.chat_id  = settings.telegram_chat_id
        self._enabled = bool(self.token and self.chat_id)
        if not self._enabled:
            logger.warning("⚠️  Telegram não configurado. Preencha TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID no .env")

    # ── HTTP ─────────────────────────────────────────────────────────────────

    async def _post(self, method: str, payload: dict) -> Optional[dict]:
        if not self._enabled:
            return None
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.post(f"{self.BASE_URL}/bot{self.token}/{method}", json=payload)
                r.raise_for_status()
                return r.json().get("result")
        except Exception as e:
            logger.error(f"❌ Telegram/{method} falhou: {e}")
            return None

    async def send_message(self, text: str, reply_markup: Optional[dict] = None) -> Optional[int]:
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        if not self._enabled:
            logger.info(f"[TELEGRAM SIMULADO]\n{text}")
            return None

        result = await self._post("sendMessage", payload)
        if result:
            msg_id = result.get("message_id")
            logger.info(f"✅ Telegram: mensagem {msg_id} enviada")
            return msg_id
        return None

    async def answer_callback(self, callback_query_id: str, text: str = ""):
        """Remove o loading spinner do botão após clique."""
        await self._post("answerCallbackQuery", {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": False,
        })

    # ── Menu resumo do ciclo ──────────────────────────────────────────────────

    async def send_cycle_summary(
        self,
        signals_by_match: dict,   # {match_label: [sig_dict, ...]}
        cycle_stats: dict,
    ) -> Optional[int]:
        """Envia mensagem de resumo com botões inline por jogo."""
        if not signals_by_match:
            return None

        duration  = cycle_stats.get("duration_s", 0)
        analyzed  = cycle_stats.get("matches_analyzed", 0)
        found     = cycle_stats.get("matches_found", 0)
        total_sig = sum(len(v) for v in signals_by_match.values())

        # Top 3 por EV
        all_sigs = [
            (match_label, sig)
            for match_label, sigs in signals_by_match.items()
            for sig in sigs
        ]
        top3 = sorted(all_sigs, key=lambda x: x[1].get("ev", 0), reverse=True)[:3]

        medals = ["🥇", "🥈", "🥉"]
        top3_lines = ""
        for i, (mlabel, sig) in enumerate(top3):
            top3_lines += (
                f"{medals[i]} <b>{mlabel}</b>\n"
                f"   {sig.get('label','?')} | Odd <b>{sig.get('odd',0):.2f}</b> | "
                f"Prob {sig.get('prob',0):.0%} | EV +{sig.get('ev',0):.0%}\n"
            )

        mins, secs = int(duration // 60), int(duration % 60)
        dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"

        text = (
            f"📊 <b>CICLO CONCLUÍDO</b>\n"
            f"{'─'*30}\n\n"
            f"⏱ Duração: <b>{dur_str}</b>\n"
            f"🔍 Analisadas: <b>{analyzed}</b> partidas ({found} encontradas)\n"
            f"🎯 Sinais gerados: <b>{total_sig}</b>\n\n"
            f"🏆 <b>TOP 3 POR EV</b>\n"
            f"{top3_lines}\n"
            f"{'─'*30}\n"
            f"👇 <i>Selecione um jogo para ver o sinal completo:</i>"
        )

        # Inline keyboard — um botão por jogo (máx 10)
        games  = list(signals_by_match.keys())[:10]
        keyboard = []
        for idx, match_label in enumerate(games):
            sigs_for_match = signals_by_match[match_label]
            n = len(sigs_for_match)
            # Badge ao vivo — qualquer sinal do jogo com is_live=True
            is_live = any(s.get("analysis", {}).get("match_status") in ("IN_PLAY", "PAUSED") for s in sigs_for_match)
            live_badge = "🔴 " if is_live else ""
            minute = next((s.get("analysis", {}).get("match_minute") for s in sigs_for_match if s.get("analysis", {}).get("match_minute")), None)
            minute_str = f" {minute}'" if (is_live and minute) else ""
            txt = f"{live_badge}⚽ {match_label}{minute_str} ({n} {'sinais' if n > 1 else 'sinal'})"
            keyboard.append([{"text": txt[:60], "callback_data": f"sig:{idx}"}])

        keyboard.append([{"text": "📋 Ver todos os sinais", "callback_data": "sig:all"}])

        return await self.send_message(text, reply_markup={"inline_keyboard": keyboard})

    # ── Formatação de sinal individual ───────────────────────────────────────

    def format_signal(self, match_label: str, sig: dict) -> str:
        a      = sig.get("analysis", {})
        league = a.get("league_name", "")
        time   = a.get("match_time", "")
        edge   = (sig.get("prob", 0) - (1 / sig.get("odd", 1))) * 100

        if sig.get("ev", 0) >= 0.15:
            conf = "🔥 ALTO"
        elif sig.get("ev", 0) >= 0.10:
            conf = "✅ MÉDIO"
        else:
            conf = "🟡 BAIXO"

        return (
            f"⚽ <b>VALUE BET</b>\n"
            f"{'─'*28}\n"
            f"🏟 <b>{match_label}</b>\n"
            f"🏆 {league}  📅 {time}\n\n"
            f"📊 Mercado: <b>{sig.get('label','?')}</b>\n"
            f"💰 Odd: <b>{sig.get('odd',0):.2f}</b> ({sig.get('bookmaker','')})\n\n"
            f"📈 <b>PROBABILIDADES</b>\n"
            f"├ Modelo:    <b>{sig.get('prob',0):.1%}</b>\n"
            f"├ Implícita: {1/sig.get('odd',1):.1%}\n"
            f"└ Edge:      <b>+{edge:.1f}pp</b>\n\n"
            f"💎 EV: <b>+{sig.get('ev',0):.1%}</b> | {conf}\n"
            f"📌 Stake: <b>R$ {sig.get('stake',0):.2f}</b> "
            f"({sig.get('kelly',0):.1%} bankroll)\n"
            f"<i>⚠️ Gerencie seu bankroll com responsabilidade.</i>"
        )

    async def send_match_signals(self, match_label: str, sigs: list):
        """Envia os sinais de um jogo específico (chamado pelo callback)."""
        for sig in sigs:
            text = self.format_signal(match_label, sig)
            await self.send_message(text)

    # ── Alertas e resumos ─────────────────────────────────────────────────────

    async def send_analysis_started(self, league_name=None, matches_found=0):
        league_str = ("Liga: <b>" + str(league_name) + "</b>") if league_name else "Todas as ligas"
        parts = [
            "🔄 <b>ANÁLISE INICIADA</b>",
            "-" * 28,
            "",
            league_str,
            "⚽ Jogos encontrados: <b>" + str(matches_found) + "</b>",
            "",
            "<i>Aguarde... os sinais chegam ao final.</i>",
        ]
        await self.send_message("\n".join(parts))

    async def send_system_alert(self, title: str, body: str):
        await self.send_message("<b>" + title + "</b>\n\n" + body)

    async def send_daily_summary(self, stats: dict):
        text = (
            f"📊 <b>RESUMO DO DIA</b>\n{'─'*28}\n\n"
            f"📤 Sinais enviados: {stats.get('signals_today', 0)}\n"
            f"✅ Ciclos executados: {stats.get('cycles_run', 0)}\n"
        )
        await self.send_message(text)
