"""
Scheduler — execução automática de análise.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from app.config import get_settings
from app.database.connection import AsyncSessionLocal
from app.services.analysis_orchestrator import AnalysisOrchestrator
from app.telegram_bot.sender import TelegramSender
from app.services.runtime_settings import get_analysis_league_id

settings = get_settings()
scheduler = AsyncIOScheduler(timezone="UTC")

_stats = {
    "cycles_run": 0,
    "signals_today": 0,
    "last_run": None,
    "last_error": None,
}


async def run_live_analysis_job():
    """
    Monitoramento ao vivo — roda em intervalo curto (padrão: 60s, mín: 30s).
    Filtra por MONITORED_LEAGUES do .env, igual ao job principal.
    """
    from app.services.live_scores import LiveScoreService

    try:
        svc = LiveScoreService()
        live_matches = await svc.get_live_matches()

        if not live_matches:
            logger.debug("[LiveJob] nenhum jogo ao vivo no momento")
            return

        # Aplica o mesmo filtro de ligas do job principal
        monitored = set(settings.monitored_leagues_list)
        if monitored:
            before = len(live_matches)
            live_matches = [
                m for m in live_matches
                if str(m.get("league_id", "")) in monitored
            ]
            skipped = before - len(live_matches)
            if skipped:
                logger.debug(f"[LiveJob] {skipped} jogo(s) ignorado(s) — liga não monitorada")

        if not live_matches:
            logger.debug("[LiveJob] nenhum jogo ao vivo nas ligas monitoradas")
            return

        logger.info(f"[LiveJob] {len(live_matches)} jogo(s) ao vivo — iniciando análise")

        signals_count = 0
        async with AsyncSessionLocal() as db:
            orchestrator = AnalysisOrchestrator(db)
            for match in live_matches:
                try:
                    result = await orchestrator._analyze_match(match)
                    if result and result > 0:
                        signals_count += result
                except Exception as e:
                    logger.warning(f"[LiveJob] erro no jogo {match.get('id', '?')}: {e}")

        if signals_count:
            logger.info(f"[LiveJob] ✅ {signals_count} sinal(is) gerado(s) ao vivo")
        else:
            logger.debug("[LiveJob] ciclo concluído sem novos sinais")

    except Exception as e:
        logger.error(f"[LiveJob] erro geral: {e}")


async def run_analysis_job(league: Optional[str] = None):
    # Prioridade: parâmetro explícito > runtime_settings.json > MONITORED_LEAGUES do .env
    if league is None:
        league = get_analysis_league_id()  # pode ser None se não selecionado pelo HUB

    _stats["last_run"] = datetime.now(timezone.utc).isoformat()
    _stats["cycles_run"] += 1

    label = f"liga {league}" if league else f"ciclo #{_stats['cycles_run']}"
    logger.info(f"🔄 Iniciando análise — {label}")

    try:
        async with AsyncSessionLocal() as db:
            orchestrator = AnalysisOrchestrator(db)
            if league:
                # Liga específica selecionada pelo HUB
                summary = await orchestrator.run_full_analysis(leagues=[league])
            else:
                # Sem seleção do HUB — usa MONITORED_LEAGUES do .env (ou tudo se vazio)
                monitored = settings.monitored_leagues_list
                if monitored:
                    summary = await orchestrator.run_full_analysis(leagues=monitored)
                else:
                    summary = await orchestrator.run_full_analysis()
            _stats["signals_today"] += summary.get("signals_generated", 0)
            _stats["last_error"] = None
    except Exception as e:
        error_msg = str(e)
        _stats["last_error"] = error_msg
        logger.error(f"❌ Erro na análise: {error_msg}")
        telegram = TelegramSender()
        await telegram.send_system_alert("⚠️ ERRO NO SISTEMA", f"Análise falhou:\n<code>{error_msg}</code>")


async def run_settlement_job():
    """Settlement automático a cada 15 min."""
    from sqlalchemy import select
    from app.models.db_models import Signal, Match
    from app.services.sokkerpro_client import SokkerProClient
    from app.services.live_scores import LiveScoreService

    try:
        svc    = LiveScoreService()
        sokker = SokkerProClient()

        live           = await svc.get_live_matches()
        finished_today = await svc.get_finished_today()

        seen_ids: set = set()
        all_matches: list = []
        for m in live + finished_today:
            mid = str(m.get("id", ""))
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                all_matches.append(m)

        live_by_id = {str(m.get("id", "")): svc.parse_score(m) for m in all_matches}

        async with AsyncSessionLocal() as db:
            cutoff = datetime.now(timezone.utc) - timedelta(days=1)
            pending_rows = (await db.execute(
                select(Signal, Match)
                .join(Match, Signal.match_id == Match.id)
                .where(Signal.result == None)
                .where(Match.commence_time >= cutoff)
            )).all()

            if not pending_rows:
                logger.debug("[Settlement] nenhum sinal pendente")
                return

            updated = 0
            for sig, match in pending_rows:
                score = live_by_id.get(str(sig.match_id))

                if score is None:
                    kickoff = match.commence_time
                    if kickoff.tzinfo is None:
                        kickoff = kickoff.replace(tzinfo=timezone.utc)
                    if datetime.now(timezone.utc) > kickoff + timedelta(hours=2):
                        raw = await sokker.get_fixture_score(str(sig.match_id))
                        if raw:
                            score = {"status": raw["status"], "home_goals": raw["home_goals"], "away_goals": raw["away_goals"]}

                if score and score["status"] == "FINISHED" and score.get("home_goals") is not None:
                    hg = int(score["home_goals"])
                    ag = int(score["away_goals"])
                    result_val = svc.determine_result(sig.market, hg, ag)
                    sig.result       = result_val
                    sig.actual_goals = hg + ag
                    sig.score_home   = hg
                    sig.score_away   = ag
                    sig.profit_loss  = (
                        round(sig.stake_units * (sig.suggested_odd - 1), 2) if result_val == "WIN"
                        else (-sig.stake_units if result_val == "LOSS" else 0.0)
                    )
                    updated += 1
                    logger.info(f"[Settlement] ✅ {match.home_team} vs {match.away_team} | {sig.market} → {result_val} ({hg}-{ag})")

            if updated:
                await db.commit()
                logger.info(f"[Settlement] {updated} sinais liquidados")

    except Exception as e:
        logger.error(f"[Settlement] erro: {e}")


async def send_daily_summary_job():
    logger.info("📊 Enviando resumo diário")
    telegram = TelegramSender()
    await telegram.send_daily_summary(_stats)
    _stats["signals_today"] = 0
    _stats["cycles_run"] = 0


def setup_scheduler():
    # Intervalo live — respeita mínimo de 30s para não sobrecarregar a API
    live_interval = max(30, settings.live_poll_interval_seconds)

    scheduler.add_job(
        run_live_analysis_job,
        trigger=IntervalTrigger(seconds=live_interval),
        id="live_analysis",
        name="Monitoramento Live",
        replace_existing=True,
        max_instances=1,          # nunca sobrepõe — se o ciclo anterior ainda correr, pula
        misfire_grace_time=15,
    )
    scheduler.add_job(
        run_analysis_job,
        trigger=IntervalTrigger(minutes=settings.analysis_interval_minutes),
        id="main_analysis",
        name="Análise de Value Bets",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=60,
    )
    scheduler.add_job(
        run_settlement_job,
        trigger=IntervalTrigger(minutes=15),
        id="settlement",
        name="Settlement Automático",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=30,
    )
    scheduler.add_job(
        send_daily_summary_job,
        trigger=CronTrigger(hour=23, minute=0),
        id="daily_summary",
        name="Resumo Diário",
        replace_existing=True,
    )
    return scheduler


def get_scheduler_status() -> dict:
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        })
    return {"running": scheduler.running, "jobs": jobs, "stats": _stats}
