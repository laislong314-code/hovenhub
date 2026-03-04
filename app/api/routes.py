"""
Rotas da API REST — FastAPI (V2 SokkerPRO)
"""
from datetime import datetime, timezone
from typing import Optional
import os
import re
import unicodedata
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from app.database.connection import get_db
from app.models.db_models import Signal, Match, MatchAnalysis, MetricsDaily
from app.models.schemas import SignalDTO, SignalStats
from app.services.analysis_orchestrator import AnalysisOrchestrator
from app.services.live_scores import LiveScoreService
from app.services.sokkerpro_client import SokkerProClient
from app.scheduler.jobs import get_scheduler_status, run_analysis_job
from loguru import logger
from app.config import get_settings

settings = get_settings()

router = APIRouter()


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(), "source": "SokkerPRO"}


@router.get("/scheduler/status")
async def scheduler_status():
    return get_scheduler_status()


# ─── Analysis ─────────────────────────────────────────────────────────────────

@router.post("/analysis/trigger")
async def trigger_analysis(
    background_tasks: BackgroundTasks,
    league_id: Optional[str] = None,
):
    if league_id:
        background_tasks.add_task(run_analysis_job, league_id)
        message = f"Análise da liga {league_id} iniciada em background"
    else:
        background_tasks.add_task(run_analysis_job)
        message = "Análise completa iniciada em background"
    return {"message": message, "league_id": league_id, "triggered_at": datetime.now(timezone.utc).isoformat()}


# ─── Live Scores ──────────────────────────────────────────────────────────────

@router.get("/live")
async def live_scores(
    db: AsyncSession = Depends(get_db),
    days_back: int = 1,
):
    """
    Placares ao vivo + auto-settlement de sinais pendentes.

    Estratégia de settlement em 3 camadas:
    1. Livescores (jogos ao vivo / recém terminados no feed ao vivo)
    2. Fixtures do dia (/mini) — inclui status FINISHED após ~60s
    3. Fallback por fixture individual (/fixture/{id}) para sinais PENDING
       cujo kickoff já passou há mais de 2h — garante settlement mesmo
       que o jogo tenha saído do feed antes de ser capturado.
    """
    from datetime import timedelta
    from app.services.sokkerpro_client import SokkerProClient

    svc    = LiveScoreService()
    sokker = SokkerProClient()

    # Camada 1: ao vivo agora (IN_PLAY, PAUSED, e FT recentes)
    live = await svc.get_live_matches()

    # Camada 2: fixtures do dia (inclui FINISHED)
    finished_today = await svc.get_finished_today()

    # Merge camadas 1+2, deduplicado por id
    seen_ids: set = set()
    all_matches: list = []
    for m in live + finished_today:
        mid = str(m.get("id", ""))
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            all_matches.append(m)

    live_by_id = {str(m.get("id", "")): svc.parse_score(m) for m in all_matches}

    # Sinais pendentes dentro da janela days_back
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(days_back, 1))
    pending_rows = (await db.execute(
        select(Signal, Match)
        .join(Match, Signal.match_id == Match.id)
        .where(Signal.result == None)
        .where(Match.commence_time >= cutoff)
    )).all()

    auto_updated = 0
    results = []

    for sig, match in pending_rows:
        score = live_by_id.get(str(sig.match_id))

        # Camada 3: fallback individual para jogos que deveriam ter terminado
        # (kickoff + 2h no passado) mas não apareceram nas camadas 1+2
        if score is None:
            kickoff = match.commence_time
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            expected_end = kickoff + timedelta(hours=2)
            if datetime.now(timezone.utc) > expected_end:
                raw = await sokker.get_fixture_score(str(sig.match_id))
                if raw:
                    score = {
                        "status":     raw["status"],
                        "home_goals": raw["home_goals"],
                        "away_goals": raw["away_goals"],
                        "minute":     None,
                    }
                    logger.info(
                        f"[Settlement] fixture fallback {sig.match_id}: "
                        f"{raw['status']} {raw['home_goals']}-{raw['away_goals']}"
                    )

        entry = {
            "signal_id":    sig.id,
            "match_id":     sig.match_id,
            "home_team":    match.home_team,
            "away_team":    match.away_team,
            "market":       sig.market,
            "market_label": sig.label or sig.market,
            "suggested_odd": sig.suggested_odd,
            "status":       score["status"] if score else "UNKNOWN",
            "minute":       score.get("minute") if score else None,
            "score": {
                "home": score["home_goals"],
                "away": score["away_goals"],
            } if score and score.get("home_goals") is not None else None,
            "result":       sig.result,
        }

        if score and score["status"] == "FINISHED" and score.get("home_goals") is not None:
            hg = int(score["home_goals"])
            ag = int(score["away_goals"])
            result_val = svc.determine_result(sig.market, hg, ag)

            sig.result      = result_val
            sig.actual_goals = hg + ag
            sig.score_home   = hg
            sig.score_away   = ag
            sig.profit_loss  = (
                round(sig.stake_units * (sig.suggested_odd - 1), 2) if result_val == "WIN"
                else (-sig.stake_units if result_val == "LOSS" else 0.0)
            )
            auto_updated += 1
            entry["result"] = result_val

        results.append(entry)

    if auto_updated:
        await db.commit()
        logger.info(f"[Settlement] {auto_updated} sinais atualizados automaticamente")

    return {
        "live_matches":    len(live),
        "finished_today":  len(finished_today),
        "pending_signals": len(pending_rows),
        "auto_updated":    auto_updated,
        "signals":         results,
    }


@router.get("/live/scores")
async def all_live_scores():
    """Todos os jogos ao vivo agora — sem filtro de sinais."""
    svc = LiveScoreService()
    live = await svc.get_live_matches()
    return [svc.parse_score(m) for m in live]


@router.get("/fixtures")
async def get_fixtures(
    date: Optional[str] = None,
    league_id: Optional[str] = None,
    timezone_offset: str = "utc-3",
):
    """Agenda de jogos de uma data (padrão: hoje). Aceita filtro por league_id."""
    sokker = SokkerProClient()
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    all_fixtures = await sokker.get_fixtures_for_date(date, timezone_offset)

    if league_id:
        all_fixtures = [f for f in all_fixtures if str(f.get("league_id")) == league_id]

    return {
        "date": date,
        "total": len(all_fixtures),
        "fixtures": all_fixtures,
    }


def _norm_team_name(s: str) -> str:
    """Normaliza nome do time para matching robusto (sem acento, sem símbolos)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


@router.get("/fixtures/find")
async def find_fixture_id(
    date: Optional[str] = None,
    home: str = "",
    away: str = "",
    timezone_offset: str = "utc-3",
    max_candidates: int = 10,
):
    """Descobre fixture_id do dia pelo par home+away (matching normalizado)."""
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not home.strip() or not away.strip():
        raise HTTPException(status_code=400, detail="Parâmetros 'home' e 'away' são obrigatórios.")

    sokker = SokkerProClient()
    fixtures = await sokker.get_fixtures_for_date(date, timezone_offset)

    nh = _norm_team_name(home)
    na = _norm_team_name(away)

    # 1) Match exato (home/away)
    for f in fixtures:
        fh = _norm_team_name((f.get("homeTeam") or {}).get("name", ""))
        fa = _norm_team_name((f.get("awayTeam") or {}).get("name", ""))
        if fh == nh and fa == na:
            return {
                "found": True,
                "date": date,
                "timezone_offset": timezone_offset,
                "fixture_id": f.get("id"),
                "league_id": f.get("league_id"),
                "league_name": f.get("league_name"),
                "kickoff_utc": f.get("utcDate"),
                "home": (f.get("homeTeam") or {}).get("name", ""),
                "away": (f.get("awayTeam") or {}).get("name", ""),
            }

    # 2) Sem match exato → lista candidatos (contém)
    candidates = []
    for f in fixtures:
        home_name = (f.get("homeTeam") or {}).get("name", "")
        away_name = (f.get("awayTeam") or {}).get("name", "")
        fh = _norm_team_name(home_name)
        fa = _norm_team_name(away_name)

        score = 0
        if nh and nh in fh:
            score += 2
        if na and na in fa:
            score += 2
        if fh and fh in nh:
            score += 1
        if fa and fa in na:
            score += 1

        if score > 0:
            candidates.append((score, f))

    candidates.sort(key=lambda x: (-x[0], str(x[1].get("utcDate",""))))
    out = []
    for score, f in candidates[:max_candidates]:
        out.append({
            "score": score,
            "fixture_id": f.get("id"),
            "league_id": f.get("league_id"),
            "league_name": f.get("league_name"),
            "kickoff_utc": f.get("utcDate"),
            "home": (f.get("homeTeam") or {}).get("name", ""),
            "away": (f.get("awayTeam") or {}).get("name", ""),
            "status": f.get("status"),
        })

    return {
        "found": False,
        "date": date,
        "timezone_offset": timezone_offset,
        "query": {"home": home, "away": away},
        "candidates": out,
    }

# ─── Signals ─────────────────────────────────────────────────────────────────

@router.get("/signals", response_model=list[SignalDTO])
async def list_signals(
    mode: str = "precision",
    limit: int = 10,
    offset: int = 0,
    result: Optional[str] = None,
    min_ev: Optional[float] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    mode:
      - "precision" (default): sinais para o dashboard.
      - "all": todos os sinais (com filtros opcionais de result/min_ev).

    Compat:
      - aceita result em qualquer casing: win/loss/void/open
      - quando HUB_INCLUDE_RESOLVED_IN_PRECISION=true, o modo precision (sem result)
        inclui também sinais resolvidos para permitir filtro local no frontend
        (Win/Loss) sem novas requests.
    """

    def _env_truthy(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

    include_resolved_in_precision = _env_truthy("HUB_INCLUDE_RESOLVED_IN_PRECISION", False)

    enable_one_per_group = _env_truthy("ENABLE_ONE_PER_MARKET_GROUP", True)

    def _market_group(market: str) -> str:
        m = (market or "").strip()
        if not m:
            return "OTHER"
        if m.startswith("Over_") or m.startswith("Under_"):
            return "GOALS"
        if m in ("BTTS", "No_BTTS"):
            return "BTTS"
        # Outcome markets (1X2 + Double Chance) must compete in the same bucket.
        # Otherwise the dashboard can show conflicting picks like "1X" and "2" for the same match.
        if m in ("1", "X", "2", "1X2", "1X", "X2", "12") or m.startswith("DC_"):
            return "OUTCOME"
        return "OTHER"

    def _pick_best_per_match_group(rows_in: list[tuple[Signal, Match]]) -> list[tuple[Signal, Match]]:
        # Keep at most 1 signal per (match_id, market_group), choosing the best by a stable score.
        best: dict[tuple[str, str], tuple[tuple, tuple[Signal, Match]]] = {}
        for sig, match in rows_in:
            mid = str(sig.match_id)
            grp = _market_group(getattr(sig, "market", "") or "")
            ev = float(getattr(sig, "ev", 0.0) or 0.0)
            prob = float(getattr(sig, "model_probability", 0.0) or 0.0)
            odd = float(getattr(sig, "suggested_odd", 0.0) or 0.0)
            # prefer odds close to suggested_odd if 'odd' exists; we only have suggested_odd in DB
            # so we use it as a neutral tie-breaker (0 distance).
            dist = 0.0
            sent = getattr(sig, "sent_at", None)
            # score tuple: higher is better
            score = (ev, prob, -dist, sent or datetime.min.replace(tzinfo=timezone.utc), sig.id)
            k = (mid, grp)
            cur = best.get(k)
            if cur is None or score > cur[0]:
                best[k] = (score, (sig, match))
        # return in recency order (sent_at desc) to keep dashboard feel
        out = [v[1] for v in best.values()]
        out.sort(key=lambda sm: (sm[0].sent_at or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        return out

    # Normalize result filter (frontend may send win/loss in different casing)
    if result is not None:
        r = str(result).strip().lower()
        if r in ("open", "pending", "null", "none"):
            result = None
        else:
            result = r.upper()

    # Base query (latest first)
    base_query = (
        select(Signal, Match)
        .join(Match, Signal.match_id == Match.id)
        .order_by(desc(Signal.sent_at))
    )

    # Preserve old behavior if requested
    if mode == "all":
        query = base_query.limit(limit).offset(offset)
        if result is not None:
            query = query.where(Signal.result == result)
        if min_ev is not None:
            query = query.where(Signal.ev >= min_ev)
        rows = (await db.execute(query)).all()
        return [
            SignalDTO(
                id=sig.id,
                match_id=sig.match_id,
                home_team=match.home_team,
                away_team=match.away_team,
                home_logo=getattr(match, "home_logo", None),
                away_logo=getattr(match, "away_logo", None),
                league=match.sport_title,
                league_id=match.sport_key,
                commence_time=match.commence_time,
                market=sig.market,
                market_label=sig.label or sig.market,
                model_probability=sig.model_probability,
                implied_probability=sig.implied_probability,
                ev=sig.ev,
                suggested_odd=sig.suggested_odd,
                bookmaker=sig.bookmaker,
                stake_pct=sig.stake_pct,
                stake_units=sig.stake_units,
                sent_at=sig.sent_at,
                result=sig.result,
                profit_loss=sig.profit_loss,
                actual_goals=getattr(sig, "actual_goals", None),
                score_home=getattr(sig, "score_home", None),
                score_away=getattr(sig, "score_away", None),
                is_live=getattr(sig, "is_live", None),
                match_minute=getattr(sig, "match_minute", None),
            )
            for sig, match in rows
        ]

    # Precision mode:
    # - If result filter is provided, return matching history (no selector).
    if result is not None:
        query = base_query.where(Signal.result == result).limit(limit).offset(offset)
        if min_ev is not None:
            query = query.where(Signal.ev >= min_ev)
        rows = (await db.execute(query)).all()

    else:
        # If HUB flag is enabled, return the latest signals (open + resolved) so the frontend
        # can filter Win/Loss locally without triggering new network calls.
        if include_resolved_in_precision:
            query = base_query.limit(limit).offset(offset)
            if min_ev is not None:
                query = query.where(Signal.ev >= min_ev)
            rows = (await db.execute(query)).all()
        else:
            # Default legacy behavior: only pending/open, then apply selector.
            query = base_query.where(Signal.result.is_(None)).limit(500).offset(0)
            if min_ev is not None:
                query = query.where(Signal.ev >= min_ev)

            raw_rows = (await db.execute(query)).all()

            from app.services.signal_selector import SignalSelector, SelectorConfig

            selector = SignalSelector(SelectorConfig.from_settings(top_n=limit))

            pending = []
            by_sig_id = {}
            for sig, match in raw_rows:
                by_sig_id[sig.id] = (sig, match)
                pending.append(
                    {
                        "analysis": {"match_id": sig.match_id, "_is_multi": str(sig.market).startswith("MULTI_")},
                        "sig": {"market_id": sig.market, "odd": sig.suggested_odd, "prob": sig.model_probability, "ev": sig.ev},
                    }
                )

            selected = selector.select(pending)
            selected_ids = []
            for it in selected:
                a = it.get("analysis") or {}
                s = it.get("sig") or {}
                mid = a.get("match_id")
                mk = s.get("market_id")
                od = float(s.get("odd") or 0.0)

                best = None
                for sig_id, (sig_obj, _m) in by_sig_id.items():
                    if sig_obj.match_id == mid and sig_obj.market == mk and float(sig_obj.suggested_odd or 0.0) == od:
                        best = sig_id
                        break
                if best is not None:
                    selected_ids.append(best)

            rows = [by_sig_id[i] for i in selected_ids if i in by_sig_id]

            if offset or (limit and len(rows) > limit):
                rows = rows[offset : offset + limit]

    # Precision UX: avoid multiple picks of the same "market family" for the same match (e.g., many GOALS lines).
    # This reduces confusion in the dashboard without changing settlement.
    if mode != "all" and enable_one_per_group:
        rows = _pick_best_per_match_group(rows)

    return [
        SignalDTO(
            id=sig.id,
            match_id=sig.match_id,
            home_team=match.home_team,
            away_team=match.away_team,
            home_logo=getattr(match, "home_logo", None),
            away_logo=getattr(match, "away_logo", None),
            league=match.sport_title,
            league_id=match.sport_key,
            commence_time=match.commence_time,
            market=sig.market,
            market_label=sig.label or sig.market,
            model_probability=sig.model_probability,
            implied_probability=sig.implied_probability,
            ev=sig.ev,
            suggested_odd=sig.suggested_odd,
            bookmaker=sig.bookmaker,
            stake_pct=sig.stake_pct,
            stake_units=sig.stake_units,
            sent_at=sig.sent_at,
            result=sig.result,
            profit_loss=sig.profit_loss,
            actual_goals=getattr(sig, "actual_goals", None),
            score_home=getattr(sig, "score_home", None),
            score_away=getattr(sig, "score_away", None),
            is_live=getattr(sig, "is_live", None),
            match_minute=getattr(sig, "match_minute", None),
        )
        for sig, match in rows
    ]



@router.get("/signals/stats", response_model=SignalStats)
async def signal_stats(db: AsyncSession = Depends(get_db)):
    total = (await db.execute(select(func.count(Signal.id)))).scalar_one()
    resolved_rows = (await db.execute(
        select(Signal).where(Signal.result.in_(["WIN", "LOSS"]))
    )).scalars().all()

    resolved   = len(resolved_rows)
    wins       = sum(1 for s in resolved_rows if s.result == "WIN")
    losses     = resolved - wins
    total_staked = sum(s.stake_units for s in resolved_rows if s.stake_units)
    total_pl   = sum(s.profit_loss for s in resolved_rows if s.profit_loss)
    roi        = (total_pl / total_staked) if total_staked > 0 else 0.0

    return SignalStats(
        total_signals=total,
        resolved_signals=resolved,
        wins=wins,
        losses=losses,
        win_rate=(wins / resolved) if resolved > 0 else 0.0,
        total_staked=round(total_staked, 2),
        total_profit_loss=round(total_pl, 2),
        roi=round(roi, 4),
        yield_pct=round(roi, 4),
    )


@router.get("/metrics/daily")
async def metrics_daily(
    days: int = 30,
    market: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Phase 1 metrics endpoint (read-only).

    No breaking changes: new endpoint only.
    When ENABLE_ADV_METRICS=false, returns empty payload.
    """
    if not settings.enable_adv_metrics:
        return {"enabled": False, "items": []}

    days = max(1, min(int(days), 365))

    q = select(MetricsDaily).order_by(desc(MetricsDaily.day_utc), desc(MetricsDaily.id))
    if market:
        q = q.where(MetricsDaily.market == market)
    q = q.limit(days * 10)  # worst-case multiple markets per day
    rows = (await db.execute(q)).scalars().all()

    # keep only last N distinct days (simple, stable)
    out = []
    seen_days = []
    for r in rows:
        if r.day_utc not in seen_days:
            seen_days.append(r.day_utc)
            if len(seen_days) > days:
                break
        out.append({
            "day_utc": r.day_utc,
            "market": r.market,
            "n_resolved": r.n_resolved,
            "roi_pct": round(r.roi, 4),
            "roi_ci_low": None if r.roi_ci_low is None else round(r.roi_ci_low, 4),
            "roi_ci_high": None if r.roi_ci_high is None else round(r.roi_ci_high, 4),
            "brier": None if r.brier is None else round(r.brier, 6),
            "logloss": None if r.logloss is None else round(r.logloss, 6),
            "brier_skill_vs_implied": None if r.brier_skill_vs_implied is None else round(r.brier_skill_vs_implied, 6),
            "spearman_rho_ev_pl": None if r.spearman_rho_ev_pl is None else round(r.spearman_rho_ev_pl, 6),
            "spearman_p_ev_pl": None if r.spearman_p_ev_pl is None else round(r.spearman_p_ev_pl, 6),
            "rolling_brier_7": None if r.rolling_brier_7 is None else round(r.rolling_brier_7, 6),
            "rolling_brier_14": None if r.rolling_brier_14 is None else round(r.rolling_brier_14, 6),
            "rolling_alert": bool(r.rolling_alert),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    # Surface alerts (latest per market)
    latest_alerts = [x for x in out if x.get("rolling_alert")]
    return {"enabled": True, "items": out, "alerts": latest_alerts[:25]}


@router.patch("/signals/{signal_id}/result")
async def update_signal_result(
    signal_id: int,
    result: str,
    actual_goals: Optional[int] = None,
    score_home: Optional[int] = None,
    score_away: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    signal = await db.get(Signal, signal_id)
    if not signal:
        raise HTTPException(404, "Sinal não encontrado")
    if result not in ("WIN", "LOSS", "VOID"):
        raise HTTPException(400, "Resultado deve ser WIN, LOSS ou VOID")

    signal.result = result
    if score_home is not None and score_away is not None:
        signal.score_home   = score_home
        signal.score_away   = score_away
        signal.actual_goals = score_home + score_away
    elif actual_goals is not None:
        signal.actual_goals = actual_goals
    signal.profit_loss = (
        round(signal.stake_units * (signal.suggested_odd - 1), 2) if result == "WIN"
        else (-signal.stake_units if result == "LOSS" else 0.0)
    )
    await db.commit()
    return {"message": "Resultado atualizado", "profit_loss": signal.profit_loss}


@router.get("/analyses")
async def list_analyses(
    limit: int = 20,
    min_ev: float = 0.0,
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(MatchAnalysis, Match)
        .join(Match, MatchAnalysis.match_id == Match.id)
        .where(MatchAnalysis.ev >= min_ev)
        .order_by(desc(MatchAnalysis.analyzed_at))
        .limit(limit)
    )
    rows = (await db.execute(query)).all()
    return [
        {
            "match":      f"{match.home_team} vs {match.away_team}",
            "league":     match.sport_title,
            "commence":   match.commence_time.isoformat(),
            "market":     a.market,
            "lambda_home": round(a.lambda_home, 3),
            "lambda_away": round(a.lambda_away, 3),
            "model_prob": round(a.model_probability, 4),
            "best_odd":   a.best_odd,
            "bookmaker":  a.best_bookmaker,
            "ev":         round(a.ev, 4),
        }
        for a, match in rows
    ]


# ─── Gerenciamento de sinais (limpar / reprocessar) ───────────────────────────

@router.delete("/signals")
async def delete_signals(
    match_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Apaga sinais do banco.
    - Sem parâmetros: apaga TODOS os sinais (e análises associadas).
    - ?match_id=X: apaga só os sinais daquele jogo.
    """
    from sqlalchemy import delete as sa_delete

    if match_id:
        await db.execute(sa_delete(Signal).where(Signal.match_id == match_id))
        await db.execute(sa_delete(MatchAnalysis).where(MatchAnalysis.match_id == match_id))
        await db.commit()
        return {"message": f"Sinais do jogo {match_id} apagados."}
    else:
        deleted_sigs = (await db.execute(sa_delete(Signal))).rowcount
        await db.execute(sa_delete(MatchAnalysis))
        await db.execute(sa_delete(Match))
        await db.commit()
        return {"message": f"Todos os sinais apagados ({deleted_sigs} registros)."}


@router.delete("/signals/{signal_id}")
async def delete_signal(
    signal_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Apaga um sinal individual pelo ID."""
    from sqlalchemy import delete as sa_delete
    result = await db.execute(sa_delete(Signal).where(Signal.id == signal_id))
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Sinal não encontrado.")
    return {"message": f"Sinal {signal_id} apagado.", "deleted": signal_id}


@router.post("/signals/reset-results")
async def reset_signal_results(
    db: AsyncSession = Depends(get_db),
):
    """Reseta Win/Loss de todos os sinais para PENDING (null) — útil para reprocessar."""
    from sqlalchemy import update as sa_update
    await db.execute(
        sa_update(Signal).values(result=None, profit_loss=None, actual_goals=None)
    )
    await db.commit()
    return {"message": "Resultados resetados. Rode /api/v1/live para recalcular."}


@router.post("/signals/backfill-scores")
async def backfill_scores(db: AsyncSession = Depends(get_db)):
    """
    Re-busca placar (score_home/score_away) para sinais que têm resultado
    mas não têm placar individual. Útil para sinais salvos antes do patch.
    """
    from app.services.sokkerpro_client import SokkerProClient
    sokker = SokkerProClient()

    # Sinais com resultado mas sem placar individual
    rows = (await db.execute(
        select(Signal, Match)
        .join(Match, Signal.match_id == Match.id)
        .where(Signal.result != None)
        .where(Signal.score_home == None)
    )).all()

    updated = 0
    failed  = 0

    for sig, match in rows:
        try:
            raw = await sokker.get_fixture_score(str(sig.match_id))
            if raw and raw.get("home_goals") is not None and raw.get("away_goals") is not None:
                sig.score_home   = int(raw["home_goals"])
                sig.score_away   = int(raw["away_goals"])
                sig.actual_goals = sig.score_home + sig.score_away
                updated += 1
            else:
                failed += 1
        except Exception as e:
            logger.warning(f"[Backfill] {sig.match_id}: {e}")
            failed += 1

    if updated:
        await db.commit()
        logger.info(f"[Backfill] {updated} placares preenchidos, {failed} sem dados")

    return {
        "updated": updated,
        "failed":  failed,
        "message": f"{updated} placares preenchidos, {failed} não encontrados na API"
    }


@router.get("/leagues/today")
async def leagues_today(db: AsyncSession = Depends(get_db)):
    """Ligas com sinais gerados hoje — para o picker do HUB."""
    from sqlalchemy import distinct
    rows = (await db.execute(
        select(Match.sport_key, Match.sport_title, func.count(Signal.id).label("matches"))
        .join(Signal, Signal.match_id == Match.id)
        .group_by(Match.sport_key, Match.sport_title)
        .order_by(func.count(Signal.id).desc())
    )).all()
    return {
        "leagues": [
            {"league_id": r.sport_key, "league_name": r.sport_title, "matches": r.matches}
            for r in rows
        ]
    }


@router.get("/settings/analysis-league")
async def get_analysis_league():
    """Retorna a liga configurada para análise (runtime)."""
    from app.services.runtime_settings import get_analysis_league_id
    val = get_analysis_league_id() or ""
    return {"analysis_league_id": val}


@router.post("/settings/analysis-league")
async def set_analysis_league(league_id: Optional[str] = None):
    """Define a liga para análise (runtime, sem reiniciar)."""
    from app.services.runtime_settings import set_analysis_league_id
    set_analysis_league_id(league_id or "")
    return {"analysis_league_id": league_id or ""}