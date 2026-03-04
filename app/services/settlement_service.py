
"""
SettlementService — fecha sinais no banco a partir do placar final do SokkerPRO.

Mercados suportados (signals.market):
- Home / Away / Draw (1X2)
- BTTS
- Over_<x.x>  (ex.: Over_1.5)
- Under_<x.x> (ex.: Under_3.5)
- DC_<A>_<B>  (Double Chance: DC_Home_Draw, DC_Home_Away, DC_Away_Draw)
- MULTI_<fixtureId>_<legMarket>_<fixtureId>_<legMarket>_... (todas as pernas precisam ganhar)

Regras de profit_loss (unidades):
- WIN  => stake_units * (odd - 1)
- LOSS => -stake_units
- VOID => 0 (ex.: jogo cancelado/adiado/abandonado/placar indisponível)

Uso:
    python -m app.services.settlement_service

Ou:
    import asyncio
    from app.services.settlement_service import SettlementService
    s = SettlementService(db_path="sports_ev.db")
    print(asyncio.run(s.settle_pending(limit=500)))
"""
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from loguru import logger
from app.services.sokkerpro_client import SokkerProClient
from app.config import get_settings
from app.performance.metrics_persistence import MetricsPersistence, MetricsConfig

FINISHED_STATUSES = {"FINISHED"}
VOID_STATUSES = {"CANCELLED", "POSTPONED", "SUSPENDED", "ABANDONED"}


def _profit(result: str, stake: float, odd: float) -> float:
    if result == "WIN":
        return float(stake) * (float(odd) - 1.0)
    if result == "LOSS":
        return -float(stake)
    return 0.0  # VOID/unknown


def _settle_1x2(market: str, home: int, away: int) -> str:
    if market == "Home":
        return "WIN" if home > away else "LOSS"
    if market == "Away":
        return "WIN" if away > home else "LOSS"
    if market == "Draw":
        return "WIN" if home == away else "LOSS"
    return "VOID"


def _settle_btts(home: int, away: int) -> str:
    return "WIN" if (home > 0 and away > 0) else "LOSS"


def _parse_ou(market: str) -> Optional[Tuple[str, float]]:
    """
    Over_1.5 / Under_3.5 -> ("Over", 1.5) / ("Under", 3.5)
    """
    if market.startswith("Over_") or market.startswith("Under_"):
        side, line = market.split("_", 1)
        try:
            return side, float(line)
        except Exception:
            return None
    return None


def _settle_ou(side: str, line: float, total_goals: int) -> str:
    if side == "Over":
        return "WIN" if total_goals > line else "LOSS"
    if side == "Under":
        # Under_2.5: win if 0,1,2
        return "WIN" if total_goals < line else "LOSS"
    return "VOID"


def _parse_dc(market: str) -> Optional[Tuple[str, str]]:
    """
    DC_Home_Draw -> ("Home","Draw")
    DC_Home_Away -> ("Home","Away")
    DC_Away_Draw -> ("Away","Draw")
    """
    if not market.startswith("DC_"):
        return None
    parts = market.split("_")
    if len(parts) != 3:
        return None
    a, b = parts[1], parts[2]
    if a not in ("Home", "Away", "Draw") or b not in ("Home", "Away", "Draw"):
        return None
    if a == b:
        return None
    return a, b


def _settle_dc(a: str, b: str, home: int, away: int) -> str:
    # Determine 1X2 outcome
    if home > away:
        outcome = "Home"
    elif away > home:
        outcome = "Away"
    else:
        outcome = "Draw"
    return "WIN" if outcome in (a, b) else "LOSS"


def _parse_multi(market: str) -> Optional[List[Tuple[str, str]]]:
    """
    MULTI_19467844_BTTS_19427164_BTTS
      -> [("19467844","BTTS"), ("19427164","BTTS")]
    MULTI_19425145_Away_19427164_BTTS
      -> [("19425145","Away"), ("19427164","BTTS")]
    MULTI_19660914_Over_1.5_19427164_BTTS  (se existir)
      -> [("19660914","Over_1.5"), ("19427164","BTTS")]

    Observação: como market pode conter "_" (Over_1.5, DC_Home_Draw),
    esta função tenta reconstruir as pernas em pares (fixture_id, market_str)
    usando heurística: fixture_id é sempre numérico.
    """
    if not market.startswith("MULTI_"):
        return None
    parts = market.split("_")[1:]
    if len(parts) < 2:
        return None

    legs: List[Tuple[str, str]] = []
    i = 0
    while i < len(parts):
        fid = parts[i]
        if not fid.isdigit():
            return None
        i += 1
        if i >= len(parts):
            return None

        # market token(s) até o próximo token numérico (próximo fixture_id) ou fim
        m_tokens = []
        while i < len(parts) and not parts[i].isdigit():
            m_tokens.append(parts[i])
            i += 1
        if not m_tokens:
            return None
        leg_market = "_".join(m_tokens)
        legs.append((fid, leg_market))

    return legs if legs else None


def _settle_market(market: str, home: int, away: int) -> Optional[str]:
    """
    Retorna WIN/LOSS/VOID ou None se mercado não suportado.
    """
    if market == "BTTS":
        return _settle_btts(home, away)
    if market in ("Home", "Away", "Draw"):
        return _settle_1x2(market, home, away)

    ou = _parse_ou(market)
    if ou:
        side, line = ou
        return _settle_ou(side, line, home + away)

    dc = _parse_dc(market)
    if dc:
        a, b = dc
        return _settle_dc(a, b, home, away)

    return None


@dataclass
class SettlementService:
    db_path: str

    async def settle_pending(self, limit: int = 500) -> Dict[str, int]:
        settings = get_settings()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, match_id, market, suggested_odd, stake_units
            FROM signals
            WHERE result IS NULL
            ORDER BY sent_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        if not rows:
            conn.close()
            return {"settled": 0, "wins": 0, "losses": 0, "void": 0, "skipped": 0, "unsupported": 0}

        client = SokkerProClient()
        counts = {"settled": 0, "wins": 0, "losses": 0, "void": 0, "skipped": 0, "unsupported": 0}

        score_cache: Dict[str, Optional[dict]] = {}

        async def get_score(fid: str) -> Optional[dict]:
            if fid in score_cache:
                return score_cache[fid]
            sc = await client.get_fixture_score(str(fid))
            score_cache[fid] = sc
            return sc

        for r in rows:
            signal_id = r["id"]
            match_id = str(r["match_id"])
            market = str(r["market"])
            odd = float(r["suggested_odd"])
            stake = float(r["stake_units"])

            try:
                # MULTI (parlay)
                if market.startswith("MULTI_"):
                    legs = _parse_multi(market)
                    if not legs:
                        counts["unsupported"] += 1
                        continue

                    leg_results: List[str] = []
                    any_void = False
                    any_unfinished = False
                    goals_sum = 0

                    for fid, leg_market in legs:
                        sc = await get_score(fid)
                        if not sc:
                            any_unfinished = True
                            break

                        status = sc.get("status")
                        if status in VOID_STATUSES:
                            any_void = True
                            continue
                        if status not in FINISHED_STATUSES:
                            any_unfinished = True
                            break

                        hg = sc.get("home_goals")
                        ag = sc.get("away_goals")
                        if hg is None or ag is None:
                            any_unfinished = True
                            break
                        hg = int(hg); ag = int(ag)
                        goals_sum += hg + ag

                        res = _settle_market(leg_market, hg, ag)
                        if res is None:
                            any_unfinished = True  # treat as not settleable
                            break
                        leg_results.append(res)

                    if any_unfinished:
                        counts["skipped"] += 1
                        continue

                    if any_void:
                        final = "VOID"
                        pl = _profit(final, stake, odd)
                        cur.execute("UPDATE signals SET result=?, profit_loss=? WHERE id=?", (final, pl, signal_id))
                        counts["settled"] += 1
                        counts["void"] += 1
                        continue

                    final = "WIN" if all(x == "WIN" for x in leg_results) else "LOSS"
                    pl = _profit(final, stake, odd)
                    cur.execute(
                        "UPDATE signals SET result=?, profit_loss=?, actual_goals=? WHERE id=?",
                        (final, pl, goals_sum, signal_id),
                    )
                    counts["settled"] += 1
                    counts["wins"] += 1 if final == "WIN" else 0
                    counts["losses"] += 1 if final == "LOSS" else 0
                    continue

                # Single match
                sc = await get_score(match_id)
                if not sc:
                    counts["skipped"] += 1
                    continue

                status = sc.get("status")
                if status in VOID_STATUSES:
                    final = "VOID"
                    pl = _profit(final, stake, odd)
                    cur.execute("UPDATE signals SET result=?, profit_loss=? WHERE id=?", (final, pl, signal_id))
                    counts["settled"] += 1
                    counts["void"] += 1
                    continue

                if status not in FINISHED_STATUSES:
                    counts["skipped"] += 1
                    continue

                hg = sc.get("home_goals")
                ag = sc.get("away_goals")
                if hg is None or ag is None:
                    counts["skipped"] += 1
                    continue
                hg = int(hg); ag = int(ag)

                res = _settle_market(market, hg, ag)
                if res is None:
                    counts["unsupported"] += 1
                    continue

                pl = _profit(res, stake, odd)
                cur.execute(
                    "UPDATE signals SET result=?, profit_loss=?, actual_goals=? WHERE id=?",
                    (res, pl, hg + ag, signal_id),
                )
                counts["settled"] += 1
                counts["wins"] += 1 if res == "WIN" else 0
                counts["losses"] += 1 if res == "LOSS" else 0

            except Exception as e:
                logger.warning(f"[Settlement] falha ao liquidar signal_id={signal_id}: {e}")
                counts["skipped"] += 1

        conn.commit()
        conn.close()

        # Phase 1 — compute advanced metrics (safe, additive)
        if settings.enable_adv_metrics:
            try:
                cfg = MetricsConfig(
                    eps=float(settings.metrics_logloss_eps),
                    n_boot=int(settings.metrics_bootstrap_n),
                    seed=int(settings.metrics_bootstrap_seed),
                )
                MetricsPersistence(self.db_path, cfg).compute_and_persist_daily()
            except Exception as e:
                logger.warning(f"[Metrics] falha ao computar metrics_daily: {e}")

        try:
            await client.close()
        except Exception:
            pass
        return counts


if __name__ == "__main__":
    import os
    db = os.getenv("SPORTS_DB", "sports_ev.db")
    s = SettlementService(db_path=db)
    out = asyncio.run(s.settle_pending(limit=500))
    print(out)
