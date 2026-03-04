"""
LiveScoreService V2 — SokkerPRO como fonte única.
Substitui ESPN completamente — sem erros de NoneType, odds Bet365/1xBet reais.
"""
from datetime import datetime, timezone, timedelta
from loguru import logger

from app.services.sokkerpro_client import SokkerProClient


class LiveScoreService:

    def __init__(self):
        self.sokker = SokkerProClient()
        logger.info("✅ LiveScoreService V2 usando SokkerPRO")

    async def get_live_matches(self) -> list[dict]:
        try:
            matches = await self.sokker.get_live_matches()
            logger.info(f"⚽ {len(matches)} jogos ao vivo agora")
            return matches
        except Exception as e:
            logger.warning(f"SokkerPRO live falhou: {e}")
            return []

    async def get_finished_today(self) -> list[dict]:
        try:
            matches = await self.sokker.get_finished_today()
            logger.info(f"✅ {len(matches)} jogos terminados hoje")
            return matches
        except Exception as e:
            logger.warning(f"SokkerPRO finished falhou: {e}")
            return []

    async def get_all_today(self, timezone_offset: str = "utc-3") -> list[dict]:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return await self.sokker.get_fixtures_for_date(today, timezone_offset)
        except Exception as e:
            logger.warning(f"SokkerPRO all_today falhou: {e}")
            return []

    def parse_score(self, match: dict) -> dict:
        score  = match.get("score", {})
        full   = score.get("fullTime", {})
        half   = score.get("halfTime", {})
        status = match.get("status", "SCHEDULED")
        minute = match.get("minute")

        if minute is None and status in ("IN_PLAY", "PAUSED"):
            try:
                utc_date = match.get("utcDate", "").strip('"')
                start = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
                elapsed = int((datetime.now(timezone.utc) - start).total_seconds() / 60)
                if elapsed > 47:
                    elapsed = elapsed - 15
                minute = max(1, min(elapsed, 90))
            except Exception:
                minute = None

        return {
            "match_id":      str(match.get("id", "")),
            "home_team":     match.get("homeTeam", {}).get("name", ""),
            "away_team":     match.get("awayTeam", {}).get("name", ""),
            "status":        status,
            "minute":        minute,
            "home_goals":    full.get("home"),
            "away_goals":    full.get("away"),
            "home_goals_ht": half.get("home"),
            "away_goals_ht": half.get("away"),
            "utc_date":      match.get("utcDate", ""),
            "statistics":    match.get("statistics", {}),
            "sokker_odds":   match.get("sokker_odds", {}),
            "league_name":   match.get("league_name", ""),
        }

    def _resolve_leg(self, leg: str, home_goals: int, away_goals: int):
        """Resolve uma perna individual. Retorna True/False ou None se desconhecida."""
        total = home_goals + away_goals
        leg = leg.strip()
        if leg in ("Over_2.5", "over_2.5"):              return total > 2
        if leg in ("Over_1.5", "over_1.5"):              return total > 1
        if leg in ("Over_3.5", "over_3.5"):              return total > 3
        if leg in ("Over_0.5", "over_0.5"):              return total > 0
        if leg in ("Under_2.5", "under_2.5"):            return total < 3
        if leg in ("Under_1.5", "under_1.5"):            return total < 2
        if leg in ("Under_3.5", "under_3.5"):            return total < 4
        if leg in ("Home", "Casa"):                      return home_goals > away_goals
        if leg in ("Draw", "Empate", "X"):               return home_goals == away_goals
        if leg in ("Away", "Fora"):                      return away_goals > home_goals
        if leg in ("BTTS", "btts", "Ambos_Marcam"):      return home_goals > 0 and away_goals > 0
        if leg in ("No_BTTS", "no_btts"):                return home_goals == 0 or away_goals == 0
        return None

    def _split_combo_legs(self, raw: str) -> list:
        """Divide raw do COMBO em legs, reagrupando Over_X.X, Under_X.X, Ambos_Marcam, No_BTTS."""
        parts = raw.split("_")
        legs = []
        i = 0
        while i < len(parts):
            if parts[i] in ("Over", "Under") and i + 1 < len(parts):
                legs.append(f"{parts[i]}_{parts[i+1]}")
                i += 2
            elif parts[i] == "Ambos" and i + 1 < len(parts) and parts[i+1] == "Marcam":
                legs.append("Ambos_Marcam")
                i += 2
            elif parts[i] == "No" and i + 1 < len(parts) and parts[i+1] == "BTTS":
                legs.append("No_BTTS")
                i += 2
            else:
                legs.append(parts[i])
                i += 1
        return legs

    def determine_result(self, market: str, home_goals: int, away_goals: int) -> str:
        # ── Mercados simples ─────────────────────────────────────────────────
        simple = self._resolve_leg(market, home_goals, away_goals)
        if simple is not None:
            return "WIN" if simple else "LOSS"

        # ── Combos: COMBO_Leg1_Leg2[_Leg3] ──────────────────────────────────
        if market.startswith("COMBO_"):
            legs = self._split_combo_legs(market[len("COMBO_"):])
            results = [self._resolve_leg(l, home_goals, away_goals) for l in legs]
            results = [r for r in results if r is not None]
            if results:
                return "WIN" if all(results) else "LOSS"

        return "VOID"
