"""
FormService V2 — fonte única: SokkerPRO (substitui ESPN completamente)

Diferenças vs V1 (ESPN):
  - Médias de gols já vêm pré-computadas no fixture (medias_home/away_goal)
  - Não precisa buscar histórico de jogos separadamente
  - Cobertura: 270 ligas vs 12 da ESPN
  - Sem erros de NoneType — dados estruturados e consistentes
"""
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger

from app.config import get_settings
from app.models.schemas import TeamForm
from app.services.sokkerpro_client import SokkerProClient

settings = get_settings()

# Mapeamento de ligas — leagueId SokkerPRO → código interno + nome display
# Cobertura muito maior que ESPN. Os mais relevantes para BR:
LEAGUE_MAP = {
    # Europa Top 5
    "8":    {"name": "Premier League",      "country": "England"},
    "564":  {"name": "La Liga",             "country": "Spain"},
    "384":  {"name": "Serie A",             "country": "Italy"},
    "82":   {"name": "Bundesliga",          "country": "Germany"},
    "301":  {"name": "Ligue 1",             "country": "France"},
    # UEFA
    "2":    {"name": "Champions League",    "country": "Europe"},
    "5":    {"name": "Europa League",       "country": "Europe"},
    "1089": {"name": "Conference League",   "country": "Europe"},
    # Outros
    "72":   {"name": "Eredivisie",          "country": "Netherlands"},
    "462":  {"name": "Primeira Liga",       "country": "Portugal"},
    # Brasil
    "325":  {"name": "Brasileirão",         "country": "Brazil"},
    "996":  {"name": "Copa do Brasil",      "country": "Brazil"},
    "348":  {"name": "MLS",                 "country": "USA"},
}

# Médias históricas por liga para fallback (quando medias não disponíveis no fixture)
LEAGUE_AVERAGES = {
    "8":    {"home_avg": 1.54, "away_avg": 1.15, "avg_total": 2.69},  # PL
    "564":  {"home_avg": 1.52, "away_avg": 1.11, "avg_total": 2.63},  # La Liga
    "384":  {"home_avg": 1.52, "away_avg": 1.11, "avg_total": 2.63},  # Serie A
    "82":   {"home_avg": 1.72, "away_avg": 1.29, "avg_total": 3.01},  # Bundesliga
    "301":  {"home_avg": 1.44, "away_avg": 1.06, "avg_total": 2.50},  # Ligue 1
    "2":    {"home_avg": 1.60, "away_avg": 1.10, "avg_total": 2.70},  # UCL
    "325":  {"home_avg": 1.41, "away_avg": 0.99, "avg_total": 2.40},  # Brasileirão
    "_default": {"home_avg": 1.45, "away_avg": 1.05, "avg_total": 2.50},
}


class FormService:
    """
    Serviço de forma dos times V2 — SokkerPRO como fonte única.

    As médias de gols são extraídas diretamente do campo medias_*
    de cada fixture, sem precisar de chamadas adicionais à API.
    """

    def __init__(self):
        self.sokker = SokkerProClient()
        logger.info("✅ FormService V2 — usando SokkerPRO (270 ligas, sem key)")

    def get_league_averages(self, league_id: str) -> dict:
        return LEAGUE_AVERAGES.get(str(league_id), LEAGUE_AVERAGES["_default"])

    async def get_todays_matches(
        self,
        league_id: str,
        date: str = None,
        timezone_offset: str = "utc-3",
    ) -> list[dict]:
        """
        Retorna jogos de hoje de uma liga específica.
        Sem league_id → retorna todas as ligas.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        all_fixtures = await self.sokker.get_fixtures_for_date(date, timezone_offset)

        if league_id:
            filtered = [f for f in all_fixtures if str(f.get("league_id")) == str(league_id)]
            logger.info(f"[FormService] Liga {league_id}: {len(filtered)} jogos em {date}")
            return filtered

        return all_fixtures

    async def get_all_todays_scheduled(self, timezone_offset: str = "utc-3") -> list[dict]:
        """
        Todos os jogos agendados dentro da janela de análise.

        Nota: o SokkerPRO separa por data. Com LOOKAHEAD_HOURS > 24,
        precisamos buscar também "amanhã" para não perder jogos pré-jogo.
        """
        now_utc = datetime.now(timezone.utc)
        today = now_utc.strftime("%Y-%m-%d")
        fixtures = await self.sokker.get_fixtures_for_date(today, timezone_offset)

        # Se a janela passa de 24h, inclui amanhã (pré-jogo) também
        lookahead = int(getattr(settings, "lookahead_hours", 24) or 24)
        if lookahead > 24:
            tomorrow = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")
            fixtures += await self.sokker.get_fixtures_for_date(tomorrow, timezone_offset)

        # apenas agendados e dedupe por id
        out = []
        seen = set()
        for f in fixtures:
            if f.get("status") != "SCHEDULED":
                continue
            mid = f.get("id")
            if mid and mid in seen:
                continue
            if mid:
                seen.add(mid)
            out.append(f)
        return out

    def get_team_form_from_fixture(
        self,
        fixture: dict,
        side: str,          # "home" or "away"
        league_id: str = "",
    ) -> TeamForm:
        """
        Calcula TeamForm a partir das médias pré-computadas do fixture.
        Muito mais eficiente que o ESPN (zero chamadas extras à API).
        """
        medias = fixture.get("medias", {})
        avgs   = self.get_league_averages(league_id)
        league_avg = avgs["avg_total"] / 2

        if side == "home":
            team_name    = fixture.get("homeTeam", {}).get("name", "Home")
            avg_scored   = medias.get("home_avg_goal")   or avgs["home_avg"]
            avg_conceded = medias.get("away_avg_goal")   or avgs["away_avg"]
        else:
            team_name    = fixture.get("awayTeam", {}).get("name", "Away")
            avg_scored   = medias.get("away_avg_goal")   or avgs["away_avg"]
            avg_conceded = medias.get("home_avg_goal")   or avgs["home_avg"]

        attack_strength  = avg_scored   / league_avg if league_avg > 0 else 1.0
        defense_weakness = avg_conceded / league_avg if league_avg > 0 else 1.0

        return TeamForm(
            team_name=team_name,
            avg_scored=round(avg_scored, 3),
            avg_conceded=round(avg_conceded, 3),
            matches_used=10,  # SokkerPRO usa a temporada toda
            attack_strength=round(attack_strength, 3),
            defense_weakness=round(defense_weakness, 3),
        )

# --- V8 helper: peso temporal (quando houver série histórica) ---
@staticmethod
def weighted_average(values, decay: float = 0.35) -> float:
    """Média ponderada com decaimento exponencial.

    values[0] deve ser o valor mais recente.
    Se o fixture não trouxer série histórica, isso fica disponível para uso futuro.
    """
    from math import exp
    if not values:
        return 0.0
    total_w = 0.0
    total = 0.0
    for i, v in enumerate(values):
        try:
            v = float(v)
        except Exception:
            continue
        w = exp(-decay * i)
        total += v * w
        total_w += w
    return (total / total_w) if total_w > 0 else 0.0
