"""
RefereeService — resolve árbitro via SokkerPRO (referee/{id}/season/{season}) + UEFA/Football-Data (fallback)

Prioridade: UEFA API (CL/EL/EC) → Football-Data.org → None
SofaScore removido — SokkerPRO é agora a fonte principal de dados.
"""
import httpx
import json
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger

from app.config import get_settings
from app.services.sokkerpro_client import SokkerProClient

settings = get_settings()

UEFA_BASE = "https://match.uefa.com/v5"
FD_BASE   = "https://api.football-data.org/v4"

UEFA_COMPETITION_IDS = {"CL": 1, "EL": 3, "EC": 2077}

FD_LEAGUE_MAP = {
    "PL": "PL", "PD": "PD", "SA": "SA", "BL1": "BL1",
    "FL1": "FL1", "PPL": "PPL", "DED": "DED", "BSA": "BSB",
}


class RefereeService:
    TIMEOUT = 10.0

    def __init__(self):
        self._cache: dict[str, tuple] = {}
        self._ttl    = timedelta(hours=getattr(settings, "sokkerpro_referee_cache_ttl_hours", 24))
        self._fd_key = getattr(settings, "football_data_api_key", "")
        self._sp_primary_type = getattr(settings, "sokkerpro_referee_primary_type_id", 6)
        self._sp = SokkerProClient()

    def _get(self, key):
        if key in self._cache:
            val, ts = self._cache[key]
            if datetime.now(timezone.utc) - ts < self._ttl:
                return val
        return None

    def _set(self, key, val):
        self._cache[key] = (val, datetime.now(timezone.utc))


    async def resolve_referee(self, fixture: dict, home_name: str, away_name: str, league_code: str, match_date) -> tuple[Optional[str], Optional[dict]]:
        """Resolve árbitro e stats.

        Ordem: SokkerPRO fixture→referee/season → UEFA → Football-Data.
        Retorna (referee_name, stats_dict).

        stats_dict inclui médias por jogo quando disponível:
          - yellow_avg, red_avg, yellowred_avg, penalties_avg, fouls_avg, matches, booking_pts_avg
        """
        name, stats = await self._from_sokkerpro_fixture(fixture)
        if name:
            return name, stats

        date_str = match_date.strftime("%Y-%m-%d")

        # UEFA (CL/EL/EC)
        if league_code.upper() in UEFA_COMPETITION_IDS:
            uefa_name = await self._from_uefa(home_name, away_name, league_code, date_str)
            if uefa_name:
                return uefa_name, None

        # Football-Data.org
        if league_code.upper() in FD_LEAGUE_MAP and self._fd_key:
            fd_name = await self._from_fd(home_name, away_name, league_code, date_str)
            if fd_name:
                return fd_name, None

        return None, None

    async def _from_sokkerpro_fixture(self, fixture: dict) -> tuple[Optional[str], Optional[dict]]:
        """Extrai referee_id + season_id do fixture SokkerPRO e resolve nome/stats via /referee/{id}/season/{season}."""
        try:
            fixture_id = fixture.get("id") or fixture.get("fixtureId") or "?"

            # dict interno usa "season_id"; fixture bruto do /fixture/{id} usa "seasonId"
            season_id = (
                fixture.get("season_id")
                or fixture.get("seasonId")
                or fixture.get("seasonID")
            )
            referees = fixture.get("referees")

            # Log detalhado para diagnóstico
            if not season_id and not referees:
                logger.debug(f"[RefereeService] fixture {fixture_id}: sem season_id e sem referees")
                return None, None

            if not season_id:
                logger.debug(f"[RefereeService] fixture {fixture_id}: season_id ausente (referees presentes)")
                return None, None

            if not referees:
                logger.debug(f"[RefereeService] fixture {fixture_id}: referees ausentes (season_id={season_id})")
                return None, None

            # `referees` pode vir como str JSON ou list[dict]
            if isinstance(referees, str):
                try:
                    referees = json.loads(referees)
                except Exception as e:
                    logger.warning(f"[RefereeService] fixture {fixture_id}: erro ao parsear referees JSON: {e}")
                    return None, None

            if not isinstance(referees, list) or not referees:
                logger.debug(f"[RefereeService] fixture {fixture_id}: referees inválidos após parse: {type(referees)}")
                return None, None

            logger.debug(f"[RefereeService] fixture {fixture_id}: {len(referees)} referee(s) encontrado(s) | season_id={season_id}")

            # ── Estratégia de seleção do árbitro principal ──────────────────
            # type_id NÃO é consistente entre ligas (PL usa 6, Ligue 1 usa 9, etc.).
            # Tenta TODOS em ordem crescente de type_id até a API responder com dados válidos.
            candidates = sorted(referees, key=lambda r: int(r.get("type_id") or 99))

            for chosen in candidates:
                referee_id = chosen.get("referee_id") or chosen.get("id")
                ref_type   = chosen.get("type_id")
                if not referee_id:
                    logger.debug(f"[RefereeService] fixture {fixture_id}: referee sem id, pulando")
                    continue

                cache_key = f"sp_ref_{referee_id}_{season_id}"
                cached = self._get(cache_key)
                if cached is not None:
                    if cached[0]:
                        return cached[0], cached[1]
                    logger.debug(f"[RefereeService] referee_id={referee_id} em cache como sem dados")
                    continue

                logger.debug(f"[RefereeService] buscando /referee/{referee_id}/season/{season_id} (type_id={ref_type})")
                data = await self._sp.get_referee_season(int(referee_id), int(season_id))

                if not data:
                    logger.debug(f"[RefereeService] referee_id={referee_id}: API retornou vazio")
                    self._set(cache_key, ("", None))
                    continue

                if not isinstance(data, dict):
                    logger.debug(f"[RefereeService] referee_id={referee_id}: resposta inesperada tipo {type(data)}: {str(data)[:200]}")
                    self._set(cache_key, ("", None))
                    continue

                # Tenta extrair nome em múltiplos campos possíveis
                name = (
                    data.get("display_name")
                    or data.get("name")
                    or data.get("common_name")
                    or data.get("fullname")
                    or data.get("full_name")
                    or data.get("short_name")
                )

                if not name:
                    logger.debug(f"[RefereeService] referee_id={referee_id}: dados sem nome. Chaves: {list(data.keys())[:10]}")

                stats = self._extract_referee_stats(data)
                if stats is None:
                    stats = self._derive_referee_stats_from_latest(data)
                    if stats is not None:
                        logger.info(f"[RefereeService] referee_id={referee_id}: stats derivadas via latest (n={stats.get('matches')})")
                self._set(cache_key, (name or "", stats))

                if name:
                    logger.info(f"👨‍⚖️ Árbitro resolvido via SokkerPRO: {name} (id={referee_id}, type_id={ref_type})")
                    return name, stats

            logger.debug(f"[RefereeService] fixture {fixture_id}: nenhum candidato resolveu nome do árbitro")
            return None, None
        except Exception as e:
            logger.warning(f"[RefereeService] _from_sokkerpro_fixture erro: {e}")
            return None, None

    def _extract_referee_stats(self, data: dict) -> Optional[dict]:
        """Extrai médias por jogo do payload de árbitro (SokkerPRO)."""
        try:
            stats = data.get("statistics")
            if not isinstance(stats, list) or not stats:
                return None
            details = stats[0].get("details") or []
            if not isinstance(details, list):
                return None

            by_code = {}
            for d in details:
                t = (d.get("type") or {})
                code = t.get("code")
                if not code:
                    continue
                val = d.get("value") or {}
                allv = val.get("all") if isinstance(val, dict) else None
                if isinstance(allv, dict):
                    by_code[code] = {"count": allv.get("count"), "average": allv.get("average")}

            def avg(code):
                v = by_code.get(code)
                if v and v.get("average") is not None:
                    return float(v["average"])
                return None

            def cnt(code, default=0):
                v = by_code.get(code)
                if v and v.get("count") is not None:
                    return int(v["count"])
                return default

            yellow = avg("yellowcards")
            red = avg("redcards")
            yellowred = avg("yellowred-cards")
            pens = avg("penalties")
            fouls = avg("fouls")
            matches = cnt("matches", 0)

            # Booking points aproximado (padrão comum): Y=10, 2Y/R=35, R=25
            bpts = None
            if yellow is not None or red is not None or yellowred is not None:
                bpts = (yellow or 0.0) * 10.0 + (red or 0.0) * 25.0 + (yellowred or 0.0) * 35.0

            return {
                "yellow_avg": yellow,
                "red_avg": red,
                "yellowred_avg": yellowred,
                "penalties_avg": pens,
                "fouls_avg": fouls,
                "matches": matches,
                "booking_pts_avg": bpts,
            }
        except Exception:
            return None


    def _derive_referee_stats_from_latest(self, data: dict) -> Optional[dict]:
        """Deriva médias do árbitro usando o bloco `latest` quando `statistics` vier vazio.

        Usa `latest[].fixture.statistics` com type_id:
          - 84: yellowcards
          - 83: redcards
          - 85: yellowred
          - 56: fouls (quando presente)
        Retorna dict no mesmo formato de _extract_referee_stats, com `source="latest_derived"`.
        """
        try:
            latest = data.get("latest")
            if not isinstance(latest, list) or not latest:
                return None

            total_y = 0.0
            total_r = 0.0
            total_yr = 0.0
            total_f = 0.0
            n = 0

            def _stat_value(s: dict) -> float:
                d = s.get("data") or {}
                v = d.get("value")
                try:
                    if v is None:
                        return 0.0
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        return float(v.strip().replace(",", "."))
                    return float(v)
                except Exception:
                    return 0.0

            for item in latest:
                fx = item.get("fixture") or {}
                stats = fx.get("statistics") or []
                if not isinstance(stats, list) or not stats:
                    continue

                y = 0.0
                r = 0.0
                yr = 0.0
                f = 0.0

                for s in stats:
                    tid = s.get("type_id")
                    if tid is None:
                        t = s.get("type") or {}
                        tid = t.get("id")
                    if tid is None:
                        continue

                    val = _stat_value(s)

                    if int(tid) == 84:
                        y += val
                    elif int(tid) == 83:
                        r += val
                    elif int(tid) == 85:
                        yr += val
                    elif int(tid) == 56:
                        f += val

                # contabiliza só se tiver alguma info de cartões (pra não diluir com jogos sem esses stats)
                if (y + r + yr) > 0.0:
                    total_y += y
                    total_r += r
                    total_yr += yr
                    total_f += f
                    n += 1

            if n <= 0:
                return None

            yellow = total_y / n
            red = total_r / n
            yellowred = total_yr / n
            fouls = (total_f / n) if total_f > 0 else None

            bpts = (yellow or 0.0) * 10.0 + (red or 0.0) * 25.0 + (yellowred or 0.0) * 35.0

            return {
                "yellow_avg": float(yellow),
                "red_avg": float(red),
                "yellowred_avg": float(yellowred),
                "penalties_avg": None,
                "fouls_avg": float(fouls) if fouls is not None else None,
                "matches": int(n),
                "booking_pts_avg": float(bpts),
                "source": "latest_derived",
            }
        except Exception:
            return None

    async def get_referee(self, home_name, away_name, league_code, match_date) -> Optional[str]:
        date_str  = match_date.strftime("%Y-%m-%d")
        cache_key = f"ref_{league_code}_{date_str}_{self._norm(home_name)}"

        cached = self._get(cache_key)
        if cached is not None:
            return cached or None

        # 1) UEFA API (CL/EL/EC)
        if league_code.upper() in UEFA_COMPETITION_IDS:
            name = await self._from_uefa(home_name, away_name, league_code, date_str)
            if name:
                logger.info(f"Arbitro (UEFA): {name} | {home_name} vs {away_name}")
                self._set(cache_key, name)
                return name

        # 2) Football-Data.org
        if league_code.upper() in FD_LEAGUE_MAP and self._fd_key:
            name = await self._from_fd(home_name, away_name, league_code, date_str)
            if name:
                logger.info(f"Arbitro (FD): {name} | {home_name} vs {away_name}")
                self._set(cache_key, name)
                return name

        self._set(cache_key, "")
        logger.debug(f"Arbitro nao encontrado: {home_name} vs {away_name} ({league_code})")
        return None

    async def _from_uefa(self, home, away, league_code, date_str) -> Optional[str]:
        comp_id = UEFA_COMPETITION_IDS[league_code.upper()]
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                r = await client.get(
                    f"{UEFA_BASE}/matches",
                    params={"competitionId": comp_id, "fromDate": date_str, "toDate": date_str},
                    headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
                )
                if r.status_code != 200:
                    return None
                data = r.json()
        except Exception as e:
            logger.warning(f"Erro UEFA API: {e}")
            return None

        matches = []
        if isinstance(data, dict):
            if "data" in data:
                matches = data["data"].get("matches", [])
            elif "matches" in data:
                matches = data.get("matches", [])
        elif isinstance(data, list):
            matches = data

        home_n = self._norm(home)
        away_n = self._norm(away)
        for m in matches:
            ht = m.get("homeTeam", {})
            at = m.get("awayTeam", {})
            mh = self._norm(
                ht.get("translations", {}).get("displayOfficialName", {}).get("EN", "")
                or ht.get("internationalName", "")
            )
            ma = self._norm(
                at.get("translations", {}).get("displayOfficialName", {}).get("EN", "")
                or at.get("internationalName", "")
            )
            if not (self._match(home_n, mh) and self._match(away_n, ma)):
                continue
            for ref in m.get("referees", []):
                if ref.get("role") == "REFEREE":
                    name = ref.get("person", {}).get("translations", {}).get("name", {}).get("EN")
                    if name:
                        return name
        return None

    async def _from_fd(self, home, away, league_code, date_str) -> Optional[str]:
        fd_code = FD_LEAGUE_MAP[league_code.upper()]
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                r = await client.get(
                    f"{FD_BASE}/competitions/{fd_code}/matches",
                    params={"dateFrom": date_str, "dateTo": date_str},
                    headers={"X-Auth-Token": self._fd_key},
                )
                if r.status_code != 200:
                    return None
                matches = r.json().get("matches", [])
        except Exception:
            return None

        home_n, away_n = self._norm(home), self._norm(away)
        for m in matches:
            mh = self._norm(m.get("homeTeam", {}).get("name", ""))
            ma = self._norm(m.get("awayTeam", {}).get("name", ""))
            if not (self._match(home_n, mh) and self._match(away_n, ma)):
                continue
            for ref in m.get("referees", []):
                if ref.get("type", "").upper() in ("REFEREE", "MAIN"):
                    return ref.get("name")
        return None

    @staticmethod
    def _norm(name: str) -> str:
        import unicodedata, re
        name = unicodedata.normalize("NFD", name)
        name = "".join(c for c in name if unicodedata.category(c) != "Mn")
        name = name.lower()
        return re.sub(r"\s+", " ", name).strip()

    @staticmethod
    def _match(a: str, b: str) -> bool:
        if a == b or a in b or b in a:
            return True
        wa = {w for w in a.split() if len(w) > 3}
        wb = {w for w in b.split() if len(w) > 3}
        return bool(wa & wb)
