"""
SofascoreClient — fonte exclusiva de xG (Expected Goals).

Responsabilidade única: buscar xg_home e xg_away para um jogo,
correlacionando o fixture do SokkerPRO com o event_id do Sofascore.

⚠️  API não-oficial. O Sofascore não fornece API pública.
    Este cliente usa endpoints internos mapeados pela comunidade.
    Risco: pode parar de funcionar se o Sofascore mudar a estrutura.

Fluxo:
    1. Busca todos os jogos do dia no Sofascore (/scheduled-events/{date})
    2. Para cada fixture do SokkerPRO, calcula score de similaridade
       contra candidatos do Sofascore (nomes dos times + horário)
    3. Se score >= MATCH_THRESHOLD → aceita correlação → busca xG
    4. Injeta xg_home / xg_away no fixture como campos extras
    5. Se correlação falhar → xg_home = xg_away = None (pipeline segue normal)

Isolamento garantido:
    - Não altera odds, médias, prognósticos ou qualquer outro campo do SokkerPRO
    - Falha silenciosa: exception → xG None → pipeline não é afetado
    - Cache por jogo evita chamadas repetidas no mesmo ciclo
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Optional

from loguru import logger

# rapidfuzz é muito mais rápido que fuzzywuzzy e já é dependência comum
try:
    from rapidfuzz import fuzz
    _FUZZ_AVAILABLE = True
except ImportError:
    _FUZZ_AVAILABLE = False
    logger.warning("[Sofascore] rapidfuzz não instalado — correlação desativada. "
                   "Execute: pip install rapidfuzz")

# curl-cffi para contornar Cloudflare (mesma estratégia do SokkerProClient)
try:
    from curl_cffi import requests as _curl_requests
    _CURL_AVAILABLE = True
except ImportError:
    _CURL_AVAILABLE = False

import httpx

BASE_URL   = "https://www.sofascore.com/api/v1"
TIMEOUT    = 12.0

# Score mínimo para aceitar a correlação (0.0 → 1.0)
# 0.75 = bom equilíbrio entre precisão e cobertura
MATCH_THRESHOLD = 0.75

# Aliases manuais para clubes com nomes muito diferentes entre as APIs
# Formato: {"nome_normalizado_sokkerpro": "nome_normalizado_sofascore"}
_ALIASES: dict[str, str] = {
    "psg":               "paris saint-germain",
    "paris sg":          "paris saint-germain",
    "man city":          "manchester city",
    "man utd":           "manchester united",
    "man united":        "manchester united",
    "spurs":             "tottenham",
    "tottenham hotspur": "tottenham",
    "atletico madrid":   "atletico de madrid",
    "atletico":          "atletico de madrid",
    "bayern":            "bayern munich",
    "fc barcelona":      "barcelona",
    "inter":             "inter milan",
    "inter milano":      "inter milan",
    "ac milan":          "milan",
    "rb leipzig":        "rasenballsport leipzig",
    "bayer leverkusen":  "bayer 04 leverkusen",
    "borussia dortmund": "dortmund",
    "bvb":               "dortmund",
    "newcastle utd":     "newcastle",
    "newcastle united":  "newcastle",
    "wolves":            "wolverhampton",
    "nott forest":       "nottingham forest",
    "brighton":          "brighton & hove albion",
    "sheffield utd":     "sheffield united",
    "west ham":          "west ham united",
    "sporting cp":       "sporting",
    "benfica":           "sl benfica",
    "porto":             "fc porto",
    "flamengo":          "clube de regatas flamengo",
    "palmeiras":         "se palmeiras",
    "corinthians":       "sport club corinthians paulista",
    "sao paulo":         "sao paulo fc",
    "atletico mineiro":  "atletico-mg",
    "atletico-go":       "atletico goianiense",
    "botafogo":          "botafogo fr",
}

_CURL_HEADERS = {
    "Accept":          "application/json",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Referer":         "https://www.sofascore.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


# ── Helpers de normalização ────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """
    Normaliza nome de time para comparação:
    - Remove acentos (São Paulo → sao paulo)
    - Lowercase
    - Remove prefixos comuns (FC, SC, CD, CF, AS, FK, SK...)
    - Remove pontuação
    - Aplica aliases manuais
    """
    if not name:
        return ""

    # Remove acentos
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))

    # Lowercase e remove pontuação exceto hífen
    cleaned = re.sub(r"[^\w\s-]", "", ascii_str.lower()).strip()

    # Remove prefixos/sufixos genéricos de clube
    for prefix in ("fc ", "sc ", "cd ", "cf ", "as ", "fk ", "sk ", "bk ",
                    "ac ", "rc ", "rcd ", "ud ", "sd ", "real ", "club ",
                    "sporting ", "atletico "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    # Colapsa espaços múltiplos
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Aplica aliases
    return _ALIASES.get(cleaned, cleaned)


def _time_proximity(dt1: datetime, dt2: datetime) -> float:
    """
    Score de proximidade horária entre dois jogos.
    1.0 = mesmo horário exato
    0.5 = 30 min de diferença
    0.0 = 60+ min de diferença
    """
    diff_minutes = abs((dt1 - dt2).total_seconds()) / 60.0
    if diff_minutes <= 5:
        return 1.0
    if diff_minutes >= 60:
        return 0.0
    return 1.0 - (diff_minutes / 60.0)


def _match_score(
    sokker_home: str, sokker_away: str, sokker_dt: datetime,
    sofa_home: str,   sofa_away: str,   sofa_dt: datetime,
) -> float:
    """
    Score de similaridade combinado entre dois jogos.
    Pesos: 40% nome casa + 40% nome fora + 20% horário.
    """
    if not _FUZZ_AVAILABLE:
        return 0.0

    sh = _normalize(sokker_home)
    sa = _normalize(sokker_away)
    fh = _normalize(sofa_home)
    fa = _normalize(sofa_away)

    # Usa token_sort_ratio para lidar com ordem de palavras diferente
    sim_home = fuzz.token_sort_ratio(sh, fh) / 100.0
    sim_away = fuzz.token_sort_ratio(sa, fa) / 100.0
    sim_time = _time_proximity(sokker_dt, sofa_dt)

    return 0.40 * sim_home + 0.40 * sim_away + 0.20 * sim_time


# ── Cliente principal ──────────────────────────────────────────────────────────

class SofascoreClient:
    """
    Cliente para buscar xG do Sofascore.

    Todos os métodos têm fallback silencioso — nunca levantam exceção
    para o caller. Se algo falhar, retorna None e o pipeline segue sem xG.
    """

    # Circuit breaker compartilhado entre instâncias
    _CB_FAILURES:  int   = 0
    _CB_LAST_FAIL: float = 0.0
    _CB_THRESHOLD: int   = 4
    _CB_COOLDOWN:  int   = 180   # 3 min de cooldown após N falhas

    # Delay mínimo entre requisições (evita rate-limit)
    _MIN_DELAY_SECONDS: float = 1.5

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        # Cache diário: date_str → list[dict]  (jogos do dia)
        self._daily_cache:   dict[str, tuple] = {}
        # Cache de correlação: sokker_fixture_id → sofa_event_id | None
        self._id_cache:      dict[str, Optional[int]] = {}
        # Cache de xG: sofa_event_id → {"sofa_xg_home": float, "sofa_xg_away": float} | None
        self._xg_cache:      dict[int, Optional[dict]] = {}
        self._last_request:  float = 0.0

    # ── Circuit breaker ───────────────────────────────────────────────────────

    @classmethod
    def _cb_open(cls) -> bool:
        if cls._CB_FAILURES < cls._CB_THRESHOLD:
            return False
        elapsed = time.time() - cls._CB_LAST_FAIL
        if elapsed >= cls._CB_COOLDOWN:
            cls._CB_FAILURES = 0
            logger.info("[Sofascore] Circuit breaker resetado")
            return False
        return True

    @classmethod
    def _cb_fail(cls):
        cls._CB_FAILURES += 1
        cls._CB_LAST_FAIL = time.time()
        if cls._CB_FAILURES == cls._CB_THRESHOLD:
            logger.warning(
                f"[Sofascore] ⚡ Circuit breaker ABERTO após {cls._CB_THRESHOLD} falhas. "
                f"Pausando por {cls._CB_COOLDOWN}s."
            )

    @classmethod
    def _cb_ok(cls):
        if cls._CB_FAILURES > 0:
            logger.info("[Sofascore] ✅ Respondeu — circuit breaker fechado.")
        cls._CB_FAILURES = 0

    # ── Transporte ────────────────────────────────────────────────────────────

    async def _get(self, path: str) -> Optional[dict | list]:
        """
        GET com rate-limit, circuit breaker e fallback curl-cffi → httpx.
        """
        if self._cb_open():
            return None

        # Rate-limit: garante delay mínimo entre requisições
        elapsed = time.time() - self._last_request
        if elapsed < self._MIN_DELAY_SECONDS:
            await asyncio.sleep(self._MIN_DELAY_SECONDS - elapsed)

        url = f"{BASE_URL}{path}"
        self._last_request = time.time()

        try:
            if _CURL_AVAILABLE:
                loop = asyncio.get_event_loop()
                def _fetch():
                    r = _curl_requests.get(
                        url,
                        headers=_CURL_HEADERS,
                        impersonate="chrome",
                        timeout=TIMEOUT,
                    )
                    r.raise_for_status()
                    return r.json()
                result = await loop.run_in_executor(None, _fetch)
            else:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=TIMEOUT,
                        headers=_CURL_HEADERS,
                        follow_redirects=True,
                    )
                resp = await self._client.get(url)
                resp.raise_for_status()
                result = resp.json()

            self._cb_ok()
            return result

        except Exception as e:
            logger.warning(f"[Sofascore] GET {path} falhou: {e}")
            self._cb_fail()
            return None

    # ── Busca de jogos do dia ─────────────────────────────────────────────────

    async def _get_daily_events(self, date_str: str) -> list[dict]:
        """
        Retorna todos os jogos de futebol de uma data no Sofascore.
        date_str: "YYYY-MM-DD"
        Cache: 10 min (jogos do dia não mudam muito)
        """
        cached = self._daily_cache.get(date_str)
        if cached:
            data, ts = cached
            if time.time() - ts < 600:  # 10 min
                return data

        payload = await self._get(f"/sport/football/scheduled-events/{date_str}")
        if not payload or not isinstance(payload, dict):
            return []

        events = payload.get("events") or []
        if not isinstance(events, list):
            return []

        # Parseia só o que precisamos para a correlação
        result = []
        for ev in events:
            try:
                event_id   = ev.get("id")
                home_name  = (ev.get("homeTeam") or {}).get("name", "")
                away_name  = (ev.get("awayTeam") or {}).get("name", "")
                start_ts   = ev.get("startTimestamp")
                tournament = (ev.get("tournament") or {}).get("name", "")

                if not event_id or not home_name or not away_name:
                    continue

                start_dt = (
                    datetime.fromtimestamp(start_ts, tz=timezone.utc)
                    if start_ts else None
                )

                result.append({
                    "event_id":   int(event_id),
                    "home_name":  home_name,
                    "away_name":  away_name,
                    "start_dt":   start_dt,
                    "tournament": tournament,
                })
            except Exception:
                continue

        logger.debug(f"[Sofascore] {len(result)} jogos carregados para {date_str}")
        self._daily_cache[date_str] = (result, time.time())
        return result

    # ── Correlação ────────────────────────────────────────────────────────────

    async def _find_event_id(self, fixture: dict) -> Optional[int]:
        """
        Correlaciona um fixture do SokkerPRO com um event_id do Sofascore.
        Retorna event_id se score >= MATCH_THRESHOLD, None caso contrário.
        """
        fixture_id = str(fixture.get("id", ""))

        # Cache de correlação — evita recalcular no mesmo ciclo
        if fixture_id in self._id_cache:
            return self._id_cache[fixture_id]

        # Extrai campos do fixture SokkerPRO
        home_name = (fixture.get("homeTeam") or {}).get("name", "")
        away_name = (fixture.get("awayTeam") or {}).get("name", "")
        utc_date  = fixture.get("utcDate", "")

        if not home_name or not away_name or not utc_date:
            self._id_cache[fixture_id] = None
            return None

        # Parse da data/hora
        try:
            sokker_dt = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
            if sokker_dt.tzinfo is None:
                sokker_dt = sokker_dt.replace(tzinfo=timezone.utc)
        except Exception:
            self._id_cache[fixture_id] = None
            return None

        date_str = sokker_dt.strftime("%Y-%m-%d")

        # Busca candidatos do Sofascore na mesma data
        candidates = await self._get_daily_events(date_str)
        if not candidates:
            self._id_cache[fixture_id] = None
            return None

        # Calcula score para cada candidato e pega o melhor
        best_score    = 0.0
        best_event_id = None

        for cand in candidates:
            if not cand.get("start_dt"):
                continue

            score = _match_score(
                home_name, away_name, sokker_dt,
                cand["home_name"], cand["away_name"], cand["start_dt"],
            )

            if score > best_score:
                best_score    = score
                best_event_id = cand["event_id"]

        if best_score >= MATCH_THRESHOLD:
            logger.debug(
                f"[Sofascore] ✅ Correlação: {home_name} vs {away_name} "
                f"→ event_id={best_event_id} (score={best_score:.2f})"
            )
            self._id_cache[fixture_id] = best_event_id
            return best_event_id
        else:
            logger.debug(
                f"[Sofascore] ❌ Sem correlação: {home_name} vs {away_name} "
                f"(melhor score={best_score:.2f} < {MATCH_THRESHOLD})"
            )
            self._id_cache[fixture_id] = None
            return None

    # ── Busca de xG ───────────────────────────────────────────────────────────

    async def _fetch_xg(self, event_id: int) -> Optional[dict]:
        """
        Busca xG de um event_id Sofascore.
        Retorna {"sofa_xg_home": float, "sofa_xg_away": float} ou None.
        """
        if event_id in self._xg_cache:
            return self._xg_cache[event_id]

        payload = await self._get(f"/event/{event_id}/statistics")
        if not payload or not isinstance(payload, dict):
            self._xg_cache[event_id] = None
            return None

        try:
            # Estrutura: {"statistics": [{"period": "ALL", "groups": [...]}]}
            stats_list = payload.get("statistics") or []
            xg_home = None
            xg_away = None

            for period_block in stats_list:
                # Só queremos o período completo (ALL)
                if period_block.get("period") != "ALL":
                    continue

                for group in (period_block.get("groups") or []):
                    for item in (group.get("statisticsItems") or []):
                        # O campo é "Expected goals" ou "xG"
                        key = (item.get("name") or "").lower()
                        if "expected" in key and "goal" in key or key == "xg":
                            try:
                                xg_home = float(item.get("home", 0) or 0)
                                xg_away = float(item.get("away", 0) or 0)
                            except (TypeError, ValueError):
                                pass

            if xg_home is None and xg_away is None:
                logger.debug(f"[Sofascore] xG não encontrado na resposta para event_id={event_id}")
                self._xg_cache[event_id] = None
                return None

            result = {"sofa_xg_home": xg_home, "sofa_xg_away": xg_away}
            logger.debug(f"[Sofascore] xG event_id={event_id}: home={xg_home} away={xg_away}")
            self._xg_cache[event_id] = result
            return result

        except Exception as e:
            logger.warning(f"[Sofascore] Erro ao parsear xG event_id={event_id}: {e}")
            self._xg_cache[event_id] = None
            return None

    # ── Interface pública ─────────────────────────────────────────────────────

    async def get_xg_for_fixture(self, fixture: dict) -> Optional[dict]:
        """
        Ponto de entrada principal.

        Recebe um fixture do SokkerPRO e retorna:
            {"sofa_xg_home": float, "sofa_xg_away": float}
        ou None se a correlação falhar ou xG não estiver disponível.

        Nunca levanta exceção — falha silenciosa garante que o pipeline
        do SokkerPRO não seja afetado.
        """
        if not _FUZZ_AVAILABLE:
            return None

        try:
            event_id = await self._find_event_id(fixture)
            if event_id is None:
                return None

            return await self._fetch_xg(event_id)

        except Exception as e:
            logger.warning(f"[Sofascore] get_xg_for_fixture falhou silenciosamente: {e}")
            return None

    async def enrich_fixture_with_xg(self, fixture: dict) -> dict:
        """
        Injeta sofa_xg_home e sofa_xg_away no fixture como campos extras.
        Retorna o fixture original se correlação falhar (sem modificação).

        Uso no orchestrator:
            fixture = await sofascore.enrich_fixture_with_xg(fixture)
            sofa_xg_home = fixture.get("sofa_xg_home")   # None se não disponível
        """
        xg = await self.get_xg_for_fixture(fixture)
        if xg is None:
            return fixture

        return {
            **fixture,
            "sofa_xg_home": xg["sofa_xg_home"],
            "sofa_xg_away": xg["sofa_xg_away"],
        }

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
