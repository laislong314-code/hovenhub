"""
SokkerProClient — fonte única de dados, substitui ESPN + Sofascore.

Endpoints:
  GET /livescores
      → jogos ao vivo agora (112+ jogos) com stats completas ao vivo,
        odds Bet365/1xBet embutidas, DAPM, prognosticos, medias históricas

  GET /home/fixtures/{date}/{timezone}/mini
      → todos os jogos de uma data (687+ jogos, 270+ ligas) com
        odds pré-jogo, medias históricas da temporada

Campos notáveis vs ESPN:
  - odds Bet365 + 1xBet reais (ESPN só tinha DraftKings US, inutilizável no BR)
  - medias_home/away_* — médias da temporada já computadas (sem precisar de ESPN team-form)
  - prognosticos — probabilidades 1X2, Over/Under, BTTS, 1T já calculadas
  - DAPM (dangerous attacks per minute) — métrica exclusiva, não existe na ESPN
  - 65 ligas ao vivo / 270 ligas na agenda diária vs ~12 da ESPN

Formato de odds: "1.87#0"  →  odd=1.87, movement=0 (0=estável, +/-=subiu/caiu)
"""

import asyncio
import json
import time
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
import httpx
from loguru import logger

from app.config import get_settings

settings = get_settings()

BASE_URL = getattr(settings, "sokkerpro_base_url", "https://m2.sokkerpro.com")

# curl-cffi — emula fingerprint TLS do Chrome, contorna Cloudflare sem browser
try:
    from curl_cffi import requests as _curl_requests
    _CURL_AVAILABLE = True
except ImportError:
    _CURL_AVAILABLE = False

_CURL_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Referer": "https://m2.sokkerpro.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}
# Mapeamento de status SokkerPRO → status interno (mesmo padrão do projeto)
STATUS_MAP = {
    "NS":   "SCHEDULED",
    "1st":  "IN_PLAY",
    "HT":   "PAUSED",
    "2nd":  "IN_PLAY",
    "FT":   "FINISHED",
    "FTP":  "FINISHED",   # full time + penalties
    "AET":  "FINISHED",   # after extra time
    "AU":   "FINISHED",   # awarded
    "AWAR": "FINISHED",
    "CANC": "CANCELLED",
    "POST": "POSTPONED",
    "SUSP": "SUSPENDED",
    "ABAN": "ABANDONED",
}


def _parse_odd(raw: str) -> tuple[float, int]:
    """
    Converte "1.87#0" → (1.87, 0)
    Converte "1.87#1" → (1.87, +1)  subiu
    Converte "1.87#-1" → (1.87, -1) caiu
    Retorna (0.0, 0) se inválido.
    """
    if not raw or raw.strip() == "":
        return 0.0, 0
    try:
        parts = str(raw).split("#")
        odd = float(parts[0])
        movement = int(parts[1]) if len(parts) > 1 else 0
        return odd, movement
    except Exception:
        return 0.0, 0


def _safe_float(val, default=None) -> Optional[float]:
    if val in (None, "", "0", 0):
        return default
    try:
        return float(val)
    except Exception:
        return default


def _safe_int(val, default=None) -> Optional[int]:
    if val in (None, ""):
        return default
    try:
        return int(float(str(val)))
    except Exception:
        return default


class SokkerProClient:
    """
    Cliente assíncrono para a API SokkerPRO.
    Cache simples em memória por TTL para reduzir chamadas.

    Transporte: curl-cffi (emula fingerprint TLS do Chrome, contorna Cloudflare).
    Fallback: httpx padrão se curl-cffi não estiver disponível.
    """

    TIMEOUT = 15.0
    _LIVESCORES_TTL = 45   # segundos — ao vivo muda rápido
    _FIXTURES_TTL   = 60   # segundos — 60s para pegar status FINISHED rápido

    # Circuit breaker — se a API falhar N vezes seguidas, pausa por _CB_COOLDOWN segundos
    _CB_FAILURES    = 0
    _CB_LAST_FAIL   = 0.0
    _CB_THRESHOLD   = 5     # falhas consecutivas para abrir o circuito
    _CB_COOLDOWN    = 120   # segundos em que o circuito fica aberto (sem bater na API)

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: dict[str, tuple] = {}  # key → (data, timestamp)
        if _CURL_AVAILABLE:
            logger.info("[SokkerPRO] fetch_mode=curl-cffi (Chrome TLS fingerprint) ✅")
        else:
            logger.warning("[SokkerPRO] curl-cffi não instalado, usando httpx (pode dar 403)")

    # ─── Circuit breaker ──────────────────────────────────────────────────────

    @classmethod
    def _cb_open(cls) -> bool:
        """Retorna True se o circuito está aberto (API em cooldown)."""
        if cls._CB_FAILURES < cls._CB_THRESHOLD:
            return False
        elapsed = time.time() - cls._CB_LAST_FAIL
        if elapsed >= cls._CB_COOLDOWN:
            cls._CB_FAILURES = 0   # reset após cooldown
            logger.info("[SokkerPRO] Circuit breaker resetado — tentando API novamente")
            return False
        return True

    @classmethod
    def _cb_record_failure(cls):
        cls._CB_FAILURES += 1
        cls._CB_LAST_FAIL = time.time()
        if cls._CB_FAILURES == cls._CB_THRESHOLD:
            logger.warning(
                f"[SokkerPRO] ⚡ Circuit breaker ABERTO após {cls._CB_THRESHOLD} falhas. "
                f"Pausando requisições por {cls._CB_COOLDOWN}s."
            )

    @classmethod
    def _cb_record_success(cls):
        if cls._CB_FAILURES > 0:
            logger.info(f"[SokkerPRO] ✅ API respondeu — circuit breaker fechado.")
        cls._CB_FAILURES = 0

    # ─── Camada de transporte centralizada ───────────────────────────────────

    async def _get_json(self, path: str, retries: int = 2) -> Optional[dict | list]:
        """
        Busca `path` no BASE_URL e retorna JSON.
        - Circuit breaker: para de bater na API se estiver fora do ar
        - Retry com backoff exponencial: tenta até `retries` vezes antes de desistir
        """
        if self._cb_open():
            remaining = int(self._CB_COOLDOWN - (time.time() - self._CB_LAST_FAIL))
            logger.debug(f"[SokkerPRO] Circuit breaker aberto — ignorando {path} ({remaining}s restantes)")
            return None

        url = f"{BASE_URL}{path}"

        for attempt in range(retries + 1):
            if attempt > 0:
                wait = 2 ** attempt   # 2s, 4s…
                logger.debug(f"[SokkerPRO] Retry {attempt}/{retries} em {wait}s para {path}")
                await asyncio.sleep(wait)

            try:
                if _CURL_AVAILABLE:
                    logger.debug(f"[SokkerPRO] curl-cffi → {url}")
                    loop = asyncio.get_event_loop()
                    def _fetch():
                        r = _curl_requests.get(
                            url,
                            headers=_CURL_HEADERS,
                            impersonate="chrome",
                            timeout=self.TIMEOUT,
                        )
                        r.raise_for_status()
                        return r.json()
                    result = await loop.run_in_executor(None, _fetch)
                else:
                    logger.debug(f"[SokkerPRO] httpx → {url}")
                    client = await self._get_client()
                    resp = await client.get(path)
                    resp.raise_for_status()
                    result = resp.json()

                self._cb_record_success()
                return result

            except Exception as e:
                is_last = attempt == retries
                if is_last:
                    logger.error(f"[SokkerPRO] curl-cffi falhou para {url}: {e}")
                    self._cb_record_failure()
                    return None
                else:
                    logger.debug(f"[SokkerPRO] Falha tentativa {attempt+1}/{retries+1}: {e}")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=self.TIMEOUT,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
                follow_redirects=True,
            )
        return self._client


    async def get_referee_season(self, referee_id: int, season_id: int):
        """Retorna dados do árbitro na temporada.
        Endpoint: /referee/{referee_id}/season/{season_id}
        A resposta costuma vir como {"success": true, "data": "<json-string>"}.
        """
        key = f"ref_{referee_id}_{season_id}"
        cached = self._cache_get(key, ttl=60*60)  # 1h
        if cached is not None:
            return cached

        try:
            payload = await self._get_json(f"/referee/{referee_id}/season/{season_id}")
            if payload is None:
                self._cache_set(key, None)
                return None
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    pass
            self._cache_set(key, data)
            return data
        except Exception as e:
            logger.warning(f"[SokkerPRO] referee/season falhou: {e}")
            self._cache_set(key, None)
            return None

    async def get_fixture(self, fixture_id: str) -> Optional[dict]:
        """Retorna fixture completo (inclui referees e seasonId).
        Endpoint: /fixture/{fixture_id}

        O payload da API pode ter várias estruturas:
          - {"success": true, "data": { ...fixture... }}
          - {"success": true, "data": "{ ...json-string... }"}
          - O fixture bruto diretamente na raiz

        O campo "referees" vem como string JSON: "[{\"referee_id\": 817491, \"type_id\": 6}]"
        O campo "seasonId" (ou "season_id") contém o ID da temporada para o endpoint
        /referee/{referee_id}/season/{season_id}

        Cache de 1h — dados pré-jogo não mudam.
        """
        key = f"fixture_{fixture_id}"
        cached = self._cache_get(key, ttl=3600)
        if cached is not None:
            return cached

        try:
            payload = await self._get_json(f"/fixture/{fixture_id}")
            if payload is None:
                logger.debug(f"[SokkerPRO] get_fixture {fixture_id}: resposta vazia")
                self._cache_set(key, None)
                return None

            # Log diagnóstico do payload bruto — remove após confirmar funcionamento
            if isinstance(payload, dict):
                data_val = payload.get("data")
                logger.debug(
                    f"[SokkerPRO] get_fixture {fixture_id} RAW: "
                    f"top_keys={list(payload.keys())[:15]} | "
                    f"data_type={type(data_val).__name__} | "
                    f"data_preview={str(data_val)[:200] if not isinstance(data_val, dict) else 'dict/' + str(list(data_val.keys())[:10])}"
                )


            # O /fixture/{id} às vezes retorna campos de metadados (referees, seasonId)
            # no nível raiz do payload E dados de stats em "data".
            # Mesclamos os dois para garantir que nada se perca.
            def _try_parse_json(obj):
                if isinstance(obj, str):
                    try:
                        return json.loads(obj)
                    except Exception:
                        return None
                return obj

            raw_data = _try_parse_json(payload.get("data") if isinstance(payload, dict) else None)

            # Monta o resultado mesclando payload raiz + data
            result: dict = {}
            if isinstance(payload, dict):
                result.update(payload)  # raiz pode ter referees, seasonId, fixtureId
            if isinstance(raw_data, dict):
                result.update(raw_data)  # stats detalhadas ficam em data

            # Remove chaves de envelope
            for meta_key in ("success", "message", "errors"):
                result.pop(meta_key, None)

            if not result:
                logger.debug(f"[SokkerPRO] get_fixture {fixture_id}: payload vazio após parse")
                self._cache_set(key, None)
                return None

            # ── Normaliza season_id ─────────────────────────────────────────
            season_id = (
                result.get("season_id")
                or result.get("seasonId")
                or result.get("seasonID")
            )
            if not season_id and isinstance(result.get("season"), dict):
                season_id = result["season"].get("id")
            if season_id:
                result["season_id"] = str(season_id)

            # ── Normaliza referees ──────────────────────────────────────────
            # Pode vir como string JSON "[{\"referee_id\": 817491, \"type_id\": 6}]"
            referees_raw = (
                result.get("referees")
                or result.get("officials")
                or result.get("Referees")
            )
            if isinstance(referees_raw, str) and referees_raw.strip():
                try:
                    referees_raw = json.loads(referees_raw)
                except Exception:
                    referees_raw = None
            result["referees"] = referees_raw if isinstance(referees_raw, list) and referees_raw else None

            refs = result.get("referees")
            sid  = result.get("season_id")
            logger.debug(
                f"[SokkerPRO] get_fixture {fixture_id}: "
                f"season_id={sid} | "
                f"referees={len(refs) if isinstance(refs, list) else 'ausente'} | "
                f"todas_chaves={list(result.keys())[:20]}"
            )

            self._cache_set(key, result)
            return result
        except Exception as e:
            logger.warning(f"[SokkerPRO] get_fixture {fixture_id} falhou: {e}")
            self._cache_set(key, None)
            return None

    async def get_fixture_score(self, fixture_id: str) -> Optional[dict]:
        """
        Busca placar atual de um fixture específico — sem cache longo.
        Usado pelo settlement para verificar resultado de jogos concluídos.
        Retorna dict com keys: status, home_goals, away_goals ou None.
        """
        key = f"score_{fixture_id}"
        # Cache curto: 90s (suficiente para evitar flood mas pega FINISHED rápido)
        cached = self._cache_get(key, ttl=90)
        if cached is not None:
            return cached

        try:
            payload = await self._get_json(f"/fixture/{fixture_id}")
            if payload is None:
                return None

            # O /fixture retorna dados em vários formatos — extrai o que precisamos
            def _get(d, *keys):
                for k in keys:
                    if isinstance(d, dict) and k in d:
                        return d[k]
                return None

            data = payload
            if isinstance(payload, dict):
                inner = payload.get("data")
                if isinstance(inner, str):
                    try:
                        import json as _json
                        inner = _json.loads(inner)
                    except Exception:
                        inner = None
                if isinstance(inner, dict):
                    # mescla
                    merged = dict(payload)
                    merged.update(inner)
                    data = merged

            if not isinstance(data, dict):
                return None

            status_raw = data.get("status", "NS")
            status = STATUS_MAP.get(status_raw, "SCHEDULED")

            # Placar — pode estar em vários campos
            home_goals = _safe_int(data.get("scoresLocalTeam") or data.get("localteam_score") or data.get("home_score"))
            away_goals = _safe_int(data.get("scoresVisitorTeam") or data.get("visitorteam_score") or data.get("away_score"))

            result = {
                "status":     status,
                "status_raw": status_raw,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
            self._cache_set(key, result)
            return result

        except Exception as e:
            logger.warning(f"[SokkerPRO] get_fixture_score {fixture_id} falhou: {e}")
            return None

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _cache_get(self, key: str, ttl: int):
        entry = self._cache.get(key)
        if entry:
            data, ts = entry
            if (datetime.now(timezone.utc) - ts).total_seconds() < ttl:
                return data
        return None

    def _cache_set(self, key: str, data):
        self._cache[key] = (data, datetime.now(timezone.utc))

    # ─── Endpoints principais ─────────────────────────────────────────────────

    async def get_livescores(self) -> list[dict]:
        """
        Retorna todos os jogos ao vivo agora, parseados.
        Cache de 45s para não sobrecarregar.
        """
        cached = self._cache_get("livescores", self._LIVESCORES_TTL)
        if cached is not None:
            return cached

        try:
            raw = await self._get_json("/livescores")
            if raw is None:
                raise ValueError("resposta vazia")
        except Exception as e:
            logger.warning(f"[SokkerPRO] livescores falhou: {e}")
            return []

        data = (raw.get("data") or {})
        fixtures_raw = []
        for cat in (data.get("sortedCategorizedFixtures") or []):
            if not cat:
                continue
            for f in (cat.get("fixtures") or []):
                if not f:
                    continue
                parsed = self._parse_fixture(f, cat)
                if parsed:
                    fixtures_raw.append(parsed)

        logger.info(f"[SokkerPRO] {len(fixtures_raw)} jogos ao vivo | total={data.get('fixtures_total',0)} live={data.get('fixtures_live',0)}")
        self._cache_set("livescores", fixtures_raw)
        return fixtures_raw

    async def get_fixtures_for_date(
        self,
        date: str,              # "2026-02-27"
        timezone_offset: str = "utc-4",
    ) -> list[dict]:
        """
        Retorna todos os jogos de uma data (agenda diária).
        date: "YYYY-MM-DD"
        timezone_offset: ex "utc-3", "utc0", "utc+1"
        Cache de 5 min.
        """
        cache_key = f"fixtures_{date}_{timezone_offset}"
        cached = self._cache_get(cache_key, self._FIXTURES_TTL)
        if cached is not None:
            return cached

        try:
            raw = await self._get_json(f"/home/fixtures/{date}/{timezone_offset}/mini")
            if raw is None:
                raise ValueError("resposta vazia")
        except Exception as e:
            logger.warning(f"[SokkerPRO] fixtures {date} falhou: {e}")
            return []

        data = (raw.get("data") or {})
        fixtures = []
        for cat in (data.get("sortedCategorizedFixtures") or []):
            if not cat:
                continue
            for f in (cat.get("fixtures") or []):
                if not f:
                    continue
                parsed = self._parse_fixture(f, cat)
                if parsed:
                    fixtures.append(parsed)

        logger.info(f"[SokkerPRO] {len(fixtures)} jogos em {date} | ligas={len(data.get('sortedCategorizedFixtures',[]))}")
        self._cache_set(cache_key, fixtures)
        return fixtures

    async def get_todays_scheduled(self, timezone_offset: str = "utc-4") -> list[dict]:
        """Jogos de hoje (agenda), status NS (não iniciados)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        all_fixtures = await self.get_fixtures_for_date(today, timezone_offset)
        return [f for f in all_fixtures if f["status"] == "SCHEDULED"]

    async def get_live_matches(self) -> list[dict]:
        """Jogos em andamento agora."""
        all_live = await self.get_livescores()
        return [f for f in all_live if f["status"] in ("IN_PLAY", "PAUSED")]

    async def get_finished_today(self) -> list[dict]:
        """Jogos terminados hoje."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        all_fixtures = await self.get_fixtures_for_date(today)
        return [f for f in all_fixtures if f["status"] == "FINISHED"]

    # ─── Parsing ──────────────────────────────────────────────────────────────

    def _parse_fixture(self, f: dict, cat: dict) -> Optional[dict]:
        """
        Converte um fixture bruto do SokkerPRO para o schema interno do projeto.
        Mantém compatibilidade máxima com o schema ESPN existente.
        """
        try:
            fixture_id  = str(f.get("fixtureId", ""))
            status_raw  = f.get("status", "NS")
            status      = STATUS_MAP.get(status_raw, "SCHEDULED")
            minute      = _safe_int(f.get("minuteSegundoTempo") or f.get("minute"))
            utc_date    = f.get("startingAtDateTime", "").strip('"')

            home_id   = _safe_int(f.get("localTeamId"), 0)
            away_id   = _safe_int(f.get("visitorTeamId"), 0)
            home_name = f.get("localTeamName", "")
            away_name = f.get("visitorTeamName", "")
            home_flag = f.get("localTeamFlag", "")
            away_flag = f.get("visitorTeamFlag", "")

            # Placar
            home_score = _safe_int(f.get("scoresLocalTeam"))   if status != "SCHEDULED" else None
            away_score = _safe_int(f.get("scoresVisitorTeam")) if status != "SCHEDULED" else None
            home_ht    = _safe_int(f.get("scoresHT"))          if status != "SCHEDULED" else None

            # Odds — formato "odd#movement"
            odds = self._parse_odds(f)

            # Estatísticas ao vivo
            stats = self._parse_statistics(f)

            # Médias históricas da temporada (pre-computadas pelo SokkerPRO)
            medias = self._parse_medias(f)

            # Prognósticos pré-computados (probabilidades + odds calculadas)
            prognosticos = self._parse_prognosticos(f.get("prognosticos", ""))

            # DAPM — dangerous attacks per minute (exclusivo SokkerPRO)
            dapm = {
                "home_dapm_1":  _safe_float(f.get("localDapm1")),
                "home_dapm_3":  _safe_float(f.get("localDapm3")),
                "home_dapm_5":  _safe_float(f.get("localDapm5")),
                "home_dapm_10": _safe_float(f.get("localDapm10")),
                "home_dapm_total": _safe_float(f.get("localDapmTotal")),
                "away_dapm_1":  _safe_float(f.get("visitorDapm1")),
                "away_dapm_3":  _safe_float(f.get("visitorDapm3")),
                "away_dapm_5":  _safe_float(f.get("visitorDapm5")),
                "away_dapm_10": _safe_float(f.get("visitorDapm10")),
                "away_dapm_total": _safe_float(f.get("visitorDapmTotal")),
            }

            league_id   = str(cat.get("leagueId") or f.get("leagueId", ""))
            league_name = cat.get("leagueName") or f.get("leagueName", "")
            country     = cat.get("countryName") or f.get("countryName", "")

            return {
                # Identificação — compatível com schema ESPN
                "id":      fixture_id,
                "utcDate": utc_date,
                "status":  status,
                "minute":  minute,
                "period":  2 if status_raw == "2nd" else (1 if status_raw == "1st" else None),

                "homeTeam": {
                    "id":        home_id,
                    "name":      home_name,
                    "shortName": home_name,
                    "form":      "",         # SokkerPRO não fornece string de forma W/D/L
                    "logo":      home_flag,
                },
                "awayTeam": {
                    "id":        away_id,
                    "name":      away_name,
                    "shortName": away_name,
                    "form":      "",
                    "logo":      away_flag,
                },

                "score": {
                    "fullTime": {"home": home_score, "away": away_score},
                    "halfTime": {"home": home_ht,    "away": None},
                },

                # Liga
                "league_id":   league_id,
                "league_name": league_name,
                "country":     country,
                "league_code": league_id,    # usado como chave interna

                # Odds (Bet365 + 1xBet, pre-jogo e ao vivo)
                "sokker_odds": odds,

                # Estatísticas ao vivo
                "statistics": stats,

                # Médias da temporada (equivalente ao form_service ESPN, pré-computado)
                "medias": medias,

                # Prognósticos pré-calculados (1X2, Over/Under, BTTS, 1T)
                "prognosticos": prognosticos,

                # DAPM (exclusive)
                "dapm": dapm,

                # Extras
                "round":     f.get("roundName", ""),
                "season_id": str(f.get("seasonId", "")),

                # Árbitros — presente no /fixture/{id}, ausente no /mini
                # Pode ser list[dict] ou str JSON; RefereeService trata os dois casos
                "referees": f.get("referees"),
            }

        except Exception as e:
            logger.debug(f"[SokkerPRO] Erro parse fixture {f.get('fixtureId')}: {e}")
            return None

    def _parse_odds(self, f: dict) -> dict:
        """
        Extrai e normaliza todas as odds do fixture para o schema interno.
        Prioridade: Bet365 pré-jogo → Bet365 ao vivo → 1xBet pré-jogo.
        """
        odds = {}

        # ── 1X2 ─────────────────────────────────────────────────────────────
        # Bet365 pré-jogo (campos sem _LIVE)
        b365_home_pre,  _ = _parse_odd(f.get("BET365_VENCEDOR_HOME", ""))
        b365_draw_pre,  _ = _parse_odd(f.get("BET365_VENCEDOR_DRAW", ""))
        b365_away_pre,  _ = _parse_odd(f.get("BET365_VENCEDOR_AWAY", ""))
        # Bet365 ao vivo (fallback quando pré-jogo zerado)
        b365_home_live, _ = _parse_odd(f.get("BET365_VENCEDOR_1_LIVE", ""))
        b365_draw_live, _ = _parse_odd(f.get("BET365_VENCEDOR_X_LIVE", ""))
        b365_away_live, _ = _parse_odd(f.get("BET365_VENCEDOR_2_LIVE", ""))
        # 1xBet pré-jogo (fallback final)
        xbet_home, _ = _parse_odd(f.get("XBET_VENCEDOR_HOME", ""))
        xbet_draw, _ = _parse_odd(f.get("XBET_VENCEDOR_DRAW", ""))
        xbet_away, _ = _parse_odd(f.get("XBET_VENCEDOR_AWAY", ""))

        def best(pre, live, xbet):
            if pre  > 1.0: return pre,  "Bet365"
            if live > 1.0: return live, "Bet365"
            if xbet > 1.0: return xbet, "1xBet"
            return None, None

        h, hb = best(b365_home_pre, b365_home_live, xbet_home)
        d, db = best(b365_draw_pre, b365_draw_live, xbet_draw)
        a, ab = best(b365_away_pre, b365_away_live, xbet_away)
        if h: odds["Home"] = {"odd": h, "bookmaker": hb}
        if d: odds["Draw"] = {"odd": d, "bookmaker": db}
        if a: odds["Away"] = {"odd": a, "bookmaker": ab}

        # ── Over/Under gols (Bet365 pré-jogo) ───────────────────────────────
        for line, key_over, key_under in [
            ("0.5",  "BET365_GOLS_OVER_0_5",  "BET365_GOLS_UNDER_0_5"),
            ("1.5",  "BET365_GOLS_OVER_1_5",  "BET365_GOLS_UNDER_1_5"),
            ("2.5",  "BET365_GOLS_OVER_2_5",  "BET365_GOLS_UNDER_2_5"),
            ("3.5",  "BET365_GOLS_OVER_3_5",  "BET365_GOLS_UNDER_3_5"),
            ("4.5",  "BET365_GOLS_OVER_4_5",  "BET365_GOLS_UNDER_4_5"),
        ]:
            ov, _ = _parse_odd(f.get(key_over,  ""))
            un, _ = _parse_odd(f.get(key_under, ""))
            if ov > 1.0: odds[f"Over_{line}"]  = {"odd": ov, "bookmaker": "Bet365"}
            if un > 1.0: odds[f"Under_{line}"] = {"odd": un, "bookmaker": "Bet365"}

        # Over/Under gols 1º tempo
        for line, key_over, key_under in [
            ("0.5", "BET365_GOLS1T_OVER_0_5", "BET365_GOLS1T_UNDER_0_5"),
            ("1.5", "BET365_GOLS1T_OVER_1_5", "BET365_GOLS1T_UNDER_1_5"),
        ]:
            ov, _ = _parse_odd(f.get(key_over,  ""))
            un, _ = _parse_odd(f.get(key_under, ""))
            if ov > 1.0: odds[f"Over_1T_{line}"]  = {"odd": ov, "bookmaker": "Bet365"}
            if un > 1.0: odds[f"Under_1T_{line}"] = {"odd": un, "bookmaker": "Bet365"}

        # ── BTTS (Ambos Marcam) ──────────────────────────────────────────────
        btts_yes, _ = _parse_odd(f.get("BET365_AMBAS_YES", ""))
        btts_no,  _ = _parse_odd(f.get("BET365_AMBAS_NO",  ""))
        if not btts_yes > 1.0:
            btts_yes, _ = _parse_odd(f.get("XBET_AMBAS_YES", ""))
            btts_no,  _ = _parse_odd(f.get("XBET_AMBAS_NO",  ""))
            bm_btts = "1xBet"
        else:
            bm_btts = "Bet365"
        if btts_yes > 1.0: odds["BTTS"]    = {"odd": btts_yes, "bookmaker": bm_btts}
        if btts_no  > 1.0: odds["No_BTTS"] = {"odd": btts_no,  "bookmaker": bm_btts}

        # ── Dupla Chance ─────────────────────────────────────────────────────
        # Pré-jogo: 1xBet usa chaves fixas — preferência por ter mais cobertura
        dc_hd_xbet, _ = _parse_odd(f.get("XBET_DUPLA_CHANCE_HOME_DRAW", ""))
        dc_da_xbet, _ = _parse_odd(f.get("XBET_DUPLA_CHANCE_DRAW_AWAY", ""))
        dc_ha_xbet, _ = _parse_odd(f.get("XBET_DUPLA_CHANCE_HOME_AWAY", ""))

        # Ao vivo: Bet365 usa 1X / X2 / 12
        dc_hd_live, _ = _parse_odd(f.get("BET365_DUPLA_CHANCE_1X_LIVE", ""))
        dc_da_live, _ = _parse_odd(f.get("BET365_DUPLA_CHANCE_X2_LIVE", ""))
        dc_ha_live, _ = _parse_odd(f.get("BET365_DUPLA_CHANCE_12_LIVE", ""))

        # Monta com prioridade: ao vivo > 1xBet pré-jogo
        dc_hd = dc_hd_live or dc_hd_xbet
        dc_da = dc_da_live or dc_da_xbet
        dc_ha = dc_ha_live or dc_ha_xbet

        bm_dc = "Bet365" if (dc_hd_live or dc_da_live or dc_ha_live) else "1xBet"
        if dc_hd > 1.0: odds["DC_Home_Draw"] = {"odd": dc_hd, "bookmaker": bm_dc}
        if dc_da > 1.0: odds["DC_Draw_Away"] = {"odd": dc_da, "bookmaker": bm_dc}
        if dc_ha > 1.0: odds["DC_Home_Away"] = {"odd": dc_ha, "bookmaker": bm_dc}

        # ── Resultado 1º Tempo ───────────────────────────────────────────────
        ht_home, _ = _parse_odd(f.get("BET365_VENCEDOR1T_HOME", ""))
        ht_draw, _ = _parse_odd(f.get("BET365_VENCEDOR1T_DRAW", ""))
        ht_away, _ = _parse_odd(f.get("BET365_VENCEDOR1T_AWAY", ""))
        if ht_home > 1.0: odds["HT_Home"] = {"odd": ht_home, "bookmaker": "Bet365"}
        if ht_draw > 1.0: odds["HT_Draw"] = {"odd": ht_draw, "bookmaker": "Bet365"}
        if ht_away > 1.0: odds["HT_Away"] = {"odd": ht_away, "bookmaker": "Bet365"}

        return odds

    def _parse_statistics(self, f: dict) -> dict:
        """
        Extrai estatísticas ao vivo — mesmo schema do método ESPN _parse_statistics.
        """
        return {
            "home_possession":      _safe_float(f.get("localBallPossession")),
            "away_possession":      _safe_float(f.get("visitorBallPossession")),
            "home_shots":           _safe_float(f.get("localShotsTotal")),
            "away_shots":           _safe_float(f.get("visitorShotsTotal")),
            "home_shots_on_target": _safe_float(f.get("localShotsOnGoal")),
            "away_shots_on_target": _safe_float(f.get("visitorShotsOnGoal")),
            "home_shots_off_target":_safe_float(f.get("localShotsOffGoal")),
            "away_shots_off_target":_safe_float(f.get("visitorShotsOffGoal")),
            "home_shots_inside_box":_safe_float(f.get("localShotsInsideBox")),
            "away_shots_inside_box":_safe_float(f.get("visitorShotsInsideBox")),
            "home_corners":         _safe_float(f.get("localCorners")),
            "away_corners":         _safe_float(f.get("visitorCorners")),
            "home_fouls":           _safe_float(f.get("localFouls")),
            "away_fouls":           _safe_float(f.get("visitorFouls")),
            "home_yellow_cards":    _safe_float(f.get("localYellowCards")),
            "away_yellow_cards":    _safe_float(f.get("visitorYellowCards")),
            "home_red_cards":       _safe_float(f.get("localRedCards")),
            "away_red_cards":       _safe_float(f.get("visitorRedCards")),
            "home_saves":           _safe_float(f.get("localSaves")),
            "away_saves":           _safe_float(f.get("visitorSaves")),
            "home_attacks":         _safe_float(f.get("localAttacksAttacks")),
            "away_attacks":         _safe_float(f.get("visitorAttacksAttacks")),
            "home_dangerous_attacks":_safe_float(f.get("localAttacksDangerousAttacks")),
            "away_dangerous_attacks":_safe_float(f.get("visitorAttacksDangerousAttacks")),
            "home_passes":          _safe_float(f.get("localPassesTotal")),
            "away_passes":          _safe_float(f.get("visitorPassesTotal")),
            "home_passes_acc":      _safe_float(f.get("localPassesAccurate")),
            "away_passes_acc":      _safe_float(f.get("visitorPassesAccurate")),
            "home_passes_pct":      _safe_float(f.get("localPassesPercentage")),
            "away_passes_pct":      _safe_float(f.get("visitorPassesPercentage")),
            "home_tackles":         _safe_float(f.get("localTackles")),
            "away_tackles":         _safe_float(f.get("visitorTackles")),
            "home_offsides":        _safe_float(f.get("localOffsides")),
            "away_offsides":        _safe_float(f.get("visitorOffsides")),
            "home_pressure_bar":    _safe_float(f.get("localPressureBar")),
            "away_pressure_bar":    _safe_float(f.get("visitorPressureBar")),
        }

    def _parse_medias(self, f: dict) -> dict:
        """
        Extrai médias históricas da temporada — substitui o ESPN FormService.
        Esses valores são equivalentes ao avg_scored/avg_conceded calculados localmente.
        """
        return {
            "home_avg_goal":        _safe_float(f.get("medias_home_goal")),
            "away_avg_goal":        _safe_float(f.get("medias_away_goal")),
            "home_avg_corners":     _safe_float(f.get("medias_home_corners")),
            "away_avg_corners":     _safe_float(f.get("medias_away_corners")),
            "home_avg_shots":       _safe_float(f.get("medias_home_shots_total")),
            "away_avg_shots":       _safe_float(f.get("medias_away_shots_total")),
            "home_avg_shots_on":    _safe_float(f.get("medias_home_shots_on_target")),
            "away_avg_shots_on":    _safe_float(f.get("medias_away_shots_on_target")),
            "home_avg_shots_off":   _safe_float(f.get("medias_home_shots_off_target")),
            "away_avg_shots_off":   _safe_float(f.get("medias_away_shots_off_target")),
            "home_avg_shots_in":    _safe_float(f.get("medias_home_shots_insidebox")),
            "away_avg_shots_in":    _safe_float(f.get("medias_away_shots_insidebox")),
            "home_avg_attacks":     _safe_float(f.get("medias_home_attacks")),
            "away_avg_attacks":     _safe_float(f.get("medias_away_attacks")),
            "home_avg_dangerous":   _safe_float(f.get("medias_home_dangerous_attacks")),
            "away_avg_dangerous":   _safe_float(f.get("medias_away_dangerous_attacks")),
            "home_avg_possession":  _safe_float(f.get("medias_home_possession")),
            "away_avg_possession":  _safe_float(f.get("medias_away_possession")),
            "home_avg_fouls":       _safe_float(f.get("medias_home_fouls")),
            "away_avg_fouls":       _safe_float(f.get("medias_away_fouls")),
            "home_avg_yellow":      _safe_float(f.get("medias_home_yellow_cards")),
            "away_avg_yellow":      _safe_float(f.get("medias_away_yellow_cards")),
            "home_avg_passes_pct":  _safe_float(f.get("medias_home_successful_passes_percentage")),
            "away_avg_passes_pct":  _safe_float(f.get("medias_away_successful_passes_percentage")),
        }

    def _parse_prognosticos(self, raw: str) -> dict:
        """
        Desserializa o campo prognosticos (JSON dentro de JSON).
        Retorna dict com probabilidades e odds pré-calculadas.
        """
        if not raw:
            return {}
        try:
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, dict):
                return raw
        except Exception:
            pass
        return {}

    # ─── Helpers para o orchestrator ─────────────────────────────────────────

    def extract_team_form_from_medias(
        self, fixture: dict, side: str  # "home" or "away"
    ) -> tuple[float, float]:
        """
        Extrai avg_scored e avg_conceded das médias SokkerPRO.
        Substitui o ESPN FormService.extract_form_from_matches().
        
        Para home: scored = medias_home_goal, conceded = medias_away_goal (oposição)
        """
        medias = fixture.get("medias", {})
        if side == "home":
            scored   = medias.get("home_avg_goal")
            conceded = medias.get("away_avg_goal")  # média do oponente como proxy
        else:
            scored   = medias.get("away_avg_goal")
            conceded = medias.get("home_avg_goal")

        return (scored or 1.2, conceded or 1.1)

    def extract_odds_for_orchestrator(self, fixture: dict) -> dict:
        """
        Retorna odds no formato compatível com o combo_engine e analysis_orchestrator.
        Prioridade: Bet365/1xBet reais (sokker_odds) > SokkerPRO prognósticos.
        """
        odds = dict(fixture.get("sokker_odds", {}))

        # Enriquece com prognósticos SokkerPRO APENAS para mercados ainda sem odds reais
        prog = fixture.get("prognosticos", {})
        if prog:
            for market_key, prog_key in [
                ("Over_1.5", "over_1_5"),
                ("Over_2.5", "over_2_5"),
                ("Over_3.5", "over_3_5"),
            ]:
                if market_key in odds:
                    continue  # já temos odd real da Bet365 — não sobrescreve
                try:
                    odd_val = prog["mercado_gols"][prog_key]["odd"]
                    if odd_val and float(odd_val) > 1.0:
                        odds[market_key] = {"odd": float(odd_val), "bookmaker": "SokkerPRO"}
                except (KeyError, TypeError):
                    pass

            # BTTS — idem: só usa SokkerPRO se Bet365 não trouxe
            if "BTTS" not in odds:
                try:
                    btts_odd = prog["mercado_ambos_marcam"]["ambos_sim"]["odd"]
                    if btts_odd and float(btts_odd) > 1.0:
                        odds["BTTS"] = {"odd": float(btts_odd), "bookmaker": "SokkerPRO"}
                except (KeyError, TypeError):
                    pass

        return odds

    def get_prognostico_probs(self, fixture: dict) -> dict:
        """
        Extrai probabilidades dos prognósticos para uso direto no orchestrator.
        Retorna dict com prob_over_25, prob_btts, etc.
        Retorna {} se não disponível (orchestrator usará Poisson próprio).
        """
        prog = fixture.get("prognosticos", {})
        if not prog:
            return {}

        result = {}
        try:
            m_gols = prog.get("mercado_gols", {})
            if "over_2_5" in m_gols:
                result["prob_over_25_sokker"] = m_gols["over_2_5"]["res"] / 100
            if "over_1_5" in m_gols:
                result["prob_over_15_sokker"] = m_gols["over_1_5"]["res"] / 100
            if "over_3_5" in m_gols:
                result["prob_over_35_sokker"] = m_gols["over_3_5"]["res"] / 100
        except Exception:
            pass

        try:
            m_1x2 = prog.get("mercado_1x2", {})
            if "casa_vencer" in m_1x2:
                result["prob_home_win_sokker"] = m_1x2["casa_vencer"]["probabilidade"] / 100
            if "empate" in m_1x2:
                result["prob_draw_sokker"] = m_1x2["empate"]["probabilidade"] / 100
            if "fora_vencer" in m_1x2:
                result["prob_away_win_sokker"] = m_1x2["fora_vencer"]["probabilidade"] / 100
        except Exception:
            pass

        try:
            m_btts = prog.get("mercado_ambos_marcam", {})
            if "ambos_sim" in m_btts:
                result["prob_btts_sokker"] = m_btts["ambos_sim"]["probabilidade"] / 100
        except Exception:
            pass

        return result
