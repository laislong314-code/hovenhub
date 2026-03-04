"""
corner_model.py — Modelo de escanteios para Beto v5.

Calcula lambda de escanteios usando:
  - Médias históricas da fixture (medias_home_corners / medias_away_corners)
  - Médias por período (1T/2T) quando disponíveis
  - Status do jogo (pré/ao vivo) para ajuste dinâmico
  - Pressão por faixa (barra015..barra4560) para contexto live
  - referee_avg_corners injetado no fixture pelo orchestrator (peso 15%)

Integra com market_registry para extrair odds reais de escanteios.

CONTRATO DE INTERFACE:
  - analyze(fixture, odds) — assinatura explícita, sem **kwargs
  - referee_avg_corners deve ser injetado no fixture pelo chamador ANTES de chamar analyze()
    Exemplo: fixture["referee_avg_corners"] = referee_profile.avg_corners_per_game
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from loguru import logger
from scipy.stats import poisson

from app.services.market_registry import (
    extract_odds_from_fixture,
    parse_odd,
    CATEGORY_CORNERS,
    get_stake_multiplier,
)


# ── Constantes ────────────────────────────────────────────────────────────────

LEAGUE_CORNER_AVERAGE = 10.3   # Média global de escanteios por jogo
MIN_EV_CORNERS        = 0.05
MIN_ODD_CORNERS       = 1.25
MAX_ODD_CORNERS       = 5.0
MIN_PROB_CORNERS      = 0.40

# Limites de sanidade para referee_avg_corners (fora disso, descarta o dado)
REFEREE_CORNERS_MIN = 7.0
REFEREE_CORNERS_MAX = 14.0


@dataclass
class CornerSignal:
    internal_key:  str
    label:         str
    line:          float
    odd:           float
    bookmaker:     str
    model_prob:    float
    implied_prob:  float
    ev:            float
    is_live:       bool
    lambda_total:  float


class CornerModel:
    """
    Modelo de Poisson para escanteios com suporte a pré-jogo e ao vivo.

    Interface:
      - O chamador (orchestrator) deve enriquecer o fixture com quaisquer dados
        contextuais extras (ex: referee_avg_corners) ANTES de chamar analyze().
      - O modelo lê tudo do fixture — nunca aceita parâmetros avulsos.
      - Isso elimina acoplamento de assinatura e risco de hot-reload inconsistente.
    """

    # ── API principal ─────────────────────────────────────────────────────────

    # Ligas com mercado de corners confiável e cobertura adequada
    CORNER_LEAGUE_WHITELIST = {
        # Europa Top 5
        "8",    # Premier League (England)
        "9",    # Championship (England)
        "12",   # League One (England)
        "14",   # League Two (England)
        "24",   # FA Cup (England)
        "564",  # La Liga (Spain)
        "567",  # La Liga 2 (Spain)
        "82",   # Bundesliga (Germany)
        "85",   # 2. Bundesliga (Germany)
        "384",  # Serie A (Italy)
        "387",  # Serie B (Italy)
        "301",  # Ligue 1 (France)
        "304",  # Ligue 2 (France)
        # UEFA
        "2",    # Champions League
        "5",    # Europa League
        "2286", # Conference League
        # Outros Europa
        "72",   # Eredivisie (Netherlands)
        "462",  # Liga Portugal
        "465",  # Liga Portugal 2
        # Americas
        "636",  # Liga Profesional Argentina
        "648",  # Brasileirao Serie A
        "325",  # Brasileirao Serie A (alt)
        "649",  # Brasileirao Serie B
        "654",  # Copa do Brasil
        "1798", # Supercopa do Brasil
        "348",  # MLS
        # Brasil estaduais principais
        "1313", # Paulista A1
        "1296", # Carioca Serie A
        "1302", # Gaucho 1
        "1307", # Mineiro 1
        "1291", # Baiano 1
        "1300", # Cearense 1
        "1311", # Paranaense 1
        "1304", # Goiano 1
    }

    def _has_corners_data(self, fixture: dict) -> bool:
        """
        Retorna True somente se:
        1. A liga está na whitelist de ligas com mercado confiavel, E
        2. Tem medias de corners reais (home e away >= 1.0), E
        3. O fixture tem mercado real de corners (flag has_corners_market)
        """
        # 1. Verifica whitelist de ligas
        league_id = str(fixture.get("league_id") or fixture.get("league_code") or "")
        if league_id not in self.CORNER_LEAGUE_WHITELIST:
            return False

        # 2. Flag injetada pelo orchestrator (BET365_CANTO_* presente no fixture)
        if not fixture.get("has_corners_market", True):
            return False

        # 3. Medias reais de corners
        medias = fixture.get("medias") or {}
        home = medias.get("home_avg_corners") or fixture.get("medias_home_corners")
        away = medias.get("away_avg_corners") or fixture.get("medias_away_corners")
        try:
            return float(home) >= 1.0 and float(away) >= 1.0
        except (TypeError, ValueError):
            return False

    def analyze(self, fixture: dict, odds: dict | None = None) -> list[CornerSignal]:
        """
        Analisa a fixture e retorna sinais de escanteios com EV positivo.
        Retorna lista vazia se nao houver dados reais de corners.
        """
        if not fixture:
            logger.debug('[CornerModel] fixture ausente; pulando analise de escanteios.')
            return []

        if not self._has_corners_data(fixture):
            home = fixture.get("homeTeam", {}).get("name", "?")
            away = fixture.get("awayTeam", {}).get("name", "?")
            logger.debug(f'[CornerModel] {home} vs {away} — sem medias_corners reais; pulando.')
            return []

        is_live      = self._is_live(fixture)
        minute       = self._get_minute(fixture)
        corners_done = self._get_corners_done(fixture)

        lambda_full = self._calc_lambda_full(fixture)

        if is_live and minute > 0:
            signals = self._analyze_live(fixture, lambda_full, minute, corners_done)
        else:
            signals = self._analyze_pre(fixture, lambda_full)

        signals.sort(key=lambda s: s.ev, reverse=True)
        logger.debug(f"[CornerModel] {len(signals)} sinais de escanteios gerados")
        return signals

    def get_lambda(self, fixture: dict) -> float:
        """Expoe lambda total. Retorna 0.0 se sem dados reais de corners."""
        if not self._has_corners_data(fixture):
            return 0.0
        return self._calc_lambda_full(fixture)

    def get_probabilities(self, fixture: dict) -> dict:
        """
        Retorna dict de probabilidades para linhas padrao.
        Retorna zeros se nao houver dados reais de corners.
        """
        if not self._has_corners_data(fixture):
            return {
                "lambda_total":   0.0,
                "prob_over_7":    0.0, "prob_over_8":    0.0,
                "prob_over_85":   0.0, "prob_over_9":    0.0,
                "prob_over_95":   0.0, "prob_over_10":   0.0,
                "prob_over_105":  0.0, "prob_over_11":   0.0,
                "prob_over_115":  0.0, "prob_under_85":  0.0,
                "prob_under_95":  0.0, "prob_under_105": 0.0,
            }
        lam = self._calc_lambda_full(fixture)
        return {
            "lambda_total":   lam,
            "prob_over_7":    self._prob_over(lam, 7.0),
            "prob_over_8":    self._prob_over(lam, 8.0),
            "prob_over_85":   self._prob_over(lam, 8.5),
            "prob_over_9":    self._prob_over(lam, 9.0),
            "prob_over_95":   self._prob_over(lam, 9.5),
            "prob_over_10":   self._prob_over(lam, 10.0),
            "prob_over_105":  self._prob_over(lam, 10.5),
            "prob_over_11":   self._prob_over(lam, 11.0),
            "prob_over_115":  self._prob_over(lam, 11.5),
            "prob_under_85":  1 - self._prob_over(lam, 8.5),
            "prob_under_95":  1 - self._prob_over(lam, 9.5),
            "prob_under_105": 1 - self._prob_over(lam, 10.5),
        }

    # ── Lambda calculation ────────────────────────────────────────────────────

    def _calc_lambda_full(self, fixture: dict) -> float:
        """
        Lambda total para o jogo completo.

        Pesos:
          - Com referee_avg_corners válido: 60% médias times + 25% média global + 15% árbitro
          - Sem referee_avg_corners:        70% médias times + 30% média global

        O dado do árbitro captura o estilo de jogo que ele permite
        (árbitros mais tolerantes -> menos interrupções -> mais escanteios).
        """
        medias   = fixture.get("medias") or {}
        home_avg = self._safe_float(
            medias.get("home_avg_corners") or fixture.get("medias_home_corners"), 5.2
        )
        away_avg = self._safe_float(
            medias.get("away_avg_corners") or fixture.get("medias_away_corners"), 5.1
        )
        raw_total = home_avg + away_avg

        # Árbitro como âncora adicional (injetado pelo orchestrator)
        referee_avg = self._safe_float(fixture.get("referee_avg_corners"), None)
        if referee_avg is not None and REFEREE_CORNERS_MIN <= referee_avg <= REFEREE_CORNERS_MAX:
            blended = raw_total * 0.60 + LEAGUE_CORNER_AVERAGE * 0.25 + referee_avg * 0.15
            logger.debug(
                f"[CornerModel] lambda com árbitro: times={raw_total:.1f} "
                f"global={LEAGUE_CORNER_AVERAGE} ref={referee_avg:.1f} -> blend={blended:.2f}"
            )
        else:
            blended = raw_total * 0.70 + LEAGUE_CORNER_AVERAGE * 0.30
            if referee_avg is not None:
                logger.debug(
                    f"[CornerModel] referee_avg_corners={referee_avg:.1f} fora do range "
                    f"[{REFEREE_CORNERS_MIN}, {REFEREE_CORNERS_MAX}] — descartado"
                )

        # Ajuste por pressão (barra de pressão indica estilo de jogo)
        pressure_factor = self._calc_pressure_factor(fixture)

        lam = max(4.0, min(blended * pressure_factor, 18.0))
        logger.debug(
            f"[CornerModel] lambda_full={lam:.2f} "
            f"(home={home_avg}, away={away_avg}, pressure={pressure_factor:.2f})"
        )
        return lam

    def _calc_lambda_remaining(self, lambda_full: float, minute: int, corners_done: int) -> float:
        """
        Lambda esperado para os minutos restantes, dado o contexto ao vivo.
        Usa projeção linear ajustada pelo ritmo atual de escanteios.
        """
        total_minutes  = 90
        elapsed_frac   = min(minute / total_minutes, 1.0)
        remaining_frac = 1.0 - elapsed_frac

        expected_so_far = lambda_full * elapsed_frac
        pace_factor     = (corners_done / max(expected_so_far, 0.5)) if expected_so_far > 0 else 1.0
        pace_factor     = max(0.6, min(pace_factor, 1.6))

        return max(0.5, (lambda_full * remaining_frac) * pace_factor)

    # ── Pré-jogo ──────────────────────────────────────────────────────────────

    def _analyze_pre(self, fixture: dict, lambda_full: float) -> list[CornerSignal]:
        odds    = extract_odds_from_fixture(fixture)
        signals = []

        for ikey, info in odds.items():
            if not ikey.startswith("Corners_") or "_LIVE" in ikey:
                continue
            if info["category"] != CATEGORY_CORNERS:
                continue

            odd  = info["odd"]
            line = self._line_from_key(ikey)
            if line is None:
                continue

            if ikey.startswith("Corners_Over_"):
                model_prob = self._prob_over(lambda_full, line)
            else:
                model_prob = 1.0 - self._prob_over(lambda_full, line)

            ev = model_prob * odd - 1
            if self._passes(model_prob, odd, ev):
                signals.append(CornerSignal(
                    internal_key=ikey,
                    label=info["label"],
                    line=line,
                    odd=odd,
                    bookmaker=info["bookmaker"],
                    model_prob=model_prob,
                    implied_prob=1/odd,
                    ev=ev,
                    is_live=False,
                    lambda_total=lambda_full,
                ))

        return signals

    # ── Ao vivo ───────────────────────────────────────────────────────────────

    def _analyze_live(self, fixture: dict, lambda_full: float, minute: int, corners_done: int) -> list[CornerSignal]:
        odds       = extract_odds_from_fixture(fixture)
        lambda_rem = self._calc_lambda_remaining(lambda_full, minute, corners_done)
        signals    = []

        for ikey, info in odds.items():
            if not ikey.startswith("Corners_") or "_LIVE" not in ikey:
                continue
            if info["category"] != CATEGORY_CORNERS:
                continue

            odd  = info["odd"]
            line = self._line_from_key(ikey)
            if line is None:
                continue

            if ikey.startswith("Corners_Over_"):
                model_prob = self._prob_over(lambda_rem, line)
            else:
                model_prob = 1.0 - self._prob_over(lambda_rem, line)

            ev = model_prob * odd - 1
            if self._passes(model_prob, odd, ev):
                signals.append(CornerSignal(
                    internal_key=ikey,
                    label=info["label"],
                    line=line,
                    odd=odd,
                    bookmaker=info["bookmaker"],
                    model_prob=model_prob,
                    implied_prob=1/odd,
                    ev=ev,
                    is_live=True,
                    lambda_total=lambda_rem,
                ))

        return signals

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prob_over(self, lam: float, line: float) -> float:
        """P(X > line) via Poisson CDF."""
        k = int(math.floor(line))
        return 1.0 - float(poisson.cdf(k, lam))

    def _passes(self, prob: float, odd: float, ev: float) -> bool:
        return (
            prob >= MIN_PROB_CORNERS and
            ev   >= MIN_EV_CORNERS   and
            MIN_ODD_CORNERS <= odd <= MAX_ODD_CORNERS
        )

    def _is_live(self, fixture: dict) -> bool:
        status = str(fixture.get("status", "")).upper()
        return status in ("LIVE", "1H", "2H", "HT", "ET", "BT", "P", "SUSP")

    def _get_minute(self, fixture: dict) -> int:
        for key in ("elapsed", "minutePrimeiroTempo", "minuteSegundoTempo", "minute"):
            v = fixture.get(key)
            if v is not None:
                try:
                    return int(v)
                except (ValueError, TypeError):
                    pass
        return 0

    def _get_corners_done(self, fixture: dict) -> int:
        v = fixture.get("cornersTl") or fixture.get("corners_total") or 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    def _calc_pressure_factor(self, fixture: dict) -> float:
        """
        Deriva fator de pressão (0.92–1.08) a partir das barras de pressão.
        Equilíbrio entre times -> jogo aberto -> mais escanteios.
        """
        barra = fixture.get("medias_barra_pressao", {})
        if not barra:
            return 1.0
        try:
            total_balance  = 0.0
            total_segments = 0
            for seg_key in ("barra015", "barra1530", "barra3045", "barra4560"):
                seg = barra.get(seg_key, {})
                if seg:
                    home_pct = float(seg.get("home", 50)) / 100.0
                    away_pct = float(seg.get("away", 50)) / 100.0
                    balance  = 1.0 - abs(home_pct - away_pct)
                    total_balance  += balance
                    total_segments += 1
            if total_segments == 0:
                return 1.0
            avg_balance = total_balance / total_segments
            return 0.92 + avg_balance * 0.16
        except Exception:
            return 1.0

    def _line_from_key(self, ikey: str) -> float | None:
        """Extrai linha numérica de chaves como Corners_Over_9.5 ou Corners_Over_9.5_LIVE."""
        try:
            parts = ikey.replace("_LIVE", "").split("_")
            return float(parts[-1])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _safe_float(val, default: float | None = 0.0) -> float | None:
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
