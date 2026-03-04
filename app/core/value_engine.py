"""
ValueEngine — Camada de cálculo exclusivo de Expected Value (EV).

Responsabilidade única: receber probabilidades + odds e retornar EV, Kelly, stake.
Não calcula probabilidades, não aplica filtros, não toma decisões de sinal.

Faz parte da refatoração da FASE 1 — Modularização da Arquitetura.
"""

import math
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class ValueResult:
    """Resultado do cálculo de valor para um mercado."""
    market: str
    model_prob: float
    implied_prob: float
    odd: float
    ev: float
    ev_pct: float
    kelly_full: float
    kelly_fraction: float  # Kelly fracionário recomendado
    stake_units: float     # unidades sugeridas (Kelly × bankroll_fraction)
    edge: float            # diferença entre model_prob e implied_prob
    is_value: bool         # True se EV > threshold configurado
    bookmaker: str


class ValueEngine:
    """
    Calcula Expected Value (EV) e dimensionamento de aposta (Kelly).

    Matemática central:
        EV = (p_model × odd) - 1

        Kelly completo: f* = (p × b - q) / b
            onde b = odd - 1, q = 1 - p

        Kelly fracionário: f = f* × kelly_fraction
            Recomendação: 0.25 (Quarter Kelly) para volatilidade controlada
    """

    def __init__(
        self,
        min_ev_threshold: float = 0.05,
        kelly_fraction: float = 0.25,
        max_stake_pct: float = 0.05,
        min_odd: float = 1.40,
        max_odd: float = 5.00,
    ):
        """
        Args:
            min_ev_threshold: EV mínimo para sinal ser válido (ex: 0.05 = 3%)
            kelly_fraction: Fração do Kelly completo a usar (0.25 = Quarter Kelly)
            max_stake_pct: Máximo % do bankroll por aposta
            min_odd: Odd mínima aceitável (filtra favoritos extremos)
            max_odd: Odd máxima aceitável (filtra longas com alta variância)
        """
        self.min_ev_threshold = min_ev_threshold
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
        self.min_odd = min_odd
        self.max_odd = max_odd

    def calculate_ev(self, model_prob: float, odd: float) -> float:
        """
        EV = (p × odd) - 1

        Interpretação:
            EV > 0: valor positivo — modelo vê edge sobre a bookmaker
            EV = 0: aposta justa
            EV < 0: a bookmaker tem vantagem
        """
        if odd <= 0 or model_prob <= 0:
            return -1.0
        return (model_prob * odd) - 1.0

    def calculate_kelly(self, model_prob: float, odd: float) -> float:
        """
        Kelly completo: f* = (p × b - q) / b
        onde b = odd - 1

        Retorna 0.0 se não há valor positivo.
        """
        b = odd - 1.0
        if b <= 0 or model_prob <= 0:
            return 0.0

        q = 1.0 - model_prob
        kelly = (model_prob * b - q) / b

        return max(0.0, kelly)

    def calculate_implied_probability(self, odd: float) -> float:
        """Probabilidade implícita na odd (sem margem removida)."""
        if odd <= 0:
            return 1.0
        return 1.0 / odd

    def remove_vig(self, odds: list[float]) -> list[float]:
        """
        Remove a margem (vig) da bookmaker e retorna odds justas.
        
        Overround = Σ(1/odd_i) — representa a margem total da casa.
        odd_justa_i = odd_i × overround

        Args:
            odds: lista de odds para um mercado (ex: [2.10, 3.40, 3.20] para 1X2)

        Returns:
            lista de odds sem margem (soma das probs implícitas = 1.0)
        """
        if not odds or any(o <= 0 for o in odds):
            return odds

        overround = sum(1.0 / o for o in odds)
        if overround <= 0:
            return odds

        return [o * overround for o in odds]

    def evaluate_market(
        self,
        market: str,
        model_prob: float,
        odd: float,
        bookmaker: str = "unknown",
    ) -> Optional[ValueResult]:
        """
        Avalia um mercado e retorna ValueResult se houver valor.

        Args:
            market: identificador do mercado (ex: "over_2_5")
            model_prob: probabilidade do modelo (0-1)
            odd: odd decimal da bookmaker
            bookmaker: nome da bookmaker

        Returns:
            ValueResult se EV > 0, None se sem valor
        """
        if model_prob <= 0 or model_prob >= 1:
            return None
        if odd < self.min_odd or odd > self.max_odd:
            return None

        implied_prob = self.calculate_implied_probability(odd)
        ev = self.calculate_ev(model_prob, odd)
        ev_pct = ev * 100

        kelly_full = self.calculate_kelly(model_prob, odd)
        kelly_frac = kelly_full * self.kelly_fraction
        stake_units = min(kelly_frac, self.max_stake_pct)

        edge = model_prob - implied_prob
        is_value = ev >= self.min_ev_threshold

        return ValueResult(
            market=market,
            model_prob=model_prob,
            implied_prob=implied_prob,
            odd=odd,
            ev=ev,
            ev_pct=ev_pct,
            kelly_full=kelly_full,
            kelly_fraction=kelly_frac,
            stake_units=stake_units,
            edge=edge,
            is_value=is_value,
            bookmaker=bookmaker,
        )

    def evaluate_all_markets(
        self,
        probabilities: dict,
        market_odds: dict,
    ) -> list[ValueResult]:
        """
        Avalia todos os mercados disponíveis.

        Args:
            probabilities: dict {market: prob} do ProbabilityEngine
            market_odds: dict {market: {bookmaker: odd}} das bookmakers

        Returns:
            lista de ValueResult ordenados por EV decrescente
        """
        results = []

        for market, odds_by_bk in market_odds.items():
            model_prob = probabilities.get(market)
            if model_prob is None:
                continue

            # Pega a melhor odd disponível
            best_odd = 0.0
            best_bk = "unknown"
            for bk, odd in odds_by_bk.items():
                try:
                    o = float(odd)
                    if o > best_odd:
                        best_odd = o
                        best_bk = bk
                except (TypeError, ValueError):
                    continue

            if best_odd <= 0:
                continue

            result = self.evaluate_market(market, model_prob, best_odd, best_bk)
            if result is not None:
                results.append(result)

        # Ordena por EV decrescente
        results.sort(key=lambda r: r.ev, reverse=True)
        return results

    def calculate_expected_profit(
        self,
        ev: float,
        stake: float,
    ) -> float:
        """Lucro esperado absoluto: EV × stake."""
        return ev * stake

    def sharpe_ratio(
        self,
        returns: list[float],
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Sharpe Ratio simplificado para sequência de apostas.

        Sharpe = (média_retorno - taxa_livre_risco) / std_retorno

        Usado no tracking de performance (FASE 7).
        """
        if len(returns) < 2:
            return 0.0
        import statistics
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        if std_r == 0:
            return 0.0
        return (mean_r - risk_free_rate) / std_r
