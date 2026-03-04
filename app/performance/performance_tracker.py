"""
PerformanceTracker — Rastreamento de performance por estratégia.

Implementa métricas completas de apostas:
  - ROI (Return on Investment)
  - Yield (ROI por unidade apostada)
  - Winrate
  - Sharpe Ratio
  - ROI por mercado, por liga, por model_version

Faz parte da FASE 7 — Tracking de Performance por Estratégia.
"""

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from loguru import logger


@dataclass
class BetRecord:
    """Registro de uma aposta individual."""
    signal_id: str
    match_id: str
    market: str
    strategy_id: str
    model_version: str
    league_id: str

    # Valores da aposta
    odd: float
    stake: float          # valor apostado
    model_prob: float
    ev: float

    # Resultado
    result: Optional[str] = None    # "WIN", "LOSS", "VOID", "PUSH"
    profit_loss: Optional[float] = None

    # Metadados
    placed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    settled_at: Optional[str] = None
    is_live: bool = False


@dataclass
class StrategyPerformance:
    """Performance consolidada de uma estratégia."""
    strategy_id: str
    model_version: str

    # Contadores
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    voids: int = 0

    # Financeiros
    total_staked: float = 0.0
    total_profit: float = 0.0
    max_drawdown: float = 0.0

    # Métricas calculadas
    roi: float = 0.0          # (lucro / total_staked) × 100
    yield_pct: float = 0.0    # ROI / número de apostas
    winrate: float = 0.0      # wins / (wins + losses) × 100
    sharpe: float = 0.0       # Sharpe Ratio
    avg_odd: float = 0.0

    # Metadados
    first_bet: Optional[str] = None
    last_bet: Optional[str] = None


class PerformanceTracker:
    """
    Rastreador de performance de apostas por múltiplas dimensões.

    Dimensões de análise:
        - Por estratégia (standard, conservative, etc.)
        - Por mercado (over_2_5, btts_yes, etc.)
        - Por liga
        - Por model_version
        - Por contexto (live vs pré-jogo)

    Uso:
        tracker = PerformanceTracker()

        # Registra uma aposta
        tracker.record_bet(bet)

        # Registra resultado
        tracker.settle_bet(signal_id, result="WIN", profit_loss=0.85)

        # Obtém performance
        perf = tracker.get_strategy_performance("standard")
        print(f"ROI: {perf.roi:.2f}%")
    """

    def __init__(self):
        self._bets: dict[str, BetRecord] = {}  # signal_id → BetRecord

        # Índices para lookup eficiente
        self._by_strategy: dict[str, list[str]] = {}
        self._by_market: dict[str, list[str]] = {}
        self._by_league: dict[str, list[str]] = {}
        self._by_version: dict[str, list[str]] = {}

    def record_bet(self, bet: BetRecord):
        """Registra uma nova aposta."""
        self._bets[bet.signal_id] = bet

        # Atualiza índices
        self._index_bet(bet.strategy_id, self._by_strategy, bet.signal_id)
        self._index_bet(bet.market, self._by_market, bet.signal_id)
        self._index_bet(bet.league_id, self._by_league, bet.signal_id)
        self._index_bet(bet.model_version, self._by_version, bet.signal_id)

    def _index_bet(self, key: str, index: dict, signal_id: str):
        """Adiciona aposta a um índice."""
        if key not in index:
            index[key] = []
        index[key].append(signal_id)

    def settle_bet(
        self,
        signal_id: str,
        result: str,
        profit_loss: float,
    ):
        """
        Registra o resultado de uma aposta.

        Args:
            signal_id: ID do sinal
            result: "WIN", "LOSS", "VOID", "PUSH"
            profit_loss: lucro/perda em unidades (positivo = lucro)
        """
        if signal_id not in self._bets:
            logger.warning(f"[Tracker] Aposta não encontrada: {signal_id}")
            return

        bet = self._bets[signal_id]
        bet.result = result.upper()
        bet.profit_loss = profit_loss
        bet.settled_at = datetime.now(timezone.utc).isoformat()

    def _get_settled_bets(self, signal_ids: list[str]) -> list[BetRecord]:
        """Retorna apostas liquidadas de uma lista de IDs."""
        return [
            self._bets[sid]
            for sid in signal_ids
            if sid in self._bets and self._bets[sid].result is not None
        ]

    def _calculate_performance(
        self,
        bets: list[BetRecord],
        strategy_id: str = "",
        model_version: str = "",
    ) -> StrategyPerformance:
        """Calcula métricas de performance para uma lista de apostas."""
        perf = StrategyPerformance(strategy_id=strategy_id, model_version=model_version)

        if not bets:
            return perf

        perf.total_bets = len(bets)
        perf.wins = sum(1 for b in bets if b.result == "WIN")
        perf.losses = sum(1 for b in bets if b.result == "LOSS")
        perf.voids = sum(1 for b in bets if b.result in ("VOID", "PUSH"))

        perf.total_staked = sum(b.stake for b in bets)
        perf.total_profit = sum(b.profit_loss or 0.0 for b in bets)
        perf.avg_odd = statistics.mean(b.odd for b in bets) if bets else 0.0

        # ROI = (lucro_total / total_apostado) × 100
        if perf.total_staked > 0:
            perf.roi = (perf.total_profit / perf.total_staked) * 100
        else:
            perf.roi = 0.0

        # Yield = ROI / num_apostas (ROI por aposta)
        resolved = perf.wins + perf.losses
        if resolved > 0:
            perf.yield_pct = perf.roi / resolved
            perf.winrate = (perf.wins / resolved) * 100
        else:
            perf.yield_pct = 0.0
            perf.winrate = 0.0

        # Sharpe Ratio sobre os retornos individuais
        returns = [(b.profit_loss or 0.0) / b.stake for b in bets if b.stake > 0]
        if len(returns) >= 2:
            mean_r = statistics.mean(returns)
            std_r = statistics.stdev(returns)
            perf.sharpe = mean_r / std_r if std_r > 0 else 0.0

        # Maximum Drawdown (maior queda do pico)
        perf.max_drawdown = self._calculate_max_drawdown(bets)

        # Datas
        dates = sorted([b.placed_at for b in bets if b.placed_at])
        if dates:
            perf.first_bet = dates[0]
            perf.last_bet = dates[-1]

        return perf

    def _calculate_max_drawdown(self, bets: list[BetRecord]) -> float:
        """
        Calcula Maximum Drawdown: maior queda percentual do pico.

        MDD = (pico - vale) / pico × 100

        Importante para gerenciamento de risco: MDD > 30% = estratégia arriscada.
        """
        if not bets:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for bet in sorted(bets, key=lambda b: b.placed_at or ""):
            cumulative += bet.profit_loss or 0.0
            if cumulative > peak:
                peak = cumulative
            if peak > 0:
                drawdown = (peak - cumulative) / peak * 100
                if drawdown > max_dd:
                    max_dd = drawdown

        return round(max_dd, 2)

    # ── Métodos públicos de consulta ──────────────────────────────────────────

    def get_strategy_performance(self, strategy_id: str) -> StrategyPerformance:
        """Retorna performance de uma estratégia."""
        ids = self._by_strategy.get(strategy_id, [])
        bets = self._get_settled_bets(ids)
        return self._calculate_performance(bets, strategy_id=strategy_id)

    def get_market_performance(
        self,
        market: str,
        strategy_id: Optional[str] = None,
    ) -> StrategyPerformance:
        """Retorna performance de um mercado específico."""
        ids = self._by_market.get(market, [])
        bets = self._get_settled_bets(ids)

        if strategy_id:
            bets = [b for b in bets if b.strategy_id == strategy_id]

        return self._calculate_performance(bets, strategy_id=strategy_id or market)

    def get_league_performance(
        self,
        league_id: str,
        strategy_id: Optional[str] = None,
    ) -> StrategyPerformance:
        """Retorna performance em uma liga específica."""
        ids = self._by_league.get(league_id, [])
        bets = self._get_settled_bets(ids)

        if strategy_id:
            bets = [b for b in bets if b.strategy_id == strategy_id]

        return self._calculate_performance(bets, strategy_id=strategy_id or league_id)

    def get_version_performance(self, model_version: str) -> StrategyPerformance:
        """Retorna performance de uma versão de modelo específica."""
        ids = self._by_version.get(model_version, [])
        bets = self._get_settled_bets(ids)
        return self._calculate_performance(bets, model_version=model_version)

    def get_summary_report(self) -> dict:
        """Gera relatório consolidado de todas as estratégias."""
        report = {
            "total_bets": len(self._bets),
            "settled_bets": sum(1 for b in self._bets.values() if b.result is not None),
            "strategies": {},
            "markets": {},
            "top_leagues": {},
        }

        # Performance por estratégia
        for strategy_id in self._by_strategy:
            perf = self.get_strategy_performance(strategy_id)
            report["strategies"][strategy_id] = {
                "bets": perf.total_bets,
                "roi": round(perf.roi, 2),
                "yield": round(perf.yield_pct, 4),
                "winrate": round(perf.winrate, 2),
                "sharpe": round(perf.sharpe, 4),
                "drawdown": perf.max_drawdown,
            }

        # Performance por mercado
        for market in self._by_market:
            perf = self.get_market_performance(market)
            if perf.total_bets >= 10:  # só exibe mercados com dados suficientes
                report["markets"][market] = {
                    "bets": perf.total_bets,
                    "roi": round(perf.roi, 2),
                    "winrate": round(perf.winrate, 2),
                }

        # Top 5 ligas por ROI
        league_perfs = []
        for league_id in self._by_league:
            perf = self.get_league_performance(league_id)
            if perf.total_bets >= 5:
                league_perfs.append((league_id, perf.roi, perf.total_bets))

        league_perfs.sort(key=lambda x: x[1], reverse=True)
        for lid, roi, n in league_perfs[:5]:
            report["top_leagues"][lid] = {"roi": round(roi, 2), "bets": n}

        return report

    def get_expected_value_accuracy(self, strategy_id: str) -> dict:
        """
        Compara EV previsto vs realizado para validar o modelo.

        Se o modelo está bem calibrado:
            avg_ev_previsto ≈ avg_roi_realizado

        Uma diferença grande indica overestimação ou underestimação sistemática.
        """
        ids = self._by_strategy.get(strategy_id, [])
        settled = self._get_settled_bets(ids)

        if not settled:
            return {}

        expected_evs = [b.ev for b in settled]
        realized_returns = [(b.profit_loss or 0.0) / b.stake for b in settled if b.stake > 0]

        return {
            "avg_expected_ev": statistics.mean(expected_evs) if expected_evs else 0.0,
            "avg_realized_return": statistics.mean(realized_returns) if realized_returns else 0.0,
            "ev_accuracy": abs(
                statistics.mean(expected_evs) - statistics.mean(realized_returns)
            ) if expected_evs and realized_returns else None,
            "n_bets": len(settled),
        }
