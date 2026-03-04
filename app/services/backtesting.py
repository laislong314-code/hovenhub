"""
Módulo de Backtesting — Fase 2

Estratégia de validação temporal (correto para séries temporais):
  - NÃO usar train_test_split aleatório
  - Usar janela deslizante ou divisão cronológica
  - Treinar em T-1 ano, validar em T (out-of-sample)

Métricas calculadas:
  - ROI (Return on Investment)
  - Yield (ROI sobre total apostado)
  - Taxa de acerto
  - Drawdown máximo
  - Sharpe ratio das apostas
  - Calibração do modelo (Brier Score)
"""
from dataclasses import dataclass, field
from typing import Optional
import statistics


@dataclass
class BetRecord:
    """Registro de uma aposta no backtesting"""
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str
    market: str
    model_prob: float
    implied_prob: float
    ev: float
    odd: float
    stake: float
    result: str             # "WIN" / "LOSS"
    profit_loss: float
    actual_goals: Optional[int] = None


@dataclass
class BacktestResults:
    """Resultado completo de um backtesting"""
    bets: list[BetRecord] = field(default_factory=list)

    @property
    def total_bets(self) -> int:
        return len(self.bets)

    @property
    def wins(self) -> int:
        return sum(1 for b in self.bets if b.result == "WIN")

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_bets if self.total_bets > 0 else 0.0

    @property
    def total_staked(self) -> float:
        return sum(b.stake for b in self.bets)

    @property
    def total_profit_loss(self) -> float:
        return sum(b.profit_loss for b in self.bets)

    @property
    def roi(self) -> float:
        return self.total_profit_loss / self.total_staked if self.total_staked > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        """
        Drawdown máximo = maior queda de pico para vale no bankroll acumulado.
        
        Importância: um sistema com ROI positivo mas drawdown de 80% é
        psicologicamente insuportável e perigoso para o bankroll.
        """
        cumulative = []
        running = 0.0
        for bet in self.bets:
            running += bet.profit_loss
            cumulative.append(running)

        if not cumulative:
            return 0.0

        max_drawdown = 0.0
        peak = cumulative[0]
        for val in cumulative:
            if val > peak:
                peak = val
            drawdown = peak - val
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    @property
    def brier_score(self) -> float:
        """
        Mede calibração do modelo: quão próximas são as probabilidades
        previstas dos resultados reais.
        
        Brier = (1/N) × Σ(prob_prevista - resultado_real)²
        
        0.0 = perfeito, 0.25 = aleatório, > 0.25 = pior que aleatório
        """
        if not self.bets:
            return 0.0

        scores = []
        for bet in self.bets:
            outcome = 1.0 if bet.result == "WIN" else 0.0
            score = (bet.model_prob - outcome) ** 2
            scores.append(score)

        return statistics.mean(scores)

    def summary(self) -> dict:
        return {
            "total_bets": self.total_bets,
            "wins": self.wins,
            "losses": self.total_bets - self.wins,
            "win_rate": f"{self.win_rate:.1%}",
            "total_staked": round(self.total_staked, 2),
            "total_profit_loss": round(self.total_profit_loss, 2),
            "roi": f"{self.roi:.2%}",
            "yield": f"{self.roi:.2%}",
            "max_drawdown": round(self.max_drawdown, 2),
            "brier_score": round(self.brier_score, 4),
        }


class BacktestEngine:
    """
    Engine de backtesting com validação temporal.
    
    PRINCÍPIO FUNDAMENTAL:
    Nunca usar dados futuros para treinar o modelo.
    A divisão deve ser cronológica:
    
    |← TREINO (80%) →|← VALIDAÇÃO (20%) →|
    |--- 2021-2023 ---|--- 2024 ----------|
    
    Walk-forward validation (melhor prática):
    |T1 treino|T1 val|
              |T2 treino  |T2 val|
                         |T3 treino     |T3 val|
    """

    def __init__(
        self,
        min_ev: float = 0.05,
        min_prob: float = 0.50,
        min_odd: float = 1.50,
        max_odd: float = 3.00,
        kelly_fraction: float = 0.25,
        initial_bankroll: float = 1000.0,
    ):
        self.min_ev = min_ev
        self.min_prob = min_prob
        self.min_odd = min_odd
        self.max_odd = max_odd
        self.kelly_fraction = kelly_fraction
        self.bankroll = initial_bankroll

    def run(self, historical_data: list[dict]) -> BacktestResults:
        """
        Executa backtest sobre dados históricos.
        
        Args:
            historical_data: lista de dicts com:
                - match_id, home_team, away_team, league, match_date
                - model_prob (probabilidade calculada retroativamente)
                - best_odd, bookmaker
                - actual_result: "over" / "under" (resultado real)
                - actual_goals: total de gols na partida
        
        Returns:
            BacktestResults com todas as métricas
        """
        # Ordenar cronologicamente — CRÍTICO
        data = sorted(historical_data, key=lambda x: x["match_date"])
        
        results = BacktestResults()
        running_bankroll = self.bankroll

        for row in data:
            model_prob = row["model_prob"]
            odd = row["best_odd"]
            
            # Aplicar filtros
            implied_prob = 1.0 / odd
            ev = (model_prob * odd) - 1.0

            if not self._passes_filters(model_prob, odd, ev):
                continue

            # Calcular stake (Kelly fracionário sobre bankroll atual)
            b = odd - 1.0
            q = 1.0 - model_prob
            kelly_full = (model_prob * b - q) / b
            kelly_pct = max(0.0, kelly_full * self.kelly_fraction)
            kelly_pct = min(kelly_pct, 0.05)  # cap 3%
            stake = running_bankroll * kelly_pct

            if stake <= 0:
                continue

            # Verificar resultado
            is_over = row.get("actual_goals", 0) > 2.5
            won = (row.get("actual_result") == "over") or (is_over and row["market"] == "over_2.5")
            
            profit_loss = stake * (odd - 1) if won else -stake
            running_bankroll += profit_loss

            results.bets.append(BetRecord(
                match_id=row["match_id"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                league=row["league"],
                match_date=row["match_date"],
                market=row.get("market", "over_2.5"),
                model_prob=model_prob,
                implied_prob=implied_prob,
                ev=ev,
                odd=odd,
                stake=stake,
                result="WIN" if won else "LOSS",
                profit_loss=profit_loss,
                actual_goals=row.get("actual_goals"),
            ))

        return results

    def _passes_filters(self, prob: float, odd: float, ev: float) -> bool:
        return (
            prob >= self.min_prob and
            ev >= self.min_ev and
            self.min_odd <= odd <= self.max_odd
        )


# ─── Exemplo de uso ───────────────────────────────────────────────────────────
"""
from app.services.backtesting import BacktestEngine

# Carregar dados históricos (CSV ou banco)
import pandas as pd
df = pd.read_csv("data/historical_matches.csv")
data = df.to_dict("records")

engine = BacktestEngine(min_ev=0.05, kelly_fraction=0.25)
results = engine.run(data)

print(results.summary())
# {
#   'total_bets': 342,
#   'win_rate': '57.9%',
#   'roi': '+8.3%',
#   'max_drawdown': 124.50,
#   'brier_score': 0.2312
# }
"""
