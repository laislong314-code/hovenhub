"""
Schemas Pydantic — validação de dados entre camadas
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, field_validator


# ─── Odds API ─────────────────────────────────────────────────────────────────

class OddsOutcome(BaseModel):
    name: str       # "Over", "Under", "Home", "Away", "Draw"
    price: float    # odd decimal
    point: Optional[float] = None  # linha (Over/Under)


class OddsMarket(BaseModel):
    key: str        # "totals", "h2h", "spreads"
    last_update: datetime
    outcomes: List[OddsOutcome]


class BookmakerOdds(BaseModel):
    key: str        # nome do bookmaker
    title: str
    last_update: datetime
    markets: List[OddsMarket]


class RawMatchOdds(BaseModel):
    id: str
    sport_key: str
    sport_title: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmakers: List[BookmakerOdds]


# ─── Análise Estatística ──────────────────────────────────────────────────────

class TeamForm(BaseModel):
    """Forma recente de um time — calculada a partir de histórico"""
    team_name: str
    avg_scored: float       # média de gols marcados
    avg_conceded: float     # média de gols sofridos
    matches_used: int       # qtd de jogos usados no cálculo
    attack_strength: float  # força ofensiva relativa à média da liga
    defense_weakness: float # fraqueza defensiva relativa à média da liga


class PoissonAnalysis(BaseModel):
    """Resultado completo do modelo Poisson para um jogo"""
    match_id: str
    home_team: str
    away_team: str
    league: str
    commence_time: datetime

    # Parâmetros do modelo
    lambda_home: float
    lambda_away: float
    lambda_total: float

    # Probabilidades
    prob_over_25: float     # P(total_gols > 2.5)
    prob_under_25: float

    # Forma dos times
    home_form: TeamForm
    away_form: TeamForm
    league_avg: float


class ValueBetAnalysis(BaseModel):
    """Análise completa de value bet"""
    analysis: PoissonAnalysis
    market: str             # "over_2.5"
    model_prob: float       # probabilidade do modelo
    best_odd: float         # melhor odd disponível
    best_bookmaker: str
    implied_prob: float     # 1 / odd (sem margem)
    ev: float               # EV = (model_prob * odd) - 1
    is_value_bet: bool      # passa nos filtros?
    kelly_stake_pct: float  # fração de Kelly ajustada
    suggested_stake: float  # valor absoluto com base no bankroll


# ─── Sinais ───────────────────────────────────────────────────────────────────

class SignalDTO(BaseModel):
    id: int
    match_id: str
    home_team: str
    away_team: str
    home_logo: Optional[str] = None
    away_logo: Optional[str] = None
    league: str
    league_id: Optional[str] = None
    commence_time: datetime
    market: str
    market_label: Optional[str] = None   # label legível ex: "Múltipla (BTTS + Vitória Fora)"
    model_probability: float
    implied_probability: float
    ev: float
    suggested_odd: float
    bookmaker: str
    stake_pct: float
    stake_units: float
    sent_at: datetime
    result: Optional[str] = None
    profit_loss: Optional[float] = None
    actual_goals: Optional[int] = None
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    is_live: Optional[bool] = None
    match_minute: Optional[int] = None


class SignalStats(BaseModel):
    """Estatísticas de performance dos sinais"""
    total_signals: int
    resolved_signals: int
    wins: int
    losses: int
    win_rate: float
    total_staked: float
    total_profit_loss: float
    roi: float
    yield_pct: float
