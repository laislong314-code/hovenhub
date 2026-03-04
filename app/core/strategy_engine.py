"""
StrategyEngine — Camada de definição de estratégias.

Responsabilidade única: definir pesos, regras e parâmetros por estratégia.
Permite múltiplas estratégias simultâneas com configurações independentes.

Faz parte da refatoração da FASE 1 — Modularização da Arquitetura.
"""

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from app.services.engines.signal_engine import SignalFilter


@dataclass
class Strategy:
    """
    Define uma estratégia de apostas com todos os parâmetros configuráveis.
    
    Cada estratégia tem seu próprio conjunto de filtros, pesos e regras.
    Isso permite rodar múltiplas estratégias simultaneamente e comparar performance.
    """
    # Identificação
    strategy_id: str
    name: str
    description: str
    version: str = "1.0.0"

    # Filtros de valor
    filters: SignalFilter = field(default_factory=SignalFilter)

    # Mercados alvo
    target_markets: list[str] = field(default_factory=list)

    # Parâmetros de modelo
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.05
    max_concurrent_bets: int = 3

    # Parâmetros Poisson
    use_dixon_coles: bool = True
    poisson_rho: float = -0.13      # coeficiente de correlação DC
    use_bivariate: bool = False     # Bivariate Poisson (mais pesado, mais preciso)

    # Pesos de forma
    form_weight_recent: float = 0.6  # peso dos últimos 5 jogos
    form_weight_season: float = 0.4  # peso da temporada completa
    exponential_form_decay: float = 0.85  # fator de decaimento exponencial

    # Flags
    active: bool = True
    is_live: bool = False  # estratégia para jogos ao vivo

    # Ligas alvo (vazio = todas)
    target_leagues: list[str] = field(default_factory=list)


class StrategyEngine:
    """
    Gerencia múltiplas estratégias e define qual aplicar para cada jogo.

    Estratégias built-in:
        - standard: estratégia principal balanceada
        - conservative: apenas sinais de alta confiança
        - aggressive: mais sinais, menor EV mínimo
        - live: otimizada para apostas ao vivo
        - value_hunter: foco em odds mais altas com edge real
    """

    # Estratégias built-in pré-configuradas
    BUILT_IN_STRATEGIES: dict[str, dict] = {
        "standard": {
            "name": "Padrão",
            "description": "Estratégia balanceada. EV mínimo 3%, Kelly Quarter.",
            "filters": {
                "min_ev": 0.05,
                "min_model_prob": 0.52,
                "min_odd": 1.50,
                "max_odd": 4.00,
                "min_edge": 0.05,
            },
            "target_markets": [
                "over_2_5", "under_2_5", "btts_yes", "btts_no",
                "home_win", "away_win", "draw",
                "over_1_5", "over_3_5",
            ],
            "kelly_fraction": 0.25,
        },
        "conservative": {
            "name": "Conservador",
            "description": "Alta confiança apenas. EV mínimo 6%, foco em odds médias.",
            "filters": {
                "min_ev": 0.06,
                "min_model_prob": 0.58,
                "min_odd": 1.60,
                "max_odd": 3.20,
                "min_edge": 0.06,
            },
            "target_markets": [
                "over_2_5", "btts_yes", "home_win", "over_1_5",
            ],
            "kelly_fraction": 0.20,
        },
        "aggressive": {
            "name": "Agressivo",
            "description": "Volume alto, EV mínimo 2%. Mais sinais, mais risco.",
            "filters": {
                "min_ev": 0.02,
                "min_model_prob": 0.48,
                "min_odd": 1.40,
                "max_odd": 5.00,
                "min_edge": 0.02,
            },
            "target_markets": [
                "over_2_5", "under_2_5", "btts_yes", "btts_no",
                "home_win", "away_win", "draw",
                "over_1_5", "over_3_5", "over_4_5",
                "home_or_draw", "away_or_draw",
            ],
            "kelly_fraction": 0.15,
        },
        "live": {
            "name": "Ao Vivo",
            "description": "Otimizada para apostas durante o jogo com momentum.",
            "filters": {
                "min_ev": 0.04,
                "min_model_prob": 0.55,
                "min_odd": 1.50,
                "max_odd": 3.50,
                "min_edge": 0.04,
            },
            "target_markets": [
                "over_0_5", "over_1_5", "over_2_5",
                "btts_yes", "home_win", "away_win",
            ],
            "kelly_fraction": 0.15,
            "is_live": True,
        },
        "value_hunter": {
            "name": "Caçador de Valor",
            "description": "Foca em odds altas com edge real. Menor volume, maior ROI.",
            "filters": {
                "min_ev": 0.05,
                "min_model_prob": 0.45,
                "min_odd": 2.20,
                "max_odd": 6.00,
                "min_edge": 0.05,
            },
            "target_markets": [
                "draw", "away_win", "btts_no",
                "under_2_5", "over_3_5",
            ],
            "kelly_fraction": 0.10,
        },
    }

    def __init__(self):
        self._strategies: dict[str, Strategy] = {}
        self._load_built_in_strategies()

    def _load_built_in_strategies(self):
        """Carrega estratégias built-in."""
        for sid, cfg in self.BUILT_IN_STRATEGIES.items():
            f = cfg.get("filters", {})
            strategy = Strategy(
                strategy_id=sid,
                name=cfg["name"],
                description=cfg["description"],
                filters=SignalFilter(
                    min_ev=f.get("min_ev", 0.05),
                    min_model_prob=f.get("min_model_prob", 0.50),
                    min_odd=f.get("min_odd", 1.50),
                    max_odd=f.get("max_odd", 5.00),
                    min_edge=f.get("min_edge", 0.05),
                ),
                target_markets=cfg.get("target_markets", []),
                kelly_fraction=cfg.get("kelly_fraction", 0.25),
                is_live=cfg.get("is_live", False),
            )
            self._strategies[sid] = strategy
            logger.debug(f"[StrategyEngine] Estratégia carregada: {sid}")

    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Retorna uma estratégia por ID."""
        return self._strategies.get(strategy_id)

    def get_active_strategies(self, is_live: bool = False) -> list[Strategy]:
        """Retorna todas as estratégias ativas para o contexto dado."""
        return [
            s for s in self._strategies.values()
            if s.active and s.is_live == is_live
        ]

    def register_strategy(self, strategy: Strategy):
        """Registra estratégia customizada."""
        self._strategies[strategy.strategy_id] = strategy
        logger.info(f"[StrategyEngine] Nova estratégia registrada: {strategy.strategy_id}")

    def get_markets_for_strategy(self, strategy_id: str) -> list[str]:
        """Retorna mercados alvo de uma estratégia."""
        s = self.get_strategy(strategy_id)
        if not s or not s.target_markets:
            return list(self.BUILT_IN_STRATEGIES["standard"]["target_markets"])
        return s.target_markets

    def should_analyze_league(self, strategy_id: str, league_id: str) -> bool:
        """Verifica se a estratégia deve analisar uma liga específica."""
        s = self.get_strategy(strategy_id)
        if not s or not s.target_leagues:
            return True  # sem filtro = todas as ligas
        return league_id in s.target_leagues

    def apply_form_weights(
        self,
        season_avg: float,
        recent_avg: float,
        strategy_id: str = "standard",
        games_recent: int = 5,
    ) -> float:
        """
        Aplica pesos de forma com decaimento exponencial.

        Fórmula:
            w_recent = w_r × (1 - decay^n) onde n = jogos recentes usados
            avg_ponderada = w_recent × recent_avg + w_season × season_avg

        Args:
            season_avg: média da temporada completa
            recent_avg: média dos últimos N jogos
            strategy_id: estratégia para buscar os pesos
            games_recent: número de jogos recentes considerados

        Returns:
            média ponderada ajustada por forma
        """
        s = self.get_strategy(strategy_id) or self.get_strategy("standard")

        decay_factor = s.form_weight_recent * (1 - s.exponential_form_decay ** games_recent)
        decay_factor = max(0.1, min(0.9, decay_factor))

        return decay_factor * recent_avg + (1 - decay_factor) * season_avg

    def list_strategies(self) -> list[dict]:
        """Lista todas as estratégias com metadados."""
        return [
            {
                "id": s.strategy_id,
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "active": s.active,
                "is_live": s.is_live,
                "min_ev": s.filters.min_ev,
                "markets": len(s.target_markets),
            }
            for s in self._strategies.values()
        ]
