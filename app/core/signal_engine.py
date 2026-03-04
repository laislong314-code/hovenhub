"""
SignalEngine — Camada de geração de sinais.

Responsabilidade única: aplicar filtros e regras de qualidade para transformar
ValueResults em Signals concretos com metadados completos.

Não calcula probabilidades, não calcula EV. Apenas filtra e gera sinais.

Faz parte da refatoração da FASE 1 — Modularização da Arquitetura.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
from loguru import logger

from app.services.engines.value_engine import ValueResult


@dataclass
class Signal:
    """
    Sinal de aposta gerado pelo SignalEngine.
    Contém todos os metadados necessários para auditoria e tracking.
    """
    # Identificação
    match_id: str
    market: str
    label: str
    model_version: str

    # Valores do modelo
    model_prob: float
    implied_prob: float
    ev: float
    odd: float
    bookmaker: str
    stake_pct: float
    stake_units: float

    # Filtros que aprovaram este sinal
    filters_passed: list[str] = field(default_factory=list)

    # Metadados de contexto
    is_live: bool = False
    match_minute: Optional[int] = None
    league_id: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Qualidade do sinal (preenchido pelo SignalEngine)
    confidence_score: float = 0.0  # 0-1 score composto de qualidade
    signal_tier: str = "C"         # A/B/C baseado em múltiplos critérios


@dataclass
class SignalFilter:
    """Configuração de filtros para o SignalEngine."""
    min_ev: float = 0.05
    min_model_prob: float = 0.50
    max_implied_prob: float = 0.85
    min_odd: float = 1.40
    max_odd: float = 5.00
    min_edge: float = 0.05       # diferença mínima entre model_prob e implied_prob
    require_volume: bool = False  # se True, exige volume de dados mínimo


class SignalEngine:
    """
    Aplica filtros de qualidade e gera sinais auditáveis.

    Fluxo:
        ValueResult → Filtros → Enriquecimento → Signal

    O SignalEngine não sabe como calcular EV ou probabilidades.
    Ele recebe ValueResults prontos e decide quais viram sinais.
    """

    def __init__(self, filters: Optional[SignalFilter] = None):
        self.filters = filters or SignalFilter()

    def process_value_results(
        self,
        value_results: list[ValueResult],
        match_id: str,
        model_version: str,
        context: Optional[dict] = None,
    ) -> list[Signal]:
        """
        Processa lista de ValueResults e retorna Signals válidos.

        Args:
            value_results: resultados do ValueEngine
            match_id: ID do jogo
            model_version: versão do modelo (do ModelRegistry)
            context: dict com metadados (league_id, home_team, etc.)

        Returns:
            lista de Signal ordenados por confidence_score decrescente
        """
        ctx = context or {}
        signals = []

        for vr in value_results:
            if not vr.is_value:
                continue

            filters_passed = self._apply_filters(vr)
            if not filters_passed:
                continue

            confidence = self._calculate_confidence(vr, filters_passed)
            tier = self._determine_tier(confidence, vr)

            signal = Signal(
                match_id=match_id,
                market=vr.market,
                label=self._generate_label(vr),
                model_version=model_version,
                model_prob=vr.model_prob,
                implied_prob=vr.implied_prob,
                ev=vr.ev,
                odd=vr.odd,
                bookmaker=vr.bookmaker,
                stake_pct=vr.stake_units,
                stake_units=vr.stake_units,
                filters_passed=filters_passed,
                is_live=ctx.get("is_live", False),
                match_minute=ctx.get("match_minute"),
                league_id=ctx.get("league_id"),
                home_team=ctx.get("home_team"),
                away_team=ctx.get("away_team"),
                confidence_score=confidence,
                signal_tier=tier,
            )
            signals.append(signal)

        # Ordena por confidence descrescente
        signals.sort(key=lambda s: s.confidence_score, reverse=True)
        return signals

    def _apply_filters(self, vr: ValueResult) -> list[str]:
        """
        Aplica filtros e retorna lista de filtros que passaram.
        Se algum filtro obrigatório falhar, retorna lista vazia.
        """
        f = self.filters
        passed = []

        # Filtro EV mínimo (obrigatório)
        if vr.ev < f.min_ev:
            logger.debug(f"[SignalEngine] {vr.market} rejeitado: EV {vr.ev:.3f} < {f.min_ev}")
            return []
        passed.append(f"ev_min_{f.min_ev}")

        # Filtro probabilidade do modelo (obrigatório)
        if vr.model_prob < f.min_model_prob:
            logger.debug(f"[SignalEngine] {vr.market} rejeitado: prob {vr.model_prob:.3f} < {f.min_model_prob}")
            return []
        passed.append(f"prob_min_{f.min_model_prob}")

        # Filtro de odd (obrigatório)
        if vr.odd < f.min_odd or vr.odd > f.max_odd:
            logger.debug(f"[SignalEngine] {vr.market} rejeitado: odd {vr.odd} fora de [{f.min_odd}, {f.max_odd}]")
            return []
        passed.append(f"odd_range_{f.min_odd}_{f.max_odd}")

        # Filtro de edge (opcional mas recomendado)
        if vr.edge >= f.min_edge:
            passed.append(f"edge_min_{f.min_edge}")

        # Filtro probabilidade implícita (evita near-certainties sobrevalorizadas)
        if vr.implied_prob <= f.max_implied_prob:
            passed.append(f"implied_max_{f.max_implied_prob}")

        return passed

    def _calculate_confidence(
        self,
        vr: ValueResult,
        filters_passed: list[str],
    ) -> float:
        """
        Score de confiança composto (0-1).

        Componentes com pesos:
            40% EV normalizado (EV 10% = score 1.0)
            30% Edge normalizado (edge 15% = score 1.0)
            20% Número de filtros adicionais passados
            10% Posição da odd (odds médias são mais confiáveis)
        """
        # Componente EV (normalizado até 10%)
        ev_score = min(1.0, vr.ev / 0.10)

        # Componente Edge
        edge_score = min(1.0, vr.edge / 0.15)

        # Componente filtros adicionais (além dos 3 obrigatórios)
        extra_filters = len(filters_passed) - 3
        filter_score = min(1.0, extra_filters / 2.0)

        # Componente odd (odds entre 1.8 e 3.0 recebem score máximo)
        if 1.8 <= vr.odd <= 3.0:
            odd_score = 1.0
        elif 1.5 <= vr.odd < 1.8 or 3.0 < vr.odd <= 4.0:
            odd_score = 0.6
        else:
            odd_score = 0.3

        confidence = (
            0.40 * ev_score +
            0.30 * edge_score +
            0.20 * filter_score +
            0.10 * odd_score
        )

        return round(min(1.0, confidence), 4)

    def _determine_tier(self, confidence: float, vr: ValueResult) -> str:
        """
        Classifica o sinal em tiers:
            A: Alta confiança (confidence >= 0.65, EV >= 7%)
            B: Média confiança (confidence >= 0.40, EV >= 4%)
            C: Baixa confiança (demais)
        """
        if confidence >= 0.65 and vr.ev >= 0.07:
            return "A"
        elif confidence >= 0.40 and vr.ev >= 0.04:
            return "B"
        else:
            return "C"

    def _generate_label(self, vr: ValueResult) -> str:
        """Gera label legível para o mercado."""
        labels = {
            "over_2_5": "Over 2.5 Gols",
            "under_2_5": "Under 2.5 Gols",
            "over_1_5": "Over 1.5 Gols",
            "under_1_5": "Under 1.5 Gols",
            "over_3_5": "Over 3.5 Gols",
            "under_3_5": "Under 3.5 Gols",
            "over_0_5": "Over 0.5 Gols",
            "over_4_5": "Over 4.5 Gols",
            "home_win": "Vitória Casa",
            "draw": "Empate",
            "away_win": "Vitória Visitante",
            "btts_yes": "Ambos Marcam (Sim)",
            "btts_no": "Ambos Marcam (Não)",
            "home_or_draw": "1X (Casa ou Empate)",
            "away_or_draw": "X2 (Empate ou Visitante)",
            "home_or_away": "12 (Dupla Hipótese)",
        }
        return labels.get(vr.market, vr.market.replace("_", " ").title())

    def filter_top_signals(
        self,
        signals: list[Signal],
        max_signals: int = 3,
        min_tier: str = "C",
    ) -> list[Signal]:
        """
        Retorna os melhores sinais após filtragem por tier.

        Args:
            signals: lista de sinais gerados
            max_signals: máximo de sinais a retornar
            min_tier: tier mínimo ("A", "B" ou "C")
        """
        tier_order = {"A": 3, "B": 2, "C": 1}
        min_tier_val = tier_order.get(min_tier, 1)

        filtered = [
            s for s in signals
            if tier_order.get(s.signal_tier, 0) >= min_tier_val
        ]

        return filtered[:max_signals]
