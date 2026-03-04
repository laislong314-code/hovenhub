"""
LiveMomentumEngine — Sistema LIVE com momentum real.

Substitui o snapshot estático por modelo dinâmico de momentum.

Problema do sistema anterior:
    Usava médias estáticas de DAPM (Dangerous Attacks Per Minute) como peso fixo.
    Não captava aceleração, reversões de momentum ou ajuste ao placar.

Solução implementada:
    1. Rolling window dos últimos 5 minutos para suavizar ruído
    2. Delta de dangerous attacks: taxa de variação (d/dt)
    3. Delta de shots on goal: variação na eficiência de finalização
    4. Velocidade de aceleração ofensiva: segunda derivada
    5. Ajuste por estado do placar (time perdendo ataca mais)
    6. Momentum score composto (0-2, onde 1.0 = neutro)

Faz parte da FASE 6 — Sistema LIVE com Momentum Real.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class MomentumWindow:
    """Janela deslizante de eventos dos últimos N minutos."""
    home_dangerous_attacks: deque = field(default_factory=lambda: deque(maxlen=5))
    away_dangerous_attacks: deque = field(default_factory=lambda: deque(maxlen=5))
    home_shots_on_goal: deque = field(default_factory=lambda: deque(maxlen=5))
    away_shots_on_goal: deque = field(default_factory=lambda: deque(maxlen=5))
    home_corners: deque = field(default_factory=lambda: deque(maxlen=5))
    away_corners: deque = field(default_factory=lambda: deque(maxlen=5))

    def add_minute_data(
        self,
        home_da: float,
        away_da: float,
        home_sog: float,
        away_sog: float,
        home_c: float = 0.0,
        away_c: float = 0.0,
    ):
        """Adiciona dados de um minuto na janela."""
        self.home_dangerous_attacks.append(home_da)
        self.away_dangerous_attacks.append(away_da)
        self.home_shots_on_goal.append(home_sog)
        self.away_shots_on_goal.append(away_sog)
        self.home_corners.append(home_c)
        self.away_corners.append(away_c)


@dataclass
class MomentumResult:
    """Resultado do cálculo de momentum."""
    # Score composto de momentum (< 1.0 = visitante domina, > 1.0 = casa domina)
    home_momentum: float
    away_momentum: float
    momentum_ratio: float  # home/away, 1.0 = equilibrado

    # Componentes individuais
    da_ratio: float          # dangerous attacks ratio
    sog_ratio: float         # shots on goal ratio
    acceleration_home: float # taxa de aceleração ofensiva do casa
    acceleration_away: float # taxa de aceleração ofensiva do visitante

    # Estado do placar e sua influência
    score_pressure_home: float  # pressão por causa do placar (time perdendo pressiona mais)
    score_pressure_away: float

    # Fator total para ajuste do λ_residual
    lambda_multiplier_home: float  # multiplica λ_home do modelo base
    lambda_multiplier_away: float  # multiplica λ_away do modelo base

    # Metadados
    minute: int
    window_size: int
    confidence: float  # 0-1, baseado em quantidade de dados


class LiveMomentumEngine:
    """
    Motor de momentum ao vivo baseado em janela deslizante.

    Fluxo:
        1. Recebe snapshot do jogo ao vivo (a cada minuto)
        2. Atualiza rolling window
        3. Calcula deltas e acelerações
        4. Ajusta λ_residual com base no momentum
        5. Retorna MomentumResult para o LiveAnalyzer

    O MomentumResult é usado para ajustar o λ calculado pelo modelo base:
        λ_live_home = λ_base_home × momentum_result.lambda_multiplier_home
    """

    # Limites do multiplicador de λ (evita extremos que distorcem o modelo)
    MIN_LAMBDA_MULTIPLIER = 0.50
    MAX_LAMBDA_MULTIPLIER = 2.00

    def __init__(
        self,
        window_size: int = 5,
        da_weight: float = 0.40,
        sog_weight: float = 0.35,
        corner_weight: float = 0.10,
        acceleration_weight: float = 0.15,
    ):
        """
        Args:
            window_size: tamanho da janela deslizante em minutos
            da_weight: peso dos dangerous attacks no score
            sog_weight: peso dos shots on goal no score
            corner_weight: peso dos corners no score
            acceleration_weight: peso da aceleração no score
        """
        self.window_size = window_size
        self.da_weight = da_weight
        self.sog_weight = sog_weight
        self.corner_weight = corner_weight
        self.acceleration_weight = acceleration_weight

        # Estado interno (por match_id)
        self._windows: dict[str, MomentumWindow] = {}

    def _get_window(self, match_id: str) -> MomentumWindow:
        """Retorna ou cria janela para um jogo."""
        if match_id not in self._windows:
            self._windows[match_id] = MomentumWindow()
        return self._windows[match_id]

    def clear_match(self, match_id: str):
        """Limpa estado de um jogo encerrado."""
        self._windows.pop(match_id, None)

    def update_and_calculate(
        self,
        match_id: str,
        minute: int,
        home_goals: int,
        away_goals: int,
        # Dados da janela atual
        home_da_total: float,      # total acumulado de dangerous attacks
        away_da_total: float,
        home_sog_total: float,     # total acumulado de shots on goal
        away_sog_total: float,
        home_corners_total: float = 0.0,
        away_corners_total: float = 0.0,
        # Dados do minuto anterior (para delta)
        prev_home_da: Optional[float] = None,
        prev_away_da: Optional[float] = None,
        prev_home_sog: Optional[float] = None,
        prev_away_sog: Optional[float] = None,
    ) -> MomentumResult:
        """
        Atualiza a janela de momentum e calcula o MomentumResult.

        Este é o método principal chamado a cada minuto para jogos ao vivo.

        Args:
            match_id: ID do jogo
            minute: minuto atual
            home_goals, away_goals: placar atual
            *_total: totais acumulados de estatísticas
            prev_*: valores do minuto anterior (para calcular delta)

        Returns:
            MomentumResult com multiplicadores de λ
        """
        window = self._get_window(match_id)

        # Calcula deltas (incremento no minuto atual)
        delta_home_da = max(0.0, (home_da_total - (prev_home_da or home_da_total * 0.9)))
        delta_away_da = max(0.0, (away_da_total - (prev_away_da or away_da_total * 0.9)))
        delta_home_sog = max(0.0, (home_sog_total - (prev_home_sog or home_sog_total * 0.9)))
        delta_away_sog = max(0.0, (away_sog_total - (prev_away_sog or away_sog_total * 0.9)))

        # Adiciona à janela deslizante
        window.add_minute_data(
            home_da=delta_home_da,
            away_da=delta_away_da,
            home_sog=delta_home_sog,
            away_sog=delta_away_sog,
        )

        return self._compute_momentum(
            match_id=match_id,
            window=window,
            minute=minute,
            home_goals=home_goals,
            away_goals=away_goals,
            home_da_total=home_da_total,
            away_da_total=away_da_total,
            home_sog_total=home_sog_total,
            away_sog_total=away_sog_total,
        )

    def _compute_momentum(
        self,
        match_id: str,
        window: MomentumWindow,
        minute: int,
        home_goals: int,
        away_goals: int,
        home_da_total: float,
        away_da_total: float,
        home_sog_total: float,
        away_sog_total: float,
    ) -> MomentumResult:
        """Calcula momentum a partir da janela atual."""

        n_window = len(window.home_dangerous_attacks)
        confidence = min(1.0, n_window / self.window_size)

        # ── 1. Ratios baseados em totais acumulados ────────────────────────────
        total_da = home_da_total + away_da_total
        total_sog = home_sog_total + away_sog_total

        da_ratio = home_da_total / total_da if total_da > 0 else 0.5
        sog_ratio = home_sog_total / total_sog if total_sog > 0 else 0.5

        # ── 2. Aceleração na janela (delta dos últimos N minutos) ──────────────
        window_home_da = list(window.home_dangerous_attacks)
        window_away_da = list(window.away_dangerous_attacks)
        window_home_sog = list(window.home_shots_on_goal)
        window_away_sog = list(window.away_shots_on_goal)

        acc_home = self._compute_acceleration(window_home_da)
        acc_away = self._compute_acceleration(window_away_da)

        recent_home_da = sum(window_home_da) / max(1, len(window_home_da))
        recent_away_da = sum(window_away_da) / max(1, len(window_away_da))
        recent_home_sog = sum(window_home_sog) / max(1, len(window_home_sog))
        recent_away_sog = sum(window_away_sog) / max(1, len(window_away_sog))

        recent_total_da = recent_home_da + recent_away_da
        recent_da_ratio = recent_home_da / recent_total_da if recent_total_da > 0 else 0.5

        recent_total_sog = recent_home_sog + recent_away_sog
        recent_sog_ratio = recent_home_sog / recent_total_sog if recent_total_sog > 0 else 0.5

        # ── 3. Score de momentum (0-1 para cada time) ─────────────────────────
        # Combinamos total acumulado (tendência de longo prazo) com janela recente
        # A janela recente tem mais peso para capturar mudanças de ritmo

        home_momentum_score = (
            0.30 * da_ratio +          # dominância total
            0.35 * recent_da_ratio +   # dominância recente (mais peso)
            0.20 * recent_sog_ratio +  # eficiência de finalização
            0.15 * self._normalize_acceleration(acc_home, acc_away, home_side=True)
        )

        away_momentum_score = 1.0 - home_momentum_score

        # ── 4. Ajuste por estado do placar ─────────────────────────────────────
        # Time perdendo tende a pressionar mais → aumenta momentum esperado
        score_diff = home_goals - away_goals

        # Fator de pressão: time perdendo recebe boost de 0-20%
        pressure_home = self._compute_score_pressure(score_diff, home_losing=(score_diff < 0))
        pressure_away = self._compute_score_pressure(-score_diff, home_losing=(score_diff > 0))

        # Aplica pressão ao momentum
        home_final = home_momentum_score * (1.0 + pressure_home * 0.20)
        away_final = away_momentum_score * (1.0 + pressure_away * 0.20)

        # Normaliza para que home + away = 1.0
        total = home_final + away_final
        if total > 0:
            home_final /= total
            away_final /= total

        momentum_ratio = home_final / away_final if away_final > 0 else 1.0

        # ── 5. Converte momentum em multiplicador de λ ─────────────────────────
        # Momentum 0.5 (neutro) → multiplicador 1.0
        # Momentum 0.7 (casa dominante) → multiplicador ~1.4
        # Momentum 0.3 (visitante dominante) → multiplicador ~0.6

        lm_home = self._momentum_to_lambda_multiplier(home_final, neutral=0.5)
        lm_away = self._momentum_to_lambda_multiplier(away_final, neutral=0.5)

        # Clamp
        lm_home = max(self.MIN_LAMBDA_MULTIPLIER, min(self.MAX_LAMBDA_MULTIPLIER, lm_home))
        lm_away = max(self.MIN_LAMBDA_MULTIPLIER, min(self.MAX_LAMBDA_MULTIPLIER, lm_away))

        logger.debug(
            f"[Momentum] {match_id} min {minute}: "
            f"home={home_final:.3f} away={away_final:.3f} "
            f"λ_mult h={lm_home:.3f} a={lm_away:.3f}"
        )

        return MomentumResult(
            home_momentum=home_final,
            away_momentum=away_final,
            momentum_ratio=momentum_ratio,
            da_ratio=da_ratio,
            sog_ratio=sog_ratio,
            acceleration_home=acc_home,
            acceleration_away=acc_away,
            score_pressure_home=pressure_home,
            score_pressure_away=pressure_away,
            lambda_multiplier_home=lm_home,
            lambda_multiplier_away=lm_away,
            minute=minute,
            window_size=n_window,
            confidence=confidence,
        )

    def _compute_acceleration(self, values: list[float]) -> float:
        """
        Calcula aceleração (segunda derivada aproximada) da série temporal.

        Aceleração positiva = time está acelerando ofensivamente.
        Usamos diferença entre primeira e segunda metade da janela.
        """
        if len(values) < 4:
            return 0.0

        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid
        second_half_avg = sum(values[mid:]) / (len(values) - mid)

        return second_half_avg - first_half_avg

    def _normalize_acceleration(
        self,
        acc_home: float,
        acc_away: float,
        home_side: bool,
    ) -> float:
        """
        Normaliza aceleração para [0-1] representando dominância.
        0.5 = equilíbrio, > 0.5 = time indicado acelerando mais.
        """
        total_acc = abs(acc_home) + abs(acc_away)
        if total_acc == 0:
            return 0.5

        if home_side:
            return max(0.0, min(1.0, 0.5 + acc_home / (2 * total_acc)))
        else:
            return max(0.0, min(1.0, 0.5 + acc_away / (2 * total_acc)))

    def _compute_score_pressure(
        self,
        score_diff: int,
        home_losing: bool,
    ) -> float:
        """
        Calcula pressão adicional por estado do placar.

        Time perdendo por mais gols → mais pressão → mais ataques.
        Returns: valor entre 0.0 e 1.0
        """
        if not home_losing or score_diff >= 0:
            return 0.0

        # Desvantagem de 1 gol = 20% de pressão
        # Desvantagem de 2+ gols = 40% de pressão
        deficit = abs(score_diff)
        return min(1.0, deficit * 0.20)

    def _momentum_to_lambda_multiplier(
        self,
        momentum: float,
        neutral: float = 0.5,
        scale: float = 1.6,
    ) -> float:
        """
        Converte score de momentum (0-1) em multiplicador de λ.

        Fórmula: multiplier = exp(scale × (momentum - neutral))

        Propriedades:
            momentum = 0.5 → multiplier = 1.0 (neutro)
            momentum = 0.7 → multiplier ≈ 1.38 (+38%)
            momentum = 0.3 → multiplier ≈ 0.73 (-27%)
            momentum = 0.8 → multiplier ≈ 1.61 (+61%)
        """
        return math.exp(scale * (momentum - neutral))

    def calculate_from_snapshot(
        self,
        match_id: str,
        minute: int,
        home_goals: int,
        away_goals: int,
        live_data: dict,
    ) -> Optional[MomentumResult]:
        """
        Interface simplificada para uso com snapshot do SokkerPRO.

        Extrai as estatísticas do snapshot e calcula momentum.

        Args:
            live_data: dict com as chaves do SokkerPRO:
                       dangerous_attacks_home, dangerous_attacks_away,
                       shots_on_goal_home, shots_on_goal_away, etc.
        """
        try:
            home_da = float(live_data.get("dangerous_attacks_home", 0))
            away_da = float(live_data.get("dangerous_attacks_away", 0))
            home_sog = float(live_data.get("shots_on_goal_home", 0))
            away_sog = float(live_data.get("shots_on_goal_away", 0))
            home_corners = float(live_data.get("corners_home", 0))
            away_corners = float(live_data.get("corners_away", 0))

            return self.update_and_calculate(
                match_id=match_id,
                minute=minute,
                home_goals=home_goals,
                away_goals=away_goals,
                home_da_total=home_da,
                away_da_total=away_da,
                home_sog_total=home_sog,
                away_sog_total=away_sog,
                home_corners_total=home_corners,
                away_corners_total=away_corners,
            )

        except Exception as e:
            logger.error(f"[Momentum] Erro ao calcular snapshot {match_id}: {e}")
            return None
