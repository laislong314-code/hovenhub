"""
ProbabilityEngine — Camada de cálculo exclusivo de probabilidades.

Responsabilidade única: receber parâmetros de força e retornar probabilidades.
Não calcula EV, não toma decisões, não envia mensagens.

Faz parte da refatoração da FASE 1 — Modularização da Arquitetura.
"""

import math
from typing import Optional
from scipy.stats import poisson
import numpy as np
from loguru import logger

_MAX_GOALS = 13  # Máximo de gols na matriz (cobre 99.9%+ dos casos reais)


class ProbabilityEngine:
    """
    Calcula probabilidades brutas de mercados de futebol via modelo Poisson.

    Entrada: lambdas (λ_home, λ_away) calculados pelo NormalizationService.
    Saída:   dicionário de probabilidades para cada mercado padrão.

    Separa completamente o cálculo probabilístico do cálculo de valor.
    """

    def __init__(self, use_dixon_coles: bool = True):
        """
        Args:
            use_dixon_coles: Se True, aplica correção Dixon-Coles para placar baixo.
                             Melhora calibração significativa para 0-0, 1-0, 0-1, 1-1.
        """
        self.use_dixon_coles = use_dixon_coles

    def build_score_matrix(
        self,
        lambda_home: float,
        lambda_away: float,
        rho: float = -0.13,
    ) -> np.ndarray:
        """
        Constrói a matriz de probabilidades de placar [home_goals x away_goals].

        Implementa Dixon-Coles (1997) com fator de correlação ρ (rho) para
        corrigir a subestimação de placares baixos pelo Poisson independente.

        Matemática:
            P(i, j) = P_pois(i, λ_h) × P_pois(j, λ_a) × τ(i, j, λ_h, λ_a, ρ)

            τ(0,0) = 1 - λ_h × λ_a × ρ
            τ(1,0) = 1 + λ_a × ρ
            τ(0,1) = 1 + λ_h × ρ
            τ(1,1) = 1 - ρ
            τ(i,j) = 1 para i+j > 2

        Args:
            lambda_home: λ esperado para o time da casa
            lambda_away: λ esperado para o time visitante
            rho: coeficiente de correlação Dixon-Coles (negativo = sub-independência)
                 Valor típico calibrado empiricamente: -0.13 (Dixon & Coles, 1997)

        Returns:
            np.ndarray de shape (_MAX_GOALS, _MAX_GOALS) com probabilidades normalizadas
        """
        lh = max(0.01, lambda_home)
        la = max(0.01, lambda_away)

        matrix = np.zeros((_MAX_GOALS, _MAX_GOALS))

        for i in range(_MAX_GOALS):
            for j in range(_MAX_GOALS):
                p_base = poisson.pmf(i, lh) * poisson.pmf(j, la)

                if self.use_dixon_coles:
                    tau = self._dixon_coles_tau(i, j, lh, la, rho)
                    p_base *= tau

                matrix[i, j] = max(0.0, p_base)

        # Normaliza para garantir que a soma seja 1.0 (correção DC pode distorcer levemente)
        total = matrix.sum()
        if total > 0:
            matrix /= total

        return matrix

    def _dixon_coles_tau(
        self,
        i: int,
        j: int,
        lh: float,
        la: float,
        rho: float,
    ) -> float:
        """
        Fator de correção τ de Dixon-Coles para baixa pontuação.
        Só afeta placares com total ≤ 2.
        """
        if i == 0 and j == 0:
            return 1.0 - lh * la * rho
        elif i == 1 and j == 0:
            return 1.0 + la * rho
        elif i == 0 and j == 1:
            return 1.0 + lh * rho
        elif i == 1 and j == 1:
            return 1.0 - rho
        else:
            return 1.0

    def calculate_all_markets(
        self,
        lambda_home: float,
        lambda_away: float,
        rho: float = -0.13,
    ) -> dict:
        """
        Calcula probabilidades para todos os mercados padrão.

        Args:
            lambda_home: λ esperado casa
            lambda_away: λ esperado visitante
            rho: coeficiente Dixon-Coles

        Returns:
            dict com probabilidades para cada mercado:
            {
                "home_win": float,
                "draw": float,
                "away_win": float,
                "over_0_5": float,
                "over_1_5": float,
                "over_2_5": float,
                "over_3_5": float,
                "over_4_5": float,
                "under_0_5": float,
                "under_1_5": float,
                "under_2_5": float,
                "under_3_5": float,
                "under_4_5": float,
                "btts_yes": float,
                "btts_no": float,
                "lambda_home": float,
                "lambda_away": float,
                "lambda_total": float,
            }
        """
        matrix = self.build_score_matrix(lambda_home, lambda_away, rho)

        probs = {
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "lambda_total": lambda_home + lambda_away,
        }

        # 1x2
        probs["home_win"] = float(np.sum(
            [matrix[i, j] for i in range(_MAX_GOALS) for j in range(_MAX_GOALS) if i > j]
        ))
        probs["draw"] = float(np.sum(
            [matrix[i, j] for i in range(_MAX_GOALS) for j in range(_MAX_GOALS) if i == j]
        ))
        probs["away_win"] = float(np.sum(
            [matrix[i, j] for i in range(_MAX_GOALS) for j in range(_MAX_GOALS) if i < j]
        ))

        # Total de gols
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            key = str(line).replace(".", "_")
            over = float(np.sum(
                [matrix[i, j] for i in range(_MAX_GOALS) for j in range(_MAX_GOALS) if (i + j) > line]
            ))
            probs[f"over_{key}"] = over
            probs[f"under_{key}"] = 1.0 - over

        # BTTS (ambos marcam)
        probs["btts_yes"] = float(np.sum(
            [matrix[i, j] for i in range(1, _MAX_GOALS) for j in range(1, _MAX_GOALS)]
        ))
        probs["btts_no"] = 1.0 - probs["btts_yes"]

        # Dupla hipótese (1X, X2, 12)
        probs["home_or_draw"] = probs["home_win"] + probs["draw"]
        probs["away_or_draw"] = probs["away_win"] + probs["draw"]
        probs["home_or_away"] = probs["home_win"] + probs["away_win"]

        return probs

    def calculate_asian_handicap(
        self,
        matrix: np.ndarray,
        handicap: float,
    ) -> dict:
        """
        Calcula probabilidade para Asian Handicap.

        AH -0.5 casa: casa vence por 1+ gols
        AH -1.0 casa: casa vence por 2+ (push em 1 gol)
        AH -1.5 casa: casa vence por 2+ gols
        """
        prob_home_ah = 0.0
        prob_away_ah = 0.0
        prob_push = 0.0

        for i in range(_MAX_GOALS):
            for j in range(_MAX_GOALS):
                adjusted = (i - j) + handicap  # perspectiva do casa
                p = matrix[i, j]
                if adjusted > 0:
                    prob_home_ah += p
                elif adjusted == 0:
                    prob_push += p
                else:
                    prob_away_ah += p

        return {
            "home_ah": prob_home_ah,
            "away_ah": prob_away_ah,
            "push": prob_push,
        }

    def calculate_correct_score(
        self,
        matrix: np.ndarray,
        home_goals: int,
        away_goals: int,
    ) -> float:
        """Probabilidade de placar exato."""
        if home_goals >= _MAX_GOALS or away_goals >= _MAX_GOALS:
            return 0.0
        return float(matrix[home_goals, away_goals])

    def get_expected_goals(self, lambda_home: float, lambda_away: float) -> dict:
        """Retorna estatísticas de gols esperados."""
        return {
            "xg_home": lambda_home,
            "xg_away": lambda_away,
            "xg_total": lambda_home + lambda_away,
            "xg_diff": lambda_home - lambda_away,
        }
