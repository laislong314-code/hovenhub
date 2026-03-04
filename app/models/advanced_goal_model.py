"""
AdvancedGoalModel — Modelo de gols avançado com Bivariate Poisson e Dixon-Coles.

Substitui o modelo Poisson simples por:
  1. Bivariate Poisson (Lee, 1999): modela correlação explícita entre gols casa/visitante
  2. Dixon-Coles (1997): correção para placares baixos (0-0, 1-0, 0-1, 1-1)
  3. Regressão à média: suaviza estimativas com poucos dados

Matemática do Bivariate Poisson:
    X = X1 + X3
    Y = X2 + X3
    onde X1 ~ Poisson(λ1), X2 ~ Poisson(λ2), X3 ~ Poisson(λ3)

    P(X=x, Y=y) = e^-(λ1+λ2+λ3) × (λ1^x/x!) × (λ2^y/y!) × Σ_{k=0}^{min(x,y)} C(x,k)C(y,k)k!(λ3/(λ1λ2))^k

    λ3 (covariance term) é estimado empiricamente ≈ 0.1 para futebol

Comparação vs Poisson Independente:
    Poisson Ind: P(X=x, Y=y) = P(X=x) × P(Y=y)   → ignora correlação
    Bivariate:   P(X=x, Y=y) considera que gols não são totalmente independentes
                 (pênaltis, situações de jogo afetam ambos os times)

Faz parte da FASE 4 — Melhorar Modelo Poisson.
"""

import math
from typing import Optional
import numpy as np
from scipy.stats import poisson
from scipy.special import comb
from loguru import logger


_MAX_GOALS = 13


class AdvancedGoalModel:
    """
    Modelo avançado de gols com suporte a:
      - Bivariate Poisson (correlação entre gols)
      - Dixon-Coles correction (baixa pontuação)
      - Fallback automático para Poisson independente

    Uso:
        model = AdvancedGoalModel()
        matrix = model.build_matrix(lambda_home=1.72, lambda_away=0.81)
        prob_over25 = model.prob_over(matrix, 2.5)
    """

    def __init__(
        self,
        use_bivariate: bool = True,
        use_dixon_coles: bool = True,
        lambda3: float = 0.10,  # covariance term do Bivariate Poisson
        rho: float = -0.13,     # coeficiente Dixon-Coles
    ):
        """
        Args:
            use_bivariate: Se True, usa Bivariate Poisson (mais preciso, mais lento)
            use_dixon_coles: Se True, aplica correção DC para baixa pontuação
            lambda3: Parâmetro de covariância do Bivariate Poisson
                     Valor típico para futebol: 0.08 a 0.15
            rho: Coeficiente de correlação Dixon-Coles
                 Valor típico calibrado: -0.13 (Dixon & Coles, 1997)
        """
        self.use_bivariate = use_bivariate
        self.use_dixon_coles = use_dixon_coles
        self.lambda3 = lambda3
        self.rho = rho

    def bivariate_poisson_pmf(
        self,
        x: int,
        y: int,
        lambda1: float,
        lambda2: float,
        lambda3: float,
    ) -> float:
        """
        PMF da Bivariate Poisson para o placar (x, y).

        P(X=x, Y=y) = e^-(λ1+λ2+λ3) × (λ1^x/x!) × (λ2^y/y!)
                     × Σ_{k=0}^{min(x,y)} C(x,k) × C(y,k) × k! × (λ3/(λ1×λ2))^k

        Args:
            x: gols do time da casa
            y: gols do time visitante
            lambda1: λ exclusivo do time casa (λ_home - λ3)
            lambda2: λ exclusivo do time visitante (λ_away - λ3)
            lambda3: λ compartilhado (covariância)

        Returns:
            probabilidade P(X=x, Y=y)
        """
        if lambda1 <= 0 or lambda2 <= 0 or lambda3 < 0:
            # Fallback para Poisson independente
            return float(poisson.pmf(x, lambda1 + lambda3) * poisson.pmf(y, lambda2 + lambda3))

        try:
            min_k = min(x, y)
            exp_term = math.exp(-(lambda1 + lambda2 + lambda3))
            pow_l1 = lambda1 ** x / math.factorial(x)
            pow_l2 = lambda2 ** y / math.factorial(y)

            # Série de Bessel (soma sobre k)
            bessel_sum = 0.0
            ratio = lambda3 / (lambda1 * lambda2)

            for k in range(min_k + 1):
                term = (
                    float(comb(x, k, exact=True)) *
                    float(comb(y, k, exact=True)) *
                    math.factorial(k) *
                    (ratio ** k)
                )
                bessel_sum += term

            result = exp_term * pow_l1 * pow_l2 * bessel_sum
            return max(0.0, result)

        except (OverflowError, ZeroDivisionError, ValueError):
            # Fallback seguro
            lh = lambda1 + lambda3
            la = lambda2 + lambda3
            return float(poisson.pmf(x, lh) * poisson.pmf(y, la))

    def dixon_coles_tau(
        self,
        x: int,
        y: int,
        lh: float,
        la: float,
    ) -> float:
        """
        Fator de correção τ de Dixon-Coles para placares baixos.

        Melhora a estimativa de P(0-0), P(1-0), P(0-1) e P(1-1) que o
        Poisson tende a subestimar ou superestimar sistematicamente.

        τ(0,0) = 1 - λh × λa × ρ
        τ(1,0) = 1 + λa × ρ
        τ(0,1) = 1 + λh × ρ
        τ(1,1) = 1 - ρ
        τ(x,y) = 1 para x+y > 2
        """
        if x == 0 and y == 0:
            return 1.0 - lh * la * self.rho
        elif x == 1 and y == 0:
            return 1.0 + la * self.rho
        elif x == 0 and y == 1:
            return 1.0 + lh * self.rho
        elif x == 1 and y == 1:
            return 1.0 - self.rho
        else:
            return 1.0

    def build_matrix(
        self,
        lambda_home: float,
        lambda_away: float,
    ) -> np.ndarray:
        """
        Constrói a matriz de probabilidades de placar.

        Usa Bivariate Poisson se habilitado, com fallback automático.
        Aplica correção Dixon-Coles se habilitado.

        Args:
            lambda_home: λ esperado para o time da casa (gols esperados)
            lambda_away: λ esperado para o time visitante

        Returns:
            np.ndarray[_MAX_GOALS, _MAX_GOALS] com probabilidades normalizadas
        """
        lh = max(0.01, float(lambda_home))
        la = max(0.01, float(lambda_away))

        matrix = np.zeros((_MAX_GOALS, _MAX_GOALS))

        # Para Bivariate Poisson, decompõe os lambdas
        # λ1 = λ_home - λ3, λ2 = λ_away - λ3
        l3 = min(self.lambda3, min(lh, la) * 0.5)  # λ3 não pode ser maior que λ1 ou λ2
        l1 = lh - l3
        l2 = la - l3

        for i in range(_MAX_GOALS):
            for j in range(_MAX_GOALS):
                if self.use_bivariate and l1 > 0 and l2 > 0:
                    p = self.bivariate_poisson_pmf(i, j, l1, l2, l3)
                else:
                    # Poisson independente (fallback)
                    p = float(poisson.pmf(i, lh) * poisson.pmf(j, la))

                # Aplica correção Dixon-Coles
                if self.use_dixon_coles:
                    tau = self.dixon_coles_tau(i, j, lh, la)
                    p = max(0.0, p * tau)

                matrix[i, j] = p

        # Normaliza
        total = matrix.sum()
        if total > 0:
            matrix /= total

        return matrix

    def expected_goals(
        self,
        matrix: np.ndarray,
    ) -> tuple[float, float]:
        """
        Calcula gols esperados a partir da matriz (não os lambdas brutos).
        Mais preciso porque já incorpora as correções DC/BP.
        """
        xg_home = 0.0
        xg_away = 0.0
        for i in range(_MAX_GOALS):
            for j in range(_MAX_GOALS):
                p = matrix[i, j]
                xg_home += i * p
                xg_away += j * p
        return xg_home, xg_away

    def prob_exact_score(self, matrix: np.ndarray, home: int, away: int) -> float:
        """P(placar exato = home:away)"""
        if home >= _MAX_GOALS or away >= _MAX_GOALS:
            return 0.0
        return float(matrix[home, away])

    def prob_over(self, matrix: np.ndarray, line: float) -> float:
        """P(total de gols > line)"""
        total = 0.0
        for i in range(_MAX_GOALS):
            for j in range(_MAX_GOALS):
                if (i + j) > line:
                    total += matrix[i, j]
        return float(total)

    def prob_under(self, matrix: np.ndarray, line: float) -> float:
        """P(total de gols < line)"""
        return 1.0 - self.prob_over(matrix, line)

    def prob_home_win(self, matrix: np.ndarray) -> float:
        """P(home_goals > away_goals)"""
        return float(np.sum([matrix[i, j] for i in range(_MAX_GOALS) for j in range(_MAX_GOALS) if i > j]))

    def prob_draw(self, matrix: np.ndarray) -> float:
        """P(home_goals == away_goals)"""
        return float(np.sum([matrix[i, i] for i in range(_MAX_GOALS)]))

    def prob_away_win(self, matrix: np.ndarray) -> float:
        """P(away_goals > home_goals)"""
        return float(np.sum([matrix[i, j] for i in range(_MAX_GOALS) for j in range(_MAX_GOALS) if j > i]))

    def prob_btts(self, matrix: np.ndarray) -> float:
        """P(ambos os times marcam)"""
        return float(np.sum([matrix[i, j] for i in range(1, _MAX_GOALS) for j in range(1, _MAX_GOALS)]))

    def all_probs(
        self,
        lambda_home: float,
        lambda_away: float,
    ) -> dict:
        """
        Calcula todas as probabilidades de mercado em uma única chamada.

        Mais eficiente que chamar cada método separadamente pois
        constrói a matriz apenas uma vez.
        """
        matrix = self.build_matrix(lambda_home, lambda_away)

        probs = {
            "home_win": self.prob_home_win(matrix),
            "draw": self.prob_draw(matrix),
            "away_win": self.prob_away_win(matrix),
            "btts_yes": self.prob_btts(matrix),
            "btts_no": 1.0 - self.prob_btts(matrix),
        }

        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            key = str(line).replace(".", "_")
            probs[f"over_{key}"] = self.prob_over(matrix, line)
            probs[f"under_{key}"] = 1.0 - probs[f"over_{key}"]

        probs["home_or_draw"] = probs["home_win"] + probs["draw"]
        probs["away_or_draw"] = probs["away_win"] + probs["draw"]

        # Gols esperados reais (pós-correção)
        xg_h, xg_a = self.expected_goals(matrix)
        probs["xg_home"] = xg_h
        probs["xg_away"] = xg_a
        probs["xg_total"] = xg_h + xg_a
        probs["lambda_home"] = lambda_home
        probs["lambda_away"] = lambda_away
        probs["lambda_total"] = lambda_home + lambda_away

        return probs

    @staticmethod
    def regression_to_mean(
        team_value: float,
        league_mean: float,
        weight: float = 0.30,
        min_games: int = 5,
        actual_games: int = 10,
    ) -> float:
        """
        Aplica regressão à média para suavizar estimativas com poucos dados.

        Quanto menos jogos, mais puxamos em direção à média da liga.

        Fórmula:
            r = weight × (1 - games/threshold)    se games < threshold
            adjusted = r × league_mean + (1-r) × team_value

        Args:
            team_value: valor observado do time
            league_mean: média da liga para aquele atributo
            weight: peso máximo de regressão (0.30 = 30% de regressão máxima)
            min_games: mínimo de jogos para plena confiança
            actual_games: jogos reais disponíveis

        Returns:
            valor ajustado com regressão à média
        """
        if actual_games >= min_games:
            # Dados suficientes: regressão mínima
            regression = weight * 0.15
        else:
            # Poucos dados: regressão maior
            data_fraction = actual_games / min_games
            regression = weight * (1.0 - data_fraction)

        adjusted = regression * league_mean + (1.0 - regression) * team_value
        return adjusted

    @staticmethod
    def exponential_weighted_average(
        values: list[float],
        decay: float = 0.85,
    ) -> float:
        """
        Média com peso exponencial decrescente.

        O jogo mais recente tem peso 1.0, o anterior tem peso 0.85,
        o anterior 0.85², etc.

        Args:
            values: lista de valores em ordem cronológica (mais antigo primeiro)
            decay: fator de decaimento (0 < decay < 1)
                   0.85: decay rápido (últimos 5 jogos dominam)
                   0.95: decay lento (toda a temporada tem peso)

        Returns:
            média ponderada exponencialmente
        """
        if not values:
            return 0.0

        n = len(values)
        weights = [decay ** (n - 1 - i) for i in range(n)]
        total_weight = sum(weights)

        if total_weight == 0:
            return sum(values) / n

        return sum(v * w for v, w in zip(values, weights)) / total_weight
