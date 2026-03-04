"""
Motor Estatístico — Modelo de Distribuição de Poisson

═══════════════════════════════════════════════════════════════════════════════
FUNDAMENTOS MATEMÁTICOS
═══════════════════════════════════════════════════════════════════════════════

1. DISTRIBUIÇÃO DE POISSON
   P(X = k) = (λᵏ × e⁻λ) / k!
   
   Onde λ é o número esperado de eventos (gols) em um período fixo (90 min).
   Assumimos que gols de casa e visitante são processos independentes de Poisson.

2. CÁLCULO DE λ (LAMBDA)
   
   Primeiro calculamos força ofensiva e defensiva relativas:
   
   attack_strength(time)  = avg_scored_time  / avg_scored_liga
   defense_weakness(time) = avg_conceded_time / avg_conceded_liga
   
   Então:
   λ_home = attack_strength(home) × defense_weakness(away) × avg_scored_liga_home
   λ_away = attack_strength(away) × defense_weakness(home) × avg_scored_liga_away
   
   Nota: avg_scored_liga_home ≠ avg_scored_liga_away pois times em casa marcam mais.
   
   Exemplo numérico (Premier League):
   - Liga: 1.55 gols/jogo casa, 1.15 gols/jogo visitante
   - Arsenal (casa): marca 1.8, sofre 0.9
   - Chelsea (fora):  marca 1.4, sofre 1.1
   
   attack_strength(Arsenal)  = 1.8 / 1.55 = 1.161
   defense_weakness(Chelsea) = 1.1 / 1.15 = 0.957
   λ_home = 1.161 × 0.957 × 1.55 = 1.72 gols esperados
   
   attack_strength(Chelsea)  = 1.4 / 1.15 = 1.217
   defense_weakness(Arsenal) = 0.9 / 1.55 = 0.581
   λ_away = 1.217 × 0.581 × 1.15 = 0.81 gols esperados
   
   λ_total = 1.72 + 0.81 = 2.53

3. PROBABILIDADE DE OVER 2.5
   
   P(Over 2.5) = 1 - P(0 gols) - P(1 gol) - P(2 gols)
   
   Como casa e visitante são independentes:
   P(total = n) = Σ P(casa=i) × P(visita=j) para todo i+j=n
   
   Calculamos a matrix de placar e somamos a diagonal inferior.

4. EXPECTED VALUE (EV)
   
   EV = (p_modelo × odd_decimal) - 1
   
   Interpretação:
   EV =  0.05 → a cada R$1 apostado, esperamos ganhar R$0.05 a longo prazo
   EV =  0.00 → aposta justa (sem edge)
   EV = -0.05 → perdemos R$0.05 em média (casa tem edge)

5. CRITÉRIO DE KELLY
   
   f* = (p × b - q) / b
   Onde:
   - p = probabilidade do modelo
   - q = 1 - p
   - b = odd - 1 (lucro por unidade)
   
   Usamos Kelly Fracionário: f = f* × fração (0.25 = quarter Kelly)
   Isso reduz volatilidade mantendo crescimento de longo prazo.

═══════════════════════════════════════════════════════════════════════════════
"""

import math
from typing import Optional
from scipy.stats import poisson
import numpy as np
from loguru import logger

from app.config import get_settings
from app.models.schemas import TeamForm, PoissonAnalysis, ValueBetAnalysis

settings = get_settings()


class PoissonModel:
    """
    Implementação do modelo de Poisson para previsão de gols.
    
    Limitações conhecidas (documentadas para evolução futura):
    - Não captura dependência entre os gols (Dixon-Coles corrige isso)
    - Não modela overtime ou pênaltis
    - Não considera situações do jogo (redução, pênalti, etc.)
    - Forma dos times é simplificada (sem peso temporal)
    """

    # Média global padrão caso não tenhamos dados históricos suficientes
    DEFAULT_LEAGUE_AVG = 2.5
    DEFAULT_HOME_AVG = 1.4
    DEFAULT_AWAY_AVG = 1.1

    def calculate_lambda(
        self,
        avg_scored: float,
        opp_avg_conceded: float,
        league_avg_scored: float,
    ) -> float:
        """
        Calcula λ esperado para um time.
        
        λ = (avg_scored / league_avg) × (opp_conceded / league_avg) × league_avg
        
        Simplifica para: (avg_scored × opp_conceded) / league_avg
        """
        if league_avg_scored <= 0:
            return avg_scored
        
        attack_strength = avg_scored / league_avg_scored
        defense_weakness = opp_avg_conceded / league_avg_scored
        
        lambda_val = attack_strength * defense_weakness * league_avg_scored
        
        # Clampar para evitar valores absurdos
        return max(0.1, min(lambda_val, 8.0))

    def prob_over_25_from_lambdas(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 10,
    ) -> float:
        """
        Calcula P(total_gols > 2.5) usando matrix de placar.
        
        Considera todos os placares de 0-0 até max_goals × max_goals.
        Soma P(home=i, away=j) para todos i+j <= 2, subtrai de 1.
        
        Args:
            lambda_home: λ do time da casa
            lambda_away: λ do time visitante
            max_goals: teto de gols por time (10 cobre 99.9%+ dos casos)
        
        Returns:
            Probabilidade de Over 2.5 (0.0 a 1.0)
        """
        prob_under_or_equal = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i + j <= 2:
                    p = (
                        poisson.pmf(i, lambda_home) *
                        poisson.pmf(j, lambda_away)
                    )
                    prob_under_or_equal += p

        return 1.0 - prob_under_or_equal

    def prob_btts(
        self,
        lambda_home: float,
        lambda_away: float,
    ) -> float:
        """
        Calcula P(Ambas Marcam) = P(home >= 1) × P(away >= 1)
        
        P(time marca) = 1 - P(X=0) = 1 - e^(-λ)
        """
        p_home_scores = 1 - poisson.pmf(0, lambda_home)
        p_away_scores = 1 - poisson.pmf(0, lambda_away)
        return p_home_scores * p_away_scores

    def prob_exact_result(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 10,
    ) -> dict[str, float]:
        """
        Retorna probabilidades de resultado: Home / Draw / Away
        Útil para análise futura de mercado 1X2.
        """
        p_home_win = p_draw = p_away_win = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                if i > j:
                    p_home_win += p
                elif i == j:
                    p_draw += p
                else:
                    p_away_win += p

        return {
            "home": p_home_win,
            "draw": p_draw,
            "away": p_away_win,
        }

    def calculate_ev(self, model_prob: float, odd: float) -> float:
        """
        EV = (p × odd) - 1
        
        Representa o retorno esperado por unidade apostada.
        EV > 0 = vantagem sobre a casa.
        """
        return (model_prob * odd) - 1.0

    def calculate_kelly_stake(
        self,
        model_prob: float,
        odd: float,
        fraction: Optional[float] = None,
    ) -> float:
        """
        Critério de Kelly Fracionário.
        
        f* = (p × b - q) / b  onde b = odd - 1
        f_adjusted = f* × fraction
        
        Retorna porcentagem do bankroll a apostar (0.0 a max_stake_pct).
        """
        fraction = fraction or settings.kelly_fraction
        b = odd - 1.0
        q = 1.0 - model_prob

        if b <= 0:
            return 0.0

        kelly_full = (model_prob * b - q) / b

        if kelly_full <= 0:
            return 0.0

        kelly_adjusted = kelly_full * fraction

        # Cap pela stake máxima configurada
        return min(kelly_adjusted, settings.max_stake_pct)

    def analyze_match(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        league: str,
        commence_time,
        home_form: TeamForm,
        away_form: TeamForm,
        league_home_avg: float,
        league_away_avg: float,
    ) -> PoissonAnalysis:
        """
        Análise completa de Poisson para um jogo.
        
        Args:
            home_form: estatísticas de forma do time da casa
            away_form: estatísticas de forma do visitante
            league_home_avg: média de gols marcados em casa na liga
            league_away_avg: média de gols marcados fora na liga
        """
        # λ do time da casa (ofensiva casa vs defensiva visitante)
        lambda_home = self.calculate_lambda(
            avg_scored=home_form.avg_scored,
            opp_avg_conceded=away_form.avg_conceded,
            league_avg_scored=league_home_avg,
        )

        # λ do visitante (ofensiva fora vs defensiva em casa)
        lambda_away = self.calculate_lambda(
            avg_scored=away_form.avg_scored,
            opp_avg_conceded=home_form.avg_conceded,
            league_avg_scored=league_away_avg,
        )

        lambda_total = lambda_home + lambda_away

        prob_over = self.prob_over_25_from_lambdas(lambda_home, lambda_away)
        prob_under = 1.0 - prob_over

        logger.debug(
            f"🔢 Poisson | {home_team} vs {away_team} | "
            f"λ_home={lambda_home:.2f} λ_away={lambda_away:.2f} "
            f"λ_total={lambda_total:.2f} | P(Over 2.5)={prob_over:.1%}"
        )

        return PoissonAnalysis(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            commence_time=commence_time,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            lambda_total=lambda_total,
            prob_over_25=prob_over,
            prob_under_25=prob_under,
            home_form=home_form,
            away_form=away_form,
            league_avg=league_home_avg + league_away_avg,
        )

    def check_value_bet(
        self,
        analysis: PoissonAnalysis,
        best_odd: float,
        best_bookmaker: str,
        market: str = "over_2.5",
        bankroll: float = None,
    ) -> ValueBetAnalysis:
        """
        Verifica se existe value bet e calcula stake sugerida.
        
        Filtros aplicados:
        1. model_prob > implied_prob (edge positivo)
        2. EV > MIN_EV_THRESHOLD
        3. odd entre MIN_ODD e MAX_ODD
        4. model_prob > MIN_MODEL_PROB
        """
        bankroll = bankroll or settings.default_bankroll

        if market == "over_2.5":
            model_prob = analysis.prob_over_25
        elif market == "under_2.5":
            model_prob = analysis.prob_under_25
        else:
            raise ValueError(f"Mercado não suportado: {market}")

        implied_prob = 1.0 / best_odd
        ev = self.calculate_ev(model_prob, best_odd)
        kelly_pct = self.calculate_kelly_stake(model_prob, best_odd)
        suggested_stake = bankroll * kelly_pct

        # Aplicar filtros
        is_value = (
            model_prob > implied_prob and
            ev >= settings.min_ev_threshold and
            settings.min_odd <= best_odd <= settings.max_odd and
            model_prob >= settings.min_model_prob
        )


        if not is_value:
            logger.debug(
                f"[SKIP] {analysis.home_team} vs {analysis.away_team} | "
                f"market={market} | prob={model_prob:.3f} implied={implied_prob:.3f} "
                f"odd={best_odd:.2f} ev={ev:.3f} | "
                f"min_ev={settings.min_ev_threshold} min_prob={settings.min_model_prob} "
                f"odd_range=[{settings.min_odd},{settings.max_odd}]"
            )
        if is_value:
            logger.info(
                f"💰 VALUE BET ENCONTRADO | {analysis.home_team} vs {analysis.away_team} | "
                f"{market} @ {best_odd} | P_modelo={model_prob:.1%} | "
                f"P_implícita={implied_prob:.1%} | EV={ev:.1%}"
            )

        return ValueBetAnalysis(
            analysis=analysis,
            market=market,
            model_prob=model_prob,
            best_odd=best_odd,
            best_bookmaker=best_bookmaker,
            implied_prob=implied_prob,
            ev=ev,
            is_value_bet=is_value,
            kelly_stake_pct=kelly_pct,
            suggested_stake=round(suggested_stake, 2),
        )
