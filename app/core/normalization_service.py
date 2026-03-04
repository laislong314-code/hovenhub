"""
NormalizationService — Normalização estatística de médias por liga.

Problema que resolve:
    Uma média de 2.5 gols/jogo na Premier League (alta pontuação) tem significado
    diferente de 2.5 gols/jogo na Serie A italiana (baixa pontuação).
    Usar médias brutas gera distorções no modelo Poisson.

Solução implementada:
    1. Z-Score por liga: normaliza cada valor em relação à média e desvio padrão da liga
    2. Ajuste de força ofensiva relativa: attack_strength = avg_scored / liga_avg_scored
    3. Ajuste de força defensiva relativa: defense_weakness = avg_conceded / liga_avg_conceded
    4. Regressão à média: suaviza com poucos jogos
    5. Ajuste por forma recente: peso exponencial nos últimos jogos

Faz parte da FASE 5 — Normalização Estatística.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class LeagueNorms:
    """Parâmetros de normalização para uma liga."""
    league_id: str
    avg_goals_home: float = 1.50    # média de gols em casa na liga
    avg_goals_away: float = 1.15    # média de gols fora na liga
    avg_goals_total: float = 2.65   # média total
    std_goals_home: float = 0.40    # desvio padrão
    std_goals_away: float = 0.35
    avg_corners: float = 10.0       # média de escanteios
    avg_yellow_cards: float = 3.5   # média de cartões amarelos
    n_games: int = 0                # jogos usados para calcular (0 = default)


@dataclass
class NormalizedForm:
    """Forma normalizada de um time para entrada no modelo."""
    # Lambdas normalizados (prontos para Poisson)
    lambda_home: float  # se jogar em casa
    lambda_away: float  # se jogar fora

    # Forças relativas (1.0 = média da liga)
    attack_strength: float
    defense_weakness: float

    # Z-scores (para comparação entre ligas)
    zscore_attack: float
    zscore_defense: float

    # Confiança nos dados (0-1, baseado em quantidade de jogos)
    data_confidence: float

    # Número de jogos usados
    n_games: int


class NormalizationService:
    """
    Normaliza estatísticas de times por liga para eliminar viés de liga forte/fraca.

    Todos os lambdas que entram no modelo Poisson devem passar por aqui.

    Exemplo de uso:
        service = NormalizationService()

        # Registra médias da liga (chamado quando dados chegam)
        service.set_league_norms("39", LeagueNorms(
            league_id="39",
            avg_goals_home=1.55,
            avg_goals_away=1.18,
            ...
        ))

        # Calcula lambda normalizado para um time
        form = service.normalize_team_form(
            league_id="39",
            avg_scored=1.8,
            avg_conceded=0.9,
            is_home=True,
        )
        print(f"λ_home = {form.lambda_home:.3f}")
    """

    # Médias globais padrão (fallback quando não há dados da liga)
    GLOBAL_DEFAULTS = LeagueNorms(
        league_id="global",
        avg_goals_home=1.45,
        avg_goals_away=1.10,
        avg_goals_total=2.55,
        std_goals_home=0.45,
        std_goals_away=0.38,
        avg_corners=10.0,
        avg_yellow_cards=3.5,
    )

    # Médias por liga (baseadas em dados históricos reais)
    LEAGUE_DEFAULTS: dict[str, dict] = {
        # Premier League (id: 39)
        "39": {"avg_home": 1.55, "avg_away": 1.18, "std_home": 0.48, "std_away": 0.41},
        # La Liga (id: 140)
        "140": {"avg_home": 1.52, "avg_away": 1.12, "std_home": 0.44, "std_away": 0.38},
        # Bundesliga (id: 78)
        "78": {"avg_home": 1.65, "avg_away": 1.28, "std_home": 0.52, "std_away": 0.44},
        # Serie A (id: 135)
        "135": {"avg_home": 1.42, "avg_away": 1.05, "std_home": 0.40, "std_away": 0.35},
        # Ligue 1 (id: 61)
        "61": {"avg_home": 1.48, "avg_away": 1.10, "std_home": 0.45, "std_away": 0.38},
        # Champions League (id: 2)
        "2": {"avg_home": 1.70, "avg_away": 1.35, "std_home": 0.55, "std_away": 0.50},
        # Brasileirao Serie A (id: 71)
        "71": {"avg_home": 1.45, "avg_away": 1.05, "std_home": 0.42, "std_away": 0.35},
    }

    def __init__(self):
        self._league_norms: dict[str, LeagueNorms] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Carrega defaults históricos por liga."""
        for league_id, data in self.LEAGUE_DEFAULTS.items():
            self._league_norms[league_id] = LeagueNorms(
                league_id=league_id,
                avg_goals_home=data["avg_home"],
                avg_goals_away=data["avg_away"],
                avg_goals_total=data["avg_home"] + data["avg_away"],
                std_goals_home=data["std_home"],
                std_goals_away=data["std_away"],
            )

    def set_league_norms(self, league_id: str, norms: LeagueNorms):
        """Atualiza as normas de uma liga com dados reais."""
        self._league_norms[str(league_id)] = norms
        logger.debug(f"[Normalization] Normas da liga {league_id} atualizadas: "
                     f"μ_home={norms.avg_goals_home:.3f}, μ_away={norms.avg_goals_away:.3f}")

    def get_league_norms(self, league_id: str) -> LeagueNorms:
        """Retorna normas da liga ou defaults globais."""
        return self._league_norms.get(str(league_id), self.GLOBAL_DEFAULTS)

    def compute_attack_strength(
        self,
        avg_scored: float,
        league_avg: float,
        n_games: int = 10,
    ) -> float:
        """
        Força ofensiva relativa: attack_strength = avg_scored / league_avg

        Interpretação:
            > 1.0: time marca mais que a média da liga
            = 1.0: time marca exatamente a média
            < 1.0: time marca menos que a média

        Aplica regressão à média se n_games < 10.
        """
        if league_avg <= 0:
            return 1.0

        raw_strength = avg_scored / league_avg

        # Regressão à média com poucos dados
        if n_games < 10:
            regression_weight = 0.3 * (1.0 - n_games / 10.0)
            raw_strength = (1.0 - regression_weight) * raw_strength + regression_weight * 1.0

        return max(0.1, min(3.0, raw_strength))

    def compute_defense_weakness(
        self,
        avg_conceded: float,
        league_avg: float,
        n_games: int = 10,
    ) -> float:
        """
        Fraqueza defensiva relativa: defense_weakness = avg_conceded / league_avg

        Interpretação:
            > 1.0: time sofre mais que a média da liga (defesa fraca)
            = 1.0: defesa média
            < 1.0: defesa acima da média (sofre menos)

        Aplica regressão à média se n_games < 10.
        """
        if league_avg <= 0:
            return 1.0

        raw_weakness = avg_conceded / league_avg

        if n_games < 10:
            regression_weight = 0.3 * (1.0 - n_games / 10.0)
            raw_weakness = (1.0 - regression_weight) * raw_weakness + regression_weight * 1.0

        return max(0.1, min(3.0, raw_weakness))

    def compute_lambda(
        self,
        attack_strength: float,
        defense_weakness: float,
        league_avg: float,
    ) -> float:
        """
        Calcula λ (gols esperados) para um time.

        λ = attack_strength × defense_weakness × league_avg

        Este é o λ que entra no modelo Poisson.
        """
        lam = attack_strength * defense_weakness * league_avg
        return max(0.01, min(8.0, lam))

    def compute_zscore(
        self,
        value: float,
        mean: float,
        std: float,
    ) -> float:
        """
        Z-Score: (valor - média) / desvio_padrão

        Permite comparar times de ligas diferentes.
        Z > 2: extremamente acima da média (99th percentile)
        Z = 0: exatamente na média
        Z < -2: extremamente abaixo da média
        """
        if std <= 0:
            return 0.0
        return (value - mean) / std

    def normalize_team_form(
        self,
        league_id: str,
        avg_scored: float,
        avg_conceded: float,
        is_home: bool,
        n_games: int = 10,
        recent_avg_scored: Optional[float] = None,
        recent_avg_conceded: Optional[float] = None,
        form_weight_recent: float = 0.35,
    ) -> NormalizedForm:
        """
        Normaliza a forma de um time para uso no modelo Poisson.

        Aplica toda a pipeline de normalização:
        1. Média ponderada (temporada + forma recente)
        2. Ajuste por força relativa à liga
        3. Z-score para auditoria
        4. Regressão à média com poucos dados

        Args:
            league_id: ID da liga
            avg_scored: média de gols marcados na temporada
            avg_conceded: média de gols sofridos na temporada
            is_home: True se este time joga em casa
            n_games: número de jogos disputados
            recent_avg_scored: média dos últimos 5 jogos (None = ignorar)
            recent_avg_conceded: média dos últimos 5 jogos (None = ignorar)
            form_weight_recent: peso dado à forma recente (0.35 = 35%)

        Returns:
            NormalizedForm com λ pronto para o modelo
        """
        norms = self.get_league_norms(league_id)

        # 1. Combina média da temporada com forma recente
        if recent_avg_scored is not None and n_games >= 3:
            eff_scored = (
                (1 - form_weight_recent) * avg_scored +
                form_weight_recent * recent_avg_scored
            )
        else:
            eff_scored = avg_scored

        if recent_avg_conceded is not None and n_games >= 3:
            eff_conceded = (
                (1 - form_weight_recent) * avg_conceded +
                form_weight_recent * recent_avg_conceded
            )
        else:
            eff_conceded = avg_conceded

        # 2. Liga avg depende de casa/fora
        league_avg_for_attack = norms.avg_goals_home if is_home else norms.avg_goals_away
        league_avg_for_defense = norms.avg_goals_away if is_home else norms.avg_goals_home

        # 3. Forças relativas
        attack = self.compute_attack_strength(eff_scored, league_avg_for_attack, n_games)
        defense = self.compute_defense_weakness(eff_conceded, league_avg_for_defense, n_games)

        # 4. Calcula λ
        lam = self.compute_lambda(attack, defense, league_avg_for_attack)

        # 5. Z-scores
        z_attack = self.compute_zscore(eff_scored, league_avg_for_attack, norms.std_goals_home)
        z_defense = self.compute_zscore(eff_conceded, league_avg_for_defense, norms.std_goals_away)

        # 6. Confiança nos dados
        confidence = min(1.0, n_games / 15.0)

        # Lambda alternativo para o outro contexto (casa → fora e vice-versa)
        alt_avg = norms.avg_goals_away if is_home else norms.avg_goals_home
        lam_alt = self.compute_lambda(attack, defense, alt_avg)

        return NormalizedForm(
            lambda_home=lam if is_home else lam_alt,
            lambda_away=lam_alt if is_home else lam,
            attack_strength=attack,
            defense_weakness=defense,
            zscore_attack=z_attack,
            zscore_defense=z_defense,
            data_confidence=confidence,
            n_games=n_games,
        )

    def normalize_match(
        self,
        league_id: str,
        home_avg_scored: float,
        home_avg_conceded: float,
        away_avg_scored: float,
        away_avg_conceded: float,
        home_n_games: int = 10,
        away_n_games: int = 10,
        home_recent_scored: Optional[float] = None,
        home_recent_conceded: Optional[float] = None,
        away_recent_scored: Optional[float] = None,
        away_recent_conceded: Optional[float] = None,
    ) -> dict:
        """
        Normaliza um jogo completo, retornando λ_home e λ_away prontos para Poisson.

        Este é o método principal — é chamado pelo orquestrador antes de qualquer
        cálculo probabilístico.

        Returns:
            {
                "lambda_home": float,
                "lambda_away": float,
                "home_attack": float,
                "home_defense": float,
                "away_attack": float,
                "away_defense": float,
                "confidence": float,  # média das confianças
            }
        """
        norms = self.get_league_norms(league_id)

        home_form = self.normalize_team_form(
            league_id=league_id,
            avg_scored=home_avg_scored,
            avg_conceded=home_avg_conceded,
            is_home=True,
            n_games=home_n_games,
            recent_avg_scored=home_recent_scored,
            recent_avg_conceded=home_recent_conceded,
        )

        away_form = self.normalize_team_form(
            league_id=league_id,
            avg_scored=away_avg_scored,
            avg_conceded=away_avg_conceded,
            is_home=False,
            n_games=away_n_games,
            recent_avg_scored=away_recent_scored,
            recent_avg_conceded=away_recent_conceded,
        )

        # λ_home = attack_home × defense_away × liga_avg_home
        # λ_away = attack_away × defense_home × liga_avg_away
        lambda_home = self.compute_lambda(
            home_form.attack_strength,
            away_form.defense_weakness,  # fraqueza do visitante vs ataque casa
            norms.avg_goals_home,
        )

        lambda_away = self.compute_lambda(
            away_form.attack_strength,
            home_form.defense_weakness,  # fraqueza do casa vs ataque visitante
            norms.avg_goals_away,
        )

        avg_confidence = (home_form.data_confidence + away_form.data_confidence) / 2

        return {
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "home_attack": home_form.attack_strength,
            "home_defense": home_form.defense_weakness,
            "away_attack": away_form.attack_strength,
            "away_defense": away_form.defense_weakness,
            "home_zscore_attack": home_form.zscore_attack,
            "home_zscore_defense": home_form.zscore_defense,
            "away_zscore_attack": away_form.zscore_attack,
            "away_zscore_defense": away_form.zscore_defense,
            "confidence": avg_confidence,
        }

    def update_league_from_results(
        self,
        league_id: str,
        results: list[dict],
    ):
        """
        Atualiza as normas de uma liga com resultados reais.

        Args:
            league_id: ID da liga
            results: lista de dicts com {"home_goals": int, "away_goals": int}
        """
        if len(results) < 10:
            return  # precisamos de dados suficientes

        home_goals = [r["home_goals"] for r in results]
        away_goals = [r["away_goals"] for r in results]

        avg_home = statistics.mean(home_goals)
        avg_away = statistics.mean(away_goals)
        std_home = statistics.stdev(home_goals) if len(home_goals) > 1 else 0.45
        std_away = statistics.stdev(away_goals) if len(away_goals) > 1 else 0.38

        norms = LeagueNorms(
            league_id=league_id,
            avg_goals_home=avg_home,
            avg_goals_away=avg_away,
            avg_goals_total=avg_home + avg_away,
            std_goals_home=std_home,
            std_goals_away=std_away,
            n_games=len(results),
        )

        self.set_league_norms(league_id, norms)
        logger.info(f"[Normalization] Liga {league_id} atualizada com {len(results)} resultados: "
                    f"μ_home={avg_home:.3f} (σ={std_home:.3f}), "
                    f"μ_away={avg_away:.3f} (σ={std_away:.3f})")
