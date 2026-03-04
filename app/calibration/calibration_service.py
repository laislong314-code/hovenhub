"""
CalibrationService — Calibração probabilística e métricas de avaliação de modelo.

Implementa:
  - Brier Score: mede acurácia probabilística (MSE entre prob e resultado)
  - Log Loss: mede calibração logarítmica (mais sensível a extremos)
  - Calibration Curve: compara probabilidades previstas vs realizadas
  - Sharpness: mede quão decisivas são as previsões
  - Platt Scaling: recalibração linear das probabilidades
  - Isotonic Regression: recalibração não-linear

Faz parte da FASE 3 — Calibração Probabilística.
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class CalibrationMetrics:
    """Métricas de calibração para um conjunto de previsões."""
    strategy_id: str
    market: str
    n_predictions: int

    # Métricas principais
    brier_score: float      # 0=perfeito, 0.25=no skill, 1=pior
    log_loss: float         # menor é melhor
    sharpness: float        # variância das probabilidades (maior = mais decisivo)

    # Calibração por bucket
    calibration_error: float  # Mean Calibration Error (MCE)
    max_calibration_error: float  # Maximum Calibration Error

    # Referências
    brier_skill_score: float  # vs baseline (0=sem melhora, 1=perfeito)
    roi: float               # ROI das apostas (se disponível)
    yield_pct: float

    # Metadados
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    buckets: list[dict] = field(default_factory=list)


@dataclass
class PlattCalibrator:
    """
    Calibração via Platt Scaling (regressão logística sobre saída do modelo).

    Transforma a probabilidade bruta do modelo usando:
        p_calibrated = 1 / (1 + exp(A × p_raw + B))

    A e B são aprendidos via MLE nos dados históricos.

    Para implementação completa, requer sklearn. Aqui implementamos
    versão simplificada com parâmetros aprendidos manualmente.
    """
    A: float = 0.0   # parâmetro de escala
    B: float = 0.0   # parâmetro de bias
    fitted: bool = False

    def calibrate(self, p_raw: float) -> float:
        """Aplica calibração Platt na probabilidade bruta."""
        if not self.fitted:
            return p_raw  # fallback: sem calibração

        logit = self.A * p_raw + self.B
        p_cal = 1.0 / (1.0 + math.exp(-logit))
        return max(0.001, min(0.999, p_cal))

    def fit_simple(self, predictions: list[float], outcomes: list[int]):
        """
        Ajusta parâmetros Platt via método simplificado (sem sklearn).
        
        Para uso com sklearn (recomendado em produção):
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
        """
        if len(predictions) < 20:
            logger.warning("[PlattCalibrator] Poucos dados para calibração (<20)")
            return

        # Implementação simplificada: encontra A e B que minimizam log loss
        # via gradient descent manual
        A, B = 1.0, 0.0
        lr = 0.01
        n = len(predictions)

        for _ in range(1000):
            grad_A = 0.0
            grad_B = 0.0

            for p, y in zip(predictions, outcomes):
                logit = A * p + B
                p_cal = 1.0 / (1.0 + math.exp(-logit))
                p_cal = max(1e-7, min(1 - 1e-7, p_cal))

                err = p_cal - y
                grad_A += err * p / n
                grad_B += err / n

            A -= lr * grad_A
            B -= lr * grad_B

        self.A = A
        self.B = B
        self.fitted = True
        logger.info(f"[PlattCalibrator] Ajustado: A={A:.4f}, B={B:.4f}")


class CalibrationService:
    """
    Serviço central de calibração e métricas de performance do modelo.

    Uso:
        service = CalibrationService()

        # Adiciona previsão (no momento do sinal)
        service.add_prediction("standard", "over_2_5", 0.65, match_id="xyz")

        # Adiciona resultado (após o jogo)
        service.add_outcome("standard", "over_2_5", match_id="xyz", outcome=1)

        # Calcula métricas
        metrics = service.calculate_metrics("standard", "over_2_5")
        print(f"Brier Score: {metrics.brier_score:.4f}")
    """

    def __init__(self):
        # Armazena previsões: {strategy_id: {market: [(prob, outcome_or_None, match_id)]}}
        self._predictions: dict[str, dict[str, list]] = {}
        self._calibrators: dict[str, dict[str, PlattCalibrator]] = {}

    def add_prediction(
        self,
        strategy_id: str,
        market: str,
        probability: float,
        match_id: str,
        outcome: Optional[int] = None,  # 1=ganhou, 0=perdeu, None=pendente
    ):
        """Registra uma previsão do modelo."""
        if strategy_id not in self._predictions:
            self._predictions[strategy_id] = {}
        if market not in self._predictions[strategy_id]:
            self._predictions[strategy_id][market] = []

        self._predictions[strategy_id][market].append({
            "probability": probability,
            "outcome": outcome,
            "match_id": match_id,
        })

    def add_outcome(
        self,
        strategy_id: str,
        market: str,
        match_id: str,
        outcome: int,
    ):
        """Atualiza o resultado de uma previsão existente."""
        if strategy_id not in self._predictions:
            return
        if market not in self._predictions[strategy_id]:
            return

        for pred in self._predictions[strategy_id][market]:
            if pred["match_id"] == match_id and pred["outcome"] is None:
                pred["outcome"] = outcome
                break

    def brier_score(
        self,
        probabilities: list[float],
        outcomes: list[int],
    ) -> float:
        """
        Brier Score = (1/N) × Σ(p_i - o_i)²

        Interpretação:
            0.00: perfeito (modelo sempre certo com 100% de confiança)
            0.25: sem habilidade (equivalente a sempre prever 50%)
            1.00: pior possível

        Um modelo que prevê sempre 50% tem BS = 0.25.
        Um bom modelo calibrado tem BS < 0.20.
        """
        if len(probabilities) != len(outcomes) or not probabilities:
            return 0.25  # fallback: no-skill

        n = len(probabilities)
        score = sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / n
        return round(score, 6)

    def brier_skill_score(
        self,
        brier: float,
        baseline_rate: float = 0.5,
    ) -> float:
        """
        BSS = 1 - (BS / BS_ref)

        BS_ref é o Brier Score de um modelo de referência (ex: climatologia).
        Para apostas, referência = sempre prever a taxa base.

        BSS = 1: modelo perfeito
        BSS = 0: sem melhora sobre referência
        BSS < 0: pior que referência
        """
        bs_ref = baseline_rate * (1 - baseline_rate)
        if bs_ref == 0:
            return 0.0
        return 1.0 - (brier / bs_ref)

    def log_loss(
        self,
        probabilities: list[float],
        outcomes: list[int],
        eps: float = 1e-7,
    ) -> float:
        """
        Log Loss = -(1/N) × Σ[o_i × log(p_i) + (1-o_i) × log(1-p_i)]

        Log Loss é mais sensível a previsões extremas erradas.
        Um modelo que prevê 99% para um evento que não ocorre é fortemente penalizado.

        Referências:
            < 0.35: modelo bem calibrado
            0.35-0.60: aceitável
            > 0.60: mal calibrado
        """
        if not probabilities:
            return 1.0

        n = len(probabilities)
        ll = 0.0
        for p, o in zip(probabilities, outcomes):
            p_clipped = max(eps, min(1 - eps, p))
            ll -= o * math.log(p_clipped) + (1 - o) * math.log(1 - p_clipped)

        return round(ll / n, 6)

    def sharpness(self, probabilities: list[float]) -> float:
        """
        Sharpness = Var(probabilidades)

        Mede quão decisivo o modelo é. Previsões concentradas perto de 0 ou 1
        têm alta sharpness. Previsões sempre em torno de 50% têm baixa sharpness.

        Alta sharpness com boa calibração = modelo valioso.
        Alta sharpness com má calibração = overconfident = perigoso.
        """
        if len(probabilities) < 2:
            return 0.0
        return statistics.variance(probabilities)

    def calibration_curve(
        self,
        probabilities: list[float],
        outcomes: list[int],
        n_buckets: int = 10,
    ) -> list[dict]:
        """
        Gera curva de calibração dividindo previsões em buckets.

        Para cada bucket [0.0-0.1, 0.1-0.2, ...]:
            - predicted_prob: média das probabilidades no bucket
            - actual_rate: taxa de acerto real
            - n: quantidade de previsões
            - calibration_error: |predicted - actual|

        Modelo perfeitamente calibrado tem predicted ≈ actual em todos os buckets.
        """
        if not probabilities:
            return []

        bucket_size = 1.0 / n_buckets
        buckets = []

        for i in range(n_buckets):
            low = i * bucket_size
            high = (i + 1) * bucket_size

            bucket_probs = []
            bucket_outcomes = []

            for p, o in zip(probabilities, outcomes):
                if low <= p < high or (i == n_buckets - 1 and p == 1.0):
                    bucket_probs.append(p)
                    bucket_outcomes.append(o)

            if not bucket_probs:
                continue

            mean_pred = statistics.mean(bucket_probs)
            actual_rate = sum(bucket_outcomes) / len(bucket_outcomes)
            cal_error = abs(mean_pred - actual_rate)

            buckets.append({
                "bucket_low": round(low, 2),
                "bucket_high": round(high, 2),
                "predicted_prob": round(mean_pred, 4),
                "actual_rate": round(actual_rate, 4),
                "calibration_error": round(cal_error, 4),
                "n": len(bucket_probs),
            })

        return buckets

    def mean_calibration_error(self, buckets: list[dict]) -> float:
        """MCE = média ponderada dos erros de calibração por bucket."""
        if not buckets:
            return 1.0
        total_n = sum(b["n"] for b in buckets)
        if total_n == 0:
            return 1.0
        weighted = sum(b["calibration_error"] * b["n"] for b in buckets)
        return round(weighted / total_n, 6)

    def calculate_metrics(
        self,
        strategy_id: str,
        market: str,
    ) -> Optional[CalibrationMetrics]:
        """
        Calcula todas as métricas para uma combinação estratégia/mercado.
        """
        preds_raw = self._predictions.get(strategy_id, {}).get(market, [])

        # Filtra apenas previsões com resultado
        completed = [(p["probability"], p["outcome"]) for p in preds_raw if p["outcome"] is not None]

        if len(completed) < 5:
            logger.warning(f"[Calibration] Poucos dados: {len(completed)} previsões para {strategy_id}/{market}")
            return None

        probs = [p for p, _ in completed]
        outcomes = [o for _, o in completed]

        brier = self.brier_score(probs, outcomes)
        ll = self.log_loss(probs, outcomes)
        sharp = self.sharpness(probs)
        bss = self.brier_skill_score(brier, statistics.mean(outcomes))

        buckets = self.calibration_curve(probs, outcomes)
        mce = self.mean_calibration_error(buckets)
        max_ce = max((b["calibration_error"] for b in buckets), default=0.0)

        return CalibrationMetrics(
            strategy_id=strategy_id,
            market=market,
            n_predictions=len(completed),
            brier_score=brier,
            log_loss=ll,
            sharpness=sharp,
            calibration_error=mce,
            max_calibration_error=max_ce,
            brier_skill_score=bss,
            roi=0.0,   # preenchido pelo PerformanceTracker
            yield_pct=0.0,
            buckets=buckets,
        )

    def get_calibrator(
        self,
        strategy_id: str,
        market: str,
    ) -> PlattCalibrator:
        """
        Retorna ou cria um calibrador Platt para uma estratégia/mercado.
        Se dados suficientes existirem, ajusta automaticamente.
        """
        if strategy_id not in self._calibrators:
            self._calibrators[strategy_id] = {}

        if market not in self._calibrators[strategy_id]:
            cal = PlattCalibrator()

            # Auto-ajusta se temos dados suficientes
            preds_raw = self._predictions.get(strategy_id, {}).get(market, [])
            completed = [(p["probability"], p["outcome"]) for p in preds_raw if p["outcome"] is not None]

            if len(completed) >= 50:
                probs = [p for p, _ in completed]
                outcomes = [o for _, o in completed]
                cal.fit_simple(probs, outcomes)
                logger.info(f"[Calibration] Platt ajustado para {strategy_id}/{market} com {len(completed)} amostras")

            self._calibrators[strategy_id][market] = cal

        return self._calibrators[strategy_id][market]

    def calibrate_probability(
        self,
        strategy_id: str,
        market: str,
        raw_probability: float,
    ) -> float:
        """
        Aplica calibração na probabilidade bruta.
        Retorna probabilidade calibrada (ou original se calibrador não ajustado).
        """
        calibrator = self.get_calibrator(strategy_id, market)
        return calibrator.calibrate(raw_probability)

    def generate_report(self) -> dict:
        """Gera relatório consolidado de calibração para todas as estratégias/mercados."""
        report = {}
        for strategy_id, markets in self._predictions.items():
            report[strategy_id] = {}
            for market in markets:
                metrics = self.calculate_metrics(strategy_id, market)
                if metrics:
                    report[strategy_id][market] = {
                        "brier_score": metrics.brier_score,
                        "log_loss": metrics.log_loss,
                        "calibration_error": metrics.calibration_error,
                        "sharpness": metrics.sharpness,
                        "brier_skill_score": metrics.brier_skill_score,
                        "n": metrics.n_predictions,
                    }
        return report
