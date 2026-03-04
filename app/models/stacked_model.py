"""StackedModel — combina probabilidades de múltiplas fontes.

Objetivo: reduzir overconfidence do pré-jogo combinando:
  - probabilidade do modelo (Poisson/Dixon-Coles ou similar)
  - probabilidade "bivariada" (quando disponível; fallback = prob_model)
  - probabilidade implícita do mercado (odds)
  - probabilidade baseada em forma (heurística simples)

Nota: Os pesos são intencionamente conservadores e podem ser ajustados depois.
"""

from __future__ import annotations


class StackedModel:
    def __init__(
        self,
        w_poisson: float = 0.35,
        w_bivariate: float = 0.25,
        w_market: float = 0.25,
        w_form: float = 0.15,
    ) -> None:
        total = w_poisson + w_bivariate + w_market + w_form
        if total <= 0:
            # fallback seguro
            w_poisson, w_bivariate, w_market, w_form = 1.0, 0.0, 0.0, 0.0
            total = 1.0
        # normaliza pesos
        self.w_poisson = w_poisson / total
        self.w_bivariate = w_bivariate / total
        self.w_market = w_market / total
        self.w_form = w_form / total

    @staticmethod
    def _clamp(p: float) -> float:
        if p is None:
            return 0.5
        try:
            p = float(p)
        except Exception:
            return 0.5
        if p < 0.01:
            return 0.01
        if p > 0.99:
            return 0.99
        return p

    def combine(
        self,
        poisson_prob: float,
        bivariate_prob: float,
        market_prob: float,
        form_prob: float,
    ) -> float:
        p1 = self._clamp(poisson_prob)
        p2 = self._clamp(bivariate_prob)
        pm = self._clamp(market_prob)
        pf = self._clamp(form_prob)

        out = (
            self.w_poisson * p1
            + self.w_bivariate * p2
            + self.w_market * pm
            + self.w_form * pf
        )
        return self._clamp(out)
