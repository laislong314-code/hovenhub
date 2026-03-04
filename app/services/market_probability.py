"""MarketProbability — converte odds em probabilidade implícita.

Importante: isso NÃO remove o vig (margem) do bookmaker.
Serve como aproximação conservadora para puxar o modelo na direção do mercado.
"""

from __future__ import annotations


class MarketProbability:
    @staticmethod
    def from_odd(odd: float) -> float:
        try:
            odd = float(odd)
        except Exception:
            return 0.0
        if odd <= 1.0:
            return 0.0
        return 1.0 / odd
