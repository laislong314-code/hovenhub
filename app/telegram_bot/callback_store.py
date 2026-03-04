"""
Store em memória para os sinais do ciclo atual.
Permite que o callback handler recupere os sinais quando o usuário clicar no botão.
"""
from typing import Optional

_cycle_signals: dict[str, list] = {}


def set_cycle_signals(signals_by_match: dict):
    global _cycle_signals
    _cycle_signals = signals_by_match


def get_cycle_signals() -> dict:
    return _cycle_signals


def get_match_signals(idx: int) -> Optional[tuple[str, list]]:
    """Retorna (match_label, signals) pelo índice."""
    keys = list(_cycle_signals.keys())
    if 0 <= idx < len(keys):
        k = keys[idx]
        return k, _cycle_signals[k]
    return None


def get_all_signals() -> list[tuple[str, list]]:
    return list(_cycle_signals.items())
