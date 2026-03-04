"""AdvancedMetrics — Phase 1 statistical validation utilities.

This module is intentionally dependency-light and deterministic.
It is used by SettlementService (sqlite3) and can be reused by offline scripts.

Rules enforced:
  - Only resolved WIN/LOSS are included for probabilistic metrics.
  - No changes to settlement logic.
  - Any risky behavior is feature-flagged via Settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> Optional[float]:
    if not probs or len(probs) != len(outcomes):
        return None
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs: Sequence[float], outcomes: Sequence[int], eps: float = 1e-7) -> Optional[float]:
    if not probs or len(probs) != len(outcomes):
        return None
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(outcomes, dtype=float)
    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(ll))


def brier_skill_vs_baseline(brier: float, baseline_brier: float) -> Optional[float]:
    """Brier Skill Score: 1 - BS / BS_ref.

    baseline_brier should be the Brier Score of a reference forecast.
    In our case we use the implied_probability as the baseline forecast.
    """
    if baseline_brier <= 0:
        return None
    return 1.0 - (float(brier) / float(baseline_brier))


def bootstrap_roi_ci(
    profit_loss: Sequence[float],
    stake: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap CI for ROI.

    ROI = sum(P&L) / sum(stake) * 100
    Returns (low, high) for 1-alpha CI.
    """
    if not profit_loss or not stake or len(profit_loss) != len(stake):
        return None
    pl = np.asarray(profit_loss, dtype=float)
    st = np.asarray(stake, dtype=float)
    if len(pl) < 2:
        return None
    if np.any(st < 0):
        return None
    if float(np.sum(st)) <= 0:
        return None

    rng = np.random.default_rng(seed)
    idx = np.arange(len(pl))
    rois = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(idx, size=len(idx), replace=True)
        st_sum = float(np.sum(st[samp]))
        rois[i] = (float(np.sum(pl[samp])) / st_sum) * 100.0 if st_sum > 0 else 0.0
    low = float(np.quantile(rois, alpha / 2.0))
    high = float(np.quantile(rois, 1.0 - alpha / 2.0))
    return low, high


def spearman_ev_vs_pl(ev: Sequence[float], profit_loss: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not ev or not profit_loss or len(ev) != len(profit_loss) or len(ev) < 3:
        return None, None
    rho, p = spearmanr(ev, profit_loss)
    # scipy may return nan if constant input
    if rho is None or p is None or (isinstance(rho, float) and math.isnan(rho)):
        return None, None
    return float(rho), float(p)


@dataclass(frozen=True)
class RollingAlert:
    bs7_threshold: float = 0.25
    consecutive: int = 3
