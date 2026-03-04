
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from app.config import get_settings


@dataclass(frozen=True)
class MarketRule:
    odd_min: float
    odd_max: float
    ev_min: float


@dataclass(frozen=True)
class SelectorConfig:
    """Centralized selection rules for "precision" mode.

    Phase 2 requirement: avoid hardcoded scattered rules.
    """

    top_n: int = 10
    allow_multi: bool = False
    one_per_match: bool = True
    allowed_markets: Tuple[str, ...] = ("DC_Home_Draw", "Under_3.5", "Over_1.5", "Over_2.5", "BTTS")
    # Optional per-market rules. If absent, global defaults apply.
    rules_by_market: Dict[str, MarketRule] = None  # type: ignore

    # Global defaults (used when market not present in rules_by_market)
    ev_min_default: float = 0.10
    odd_min_default: float = 1.45
    odd_max_default: float = 2.60

    @classmethod
    def from_settings(cls, top_n: Optional[int] = None) -> "SelectorConfig":
        s = get_settings()
        allowed = tuple([x.strip() for x in (s.precision_allowed_markets or "").split(",") if x.strip()])
        if not allowed:
            allowed = ("DC_Home_Draw", "Under_3.5", "Over_1.5", "Over_2.5", "BTTS")

        # Start with global defaults; allow later to specialize per market.
        rules = {m: MarketRule(
            odd_min=float(s.precision_odd_min),
            odd_max=float(s.precision_odd_max),
            ev_min=float(s.precision_ev_min),
        ) for m in allowed}

        return cls(
            top_n=int(top_n if top_n is not None else s.precision_top_n_default),
            allow_multi=bool(s.precision_allow_multi),
            one_per_match=bool(s.precision_one_per_match),
            allowed_markets=allowed,
            rules_by_market=rules,
            ev_min_default=float(s.precision_ev_min),
            odd_min_default=float(s.precision_odd_min),
            odd_max_default=float(s.precision_odd_max),
        )


class SignalSelector:
    """
    Filters and ranks pending signals for a 'precision' strategy.

    Expected input items (from AnalysisOrchestrator._pending_signals):
      {
        "analysis": {..., "match_id": "...", "_is_multi": bool?},
        "sig": {"market_id": "...", "odd": float, "prob": float, "ev": float, ...},
        "stake": float,
        "kelly": float,
        "match_id": "...",  # sometimes duplicated
      }
    """

    def __init__(self, config: SelectorConfig | None = None):
        self.cfg = config or SelectorConfig.from_settings()

    def _is_allowed(self, item: Dict[str, Any]) -> bool:
        a = item.get("analysis") or {}
        sig = item.get("sig") or {}

        if not self.cfg.allow_multi and a.get("_is_multi"):
            return False

        market = sig.get("market_id") or sig.get("market") or ""
        if market not in self.cfg.allowed_markets:
            return False

        try:
            ev = float(sig.get("ev", 0.0))
            odd = float(sig.get("odd", 0.0))
        except Exception:
            return False

        rule = None
        if self.cfg.rules_by_market:
            rule = self.cfg.rules_by_market.get(market)
        ev_min = rule.ev_min if rule else self.cfg.ev_min_default
        odd_min = rule.odd_min if rule else self.cfg.odd_min_default
        odd_max = rule.odd_max if rule else self.cfg.odd_max_default

        if ev < ev_min:
            return False
        if odd < odd_min or odd > odd_max:
            return False

        return True

    def _score(self, item: Dict[str, Any]) -> float:
        """
        Hybrid score prioritizing EV but rewarding 'edge' (prob - implied_prob).
        Keeps it simple and robust.
        """
        sig = item.get("sig") or {}
        ev = float(sig.get("ev", 0.0))
        odd = float(sig.get("odd", 0.0))
        prob = float(sig.get("prob", 0.0))

        implied = (1.0 / odd) if odd > 0 else 0.0
        edge = prob - implied

        # small bonus for edge, but EV still dominates
        return ev + 0.25 * edge

    def select(self, pending: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Filter
        filtered = [it for it in pending if self._is_allowed(it)]

        # Sort by hybrid score
        ranked = sorted(filtered, key=self._score, reverse=True)

        # Enforce one per match (best ranked wins)
        if self.cfg.one_per_match:
            out: List[Dict[str, Any]] = []
            seen = set()
            for it in ranked:
                mid = (it.get("analysis") or {}).get("match_id") or it.get("match_id")
                if not mid:
                    continue
                if mid in seen:
                    continue
                seen.add(mid)
                out.append(it)
            ranked = out

        return ranked[: self.cfg.top_n]
