"""
ModularOrchestrator: a compatibility wrapper that allows gradual adoption of the new modular engines
without breaking the existing v7 orchestration flow.

Usage:
  from app.services.modular_orchestrator import ModularOrchestrator
  orch = ModularOrchestrator(db=..., settings=...)
  result = orch.analyze_fixture(fixture_id, forced_minute=None)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Engines (new)
from app.core.strategy_engine import StrategyEngine
from app.core.probability_engine import ProbabilityEngine
from app.core.value_engine import ValueEngine
from app.core.signal_engine import SignalEngine
from app.core.model_registry import ModelRegistry
from app.core.normalization_service import NormalizationService

from app.calibration.calibration_service import CalibrationService
from app.live.live_momentum_engine import LiveMomentumEngine
from app.performance.performance_tracker import PerformanceTracker

# NOTE: We intentionally do NOT import the legacy orchestrator here to avoid circular imports.
# The legacy orchestrator remains the default; this modular orchestrator can be wired in later.


@dataclass
class ModularOrchestrator:
    """
    Coordinates modular components. Designed to be plugged into existing API endpoints over time.
    """
    db: Any = None
    settings: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.registry = ModelRegistry()
        self.strategy = StrategyEngine(settings=self.settings or {})
        self.normalizer = NormalizationService()
        self.prob_engine = ProbabilityEngine(strategy=self.strategy, normalizer=self.normalizer)
        self.value_engine = ValueEngine(strategy=self.strategy)
        self.signal_engine = SignalEngine(strategy=self.strategy)
        self.calibration = CalibrationService(db=self.db)
        self.live_momentum = LiveMomentumEngine()
        self.performance = PerformanceTracker(db=self.db)

    def analyze_fixture(self, fixture_id: int, forced_minute: Optional[int] = None, live_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Returns a dict compatible with the current signal payload style:
          - probabilities
          - ev
          - signals
          - meta (includes model_version/strategy)
        """
        # 1) Determine model/strategy
        model_version = self.registry.current_version()
        strategy_name = self.strategy.name

        # 2) Live momentum features if available
        live_features = {}
        if live_snapshot:
            live_features = self.live_momentum.compute_features(live_snapshot, minute=forced_minute)

        # 3) Probabilities
        probs = self.prob_engine.compute_probabilities(fixture_id=fixture_id, live_features=live_features)

        # 4) Optional calibration
        probs_cal = self.calibration.apply_calibration(strategy=strategy_name, probabilities=probs)

        # 5) EV from odds (if present in probs payload / upstream)
        ev = self.value_engine.compute_ev(probabilities=probs_cal)

        # 6) Signals
        signals = self.signal_engine.generate_signals(probabilities=probs_cal, ev=ev, fixture_id=fixture_id)

        # 7) Persist tracking (best-effort)
        try:
            self.performance.record_signal(
                fixture_id=fixture_id,
                strategy=strategy_name,
                model_version=model_version,
                signals=signals,
                probabilities=probs_cal,
                ev=ev,
            )
        except Exception:
            pass

        return {
            "fixture_id": fixture_id,
            "forced_minute": forced_minute,
            "model_version": model_version,
            "strategy": strategy_name,
            "probabilities": probs_cal,
            "ev": ev,
            "signals": signals,
        }
