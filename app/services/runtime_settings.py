"""Runtime settings persisted to data/runtime_settings.json

Used by HUB to persist user choices (e.g., analyze only one league).
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, Any

_SETTINGS_PATH = Path("data/runtime_settings.json")
_LOCK = Lock()

_DEFAULT: Dict[str, Any] = {
    "analysis_league_id": None,  # None => analyze normal (monitored_leagues or all)
}

def _read() -> Dict[str, Any]:
    if not _SETTINGS_PATH.exists():
        return dict(_DEFAULT)
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(_DEFAULT)
        merged = dict(_DEFAULT)
        merged.update(data)
        return merged
    except Exception:
        return dict(_DEFAULT)

def _write(data: Dict[str, Any]) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def get_analysis_league_id() -> Optional[str]:
    with _LOCK:
        data = _read()
        v = data.get("analysis_league_id")
        if v is None:
            return None
        s = str(v).strip()
        return s or None

def set_analysis_league_id(league_id: Optional[str]) -> Optional[str]:
    """Set preferred league_id. Pass None/"" to analyze all (default behavior)."""
    lid = None
    if league_id is not None:
        s = str(league_id).strip()
        lid = s if s else None
    with _LOCK:
        data = _read()
        data["analysis_league_id"] = lid
        _write(data)
    return lid
