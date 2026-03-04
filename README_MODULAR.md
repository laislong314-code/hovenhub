# Betoven v7 — Modular Engines (Fused)

This fused package adds new modular engines under `app/core`, `app/live`, `app/calibration`, `app/performance` and `app/database`.

**Important:** The legacy v7 flow is NOT removed. A compatibility wrapper is provided:

- `app/services/modular_orchestrator.py`

You can wire it into existing FastAPI routes gradually (recommended), so production doesn't break.
