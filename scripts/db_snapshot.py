"""Phase 0 — DB schema snapshot.

Usage:
  python scripts/db_snapshot.py --db data/sports_ev.db --out data/schema_snapshot_signals.json
"""

from __future__ import annotations

import argparse
from app.performance.metrics_persistence import MetricsPersistence, MetricsConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sports_ev.db")
    ap.add_argument("--out", default="data/schema_snapshot_signals.json")
    args = ap.parse_args()

    MetricsPersistence(args.db, MetricsConfig()).ensure_schema_snapshot(args.out)
    print(f"OK: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
