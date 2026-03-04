"""MetricsPersistence — compute & persist Phase-1 metrics into SQLite.

Designed to be called from SettlementService (which already uses sqlite3 directly).

Data source:
  signals table with fields: market, model_probability, implied_probability, ev,
  stake_units, profit_loss, result, sent_at.

Storage:
  metrics_daily table (created by init_db migration).

Safety:
  - Read-only on signals
  - Upserts metrics by (day_utc, market)
  - Only computes metrics on WIN/LOSS
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger

from app.performance.advanced_metrics import (
    brier_score,
    log_loss,
    brier_skill_vs_baseline,
    bootstrap_roi_ci,
    spearman_ev_vs_pl,
    RollingAlert,
)


def _utc_day_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class MetricsConfig:
    eps: float = 1e-7
    n_boot: int = 5000
    seed: int = 1337
    rolling_windows: Tuple[int, ...] = (7, 14)
    rolling_alert: RollingAlert = RollingAlert()


class MetricsPersistence:
    def __init__(self, db_path: str, cfg: MetricsConfig):
        self.db_path = db_path
        self.cfg = cfg

    def ensure_schema_snapshot(self, out_path: str = "data/schema_snapshot_signals.json") -> None:
        """Phase 0: save DB schema + sample rows for audit."""
        import json
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(signals)")
        cols = cur.fetchall()
        cur.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 5")
        rows = cur.fetchall()
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "signals_table_info": cols,
            "signals_sample_last5": rows,
        }
        conn.close()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"[Metrics] schema snapshot saved: {out_path}")

    def compute_and_persist_daily(self, day_utc: Optional[str] = None) -> Dict[str, int]:
        day = day_utc or _utc_day_now()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Fetch resolved WIN/LOSS for the day
        cur.execute(
            """
            SELECT market, model_probability, implied_probability, ev, stake_units, profit_loss
            FROM signals
            WHERE result IN ('WIN','LOSS')
              AND substr(sent_at, 1, 10) = ?
            ORDER BY sent_at ASC
            """,
            (day,),
        )
        rows = cur.fetchall()
        if not rows:
            conn.close()
            return {"markets": 0, "rows": 0}

        by_market: Dict[str, List[sqlite3.Row]] = {}
        for r in rows:
            by_market.setdefault(str(r["market"]), []).append(r)

        # Also compute ALL markets aggregate
        by_market["ALL"] = rows

        now_ts = datetime.now(timezone.utc).isoformat()
        upserts = 0

        for market, mrows in by_market.items():
            probs = [float(x["model_probability"]) for x in mrows]
            probs_base = [float(x["implied_probability"]) for x in mrows]
            outcomes = [1 if float(x["profit_loss"]) > -1e-12 else 0 for x in mrows]  # WIN=1, LOSS=0
            evs = [float(x["ev"]) for x in mrows]
            pls = [float(x["profit_loss"]) for x in mrows]
            stakes = [float(x["stake_units"]) for x in mrows]

            n_resolved = len(mrows)
            st_sum = sum(stakes)
            roi = (sum(pls) / st_sum) * 100.0 if st_sum > 0 else 0.0

            bs = brier_score(probs, outcomes)
            ll = log_loss(probs, outcomes, eps=self.cfg.eps)
            bs_base = brier_score(probs_base, outcomes)
            bss = brier_skill_vs_baseline(bs, bs_base) if (bs is not None and bs_base is not None) else None
            rho, pval = spearman_ev_vs_pl(evs, pls)
            ci = bootstrap_roi_ci(pls, stakes, n_boot=self.cfg.n_boot, seed=self.cfg.seed)
            ci_low, ci_high = (ci if ci else (None, None))

            # Rolling brier: last N resolved for that market (not limited to day)
            rb7, rb14, alert = self._rolling_brier_alert(cur, market)

            cur.execute(
                """
                INSERT OR REPLACE INTO metrics_daily(
                    id,
                    day_utc, market,
                    n_resolved, roi, roi_ci_low, roi_ci_high,
                    brier, logloss, brier_skill_vs_implied,
                    spearman_rho_ev_pl, spearman_p_ev_pl,
                    rolling_brier_7, rolling_brier_14, rolling_alert,
                    created_at
                )
                VALUES(
                    (SELECT id FROM metrics_daily WHERE day_utc=? AND market=?),
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?
                )
                """,
                (
                    day, market,
                    day, market,
                    n_resolved, float(roi), ci_low, ci_high,
                    bs, ll, bss,
                    rho, pval,
                    rb7, rb14, 1 if alert else 0,
                    now_ts,
                ),
            )
            upserts += 1

        conn.commit()
        conn.close()
        logger.info(f"[Metrics] metrics_daily upserted: day={day} rows={upserts}")
        return {"markets": len(by_market), "rows": upserts}

    def _rolling_brier_alert(self, cur: sqlite3.Cursor, market: str) -> Tuple[Optional[float], Optional[float], bool]:
        if market == "ALL":
            # Rolling metrics are more meaningful per-market. Keep None for ALL.
            return None, None, False

        cur.execute(
            """
            SELECT model_probability, implied_probability, profit_loss
            FROM signals
            WHERE result IN ('WIN','LOSS') AND market = ?
            ORDER BY sent_at ASC
            """,
            (market,),
        )
        rows = cur.fetchall()
        if not rows:
            return None, None, False

        probs = [float(r[0]) for r in rows]
        outcomes = [1 if float(r[2]) > -1e-12 else 0 for r in rows]

        def rolling(window: int) -> Optional[float]:
            if len(probs) < window:
                return None
            p = probs[-window:]
            y = outcomes[-window:]
            return brier_score(p, y)

        rb7 = rolling(7)
        rb14 = rolling(14)

        # Alert: if rolling_bs_7 > threshold for X consecutive windows
        alert = False
        th = self.cfg.rolling_alert.bs7_threshold
        k = int(self.cfg.rolling_alert.consecutive)
        if len(probs) >= (7 + k - 1) and rb7 is not None:
            # compute last k rolling BS7 values
            vals = []
            for i in range(k):
                start = -(7 + i)
                end = None if i == 0 else -(i)
                p = probs[start:end]
                y = outcomes[start:end]
                vals.append(brier_score(p, y))
            if all(v is not None and v > th for v in vals):
                alert = True
        return rb7, rb14, alert
