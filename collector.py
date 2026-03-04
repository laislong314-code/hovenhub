"""
collector.py — Coleta contínua de jogos + stats de jogadores do SofaScore
Salva direto no Postgres (DATABASE_URL via env var).

Fluxo:
  1. A cada ciclo, busca jogos de hoje e dos últimos HISTORY_DAYS dias
  2. Para cada jogo finalizado ainda sem stats, coleta estatísticas por jogador
  3. Dorme SLEEP_SECONDS e repete
"""

import os
import time
import datetime as dt
import logging
from typing import Optional

import psycopg2
import psycopg2.extras
from curl_cffi import requests as cfrequests

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL  = os.environ["DATABASE_URL"]
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "120"))   # intervalo entre ciclos
HISTORY_DAYS  = int(os.getenv("HISTORY_DAYS", "7"))      # dias passados a varrer
BASE          = "https://api.sofascore.com/api/v1"

# Ligas principais por category slug (adicione ou remova conforme necessidade)
TRACKED_CATEGORIES = {
    "england", "spain", "germany", "italy", "france",
    "brazil", "argentina", "portugal", "netherlands",
    "champions-league", "europa-league", "conference-league",
    "world-cup", "euro", "copa-america",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── HTTP session ──────────────────────────────────────────────────────────────
SESSION = cfrequests.Session()

def sofa_get(path: str) -> Optional[dict]:
    url = f"{BASE}{path}"
    try:
        r = SESSION.get(url, impersonate="chrome124", timeout=25)
        if r.status_code == 200:
            return r.json()
        log.warning(f"HTTP {r.status_code} {path} — {r.text[:100]}")
        return None
    except Exception as e:
        log.error(f"Request error {path}: {e}")
        return None

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL)

def upsert_match(cur, ev: dict):
    home  = ev.get("homeTeam") or {}
    away  = ev.get("awayTeam") or {}
    trn   = ev.get("tournament") or {}
    cat   = trn.get("category") or {}
    ts    = ev.get("startTimestamp")
    cur.execute("""
        INSERT INTO matches
            (event_id, start_ts, start_date, status,
             home_id, away_id, home_name, away_name,
             home_score, away_score,
             tournament_id, tournament_name, category_name, updated_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
        ON CONFLICT (event_id) DO UPDATE SET
            status          = EXCLUDED.status,
            home_score      = EXCLUDED.home_score,
            away_score      = EXCLUDED.away_score,
            updated_at      = NOW()
    """, (
        ev.get("id"),
        ts,
        dt.datetime.utcfromtimestamp(ts).date() if ts else None,
        (ev.get("status") or {}).get("type"),
        home.get("id"), away.get("id"),
        home.get("name"), away.get("name"),
        (ev.get("homeScore") or {}).get("current"),
        (ev.get("awayScore") or {}).get("current"),
        trn.get("id"), trn.get("name"), cat.get("slug"),
    ))

def match_has_stats(cur, event_id: int) -> bool:
    cur.execute("SELECT 1 FROM player_stats WHERE event_id=%s LIMIT 1", (event_id,))
    return cur.fetchone() is not None

def upsert_player_stats(cur, event_id: int, team_id: int, player: dict, stats: dict):
    def g(key):
        return stats.get(key)
    cur.execute("""
        INSERT INTO player_stats
            (event_id, team_id, player_id, player_name, position,
             minutes_played, rating,
             goals, assists, shots_on_goal, shots_total,
             passes_total, passes_accurate, key_passes,
             tackles, interceptions, clearances,
             saves, yellow_cards, red_cards, updated_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
        ON CONFLICT (event_id, player_id) DO UPDATE SET
            minutes_played  = EXCLUDED.minutes_played,
            rating          = EXCLUDED.rating,
            goals           = EXCLUDED.goals,
            assists         = EXCLUDED.assists,
            updated_at      = NOW()
    """, (
        event_id, team_id,
        player.get("id"), player.get("name"),
        player.get("position"),
        g("minutesPlayed"), g("rating"),
        g("goals"), g("goalAssist"),
        g("onTargetScoringAttempt"), g("totalScoringAttempt") ,
        g("totalPass"), g("accuratePass"), g("keyPass"),
        g("totalTackle"), g("interceptionWon"), g("totalClearance"),
        g("savedShotsFromInsideTheBox"), g("yellowCard"), g("redCard"),
    ))

# ── Coleta de stats ───────────────────────────────────────────────────────────
def collect_stats(conn, event_id: int):
    data = sofa_get(f"/event/{event_id}/lineups")
    if not data:
        return 0
    count = 0
    with conn.cursor() as cur:
        for side in ("home", "away"):
            side_data = data.get(side) or {}
            team_id   = (side_data.get("team") or {}).get("id")
            players   = side_data.get("players") or []
            for entry in players:
                player     = entry.get("player") or {}
                statistics = entry.get("statistics") or {}
                if not player.get("id"):
                    continue
                upsert_player_stats(cur, event_id, team_id, player, statistics)
                count += 1
        conn.commit()
    return count

# ── Ciclo principal ───────────────────────────────────────────────────────────
def is_finished(ev: dict) -> bool:
    st = (ev.get("status") or {}).get("type") or ""
    return st.lower() in ("finished", "afterextra", "afterpenalties", "ended")

def is_tracked(ev: dict) -> bool:
    cat_slug = ((ev.get("tournament") or {}).get("category") or {}).get("slug") or ""
    return cat_slug in TRACKED_CATEGORIES

def run_cycle():
    today = dt.date.today()
    dates = [today - dt.timedelta(days=i) for i in range(HISTORY_DAYS)]

    conn = get_conn()
    matches_upserted = 0
    stats_collected  = 0

    try:
        for d in dates:
            ds   = d.strftime("%Y-%m-%d")
            data = sofa_get(f"/sport/football/scheduled-events/{ds}")
            if not data:
                time.sleep(0.5)
                continue

            events = data.get("events") or []
            relevant = [ev for ev in events if is_tracked(ev)]

            with conn.cursor() as cur:
                for ev in relevant:
                    upsert_match(cur, ev)
                    matches_upserted += 1
                conn.commit()

            # Coleta stats dos jogos já finalizados sem stats ainda
            for ev in relevant:
                if not is_finished(ev):
                    continue
                event_id = ev.get("id")
                with conn.cursor() as cur:
                    has = match_has_stats(cur, event_id)
                if not has:
                    n = collect_stats(conn, event_id)
                    stats_collected += n
                    if n:
                        log.info(f"Stats coletadas: event_id={event_id} players={n}")
                    time.sleep(0.3)

            time.sleep(0.2)

    finally:
        conn.close()

    log.info(f"Ciclo concluído — matches={matches_upserted} stats={stats_collected}")

def main():
    log.info(f"Iniciando coletor | ciclo={SLEEP_SECONDS}s | history={HISTORY_DAYS}d")
    while True:
        try:
            run_cycle()
        except Exception as e:
            log.error(f"Erro no ciclo: {e}", exc_info=True)
        log.info(f"Aguardando {SLEEP_SECONDS}s...")
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
