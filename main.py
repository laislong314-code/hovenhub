import sys
import time
import datetime as dt
from typing import List, Dict, Optional
from curl_cffi import requests

BASE = "https://api.sofascore.com/api/v1"
SESSION = requests.Session()

def get_scheduled(date_str):
    url = f"{BASE}/sport/football/scheduled-events/{date_str}"
    r = SESSION.get(url, impersonate="chrome124", timeout=25)
    if r.status_code != 200:
        print(f"[{r.status_code}] {date_str}")
        return None
    return r.json()

def is_finished(ev):
    st = (ev.get("status") or {}).get("type") or ""
    return st.lower() in ("finished", "afterextra", "afterpenalties", "ended")

def extract(ev):
    home = ev.get("homeTeam") or {}
    away = ev.get("awayTeam") or {}
    return {
        "event_id": ev.get("id"),
        "startTimestamp": ev.get("startTimestamp"),
        "home_id": home.get("id"),
        "away_id": away.get("id"),
        "home": home.get("name"),
        "away": away.get("name"),
        "home_score": (ev.get("homeScore") or {}).get("current"),
        "away_score": (ev.get("awayScore") or {}).get("current"),
        "tournament": (ev.get("tournament") or {}).get("name"),
        "category": ((ev.get("tournament") or {}).get("category") or {}).get("name"),
    }

def get_team_history(team_id, n_matches=20, days_back_max=120):
    out = []
    today = dt.date.today()
    for i in range(days_back_max):
        d = today - dt.timedelta(days=i)
        data = get_scheduled(d.strftime("%Y-%m-%d"))
        if not data:
            time.sleep(0.3)
            continue
        for ev in data.get("events") or []:
            home = (ev.get("homeTeam") or {}).get("id")
            away = (ev.get("awayTeam") or {}).get("id")
            if home != team_id and away != team_id:
                continue
            if not is_finished(ev):
                continue
            out.append(extract(ev))
            if len(out) >= n_matches:
                return out
        time.sleep(0.20)
    return out

# Teste: team_id=44 (Manchester United), 20 jogos
results = get_team_history(44, 20, 120)
print(f"found={len(results)}")
for row in results:
    ts = row["startTimestamp"]
    d = dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    print(f"{d} | {row['home']} {row['home_score']} x {row['away_score']} {row['away']}")