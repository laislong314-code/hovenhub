# CHECKLIST executável — Verificação (Local + AWS)

> Objetivo: validar que nada quebrou e que as métricas estão sendo geradas **sem mudar** o settlement.

## 0) Pré-requisitos
```bash
cd betoven_hub
python -V
pip install -r requirements.txt
```

## 1) FASE 0 — Segurança e Diagnóstico
### 1.1 Compile
```bash
python -m compileall .
```

### 1.2 Snapshot do schema do DB (auditoria)
```bash
python scripts/db_snapshot.py --db data/sports_ev.db --out data/schema_snapshot_signals.json
```

### 1.3 Conferir schema atual
```bash
python - <<'PY'
import sqlite3
conn=sqlite3.connect('data/sports_ev.db')
cur=conn.cursor()
print('signals columns:')
for r in cur.execute('PRAGMA table_info(signals)').fetchall():
    print(r)
print('\nmetrics_daily exists?:', bool(cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metrics_daily'").fetchone()))
conn.close()
PY
```

## 2) Smoke test do servidor
> Se TELEGRAM não estiver configurado, ele loga "[TELEGRAM SIMULADO]" (ok).

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Em outro terminal:
```bash
curl -s http://127.0.0.1:8000/api/v1/health || true
curl -s "http://127.0.0.1:8000/api/v1/signals?mode=precision&limit=10" | head
curl -s "http://127.0.0.1:8000/api/v1/signals?mode=all&limit=10" | head
curl -s "http://127.0.0.1:8000/api/v1/metrics/daily?days=30" | head
```

## 3) Settlement + métricas (DB real)
### 3.1 Contagens antes
```bash
python - <<'PY'
import sqlite3
conn=sqlite3.connect('data/sports_ev.db')
cur=conn.cursor()
print('Before:', cur.execute("SELECT result, COUNT(*) FROM signals GROUP BY result ORDER BY result").fetchall())
conn.close()
PY
```

### 3.2 Rodar settlement (contra DB real)
```bash
python - <<'PY'
import asyncio
from app.services.settlement_service import SettlementService

async def main():
    svc = SettlementService(db_path='data/sports_ev.db')
    out = await svc.settle_pending(limit=500)
    print('settlement:', out)

asyncio.run(main())
PY
```

### 3.3 Contagens depois (não pode "sumir" nada)
```bash
python - <<'PY'
import sqlite3
conn=sqlite3.connect('data/sports_ev.db')
cur=conn.cursor()
print('After:', cur.execute("SELECT result, COUNT(*) FROM signals GROUP BY result ORDER BY result").fetchall())
print('metrics_daily rows:', cur.execute('SELECT COUNT(*) FROM metrics_daily').fetchone()[0])
print('latest metrics:', cur.execute('SELECT day_utc, market, n_resolved, roi, brier, logloss, rolling_alert FROM metrics_daily ORDER BY id DESC LIMIT 5').fetchall())
conn.close()
PY
```

## 4) Testes unitários
```bash
pytest -q
```

## 5) Rollback rápido
```bash
# 1) pare o serviço
# 2) restaure o zip anterior do projeto
# 3) se quiser remover apenas as métricas:
python - <<'PY'
import sqlite3
conn=sqlite3.connect('data/sports_ev.db')
cur=conn.cursor()
cur.execute('DROP TABLE IF EXISTS metrics_daily')
conn.commit()
conn.close()
print('OK: metrics_daily removida')
PY
```
