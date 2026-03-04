# CHANGELOG — Betoven Hub v7 (Patch Março 2026)

## 2026-03-03

### FASE 0 — Segurança e Diagnóstico
- **Novo**: feature flags e configs de seleção/validação em `app/config.py`.
- **Novo**: variáveis correspondentes em `.env.EXAMPLE`.
- **Novo**: script de snapshot do schema do DB: `scripts/db_snapshot.py`.
- **Migração aditiva**: criação de `metrics_daily` + índice único via `app/database/connection.py`.

### FASE 1 — Métricas imediatas (baixa complexidade, alto impacto)
- **Novo**: utilitários de métricas em `app/performance/advanced_metrics.py`:
  - Brier Score
  - Log Loss (com clipping)
  - Bootstrap CI (percentile) para ROI
  - Spearman EV vs P&L
- **Novo**: persistência e cálculo das métricas em `app/performance/metrics_persistence.py`:
  - `metrics_daily` por mercado e agregado `ALL`
  - Rolling Brier (7/14) + alerta configurável
- **Integração**: `SettlementService` agora (quando `ENABLE_ADV_METRICS=true`) calcula e grava métricas sem alterar o settlement.
- **Novo endpoint**: `GET /api/v1/metrics/daily` (somente leitura).

### FASE 2 — Ajustes controlados de seleção (sem overfitting)
- **Mudança controlada**: `EV>=0.10` como default no modo `precision` (configurável via `PRECISION_EV_MIN`).
- **Centralização**: regras de seleção do modo `precision` saíram do hardcode em `routes.py` e foram centralizadas em `app/services/signal_selector.py` via `SelectorConfig.from_settings()`.

### TESTES
- **Fix**: suíte legacy `tests/test_betoven_v7.py` marcada como *skipped* (estrutura antiga).
- **Novo**: testes executáveis mínimos em `tests/test_settlement_and_metrics.py`:
  - settlement de mercados (BTTS, Over/Under, 1X2, DC_*)
  - brier/logloss com clipping
  - bootstrap CI (low<=roi<=high)


## Hotfix (2026-03-03)
- Fix: AnalysisOrchestrator now builds SelectorConfig via SelectorConfig.from_settings() (removed legacy ev_min/odd_min/odd_max kwargs) to prevent runtime crash.
