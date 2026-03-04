"""
Runtime Audit Log — Tabela de auditoria de alterações de settings em runtime.

Registra toda mudança de configuração com:
  - timestamp de quando foi feita
  - setting alterado
  - valor anterior
  - valor novo
  - usuário responsável (se disponível)

Faz parte da FASE 8 — Auditoria de Runtime Settings.
"""

from datetime import datetime
from sqlalchemy import String, Float, DateTime, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column

# Importa a Base existente para manter compatibilidade
from app.models.db_models import Base


class RuntimeAuditLog(Base):
    """
    Tabela de auditoria de alterações em runtime settings.

    Toda chamada ao endpoint de atualização de settings deve gerar
    um registro nesta tabela. Nunca deletar registros.
    """
    __tablename__ = "runtime_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Identificação da mudança
    setting_key: Mapped[str] = mapped_column(String(128))       # ex: "min_ev_threshold"
    setting_path: Mapped[str] = mapped_column(String(256))      # ex: "strategies.standard.min_ev"

    # Valores (serializados como string para flexibilidade de tipo)
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_value: Mapped[str] = mapped_column(Text)

    # Contexto da mudança
    changed_by: Mapped[str] = mapped_column(String(128), default="system")  # usuário ou "system"
    source: Mapped[str] = mapped_column(String(64), default="api")           # "api", "telegram", "migration"
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)          # motivo opcional

    # Metadata
    changed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)


# ── Migração SQL ──────────────────────────────────────────────────────────────

MIGRATION_SQL = """
-- FASE 8: Tabela de auditoria de runtime settings
-- Execute este SQL no banco de dados para criar a tabela

CREATE TABLE IF NOT EXISTS runtime_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setting_key TEXT NOT NULL,
    setting_path TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT NOT NULL,
    changed_by TEXT DEFAULT 'system',
    source TEXT DEFAULT 'api',
    reason TEXT,
    changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_setting_key ON runtime_audit_log(setting_key);
CREATE INDEX IF NOT EXISTS idx_audit_changed_at ON runtime_audit_log(changed_at);
CREATE INDEX IF NOT EXISTS idx_audit_changed_by ON runtime_audit_log(changed_by);
"""

# ── Modelos adicionais para performance tracking ──────────────────────────────

MIGRATION_SQL_PERFORMANCE = """
-- FASE 7: Tabela de performance tracking por estratégia

CREATE TABLE IF NOT EXISTS bet_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL UNIQUE,
    match_id TEXT NOT NULL,
    market TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    league_id TEXT DEFAULT '',
    
    -- Valores da aposta
    odd REAL NOT NULL,
    stake REAL NOT NULL,
    model_prob REAL NOT NULL,
    ev REAL NOT NULL,
    
    -- Resultado (preenchido após o jogo)
    result TEXT,          -- WIN/LOSS/VOID/PUSH
    profit_loss REAL,
    
    -- Metadata
    placed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    settled_at DATETIME,
    is_live INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_bet_strategy ON bet_records(strategy_id);
CREATE INDEX IF NOT EXISTS idx_bet_market ON bet_records(market);
CREATE INDEX IF NOT EXISTS idx_bet_league ON bet_records(league_id);
CREATE INDEX IF NOT EXISTS idx_bet_version ON bet_records(model_version);

-- FASE 2: Adicionar model_version na tabela de sinais existente
-- (migration segura com fallback)
ALTER TABLE signals ADD COLUMN model_version TEXT DEFAULT 'legacy';
ALTER TABLE signals ADD COLUMN strategy_id TEXT DEFAULT 'standard';
ALTER TABLE signals ADD COLUMN confidence_score REAL DEFAULT 0.0;
ALTER TABLE signals ADD COLUMN signal_tier TEXT DEFAULT 'C';
"""

MIGRATION_SQL_CALIBRATION = """
-- FASE 3: Tabela de métricas de calibração

CREATE TABLE IF NOT EXISTS calibration_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    market TEXT NOT NULL,
    model_version TEXT NOT NULL,
    
    -- Métricas
    brier_score REAL,
    log_loss REAL,
    sharpness REAL,
    calibration_error REAL,
    max_calibration_error REAL,
    brier_skill_score REAL,
    
    -- Performance financeira
    roi REAL DEFAULT 0.0,
    yield_pct REAL DEFAULT 0.0,
    n_predictions INTEGER DEFAULT 0,
    
    -- Período
    period_start DATETIME,
    period_end DATETIME,
    calculated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cal_strategy ON calibration_metrics(strategy_id);
CREATE INDEX IF NOT EXISTS idx_cal_market ON calibration_metrics(market);
"""
