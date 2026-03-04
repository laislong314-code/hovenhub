"""
Modelos de banco de dados — SQLAlchemy 2.0 (async)
"""
from datetime import datetime
from sqlalchemy import String, Float, DateTime, Boolean, Integer, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Match(Base):
    """
    Partida coletada da API.
    Armazena dados brutos + status de processamento.
    """
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # ID da API
    sport_key: Mapped[str] = mapped_column(String(64))             # ex: soccer_epl
    sport_title: Mapped[str] = mapped_column(String(128))
    home_team: Mapped[str] = mapped_column(String(128))
    away_team: Mapped[str] = mapped_column(String(128))
    home_logo: Mapped[str | None] = mapped_column(String(512), nullable=True)
    away_logo: Mapped[str | None] = mapped_column(String(512), nullable=True)
    commence_time: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relacionamentos
    odds: Mapped[list["OddsSnapshot"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    signals: Mapped[list["Signal"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    analysis: Mapped[list["MatchAnalysis"]] = relationship(back_populates="match", cascade="all, delete-orphan")


class OddsSnapshot(Base):
    """
    Snapshot de odds no momento da coleta.
    Permite rastrear movimentação de linha ao longo do tempo.
    """
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(String(64), ForeignKey("matches.id"))
    bookmaker: Mapped[str] = mapped_column(String(64))
    market: Mapped[str] = mapped_column(String(64))       # ex: totals, h2h
    outcome: Mapped[str] = mapped_column(String(64))      # ex: Over, Under, Home
    price: Mapped[float] = mapped_column(Float)           # odd decimal
    point: Mapped[float | None] = mapped_column(Float, nullable=True)  # linha (ex: 2.5)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    match: Mapped["Match"] = relationship(back_populates="odds")


class MatchAnalysis(Base):
    """
    Resultado da análise estatística Poisson para um jogo.
    """
    __tablename__ = "match_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(String(64), ForeignKey("matches.id"))
    market: Mapped[str] = mapped_column(String(64))            # ex: over_2.5
    lambda_home: Mapped[float] = mapped_column(Float)          # λ esperado time casa
    lambda_away: Mapped[float] = mapped_column(Float)          # λ esperado time visitante
    lambda_total: Mapped[float] = mapped_column(Float)         # λ total da partida
    model_probability: Mapped[float] = mapped_column(Float)    # P(Over 2.5) pelo modelo
    best_odd: Mapped[float] = mapped_column(Float)             # Melhor odd encontrada
    best_bookmaker: Mapped[str] = mapped_column(String(64))
    implied_probability: Mapped[float] = mapped_column(Float)  # 1 / odd
    ev: Mapped[float] = mapped_column(Float)                   # EV calculado
    analyzed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Dados de forma usados no cálculo (para auditoria)
    home_avg_scored: Mapped[float] = mapped_column(Float)
    home_avg_conceded: Mapped[float] = mapped_column(Float)
    away_avg_scored: Mapped[float] = mapped_column(Float)
    away_avg_conceded: Mapped[float] = mapped_column(Float)
    league_avg_goals: Mapped[float] = mapped_column(Float)

    # xG do Sofascore (opcional) — para auditoria e stacked models
    sofa_xg_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    sofa_xg_away: Mapped[float | None] = mapped_column(Float, nullable=True)

    match: Mapped["Match"] = relationship(back_populates="analysis")


class Signal(Base):
    """
    Sinal gerado e enviado via Telegram.
    Controle de duplicidade + resultado posterior.
    """
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(String(64), ForeignKey("matches.id"))
    market: Mapped[str] = mapped_column(String(64))
    label: Mapped[str | None] = mapped_column(String(256), nullable=True)  # label legível ex: "Múltipla (BTTS + Vitória Fora)"
    model_probability: Mapped[float] = mapped_column(Float)
    implied_probability: Mapped[float] = mapped_column(Float)
    ev: Mapped[float] = mapped_column(Float)
    suggested_odd: Mapped[float] = mapped_column(Float)
    bookmaker: Mapped[str] = mapped_column(String(64))
    stake_pct: Mapped[float] = mapped_column(Float)         # % do bankroll
    stake_units: Mapped[float] = mapped_column(Float)       # valor absoluto sugerido
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    telegram_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Contexto live (preenchido quando o sinal é gerado durante o jogo)
    is_live: Mapped[bool] = mapped_column(Boolean, default=False)
    match_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)  # minuto do jogo no momento do sinal

    # Resultado (preenchido após o jogo)
    result: Mapped[str | None] = mapped_column(String(16), nullable=True)  # WIN/LOSS/VOID
    actual_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_home: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_away: Mapped[int | None] = mapped_column(Integer, nullable=True)
    profit_loss: Mapped[float | None] = mapped_column(Float, nullable=True)

    match: Mapped["Match"] = relationship(back_populates="signals")


class LeagueStats(Base):
    """
    Estatísticas agregadas por liga — usadas no ajuste de força.
    Recalculadas periodicamente.
    """
    __tablename__ = "league_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport_key: Mapped[str] = mapped_column(String(64))
    season: Mapped[str] = mapped_column(String(16))          # ex: 2024
    avg_goals_per_match: Mapped[float] = mapped_column(Float)
    total_matches: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MetricsDaily(Base):
    """Métricas diárias por mercado para validação estatística avançada.

    IMPORTANT:
      - Não impacta endpoints existentes.
      - É preenchida pelo SettlementService (ou scripts offline) quando habilitado.
      - Chave composta (day_utc, market) para permitir upsert.
    """

    __tablename__ = "metrics_daily"
    __table_args__ = (
        # enables safe upsert semantics (INSERT OR REPLACE) in sqlite usage
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    day_utc: Mapped[str] = mapped_column(String(10))           # YYYY-MM-DD
    market: Mapped[str] = mapped_column(String(64))

    n_resolved: Mapped[int] = mapped_column(Integer)
    roi: Mapped[float] = mapped_column(Float)
    roi_ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi_ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)

    brier: Mapped[float | None] = mapped_column(Float, nullable=True)
    logloss: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_skill_vs_implied: Mapped[float | None] = mapped_column(Float, nullable=True)

    spearman_rho_ev_pl: Mapped[float | None] = mapped_column(Float, nullable=True)
    spearman_p_ev_pl: Mapped[float | None] = mapped_column(Float, nullable=True)

    rolling_brier_7: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_brier_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_alert: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)