"""
Conexão assíncrona com o banco de dados — SQLAlchemy 2.0 + asyncpg (Postgres)
"""
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
from loguru import logger

from app.config import get_settings

settings = get_settings()

# Converte DATABASE_URL do formato postgres:// para postgresql+asyncpg://
def _build_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url

_db_url = _build_db_url(os.environ.get("DATABASE_URL", settings.database_url))

engine = create_async_engine(
    _db_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Cria todas as tabelas definidas nos models se não existirem."""
    from app.models.db_models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Migrations: adiciona colunas novas sem apagar dados existentes
        migrations = [
            ("ALTER TABLE signals ADD COLUMN label VARCHAR(256)",      "label"),
            ("ALTER TABLE signals ADD COLUMN score_home INTEGER",      "score_home"),
            ("ALTER TABLE signals ADD COLUMN score_away INTEGER",      "score_away"),
            ("ALTER TABLE signals ADD COLUMN actual_goals INTEGER",    "actual_goals"),
            ("ALTER TABLE signals ADD COLUMN profit_loss FLOAT",       "profit_loss"),
        ]
        for sql, col in migrations:
            try:
                await conn.execute(text(sql))
                logger.info(f"✅ Migration: coluna '{col}' adicionada em signals")
            except Exception:
                pass  # coluna já existe

        # Metrics table (Phase 1) — safe, additive migration
        try:
            await conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS metrics_daily (
                    id SERIAL PRIMARY KEY,
                    day_utc VARCHAR(10) NOT NULL,
                    market VARCHAR(64) NOT NULL,
                    n_resolved INTEGER NOT NULL,
                    roi FLOAT NOT NULL,
                    roi_ci_low FLOAT NULL,
                    roi_ci_high FLOAT NULL,
                    brier FLOAT NULL,
                    logloss FLOAT NULL,
                    brier_skill_vs_implied FLOAT NULL,
                    spearman_rho_ev_pl FLOAT NULL,
                    spearman_p_ev_pl FLOAT NULL,
                    rolling_brier_7 FLOAT NULL,
                    rolling_brier_14 FLOAT NULL,
                    rolling_alert BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP NOT NULL
                )
                """
            ))
            await conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_metrics_daily_day_market ON metrics_daily(day_utc, market)"
            ))
            logger.info("✅ Migration: metrics_daily pronta")
        except Exception as e:
            logger.warning(f"⚠️ Migration metrics_daily falhou: {e}")
    logger.info("✅ Banco de dados inicializado")


async def health_check() -> bool:
    """Verifica se o banco responde corretamente."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"❌ Falha no health check do banco: {e}")
        return False


async def get_db():
    """Dependency injection do FastAPI — fornece sessão do banco por request."""
    async with AsyncSessionLocal() as session:
        yield session