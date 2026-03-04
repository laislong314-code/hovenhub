"""
DatabaseAdapter — Camada de abstração de banco de dados.

Permite troca transparente entre SQLite (atual) e PostgreSQL (futuro)
sem quebrar a API existente.

Separa as queries da engine de banco, facilitando:
  1. Migração para PostgreSQL sem reescrever toda a lógica
  2. Testes com banco em memória
  3. Potencial suporte a outros bancos no futuro

Faz parte da FASE 9 — Preparar Migração para PostgreSQL.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Any, AsyncGenerator
from loguru import logger

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy import text


class DatabaseConfig:
    """
    Configuração de banco de dados com suporte a múltiplos backends.

    Detecção automática baseada em DATABASE_URL:
        sqlite+aiosqlite:///./betoven.db  → SQLite (padrão)
        postgresql+asyncpg://user:pass@host/db → PostgreSQL
    """

    def __init__(self):
        self.url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./betoven.db")
        self.is_sqlite = self.url.startswith("sqlite")
        self.is_postgres = self.url.startswith("postgresql") or self.url.startswith("postgres")

    def get_engine_kwargs(self) -> dict:
        """Retorna kwargs otimizados para o backend detectado."""
        if self.is_sqlite:
            return {
                "connect_args": {"check_same_thread": False},
                "echo": False,
                "pool_pre_ping": True,
            }
        elif self.is_postgres:
            return {
                "echo": False,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_pre_ping": True,
                "pool_recycle": 300,
            }
        return {}

    def __repr__(self):
        backend = "SQLite" if self.is_sqlite else "PostgreSQL" if self.is_postgres else "Unknown"
        return f"DatabaseConfig({backend})"


class BaseDatabaseAdapter(ABC):
    """Interface abstrata para operações de banco de dados."""

    @abstractmethod
    async def get_session(self) -> AsyncSession:
        """Retorna uma sessão de banco de dados."""
        ...

    @abstractmethod
    async def execute_query(self, query: str, params: dict = None) -> list:
        """Executa uma query SQL bruta."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verifica se o banco está acessível."""
        ...


class SQLAlchemyAdapter(BaseDatabaseAdapter):
    """
    Adapter concreto para SQLAlchemy Async (SQLite ou PostgreSQL).

    A mesma implementação funciona para ambos os bancos pois:
    - SQLite: usa aiosqlite como driver async
    - PostgreSQL: usa asyncpg como driver async
    - A API do SQLAlchemy é idêntica para ambos
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None

        logger.info(f"[DatabaseAdapter] Inicializando: {self.config}")

    def _get_engine(self) -> AsyncEngine:
        """Cria a engine (lazy initialization)."""
        if self._engine is None:
            kwargs = self.config.get_engine_kwargs()
            self._engine = create_async_engine(self.config.url, **kwargs)
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._engine

    async def get_session(self) -> AsyncSession:
        """Retorna nova sessão de banco."""
        engine = self._get_engine()
        async with self._session_factory() as session:
            return session

    async def execute_query(
        self,
        query: str,
        params: Optional[dict] = None,
    ) -> list:
        """
        Executa query SQL bruta (compatível com ambos os backends).

        IMPORTANTE: para queries com placeholders, use sintaxe compatível:
            SQLite: :param_name
            PostgreSQL: :param_name (funciona igual via SQLAlchemy)
        """
        engine = self._get_engine()
        async with self._session_factory() as session:
            result = await session.execute(text(query), params or {})
            return result.fetchall()

    async def health_check(self) -> bool:
        """Verifica conectividade com o banco."""
        try:
            engine = self._get_engine()
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"[DatabaseAdapter] Health check falhou: {e}")
            return False

    async def create_tables(self, base):
        """Cria todas as tabelas definidas no Base do SQLAlchemy."""
        engine = self._get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(base.metadata.create_all)
        logger.info("[DatabaseAdapter] Tabelas criadas/verificadas")

    async def run_migration(self, sql: str):
        """
        Executa uma migration SQL.

        Agrupa os statements e executa um por um para compatibilidade.
        Ignora erros de "column already exists" (idempotente).
        """
        engine = self._get_engine()
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        async with engine.begin() as conn:
            for stmt in statements:
                if not stmt:
                    continue
                try:
                    await conn.execute(text(stmt))
                    logger.debug(f"[Migration] Executado: {stmt[:60]}...")
                except Exception as e:
                    err_str = str(e).lower()
                    if "already exists" in err_str or "duplicate column" in err_str:
                        logger.debug(f"[Migration] Ignorado (já existe): {stmt[:60]}...")
                    else:
                        logger.warning(f"[Migration] Aviso: {e}")

        logger.info("[DatabaseAdapter] Migration concluída")

    def get_backend_name(self) -> str:
        """Retorna o nome do backend atual."""
        if self.config.is_sqlite:
            return "SQLite"
        elif self.config.is_postgres:
            return "PostgreSQL"
        return "Unknown"

    async def dispose(self):
        """Fecha todas as conexões."""
        if self._engine:
            await self._engine.dispose()
            logger.info("[DatabaseAdapter] Conexões fechadas")


# ── Queries compatíveis com ambos os backends ─────────────────────────────────

class CompatibleQueries:
    """
    Queries SQL escritas para máxima compatibilidade SQLite/PostgreSQL.

    Diferenças principais:
        SQLite: AUTOINCREMENT, DATETIME, TEXT
        PostgreSQL: SERIAL, TIMESTAMP, VARCHAR

    SQLAlchemy ORM abstrai isso, mas para queries raw precisamos de cuidado.
    """

    @staticmethod
    def get_recent_signals(n: int = 50) -> str:
        """Últimos N sinais gerados."""
        return """
        SELECT
            s.id,
            s.match_id,
            s.market,
            s.model_probability,
            s.ev,
            s.suggested_odd,
            s.sent_at,
            s.result,
            s.profit_loss,
            COALESCE(s.model_version, 'legacy') as model_version,
            COALESCE(s.strategy_id, 'standard') as strategy_id
        FROM signals s
        ORDER BY s.sent_at DESC
        LIMIT :n
        """

    @staticmethod
    def get_performance_by_strategy() -> str:
        """Performance agregada por estratégia."""
        return """
        SELECT
            COALESCE(strategy_id, 'standard') as strategy_id,
            COUNT(*) as total_bets,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
            AVG(ev) as avg_ev,
            SUM(profit_loss) as total_profit
        FROM signals
        WHERE result IS NOT NULL
        GROUP BY strategy_id
        ORDER BY SUM(profit_loss) DESC
        """

    @staticmethod
    def get_calibration_data(strategy_id: str, market: str) -> str:
        """Dados de calibração para uma estratégia/mercado."""
        return """
        SELECT
            s.model_probability,
            CASE WHEN s.result = 'WIN' THEN 1 ELSE 0 END as outcome,
            s.market,
            s.sent_at
        FROM signals s
        WHERE s.result IS NOT NULL
          AND COALESCE(s.strategy_id, 'standard') = :strategy_id
          AND s.market = :market
        ORDER BY s.sent_at ASC
        """

    @staticmethod
    def get_audit_log(limit: int = 100) -> str:
        """Histórico de alterações de settings."""
        return """
        SELECT
            id,
            setting_key,
            setting_path,
            old_value,
            new_value,
            changed_by,
            source,
            reason,
            changed_at
        FROM runtime_audit_log
        ORDER BY changed_at DESC
        LIMIT :limit
        """


# ── Singleton ─────────────────────────────────────────────────────────────────

_adapter_instance: Optional[SQLAlchemyAdapter] = None


def get_database_adapter() -> SQLAlchemyAdapter:
    """Retorna a instância singleton do DatabaseAdapter."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = SQLAlchemyAdapter()
    return _adapter_instance


# ── Guia de migração para PostgreSQL ─────────────────────────────────────────

POSTGRES_MIGRATION_GUIDE = """
GUIA DE MIGRAÇÃO SQLite → PostgreSQL

1. Instalar dependências:
   pip install asyncpg sqlalchemy[asyncpg]

2. Configurar DATABASE_URL no .env:
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/betoven

3. Criar banco PostgreSQL:
   CREATE DATABASE betoven;
   CREATE USER betoven_user WITH PASSWORD 'sua_senha';
   GRANT ALL PRIVILEGES ON DATABASE betoven TO betoven_user;

4. Exportar dados do SQLite:
   python scripts/migrate_sqlite_to_postgres.py

5. Reiniciar a aplicação — o DatabaseAdapter detecta automaticamente o backend.

DIFERENÇAS CONHECIDAS SQLite vs PostgreSQL:
  - AUTOINCREMENT → SERIAL (transparente via SQLAlchemy)
  - TEXT → VARCHAR (transparente via SQLAlchemy)
  - DATETIME → TIMESTAMP (transparente via SQLAlchemy)
  - Queries com LIKE: case-sensitive no PostgreSQL por padrão
  - JSON: SQLite usa TEXT, PostgreSQL usa JSONB (mais eficiente)

BENEFÍCIOS DO POSTGRESQL:
  - Suporte nativo a JSON com queries
  - Full-text search nativo
  - LISTEN/NOTIFY para eventos em tempo real
  - Melhor performance em grandes volumes
  - Suporte a concurrent writes
  - Backup incremental nativo
"""
