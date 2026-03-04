"""
RuntimeAuditService — Serviço de auditoria de runtime settings.

Toda alteração de configuração passa por aqui.
Gera registro imutável no banco para rastreabilidade completa.

Faz parte da FASE 8 — Auditoria de Runtime Settings.
"""

import json
from datetime import datetime, timezone
from typing import Optional, Any
from loguru import logger


class RuntimeAuditService:
    """
    Serviço de auditoria de alterações em runtime settings.

    Uso:
        audit = RuntimeAuditService(db)

        # Registra uma mudança de setting
        await audit.log_change(
            setting_key="min_ev_threshold",
            setting_path="strategies.standard.min_ev",
            old_value=0.05,
            new_value=0.05,
            changed_by="admin",
            reason="Performance abaixo do esperado",
        )
    """

    def __init__(self, db=None):
        """
        Args:
            db: sessão AsyncSession do SQLAlchemy (opcional para modo offline)
        """
        self.db = db
        self._in_memory_log: list[dict] = []  # fallback se banco indisponível

    async def log_change(
        self,
        setting_key: str,
        setting_path: str,
        old_value: Any,
        new_value: Any,
        changed_by: str = "system",
        source: str = "api",
        reason: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> dict:
        """
        Registra uma alteração de setting.

        Args:
            setting_key: chave do setting (ex: "min_ev_threshold")
            setting_path: caminho completo (ex: "strategies.standard.filters.min_ev")
            old_value: valor anterior (qualquer tipo, será serializado)
            new_value: novo valor
            changed_by: usuário/sistema que fez a mudança
            source: origem da mudança ("api", "telegram", "migration", "system")
            reason: motivo da mudança (opcional mas recomendado)
            ip_address: IP da requisição (se disponível)

        Returns:
            dict com o registro criado
        """
        record = {
            "setting_key": setting_key,
            "setting_path": setting_path,
            "old_value": self._serialize(old_value),
            "new_value": self._serialize(new_value),
            "changed_by": changed_by,
            "source": source,
            "reason": reason,
            "ip_address": ip_address,
            "changed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Tenta salvar no banco
        if self.db is not None:
            try:
                from app.models.audit_models import RuntimeAuditLog
                audit_entry = RuntimeAuditLog(
                    setting_key=setting_key,
                    setting_path=setting_path,
                    old_value=record["old_value"],
                    new_value=record["new_value"],
                    changed_by=changed_by,
                    source=source,
                    reason=reason,
                    ip_address=ip_address,
                    changed_at=datetime.now(timezone.utc),
                )
                self.db.add(audit_entry)
                await self.db.commit()
                logger.info(
                    f"[Audit] {setting_key}: {old_value!r} → {new_value!r} "
                    f"(by: {changed_by}, via: {source})"
                )
            except Exception as e:
                logger.error(f"[Audit] Erro ao salvar no banco: {e}")
                # Fallback: salva em memória
                self._in_memory_log.append(record)
        else:
            # Modo offline: apenas log e memória
            self._in_memory_log.append(record)
            logger.info(
                f"[Audit][OFFLINE] {setting_key}: {old_value!r} → {new_value!r} "
                f"(by: {changed_by})"
            )

        return record

    def _serialize(self, value: Any) -> str:
        """Serializa qualquer valor para string."""
        if value is None:
            return "null"
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str)
        return str(value)

    async def get_history(
        self,
        setting_key: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Retorna histórico de alterações.

        Args:
            setting_key: filtrar por chave específica (None = todas)
            limit: máximo de registros a retornar
        """
        if self.db is not None:
            try:
                from sqlalchemy import select, desc
                from app.models.audit_models import RuntimeAuditLog

                query = select(RuntimeAuditLog).order_by(desc(RuntimeAuditLog.changed_at)).limit(limit)
                if setting_key:
                    query = query.where(RuntimeAuditLog.setting_key == setting_key)

                result = await self.db.execute(query)
                rows = result.scalars().all()

                return [
                    {
                        "id": r.id,
                        "setting_key": r.setting_key,
                        "setting_path": r.setting_path,
                        "old_value": r.old_value,
                        "new_value": r.new_value,
                        "changed_by": r.changed_by,
                        "source": r.source,
                        "reason": r.reason,
                        "changed_at": r.changed_at.isoformat() if r.changed_at else None,
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.error(f"[Audit] Erro ao buscar histórico: {e}")

        # Fallback: retorna memória
        records = self._in_memory_log
        if setting_key:
            records = [r for r in records if r["setting_key"] == setting_key]
        return records[-limit:]

    async def log_strategy_change(
        self,
        strategy_id: str,
        field: str,
        old_val: Any,
        new_val: Any,
        changed_by: str = "system",
    ):
        """Atalho para logar mudança em parâmetro de estratégia."""
        await self.log_change(
            setting_key=f"strategy.{strategy_id}.{field}",
            setting_path=f"strategies.{strategy_id}.{field}",
            old_value=old_val,
            new_value=new_val,
            changed_by=changed_by,
            source="api",
        )

    def get_in_memory_log(self) -> list[dict]:
        """Retorna log em memória (para debug ou quando banco está indisponível)."""
        return list(self._in_memory_log)
