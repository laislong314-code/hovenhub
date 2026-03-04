"""
ModelRegistry — Versionamento de modelos e rastreabilidade de execuções.

Cada execução de análise é registrada com:
  - model_version: identificador único da versão do modelo
  - strategy_id: estratégia usada
  - parameters: snapshot dos parâmetros relevantes

Isso permite:
  - Rastrear qual modelo gerou cada sinal
  - Comparar performance entre versões
  - Rollback para versão anterior se performance cair
  - Auditoria completa de mudanças

Faz parte da FASE 2 — Versionamento de Modelo.
"""

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger


@dataclass
class ModelVersion:
    """Representa uma versão do modelo com seus parâmetros."""
    version_id: str       # hash SHA-256 dos parâmetros (deterministico)
    version_tag: str      # tag legível, ex: "v7.2.1-dixon-coles"
    strategy_id: str
    parameters: dict
    created_at: str
    description: str = ""
    is_active: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """
    Registro centralizado de versões de modelo.

    Princípio: qualquer mudança nos parâmetros cria uma nova versão.
    A versão é determinada pelo hash dos parâmetros, então é reproduzível.

    Uso:
        registry = ModelRegistry()

        # Registra versão
        version = registry.register_version(
            strategy_id="standard",
            parameters={"poisson_rho": -0.13, "kelly_fraction": 0.25, ...},
            tag="v7.2.1",
        )

        # Usa em sinais
        signal.model_version = version.version_id
    """

    # Versão base do sistema — incrementar manualmente em releases maiores
    BASE_VERSION = "v7.2"

    def __init__(self):
        self._registry: dict[str, ModelVersion] = {}
        self._active_versions: dict[str, str] = {}  # strategy_id -> version_id
        self._initialize_default_versions()

    def _initialize_default_versions(self):
        """Registra versões padrão para cada estratégia built-in."""
        default_params = {
            "poisson_version": "bivariate_dc",  # Dixon-Coles + Bivariate
            "rho": -0.13,
            "use_dixon_coles": True,
            "kelly_fraction": 0.25,
            "normalization": "zscore_per_league",
            "form_window": 5,
            "exponential_decay": 0.85,
            "regression_to_mean": True,
        }

        strategies = ["standard", "conservative", "aggressive", "live", "value_hunter"]
        for sid in strategies:
            params = {**default_params, "strategy_id": sid}
            v = self.register_version(
                strategy_id=sid,
                parameters=params,
                tag=f"{self.BASE_VERSION}.0",
                description=f"Versão inicial para estratégia {sid}",
            )
            self._active_versions[sid] = v.version_id
            logger.debug(f"[ModelRegistry] Versão padrão registrada: {sid} → {v.version_id[:8]}...")

    def _compute_version_id(self, strategy_id: str, parameters: dict) -> str:
        """
        Computa ID determinístico baseado em hash SHA-256 dos parâmetros.
        Mesmo conjunto de parâmetros sempre gera mesmo ID.
        """
        payload = {
            "strategy_id": strategy_id,
            "parameters": parameters,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def register_version(
        self,
        strategy_id: str,
        parameters: dict,
        tag: str = "",
        description: str = "",
    ) -> ModelVersion:
        """
        Registra uma versão de modelo.

        Se os parâmetros já existirem, retorna a versão existente (idempotente).

        Args:
            strategy_id: ID da estratégia
            parameters: dict com todos os parâmetros relevantes
            tag: tag legível (ex: "v7.2.1")
            description: descrição das mudanças

        Returns:
            ModelVersion registrada
        """
        version_id = self._compute_version_id(strategy_id, parameters)

        if version_id in self._registry:
            return self._registry[version_id]

        version_tag = tag or f"{self.BASE_VERSION}-{strategy_id}-{version_id[:6]}"

        version = ModelVersion(
            version_id=version_id,
            version_tag=version_tag,
            strategy_id=strategy_id,
            parameters=parameters,
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description,
            is_active=True,
        )

        self._registry[version_id] = version
        logger.info(f"[ModelRegistry] Nova versão: {version_tag} ({version_id[:8]}...)")

        return version

    def get_active_version(self, strategy_id: str) -> Optional[ModelVersion]:
        """Retorna a versão ativa para uma estratégia."""
        vid = self._active_versions.get(strategy_id)
        if not vid:
            return None
        return self._registry.get(vid)

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Busca versão por ID completo."""
        return self._registry.get(version_id)

    def set_active_version(self, strategy_id: str, version_id: str) -> bool:
        """Define a versão ativa para uma estratégia."""
        if version_id not in self._registry:
            logger.error(f"[ModelRegistry] Versão não encontrada: {version_id}")
            return False

        self._active_versions[strategy_id] = version_id
        logger.info(f"[ModelRegistry] Versão ativa alterada: {strategy_id} → {version_id[:8]}...")
        return True

    def get_version_id_for_signal(self, strategy_id: str) -> str:
        """
        Retorna o version_id para ser salvo no sinal.
        Fallback para string descritiva se não encontrar.
        """
        v = self.get_active_version(strategy_id)
        if v:
            return v.version_id
        return f"unknown-{strategy_id}-{datetime.now().strftime('%Y%m%d')}"

    def list_versions(self, strategy_id: Optional[str] = None) -> list[dict]:
        """Lista versões registradas com filtro opcional por estratégia."""
        versions = self._registry.values()
        if strategy_id:
            versions = [v for v in versions if v.strategy_id == strategy_id]
        return [v.to_dict() for v in versions]

    def deprecate_version(self, version_id: str):
        """Marca uma versão como inativa (não remove, mantém histórico)."""
        if version_id in self._registry:
            self._registry[version_id].is_active = False
            logger.info(f"[ModelRegistry] Versão deprecada: {version_id[:8]}...")

    def snapshot_current_config(self, strategy_id: str, config: dict) -> str:
        """
        Cria snapshot da configuração atual e retorna o version_id.
        Usado no início de cada ciclo de análise.
        """
        v = self.register_version(
            strategy_id=strategy_id,
            parameters=config,
            tag=f"{self.BASE_VERSION}-{datetime.now().strftime('%Y%m%d')}",
        )
        self._active_versions[strategy_id] = v.version_id
        return v.version_id


# Singleton global
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Retorna a instância singleton do ModelRegistry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
