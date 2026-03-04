from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SokkerPRO
    sokkerpro_base_url: str = "https://m2.sokkerpro.com"
    sokkerpro_timezone: str = "utc-3"
    sokkerpro_referee_primary_type_id: int = 6
    sokkerpro_referee_cache_ttl_hours: int = 24

    # Football-Data.org (árbitros fallback)
    football_data_api_key: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://user:pass@host/db"

    # Scheduler
    analysis_interval_minutes: int = 360
    lookahead_hours: int = 24
    live_poll_interval_seconds: int = 60   # intervalo do monitoramento live (mín: 30s)

    force_all_leagues_today: bool = False

    # Melhores sinais
    send_only_best_signals: bool = True
    best_signals_limit: int = 15

    # ── Filtros globais ───────────────────────────────────────────────────────
    min_ev_threshold: float = 0.05
    min_model_prob: float = 0.50
    min_odd: float = 1.30
    max_odd: float = 4.00

    # ── Goals (Over/Under) ────────────────────────────────────────────────────
    # EV mínimo de 3%: captura mercados com edge real sem exigir 8% impossíveis
    # Prob mínima de 52%: elimina apostas sem convicção sem cortar bons sinais
    goals_min_odd: float = 1.30
    goals_max_odd: float = 3.50
    goals_min_prob: float = 0.52
    goals_min_ev: float = 0.05

    # ── BTTS ──────────────────────────────────────────────────────────────────
    btts_min_odd: float = 1.40
    btts_max_odd: float = 3.00
    btts_min_prob: float = 0.52
    btts_min_ev: float = 0.05

    # ── Resultado 1X2 ─────────────────────────────────────────────────────────
    # Result é o mercado mais eficiente — EV de 3% já é relevante
    result_min_odd: float = 1.40
    result_max_odd: float = 4.00
    result_min_prob: float = 0.50
    result_min_ev: float = 0.05

    # ── Escanteios (odds estimadas — sem bookmaker real) ──────────────────────
    # Mantém critério um pouco mais exigente pois as odds são estimadas (sem bookie real)
    corners_min_odd: float = 1.40
    corners_max_odd: float = 3.00
    corners_min_prob: float = 0.55
    corners_min_ev: float = 0.05

    # ── Cartões (odds estimadas — sem bookmaker real) ─────────────────────────
    cards_min_odd: float = 1.40
    cards_max_odd: float = 3.00
    cards_min_prob: float = 0.55
    cards_min_ev: float = 0.05

    # Kelly
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.05
    default_bankroll: float = 1000.0

    # Ligas monitoradas — IDs numéricos SokkerPRO
    monitored_leagues: str = ""

    # Log
    log_level: str = "INFO"
    log_file: str = "logs/system.log"

    # ── Feature flags (Advanced Validation / March 2026) ────────────────────
    # Defaults chosen to avoid runtime/behavior changes unless explicitly enabled.
    enable_adv_metrics: bool = True
    enable_auto_calibration: bool = False
    enable_temp_scaling: bool = False
    enable_cusum: bool = False
    enable_bayes_kelly: bool = False

    # ── Precision selector controls (Phase 2) ───────────────────────────────
    # Doc default: EV>=0.10 (keep configurable)
    precision_ev_min: float = 0.10
    precision_odd_min: float = 1.45
    precision_odd_max: float = 2.60
    precision_top_n_default: int = 10
    # Comma-separated markets for "precision" mode.
    precision_allowed_markets: str = "DC_Home_Draw,Under_3.5,Over_1.5,Over_2.5,BTTS"
    precision_one_per_match: bool = True
    precision_allow_multi: bool = False

    # ── Metrics controls (Phase 1) ──────────────────────────────────────────
    metrics_logloss_eps: float = 1e-7
    metrics_bootstrap_n: int = 5000
    metrics_bootstrap_seed: int = 1337
    metrics_rolling_windows: str = "7,14"
    metrics_rolling_alert_bs7_threshold: float = 0.25
    metrics_rolling_alert_consecutive: int = 3

    @property
    def monitored_leagues_list(self) -> List[str]:
        if not self.monitored_leagues.strip():
            return []
        return [s.strip() for s in self.monitored_leagues.split(",") if s.strip()]


_settings_instance: "Settings | None" = None


def get_settings() -> Settings:
    """
    Retorna a instância singleton de Settings.

    Diferente de @lru_cache, permite forçar recarga via reload_settings()
    sem precisar reiniciar o processo.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reload_settings() -> Settings:
    """Força releitura do .env. Útil após alterar variáveis em runtime."""
    global _settings_instance
    _settings_instance = Settings()
    return _settings_instance
