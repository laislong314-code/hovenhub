"""
AnalysisOrchestrator V2 — SokkerPRO como fonte única de dados.

Principais mudanças vs V1 (ESPN):
  - FormService.get_team_form_from_fixture() — extrai médias do próprio fixture
    → zero chamadas extras à API para forma dos times
  - Odds Bet365/1xBet reais (ESP só tinha DraftKings US, inútil para BR)
  - Prognósticos SokkerPRO usados para calibrar/validar modelo Poisson
  - Cobertura de 270 ligas vs 12 da ESPN
  - Sem ESPN, sem Sofascore, sem erros de NoneType
  - Árbitro: mantido via RefereeService (UEFA + Football-Data.org)
"""
import json
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from loguru import logger

from app.config import get_settings
from app.models.db_models import Match, MatchAnalysis, Signal
from app.services.form_service import FormService, LEAGUE_MAP, LEAGUE_AVERAGES
from app.models.stacked_model import StackedModel
from app.models.advanced_goal_model import AdvancedGoalModel
from app.services.market_probability import MarketProbability
from app.services.poisson_model import PoissonModel
from app.services.special_markets import SpecialMarketsModel
from app.services.referee_service import RefereeService
from app.services.corner_model import CornerModel
from app.services.combo_engine import ComboEngine, build_matrix, market_single_prob, combo_prob_correlated
from app.services.live_analyzer import LiveAnalyzer
from app.services.sofascore_client import SofascoreClient
from app.telegram_bot.sender import TelegramSender

settings      = get_settings()
PROFILES_PATH = Path("data/profiles.json")


def _lambda_for_over25(prob_over25: float) -> float:
    """
    Estima o lambda total de Poisson que produz a probabilidade dada de Over 2.5.
    Resolve numericamente: P(N > 2.5) = 1 - e^-λ(1 + λ + λ²/2)
    """
    import math
    if prob_over25 <= 0 or prob_over25 >= 1:
        return 0.0
    # Busca binária simples em [0.3, 8.0]
    lo, hi = 0.3, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p = 1 - math.exp(-mid) * (1 + mid + mid**2 / 2)
        if p < prob_over25:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class AnalysisOrchestrator:

    def __init__(self, db: AsyncSession):
        self.db           = db
        self.form_service = FormService()
        self.stacked        = StackedModel()
        self.market_prob    = MarketProbability()
        self.advanced_model = AdvancedGoalModel(use_bivariate=True, use_dixon_coles=True)
        self.poisson      = PoissonModel()
        self.special      = SpecialMarketsModel()
        self.corner_model = CornerModel()   # instância por sessão — sem singleton global
        self.telegram     = TelegramSender()
        self.referee_svc  = RefereeService()
        self.combo_engine = ComboEngine(settings)
        self.live_analyzer  = LiveAnalyzer()
        self.sofascore      = SofascoreClient()   # xG only — não interfere com SokkerPRO
        self._referee_profiles = self._load_referee_profiles()
        self._cycle_signals: list[dict] = []
        self._pending_signals: list[dict] = []  # candidatos (para enviar só os melhores)

        # Cache leve para estabilizar o PRÉ quando o SokkerPro oscila/retorna vazio.
        # Chave: match_id -> dict(probs)
        self._sokker_probs_cache: dict[str, dict] = {}

    def _load_referee_profiles(self) -> dict:
        if not PROFILES_PATH.exists():
            return {}
        try:
            data = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
            logger.info(f"👨‍⚖️ {len(data.get('referees', {}))} perfis de árbitros carregados")
            return data.get("referees", {})
        except Exception as e:
            logger.error(f"Erro ao carregar perfis: {e}")
            return {}

    def _get_referee_profile(self, referee_name: str):
        from app.services.special_markets import RefereeProfile
        if not referee_name or referee_name == "Unknown":
            return None
        data = self._referee_profiles.get(referee_name)
        if data:
            return RefereeProfile(
                name=referee_name,
                avg_yellow_per_game=data.get("avg_yellow", 3.5),
                avg_red_per_game=data.get("avg_red", 0.15),
                avg_booking_pts_per_game=data.get("avg_booking_pts", 38.0),
                avg_corners_per_game=data.get("avg_corners", 10.0),
                matches_officiated=data.get("matches", 0),
                strictness_index=data.get("strictness", 1.0),
            )
        return None

    # ── Loop principal ────────────────────────────────────────────────────────

    async def run_full_analysis(self, leagues: list[str] = None) -> dict:
        summary = {
            "started_at":                datetime.now(timezone.utc).isoformat(),
            "_start_dt":                 datetime.now(timezone.utc),
            "leagues_analyzed":          0,
            "matches_found":             0,
            "matches_analyzed":          0,
            "signals_generated":         0,
            "signals_skipped_duplicate": 0,
            "errors":                    [],
        }

        self._cycle_signals = []
        self._pending_signals = []

        # SokkerPRO: busca TODOS os jogos de hoje em uma única chamada
        try:
            # Jogos agendados (mini) + jogos ao vivo (livescores)
            # Jogos em andamento saem do mini e migram pro livescores — precisamos dos dois
            scheduled   = await self.form_service.get_all_todays_scheduled()
            live        = await self.form_service.sokker.get_live_matches()

            # Deduplica por id (um jogo não deve ser analisado duas vezes)
            seen = set()
            all_matches = []
            for m in scheduled + live:
                mid = m.get("id")
                if mid and mid not in seen:
                    seen.add(mid)
                    all_matches.append(m)
        except Exception as e:
            logger.error(f"Erro ao buscar jogos SokkerPRO: {e}")
            summary["errors"].append(str(e))
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            return summary

        # Filtra por ligas configuradas
        # Prioridade: parâmetro da chamada > MONITORED_LEAGUES do .env > tudo
        if leagues:
            monitored_ids = set(leagues)
        else:
            monitored_ids = set(settings.monitored_leagues_list)

        if monitored_ids:
            matches_to_analyze = [
                m for m in all_matches
                if str(m.get("league_id")) in monitored_ids
            ]
            logger.info(f"🔎 Filtro ativo: {len(monitored_ids)} ligas monitoradas → {len(matches_to_analyze)}/{len(all_matches)} jogos selecionados")
        else:
            matches_to_analyze = all_matches
            logger.info(f"🔎 Sem filtro de liga — analisando todos os {len(all_matches)} jogos do dia")

        summary["matches_found"] = len(matches_to_analyze)
        logger.info(f"📋 {len(matches_to_analyze)} jogos para analisar hoje")

        # Avisa no Telegram que a análise começou
        league_label = None
        if leagues and len(leagues) == 1:
            # Tenta pegar o nome da liga do primeiro jogo
            for m in matches_to_analyze[:1]:
                league_label = m.get("league_name") or leagues[0]
        await self.telegram.send_analysis_started(
            league_name=league_label,
            matches_found=len(matches_to_analyze),
        )

        # Agrupa por liga para log
        leagues_seen = set()
        matches_skipped = 0
        for match in matches_to_analyze:
            try:
                n = await self._analyze_match(match)
                if n != -1:
                    summary["matches_analyzed"] += 1
                    summary["signals_generated"] += n
                    leagues_seen.add(match.get("league_id", "?"))
                else:
                    matches_skipped += 1
            except Exception as e:
                logger.error(f"Erro jogo {match.get('id')}: {e}")
                summary["errors"].append(str(e))

        summary["leagues_analyzed"] = len(leagues_seen)

        # Combos multi-jogo desativados — foco em sinais simples por jogo
        # Para reativar: descomentar o bloco abaixo
        # if len(self._cycle_signals) >= 2:
        #     multi_count = await self._process_multi_game_combos()
        #     if not settings.send_only_best_signals:
        #         summary["signals_generated"] += multi_count

        # Modo "melhores sinais": envia menu Telegram com inline buttons por jogo
        if settings.send_only_best_signals and self._pending_signals:
            duration_s = (datetime.now(timezone.utc) - summary["_start_dt"]).total_seconds()
            cycle_stats = {
                "duration_s":       duration_s,
                "matches_analyzed": summary["matches_analyzed"],
                "matches_found":    summary["matches_found"],
            }
            await self._send_best_pending_signals(cycle_stats)

        summary.pop("_start_dt", None)

        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"📊 Ciclo concluído: {summary}")
        return summary

    # ── Análise de um jogo ────────────────────────────────────────────────────

    async def _analyze_match(self, match: dict) -> int:
        home_name = match.get("homeTeam", {}).get("name", "Home")
        away_name = match.get("awayTeam", {}).get("name", "Away")
        home_logo = match.get("homeTeam", {}).get("logo", "")
        away_logo = match.get("awayTeam", {}).get("logo", "")
        match_id  = str(match.get("id"))
        utc_date  = match.get("utcDate", "").strip('"')

        try:
            commence_time = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
            if commence_time.tzinfo is None:
                commence_time = commence_time.replace(tzinfo=timezone.utc)
        except Exception:
            logger.warning(f"⚠️ Data inválida {match_id}: '{utc_date}'")
            return 0

        if not self._is_within_window(commence_time):
            logger.debug(f"⏭️  Fora da janela: {home_name} vs {away_name} | kickoff={utc_date} | lookahead={settings.lookahead_hours}h")
            return -1

        league_id   = str(match.get("league_id", ""))
        league_name = match.get("league_name", league_id)
        date_str    = commence_time.strftime("%Y-%m-%d")

        # ── xG Sofascore (enriquecimento opcional — falha silenciosa)
        # Injeta sofa_xg_home / sofa_xg_away no match antes de qualquer cálculo.
        # Se a correlação falhar, match permanece inalterado e pipeline segue normal.
        try:
            match = await self.sofascore.enrich_fixture_with_xg(match)
            if match.get("sofa_xg_home") is not None:
                logger.debug(
                    f"[xG] {home_name} vs {away_name}: "
                    f"xG home={match['sofa_xg_home']:.2f} away={match['sofa_xg_away']:.2f}"
                )
        except Exception:
            pass  # nunca bloqueia o pipeline

        # ── Forma dos times — direto das médias SokkerPRO (zero chamadas extras)
        home_form = self.form_service.get_team_form_from_fixture(match, "home", league_id)
        away_form = self.form_service.get_team_form_from_fixture(match, "away", league_id)
        league_avgs = self.form_service.get_league_averages(league_id)

        # ── Sanidade de dados PRÉ
        # Se o fixture vier sem "medias" (ou incompleto), o FormService cai em
        # média da liga e o modelo pré fica muito instável (favorito pode "virar").
        # Nesses casos, mantemos pré para totais/BTTS/cantos/cartões, mas bloqueamos 1X2/DC.
        medias = match.get("medias", {}) if isinstance(match.get("medias", {}), dict) else {}
        pre_data_ok = ("home_avg_goal" in medias and "away_avg_goal" in medias)

        # ── Skip pré-jogo com dados históricos insuficientes (< MIN_RECENT_GAMES)
        # SokkerPRO seta matches_used=10 como padrão quando não há série histórica real.
        # Bloqueamos completamente o pré quando ambos os times têm poucos jogos
        # E o fixture também não tem médias confiáveis (dupla ausência de dados).
        MIN_RECENT_GAMES = 6
        home_recent = home_form.matches_used if home_form else 0
        away_recent = away_form.matches_used if away_form else 0
        insufficient_data = (home_recent < MIN_RECENT_GAMES or away_recent < MIN_RECENT_GAMES)
        if insufficient_data and not pre_data_ok:
            logger.warning(
                f"[PRE-SKIP] {match.get('homeTeam', {}).get('name', '?')} vs "
                f"{match.get('awayTeam', {}).get('name', '?')}: "
                f"dados insuficientes (home={home_recent}, away={away_recent} jogos recentes) "
                f"— sinal pré-jogo bloqueado."
            )
            return 0

        # ── Fixture completo: contém odds Bet365 pré-jogo + árbitro
        # O /mini retorna payload leve (~20 campos) sem BET365_VENCEDOR_HOME,
        # BET365_GOLS_*, BET365_AMBAS_* etc. Esses campos existem apenas no
        # /fixture/{id} completo. Buscamos sempre para garantir odds reais.
        full_fixture = await self.form_service.sokker.get_fixture(match_id)
        if full_fixture:
            # Re-parseia odds a partir do fixture completo (600+ campos Bet365)
            real_odds = self.form_service.sokker._parse_odds(full_fixture)
            if real_odds:
                match = {**match, "sokker_odds": real_odds}
                logger.debug(
                    f"[Orchestrator] Odds reais do fixture completo: "
                    f"{list(real_odds.keys())} para {home_name} vs {away_name}"
                )
            else:
                logger.debug(f"[Orchestrator] /fixture/{match_id} sem odds Bet365 — usando mini")

            # Logos do fixture completo têm prioridade (mini pode vir sem elas)
            full_home_logo = full_fixture.get("localTeamFlag") or full_fixture.get("homeTeam", {}).get("logo", "")
            full_away_logo = full_fixture.get("visitorTeamFlag") or full_fixture.get("awayTeam", {}).get("logo", "")
            if full_home_logo: home_logo = full_home_logo
            if full_away_logo: away_logo = full_away_logo
        else:
            full_fixture = {}
            logger.debug(f"[Orchestrator] /fixture/{match_id} retornou vazio")

        # ── Árbitro (usa full_fixture já buscado acima)
        referee_name, referee_stats = await self.referee_svc.resolve_referee(
            match, home_name, away_name, league_id, commence_time
        )
        if not referee_name and full_fixture:
            referee_name, referee_stats = await self.referee_svc.resolve_referee(
                full_fixture, home_name, away_name, league_id, commence_time
            )


        referee_name = referee_name or "Unknown"
        logger.info(f"🔍 Analisando: {home_name} vs {away_name} | Liga: {league_name} | Árbitro: {referee_name}")

        # ── Determina modo de análise ─────────────────────────────────────────
        match_status = match.get("status", "SCHEDULED")
        match_minute = match.get("minute")
        is_live      = match_status in ("IN_PLAY", "PAUSED")

        live_ctx: dict = {}
        mercados_viaveis: set = set()  # vazio = sem filtro (pré-jogo aceita tudo)

        if is_live:
            # ── AO VIVO: Poisson Dinâmico — λ residual baseado no estado atual
            # Bypassa o Poisson histórico completamente.
            # O LiveAnalyzer recalcula lambdas considerando: placar, minuto,
            # DAPM janelado (ritmo atual) e médias históricas da temporada.
            live_ctx         = self.live_analyzer.calculate(match, full_fixture)
            lh               = live_ctx["lambda_home"]
            la               = live_ctx["lambda_away"]
            mat              = live_ctx["mat"]
            mercados_viaveis = live_ctx["mercados_viaveis"]
            sokker_probs     = {}  # probs pré-jogo não refletem estado atual

            goals_markets = {
                "prob_over_15":  live_ctx["prob_over_15"],
                "prob_over_25":  live_ctx["prob_over_25"],
                "prob_over_35":  live_ctx["prob_over_35"],
                "prob_under_25": live_ctx["prob_under_25"],
                "prob_btts":     live_ctx["prob_btts"],
                "prob_no_btts":  live_ctx["prob_no_btts"],
                "prob_home_win": live_ctx["prob_home_win"],
                "prob_draw":     live_ctx["prob_draw"],
                "prob_away_win": live_ctx["prob_away_win"],
            }

            logger.info(
                f"🔴 [AO VIVO] {home_name} vs {away_name} | "
                f"{live_ctx['gols_home']}x{live_ctx['gols_away']} {match_minute}' | "
                f"λ_res={lh:.2f}/{la:.2f} | "
                f"ritmo home={live_ctx['fator_ritmo_home']:.2f} away={live_ctx['fator_ritmo_away']:.2f} | "
                f"{len(mercados_viaveis)} mercados viáveis"
            )

        else:
            # ── PRÉ-JOGO: Poisson histórico (fluxo original, sem alteração)
            pa = self.poisson.analyze_match(
                match_id=match_id,
                home_team=home_name, away_team=away_name,
                league=league_name, commence_time=commence_time,
                home_form=home_form, away_form=away_form,
                league_home_avg=league_avgs["home_avg"],
                league_away_avg=league_avgs["away_avg"],
            )

            lh = max(0.1, pa.lambda_home)
            la = max(0.1, pa.lambda_away)

            # ── Ajuste de λ via xG Sofascore (quando disponível)
            # Blenda o λ do Poisson histórico com o xG real do jogo.
            # Peso conservador (30% xG / 70% Poisson) para não depender
            # exclusivamente de uma fonte não-oficial.
            sofa_xg_home = match.get("sofa_xg_home")
            sofa_xg_away = match.get("sofa_xg_away")
            if sofa_xg_home is not None and sofa_xg_away is not None:
                xg_h = max(0.1, float(sofa_xg_home))
                xg_a = max(0.1, float(sofa_xg_away))
                lh = round(0.70 * lh + 0.30 * xg_h, 4)
                la = round(0.70 * la + 0.30 * xg_a, 4)
                logger.debug(
                    f"[xG blend] {home_name} vs {away_name}: "
                    f"λh {pa.lambda_home:.2f}→{lh:.2f} "
                    f"λa {pa.lambda_away:.2f}→{la:.2f}"
                )

            # SokkerPro às vezes retorna {} intermitentemente.
            # Para evitar "saltos" no pré (favorito muda do nada), reutilizamos
            # o último prognóstico válido do mesmo match.
            sokker_probs = self.form_service.sokker.get_prognostico_probs(match) or {}
            if sokker_probs:
                self._sokker_probs_cache[match_id] = sokker_probs
            else:
                sokker_probs = self._sokker_probs_cache.get(match_id, {})
            has_1x2 = sokker_probs.get("prob_home_win_sokker", 0) > 0 or sokker_probs.get("prob_away_win_sokker", 0) > 0
            has_ou  = sokker_probs.get("prob_over_25_sokker", 0) > 0
            if has_1x2 or has_ou:
                lh, la = self._calibrate_lambdas_with_sokker(lh, la, sokker_probs)

            mat           = build_matrix(lh, la)
            goals_markets = self.special.calculate_goals_markets(lh, la)

        # ── Árbitro profile (profiles.json) + fallback via stats SokkerPRO
        referee_profile = self._get_referee_profile(referee_name)

        if referee_profile is None and referee_name != "Unknown":
            try:
                from app.services.special_markets import RefereeProfile
                # Se veio stats do SokkerPRO, monta um profile "on the fly"
                if referee_stats:
                    y = referee_stats.get("yellow_avg")
                    r = referee_stats.get("red_avg")
                    yr = referee_stats.get("yellowred_avg")
                    bpts = referee_stats.get("booking_pts_avg")
                    matches_off = referee_stats.get("matches") or 0

                    # Strictness: normaliza pelo league avg se existir; senão usa 1.0
                    league_avg_bpts = 38.0
                    try:
                        league_avg_bpts = float(league_avgs.get("avg_booking_pts", 38.0)) if isinstance(league_avgs, dict) else 38.0
                    except Exception:
                        league_avg_bpts = 38.0

                    strict = 1.0
                    if bpts and league_avg_bpts and league_avg_bpts > 0:
                        strict = max(0.75, min(float(bpts) / float(league_avg_bpts), 1.45))

                    referee_profile = RefereeProfile(
                        name=referee_name,
                        avg_yellow_per_game=float(y) if y is not None else 3.5,
                        avg_red_per_game=float(r) if r is not None else 0.15,
                        avg_booking_pts_per_game=float(bpts) if bpts is not None else 38.0,
                        matches_officiated=int(matches_off),
                        strictness_index=float(strict),
                    )
            except Exception:
                pass

        # ── Escanteios via CornerModel (modelo Poisson principal)
        # Injeta referee_avg_corners no fixture para que o modelo consuma diretamente.
        # Isso mantém a assinatura do modelo limpa e evita qualquer acoplamento de parâmetros.
        from app.services.special_markets import TeamCornerProfile, TeamCardProfile
        match_enriched = {
            **match,
            "referee_avg_corners": (
                referee_profile.avg_corners_per_game if referee_profile else None
            ),
        }
        corner_probs   = self.corner_model.get_probabilities(match_enriched)
        lambda_corners = corner_probs["lambda_total"]
        corners_data   = {
            "lambda_total":    lambda_corners,
            "prob_over_85":    corner_probs["prob_over_85"],
            "prob_over_95":    corner_probs["prob_over_95"],
            "prob_over_105":   corner_probs["prob_over_105"],
            "prob_under_95":   corner_probs["prob_under_95"],
        }
        # ── Cartões
        cards_data = self.special.calculate_cards(
            TeamCardProfile(
                team_name=home_name,
                avg_yellow_for=match.get("medias", {}).get("home_avg_yellow") or 1.8,
            ),
            TeamCardProfile(
                team_name=away_name,
                avg_yellow_for=match.get("medias", {}).get("away_avg_yellow") or 1.8,
            ),
            referee_profile,
        )

        # Ao vivo: substitui corners e cards pelos valores residuais do LiveAnalyzer
        # (refletem apenas os minutos restantes, não o jogo completo)
        if is_live:
            corners_data = {
                "lambda_total":  live_ctx["lambda_corners"],
                "prob_over_85":  live_ctx["prob_over_85_corners"],
                "prob_over_95":  live_ctx["prob_over_95_corners"],
                "prob_over_105": live_ctx["prob_over_105_corners"],
                "prob_under_95": live_ctx["prob_under_95_corners"],
            }
            cards_data = {
                **cards_data,   # mantém referee_factor
                "lambda_booking_pts": live_ctx["lambda_booking_pts"],
                "prob_over_20_bpts":  live_ctx["prob_over_20_bpts"],
                "prob_over_30_bpts":  live_ctx["prob_over_30_bpts"],
                "prob_over_40_bpts":  live_ctx["prob_over_40_bpts"],
                "prob_over_50_bpts":  live_ctx["prob_over_50_bpts"],
                "prob_under_40_bpts": live_ctx["prob_under_40_bpts"],
            }

        # ── Odds SokkerPRO (Bet365 + 1xBet + Over/Under dos prognósticos)
        all_odds = self.form_service.sokker.extract_odds_for_orchestrator(match)
        all_odds = self._normalize_odds_keys(all_odds)

        # ── Sinais diretos do CornerModel (odds reais extraídas da fixture)
        # Esses sinais usam exclusivamente odds reais da Bet365/1xBet (linhas inteiras).
        # Têm prioridade absoluta — sobrescrevem qualquer valor anterior no all_odds.
        corner_signals_direct = self.corner_model.analyze(fixture=match_enriched)
        for cs in corner_signals_direct:
            all_odds[cs.internal_key] = {
                "odd":        cs.odd,
                "bookmaker":  cs.bookmaker,
                "model_prob": cs.model_prob,
                "label":      cs.label,
            }
        if corner_signals_direct:
            logger.debug(
                f"🎯 CornerModel sinais reais para {home_name} vs {away_name}: "
                f"{[cs.internal_key for cs in corner_signals_direct]}"
            )

        if all_odds:
            logger.debug(f"💰 Odds para {home_name} vs {away_name}: {list(all_odds.keys())}")
        else:
            logger.warning(f"⚠️ Sem odds SokkerPRO para {home_name} vs {away_name}")

        full_analysis = {
            "match_id":    match_id,
            "pre_data_ok": pre_data_ok,
            "home_team":   home_name,
            "away_team":   away_name,
            "league":      league_name,
            "league_id":   league_id,
            "sofa_xg_home": match.get("sofa_xg_home"),
            "sofa_xg_away": match.get("sofa_xg_away"),
            "commence_time": commence_time,
            "referee":     referee_name,
            "referee_strictness":   (referee_profile.strictness_index if referee_profile else 1.0),
            "referee_avg_yellow":   (referee_profile.avg_yellow_per_game if referee_profile else 3.5),
            "referee_avg_red":      (referee_profile.avg_red_per_game if referee_profile else 0.15),
            "referee_penalties_avg": (referee_stats.get("penalties_avg") if referee_stats else None),
            "referee_fouls_avg":     (referee_stats.get("fouls_avg") if referee_stats else None),
            "referee_avg_bpts":     (referee_profile.avg_booking_pts_per_game if referee_profile else 38.0),
            "referee_avg_corners":  (referee_profile.avg_corners_per_game if referee_profile else 10.3),
            "referee_matches":      (referee_profile.matches_officiated if referee_profile else 0),
            "lambda_home":   lh,
            "lambda_away":   la,
            "lambda_total":  lh + la,
            "prob_over_15":  goals_markets["prob_over_15"],
            "prob_over_25":  goals_markets["prob_over_25"],
            "prob_over_35":  goals_markets["prob_over_35"],
            "prob_under_25": goals_markets["prob_under_25"],
            "prob_btts":     goals_markets["prob_btts"],
            "prob_no_btts":  goals_markets["prob_no_btts"],
            "prob_home_win": goals_markets["prob_home_win"],
            "prob_draw":     goals_markets["prob_draw"],
            "prob_away_win": goals_markets["prob_away_win"],
            "lambda_corners":        corners_data["lambda_total"],
            "prob_over_85_corners":  corners_data["prob_over_85"],
            "prob_over_95_corners":  corners_data["prob_over_95"],
            "prob_over_105_corners": corners_data["prob_over_105"],
            "prob_under_95_corners": corners_data["prob_under_95"],
            "lambda_booking_pts":    cards_data["lambda_booking_pts"],
            "referee_factor":        cards_data["referee_factor"],
            "prob_over_20_bpts":     cards_data["prob_over_20_bpts"],
            "prob_over_30_bpts":     cards_data["prob_over_30_bpts"],
            "prob_over_40_bpts":     cards_data["prob_over_40_bpts"],
            "prob_over_50_bpts":     cards_data["prob_over_50_bpts"],
            "prob_under_40_bpts":    cards_data["prob_under_40_bpts"],
            "odds":          all_odds,
            "mat":           mat,
            "home_injuries": [],   # SokkerPRO não fornece lesões individuais
            "away_injuries": [],
            "home_out_count": 0,
            "away_out_count": 0,
            "h2h":     [],         # SokkerPRO não fornece H2H direto
            "venue":   {},
            "lineups": {},
            "home_form_str": "",
            "away_form_str": "",
            # Extras SokkerPRO
            "sokker_probs":  sokker_probs,
            "dapm":          match.get("dapm", {}),
            "medias":        match.get("medias", {}),
            # Logos dos times (CDN Sportmonks via SokkerPRO)
            "home_logo":     home_logo,
            "away_logo":     away_logo,
            # Contexto live
            "match_status":  match_status,
            "match_minute":  match_minute,
            "is_live":       is_live,
            # Mercados viáveis ao vivo (vazio = sem restrição no pré-jogo)
            "mercados_viaveis": mercados_viaveis,
            # Contexto detalhado ao vivo (fatores de ajuste para debug/log)
            "live_ctx": live_ctx,
        }

        await self._persist_match_and_analysis(full_analysis, home_form, away_form, league_avgs)
        signals_sent = await self._process_all_signals(full_analysis)
        return signals_sent

    def _calibrate_lambdas_with_sokker(
        self, lh: float, la: float, sokker_probs: dict
    ) -> tuple[float, float]:
        """
        Deriva lambdas diretamente das probabilidades do SokkerPRO.

        O SokkerPRO já calculou P(home win), P(away win) e P(Over 2.5) corretamente
        a partir do histórico real. Nosso modelo Poisson usa médias de gols que chegam
        mal mapeadas (avg_conceded = avg do adversário, não média concedida pelo time),
        gerando lambdas inflados. Por isso o SokkerPRO é a fonte primária aqui.

        Estratégia:
          1. Deriva lambda_total via P(Over 2.5) do SokkerPRO (busca binária)
          2. Divide lambda_total usando a razão home/away implícita em P(1X2)
          3. Fallback para lambdas originais se SokkerPRO não tiver dados suficientes
        """
        try:
            p_home   = sokker_probs.get("prob_home_win_sokker", 0)
            p_away   = sokker_probs.get("prob_away_win_sokker", 0)
            p_over25 = sokker_probs.get("prob_over_25_sokker", 0)

            if p_home <= 0 and p_away <= 0 and p_over25 <= 0:
                return lh, la

            # ── Passo 1: lambda_total a partir de P(Over 2.5)
            if p_over25 > 0:
                lambda_total = _lambda_for_over25(p_over25)
            else:
                lambda_total = lh + la  # mantém total original

            # ── Passo 2: razão lh/la a partir das probabilidades 1X2
            # Em Poisson simétrico: P(home) cresce com lh/la
            # Usamos a razão de probs como proxy direto da razão de lambdas
            if p_home > 0 and p_away > 0:
                # Razão das probs 1X2 (excluindo empate) reflete a razão de lambdas
                ratio = p_home / p_away  # ex: 0.29/0.41 = 0.71 (away favorito)
                lh_new = lambda_total * ratio / (1 + ratio)
                la_new = lambda_total - lh_new
            elif p_home > 0:
                lh_new = lambda_total * 0.55
                la_new = lambda_total * 0.45
            else:
                lh_new = lambda_total * 0.45
                la_new = lambda_total * 0.55

            lh_new = max(0.3, lh_new)
            la_new = max(0.3, la_new)

            logger.debug(
                f"[Calibração] SokkerPRO → λ_home={lh_new:.2f} λ_away={la_new:.2f} "
                f"(original: {lh:.2f}/{la:.2f}) | "
                f"P(home)={p_home:.0%} P(away)={p_away:.0%} P(o2.5)={p_over25:.0%}"
            )
            return lh_new, la_new

        except Exception as e:
            logger.debug(f"[Calibração] Erro: {e} — usando lambdas originais")
            return lh, la

    # ── Geração de sinais (idêntico ao V1) ────────────────────────────────────

    # Pares de mercados mutuamente exclusivos — apenas um pode ser sinal válido por jogo
    _CONFLICT_PAIRS: list[tuple[str, str]] = [
        # Opostos simétricos — só um pode ser verdade
        ("Over_1.5",  "Under_1.5"),
        ("Over_2.5",  "Under_2.5"),
        ("Over_3.5",  "Under_3.5"),
        ("Over_4.5",  "Under_4.5"),
        ("Over_5.5",  "Under_5.5"),
        ("BTTS",      "No_BTTS"),
        ("Corners_Over_8.5",  "Corners_Under_8.5"),
        ("Corners_Over_9.5",  "Corners_Under_9.5"),
        ("Corners_Over_10.5", "Corners_Under_10.5"),
    ]

    # Sobreposição assimétrica: 1X2 contido dentro de Dupla Chance.
    # Chave = mercado simples (1X2); Valor = DC que o contém (superset).
    # Se ambos existirem no mesmo jogo, o DC é redundante com o 1X2 — descarta o DC.
    # Exemplo: "Home" já está dentro de "DC_Home_Draw" e "DC_Home_Away".
    #   → se temos sinal "Home" (mais específico e geralmente maior EV), DC vira ruído.
    #   → se só temos DC sem o 1X2, DC fica (é sinal válido por si só).
    _DC_OVERLAP: dict[str, list[str]] = {
        "Home":  ["DC_Home_Draw", "DC_Home_Away"],   # vitória casa ⊂ 1X e 12
        "Away":  ["DC_Draw_Away", "DC_Home_Away"],   # vitória fora ⊂ X2 e 12
        "Draw":  ["DC_Home_Draw", "DC_Draw_Away"],   # empate       ⊂ 1X e X2
    }

    @staticmethod
    def _resolve_conflicts(signals: list[dict]) -> list[dict]:
        """
        Remove sinais redundantes ou mutuamente exclusivos do mesmo jogo.

        Dois tipos de conflito tratados:

        1. Opostos simétricos (Over/Under, BTTS/No_BTTS):
           ambos não podem ser verdade — mantém o de maior EV.

        2. Sobreposição assimétrica (1X2 vs Dupla Chance):
           um sinal específico (ex: "Away") já está contido no DC correspondente
           (ex: "DC_Draw_Away" = X2). Gerar os dois é apostar no mesmo resultado
           duas vezes com capital diferente — quebra a gestão de banca.
           Regra: se o 1X2 específico passou nos filtros, descarta o(s) DC que o contêm.
           Mantém o DC se e somente se o 1X2 correspondente NÃO gerou sinal.
        """
        by_market: dict[str, dict] = {s["market_id"]: s for s in signals}
        to_remove: set[str] = set()

        # ── 1. Opostos simétricos ─────────────────────────────────────────────
        for mkt_a, mkt_b in AnalysisOrchestrator._CONFLICT_PAIRS:
            if mkt_a in by_market and mkt_b in by_market:
                sig_a = by_market[mkt_a]
                sig_b = by_market[mkt_b]
                ev_a  = sig_a.get("ev", 0)
                ev_b  = sig_b.get("ev", 0)

                if ev_a >= ev_b:
                    loser, winner = mkt_b, mkt_a
                else:
                    loser, winner = mkt_a, mkt_b

                to_remove.add(loser)
                logger.warning(
                    f"⚡ Conflito simétrico: [{mkt_a} EV={ev_a:.1%}] vs "
                    f"[{mkt_b} EV={ev_b:.1%}] → mantendo [{winner}], descartando [{loser}]"
                )

        # ── 2. Sobreposição assimétrica 1X2 ⊂ DC ─────────────────────────────
        for simple_mkt, dc_markets in AnalysisOrchestrator._DC_OVERLAP.items():
            if simple_mkt not in by_market:
                continue  # sem sinal 1X2 específico → DC pode ficar

            ev_simple = by_market[simple_mkt].get("ev", 0)

            for dc_mkt in dc_markets:
                if dc_mkt not in by_market:
                    continue

                ev_dc = by_market[dc_mkt].get("ev", 0)
                to_remove.add(dc_mkt)
                logger.warning(
                    f"⚡ Sobreposição DC: [{simple_mkt} EV={ev_simple:.1%}] contém "
                    f"[{dc_mkt} EV={ev_dc:.1%}] → descartando DC redundante"
                )

        if to_remove:
            filtered = [s for s in signals if s["market_id"] not in to_remove]
            logger.info(f"🚫 {len(to_remove)} sinal(is) redundante(s) removido(s): {to_remove}")
            return filtered

        return signals

    async def _process_all_signals(self, analysis: dict) -> int:
        mat      = analysis["mat"]
        odds     = analysis["odds"]
        btts_p   = analysis["prob_btts"]
        match_id = analysis["match_id"]
        sent     = 0

        singles = self.combo_engine.get_single_signals(match_id, odds, mat, btts_p)

        # ── Filtra mercados inviáveis ao vivo ─────────────────────────────────
        # Ao vivo, remove sinais de mercados que fisicamente não podem mais acontecer
        # (ex: Over 2.5 com 0x0 aos 83', BTTS com time sem gol aos 82', etc.)
        mercados_viaveis = analysis.get("mercados_viaveis", set())
        if mercados_viaveis:
            antes = len(singles)
            singles = [
                s for s in singles
                if s["market_id"] in mercados_viaveis
            ]
            removidos = antes - len(singles)
            if removidos:
                logger.info(
                    f"[Filtro Viabilidade] {analysis['home_team']} vs {analysis['away_team']}: "
                    f"{removidos} sinal(is) removido(s) por inviabilidade física ao vivo"
                )

        # ── Filtra conflitos antes de qualquer persistência ou envio
        before = len(singles)
        singles = self._resolve_conflicts(singles)
        if len(singles) < before:
            logger.info(
                f"[Anti-conflito] {analysis['home_team']} vs {analysis['away_team']}: "
                f"{before - len(singles)} sinal(is) removido(s) por conflito de mercado"
            )

        # Contexto live — extraído uma vez para todos os sinais do jogo
        match_status = analysis.get("match_status", "SCHEDULED")
        match_minute = analysis.get("match_minute")
        is_live      = match_status in ("IN_PLAY", "PAUSED")

        # ── Estabilização PRÉ: não gerar 1X2/DC quando os dados do fixture estão incompletos
        # (ex.: sem "medias" → FormService cai para média da liga e distorce o favorito).
        if (not is_live) and (not analysis.get("pre_data_ok", True)):
            before_pd = len(singles)
            singles = [
                s for s in singles
                if s.get("market_id") not in ("Home", "Draw", "Away")
                and not str(s.get("market_id", "")).startswith("DC_")
            ]
            removed = before_pd - len(singles)
            if removed:
                logger.warning(
                    f"[PRE-DATA] {analysis['home_team']} vs {analysis['away_team']}: "
                    f"{removed} sinal(is) 1X2/DC removido(s) por falta de medias no fixture"
                )

        for sig in singles:
            if await self._check_duplicate(match_id, sig["market_id"]):
                continue
            kelly = self.poisson.calculate_kelly_stake(sig["prob"], sig["odd"])
            stake = settings.default_bankroll * kelly

            # Sempre persiste no banco (HUB depende disso)
            implied = 1 / sig["odd"] if sig.get("odd", 0) > 0 else 0.0
            try:
                await self._persist_signal(
                    analysis, sig["market_id"], sig["prob"], implied,
                    sig["ev"], sig["odd"], sig.get("bookmaker", ""),
                    kelly, stake, None, label=sig.get("label"),
                    is_live=is_live, match_minute=match_minute,
                )
            except Exception as pe:
                logger.warning(f"[Persist] Erro ao salvar sinal {sig.get('market_id')}: {pe}")
                try:
                    await self.db.rollback()
                except Exception:
                    pass

            # Acumula para Telegram e combos
            self._pending_signals.append({
                "analysis": analysis, "sig": {
                    **sig,
                    "analysis": analysis,
                    "kelly": kelly,
                    "stake": stake,
                },
                "kelly": kelly, "stake": stake, "match_id": match_id,
            })
            self._cycle_signals.append({
                "match_id":    match_id,
                "match_label": f"{analysis['home_team']} vs {analysis['away_team']}",
                "signal":      sig,
                "analysis":    analysis,
            })

            # Envia imediatamente pelo Telegram se não for modo "melhores"
            if not settings.send_only_best_signals:
                await self._send_signal(analysis, sig, kelly, stake)

            sent += 1

        # Combos do mesmo jogo desativados — apenas sinais simples
        # Para reativar: descomentar o bloco abaixo
        # combos = self.combo_engine.get_combo_signals_single_game(match_id, odds, mat, btts_p)
        # for sig in combos:
        #     ...

        if sent:
            logger.info(f"✅ {sent} sinal(is) persistido(s) — {analysis['home_team']} vs {analysis['away_team']}")

        return sent


    async def _send_best_pending_signals(self, cycle_stats: dict = None) -> int:
        """
        Envia menu resumo do ciclo com inline buttons por jogo.
        O usuário clica no jogo para receber os sinais — evita flood de mensagens.
        """
        if not self._pending_signals:
            return 0

        # Seleção de precisão: filtra e ranqueia (Top 10) por score híbrido
        from app.services.signal_selector import SignalSelector, SelectorConfig
        selector = SignalSelector(SelectorConfig.from_settings(top_n=10))
        top = selector.select(self._pending_signals)

        # Agrupa por jogo
        signals_by_match: dict[str, list] = {}
        for item in top:
            a = item["analysis"]
            if a.get("_is_multi"):
                label = a["home_team"]  # já é o label composto dos jogos
            elif a.get("away_team"):
                label = f"{a['home_team']} vs {a['away_team']}"
            else:
                label = a["home_team"]
            if label not in signals_by_match:
                signals_by_match[label] = []
            signals_by_match[label].append({
                "label":      item["sig"].get("label", item["sig"].get("market_id", "?")),
                "market_id":  item["sig"].get("market_id", ""),
                "odd":        item["sig"].get("odd", 0),
                "prob":       item["sig"].get("prob", 0),
                "ev":         item["sig"].get("ev", 0),
                "bookmaker":  item["sig"].get("bookmaker", ""),
                "stake":      item["stake"],
                "kelly":      item["kelly"],
                "analysis":   a,
                "sig":        item["sig"],
            })

        # Sinais já foram persistidos em _process_all_signals — apenas monta o menu Telegram
        logger.info(f"[Telegram] Enviando menu com {len(top)} sinais de {len(signals_by_match)} jogos")

        # Salva no store global para o callback handler recuperar
        from app.telegram_bot.callback_store import set_cycle_signals
        set_cycle_signals(signals_by_match)

        # Envia menu resumo
        stats = cycle_stats or {}
        msg_id = await self.telegram.send_cycle_summary(signals_by_match, stats)
        logger.info(f"📋 Menu resumo enviado: {len(signals_by_match)} jogos | {len(top)} sinais")

        self._pending_signals = []
        return len(top)

    async def _process_multi_game_combos(self) -> int:
        if len(self._cycle_signals) < 2:
            return 0

        multi_combos = self.combo_engine.get_multi_game_combos(self._cycle_signals)
        sent = 0

        # Em modo "melhores sinais", apenas acumula e decide no fim
        if settings.send_only_best_signals:
            base_analysis = self._cycle_signals[0]["analysis"]
            for combo in multi_combos:
                kelly = self.poisson.calculate_kelly_stake(combo["prob"], combo["odd"])
                stake = settings.default_bankroll * kelly * 0.5
                # Usa label dos jogos envolvidos para não misturar com sinais simples
                games = combo.get("games", [])
                multi_label_key = " + ".join(games) if games else "Multi-jogo"
                combo_analysis = {
                    **base_analysis,
                    "home_team": multi_label_key,
                    "away_team": "",
                    "_is_multi": True,
                }
                self._pending_signals.append({
                    "analysis": combo_analysis,
                    "sig": combo,
                    "kelly": kelly * 0.5,
                    "stake": stake,
                    "match_id": self._cycle_signals[0]["match_id"],
                })
            return 0

        for combo in multi_combos:
            first_match = self._cycle_signals[0]["match_id"]
            if await self._check_duplicate(first_match, combo["market_id"]):
                continue
            base_analysis = self._cycle_signals[0]["analysis"]
            kelly = self.poisson.calculate_kelly_stake(combo["prob"], combo["odd"])
            stake = settings.default_bankroll * kelly * 0.5
            await self._send_multi_signal(base_analysis, combo, kelly * 0.5, stake)
            sent += 1

        if sent:
            logger.info(f"🎲 {sent} combo(s) multi-jogo gerado(s)")

        return sent

    # ── Formatação e envio (idêntico ao V1 + bloco DAPM novo) ─────────────────

    async def _send_signal(self, analysis: dict, sig: dict, kelly_pct: float, stake: float):
        ref_name        = analysis.get("referee", "Unknown")
        ref_strictness  = analysis.get("referee_strictness", 1.0)
        ref_avg_yellow  = analysis.get("referee_avg_yellow", 3.5)
        ref_avg_bpts    = analysis.get("referee_avg_bpts", 38.0)
        ref_avg_corners = analysis.get("referee_avg_corners", 10.3)
        ref_matches     = analysis.get("referee_matches", 0)

        if ref_strictness >= 1.15:   ref_emoji, ref_tag = "🟥", "RIGOROSO"
        elif ref_strictness >= 1.05: ref_emoji, ref_tag = "🟠", "EXIGENTE"
        elif ref_strictness <= 0.85: ref_emoji, ref_tag = "🟢", "PERMISSIVO"
        else:                        ref_emoji, ref_tag = "🟡", "NEUTRO"

        prob    = sig["prob"]
        odd     = sig["odd"]
        ev      = sig["ev"]
        implied = 1 / odd
        edge    = (prob - implied) * 100
        n_legs  = sig.get("n_legs", 1)

        ev_emoji = "🔥🔥" if ev >= 0.20 else ("🔥" if ev >= 0.10 else "✅")
        type_tag = "📌 SINAL SIMPLES" if sig["type"] == "single" else f"🎯 COMBO {n_legs} PERNAS"
        time_str = analysis["commence_time"].strftime("%d/%m %H:%M") + " UTC"

        # ── Bloco live ────────────────────────────────────────────────────────
        match_status = analysis.get("match_status", "SCHEDULED")
        match_minute = analysis.get("match_minute")
        is_live      = match_status in ("IN_PLAY", "PAUSED")

        if is_live:
            minute_str = f" {match_minute}'" if match_minute else ""
            half_str   = " (Intervalo)" if match_status == "PAUSED" else ""
            live_block = f"🔴 <b>SINAL AO VIVO{minute_str}{half_str}</b>\n"
            type_tag   = f"📌 SINAL AO VIVO" if sig["type"] == "single" else f"🎯 COMBO AO VIVO {n_legs} PERNAS"
        else:
            live_block = ""

        ref_block = f"👨‍⚖️ <b>{ref_name}</b> {ref_emoji} {ref_tag}\n"
        if ref_matches >= 5:
            ref_block += (
                f"├ Amarelos: {ref_avg_yellow:.1f}/jogo  "
                f"Bk.Pts: {ref_avg_bpts:.0f}/jogo  "
                f"Esc: {ref_avg_corners:.1f}/jogo\n"
            )

        # DAPM block — exclusivo SokkerPRO
        dapm = analysis.get("dapm", {}) or {}
        home_dapm_total = dapm.get("home_dapm_total") or 0
        away_dapm_total = dapm.get("away_dapm_total") or 0
        home_dapm_10    = dapm.get("home_dapm_10")    or 0
        away_dapm_10    = dapm.get("away_dapm_10")    or 0
        dapm_block = ""
        if home_dapm_total or away_dapm_total:
            dapm_block = (
                f"\n⚡ <b>DAPM (Ataques Perigosos/min)</b>\n"
                f"├ {analysis['home_team']}: {home_dapm_total:.2f} total | "
                f"ult.10min: {home_dapm_10:.2f}\n"
                f"└ {analysis['away_team']}: {away_dapm_total:.2f} total | "
                f"ult.10min: {away_dapm_10:.2f}\n"
            )

        legs_block = ""
        if n_legs > 1:
            legs_lines = "\n".join(f"  └ {l}" for l in sig.get("legs", []))
            legs_block = f"\n📋 <b>PERNAS:</b>\n{legs_lines}\n"

        text = (
            f"{type_tag}\n"
            f"{'─'*30}\n\n"
            f"{live_block}"
            f"⚽ <b>{analysis['home_team']}</b> x <b>{analysis['away_team']}</b>\n"
            f"🏆 {analysis['league']}\n"
            f"📅 {time_str}\n"
            f"{dapm_block}"
            f"\n{ref_block}"
            f"\n{'─'*30}\n"
            f"🎯 <b>{sig['label']}</b>\n"
            f"💰 Odd: <b>{odd:.2f}</b> ({sig.get('bookmaker','')})\n"
            f"{legs_block}\n"
            f"📈 <b>PROBABILIDADES</b>\n"
            f"├ Modelo:    <b>{prob:.1%}</b>\n"
            f"├ Implícita: {implied:.1%}\n"
            f"└ Edge:      <b>+{edge:.1f}pp</b>\n\n"
            f"{'─'*30}\n"
            f"🧮 <b>QUADRO DO JOGO</b>\n"
            f"├ λ Casa/Fora: {analysis['lambda_home']:.2f} / {analysis['lambda_away']:.2f}\n"
            f"├ Over 1.5: {analysis['prob_over_15']:.1%}  │  Over 2.5: {analysis['prob_over_25']:.1%}  │  Over 3.5: {analysis['prob_over_35']:.1%}\n"
            f"├ BTTS: {analysis['prob_btts']:.1%}  │  1X2: {analysis['prob_home_win']:.1%}/{analysis['prob_draw']:.1%}/{analysis['prob_away_win']:.1%}\n"
            f"├ 🚩 Esc. λ={analysis['lambda_corners']:.1f}  O8.5:{analysis['prob_over_85_corners']:.1%}  O9.5:{analysis['prob_over_95_corners']:.1%}  O10.5:{analysis['prob_over_105_corners']:.1%}\n"
            f"└ 🟨 Bk.Pts λ={analysis['lambda_booking_pts']:.0f}  O30:{analysis['prob_over_30_bpts']:.1%}  O40:{analysis['prob_over_40_bpts']:.1%}  O50:{analysis['prob_over_50_bpts']:.1%}\n\n"
            f"{ev_emoji} EV: <b>+{ev:.1%}</b>\n"
            f"📌 Stake: <b>R$ {stake:.2f}</b> ({kelly_pct:.1%} bankroll)\n\n"
            f"<i>⚠️ Sistema estatístico. Aposte com responsabilidade.</i>"
        )

        msg_id = await self.telegram.send_message(text)
        await self._persist_signal(analysis, sig["market_id"], prob, implied, ev, odd,
                                   sig.get("bookmaker", ""), kelly_pct, stake, msg_id,
                                   label=sig.get("label"), is_live=is_live,
                                   match_minute=match_minute)

    async def _send_multi_signal(self, base_analysis: dict, combo: dict, kelly_pct: float, stake: float):
        prob    = combo["prob"]
        odd     = combo["odd"]
        ev      = combo["ev"]
        implied = 1 / odd
        edge    = (prob - implied) * 100
        ev_emoji = "🔥🔥" if ev >= 0.20 else ("🔥" if ev >= 0.10 else "✅")
        n_type  = "2 JOGOS" if combo["type"] == "multi_2" else "3 JOGOS"

        games_block = "\n".join(
            f"  └ {g}: {l}"
            for g, l in zip(combo.get("games", []), combo.get("legs", []))
        )

        text = (
            f"🎲 <b>COMBO MULTI-JOGO — {n_type}</b>  [{combo.get('target_band','')}]\n"
            f"{'─'*30}\n\n"
            f"📋 <b>PERNAS:</b>\n{games_block}\n\n"
            f"💰 Odd Combinada: <b>{odd:.2f}</b>\n"
            f"├ Bookmaker: {combo.get('bookmaker','')}\n\n"
            f"📈 <b>PROBABILIDADES</b>\n"
            f"├ Modelo (independente): <b>{prob:.1%}</b>\n"
            f"├ Implícita: {implied:.1%}\n"
            f"└ Edge: <b>+{edge:.1f}pp</b>\n\n"
            f"{ev_emoji} EV: <b>+{ev:.1%}</b>\n"
            f"📌 Stake: <b>R$ {stake:.2f}</b> ({kelly_pct:.1%} bankroll)\n\n"
            f"<i>⚠️ Sistema estatístico. Aposte com responsabilidade.</i>"
        )

        msg_id = await self.telegram.send_message(text)
        # Label legível para a UI
        legs_labels = combo.get("legs", [])
        tipo = "Dupla" if combo.get("type") == "multi_2" else "Tripla"
        multi_label = f"{tipo}: " + " + ".join(legs_labels)
        await self._persist_signal(
            base_analysis, combo["market_id"], prob, implied, ev, odd,
            combo.get("bookmaker", ""), kelly_pct, stake, msg_id,
            label=multi_label,
        )

    # ── Persistência (idêntico ao V1) ─────────────────────────────────────────

    async def _persist_match_and_analysis(self, analysis, home_form, away_form, league_avgs):
        match_id = analysis["match_id"]
        # Pode rodar concorrente para o mesmo match_id.
        # Evita UNIQUE constraint failed: matches.id com upsert "do nothing".
        try:
            stmt = (
                sqlite_insert(Match.__table__)
                .values(
                    id=match_id,
                    sport_key=analysis["league_id"],
                    sport_title=analysis["league"],
                    home_team=analysis["home_team"],
                    away_team=analysis["away_team"],
                    home_logo=analysis.get("home_logo") or None,
                    away_logo=analysis.get("away_logo") or None,
                    commence_time=analysis["commence_time"],
                )
            )
            # SQLite: INSERT OR IGNORE
            stmt = stmt.prefix_with("OR IGNORE")
            await self.db.execute(stmt)
            await self.db.flush()
        except IntegrityError:
            await self.db.rollback()

        league_avg = league_avgs.get("avg_total", 2.5)
        odds = analysis["odds"]

        market_map = {
            "over_1.5":    ("Over_1.5",  analysis["prob_over_15"]),
            "over_2.5":    ("Over_2.5",  analysis["prob_over_25"]),
            "over_3.5":    ("Over_3.5",  analysis["prob_over_35"]),
            "under_2.5":   ("Under_2.5", analysis["prob_under_25"]),
            "home_win":    ("Home",      analysis["prob_home_win"]),
            "draw":        ("Draw",      analysis["prob_draw"]),
            "away_win":    ("Away",      analysis["prob_away_win"]),
            "btts":        ("BTTS",      analysis["prob_btts"]),
            "corners_o95": ("Corners_Over_9.5", analysis["prob_over_95_corners"]),
            "cards_o40":   ("Cards_Over_40",    analysis["prob_over_40_bpts"]),
        }

        # Se o PRÉ estiver com dados fracos (fixture sem medias), 1X2/DC fica instável.
        # Bloqueamos 1X2 na persistência (e por consequência na seleção de sinais).
        if not analysis.get("pre_data_ok", True):
            for k in ("home_win", "draw", "away_win"):
                market_map.pop(k, None)

        for mkey, (okey, prob) in market_map.items():
            info     = odds.get(okey, {})
            best_odd = info.get("odd", 0.0) if isinstance(info, dict) else 0.0
            bookmaker= info.get("bookmaker", "") if isinstance(info, dict) else ""
            implied  = 1 / best_odd if best_odd > 0 else 0.0
            # --- V8.2: stacked probability com Bivariate Poisson real ---
            market_p = self.market_prob.from_odd(best_odd) if best_odd > 0 else 0.0

            # Bivariate Poisson real — mesmos lambdas, mas com covariância λ3
            try:
                bivariate_probs = self.advanced_model.all_probs(
                    analysis["lambda_home"], analysis["lambda_away"]
                )
                # Mapeia mkey para a chave correspondente no bivariate
                _bivariate_key_map = {
                    "home_win":  "home_win",
                    "draw":      "draw",
                    "away_win":  "away_win",
                    "btts":      "btts_yes",
                    "over_1.5":  "over_1_5",
                    "over_2.5":  "over_2_5",
                    "over_3.5":  "over_3_5",
                    "under_2.5": "under_2_5",
                }
                bivariate_p = bivariate_probs.get(_bivariate_key_map.get(mkey, ""), prob)
            except Exception:
                bivariate_p = prob  # fallback seguro

            # Heurística de forma: ratio gols esperados vs média da liga
            try:
                expected_total_form = float(home_form.avg_scored) + float(away_form.avg_conceded)
            except Exception:
                expected_total_form = 0.0
            ratio = (expected_total_form / league_avg) if league_avg > 0 else 1.0
            ratio = max(0.85, min(1.15, ratio))
            form_p = prob * ratio

            final_prob = self.stacked.combine(prob, bivariate_p, market_p, form_p)

            ev       = (final_prob * best_odd) - 1 if best_odd > 0 else 0.0

            self.db.add(MatchAnalysis(
                match_id=match_id, market=mkey,
                lambda_home=analysis["lambda_home"],
                lambda_away=analysis["lambda_away"],
                lambda_total=analysis["lambda_total"],
                model_probability=final_prob,
                best_odd=best_odd, best_bookmaker=bookmaker,
                implied_probability=implied, ev=ev,
                home_avg_scored=home_form.avg_scored,
                home_avg_conceded=home_form.avg_conceded,
                away_avg_scored=away_form.avg_scored,
                away_avg_conceded=away_form.avg_conceded,
                league_avg_goals=league_avg,
                sofa_xg_home=analysis.get("sofa_xg_home"),
                sofa_xg_away=analysis.get("sofa_xg_away"),
            ))

        try:
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            logger.warning(f"Erro ao persistir análise {match_id}: {e}")

    async def _check_duplicate(self, match_id: str, market: str) -> bool:
        result = await self.db.execute(
            select(Signal).where(and_(Signal.match_id == match_id, Signal.market == market))
        )
        return result.scalar_one_or_none() is not None

    async def _persist_signal(self, analysis, market_id, model_prob, implied_prob,
                               ev, odd, bookmaker, kelly_pct, stake, message_id, label=None,
                               is_live: bool = False, match_minute: int | None = None):
        match_id = analysis["match_id"]
        # Evita IntegrityError em matches.id em cenários concorrentes
        try:
            stmt = (
                sqlite_insert(Match.__table__)
                .values(
                    id=match_id,
                    sport_key=analysis["league_id"],
                    sport_title=analysis["league"],
                    home_team=analysis["home_team"],
                    away_team=analysis["away_team"],
                    home_logo=analysis.get("home_logo") or None,
                    away_logo=analysis.get("away_logo") or None,
                    commence_time=analysis["commence_time"],
                )
            )
            stmt = stmt.prefix_with("OR IGNORE")
            await self.db.execute(stmt)
        except IntegrityError:
            await self.db.rollback()
        self.db.add(Signal(
            match_id=match_id, market=market_id,
            label=label,
            model_probability=model_prob, implied_probability=implied_prob,
            ev=ev, suggested_odd=odd, bookmaker=bookmaker,
            stake_pct=kelly_pct, stake_units=stake,
            telegram_message_id=message_id,
            is_live=is_live,
            match_minute=match_minute,
        ))
        try:
            await self.db.commit()
        except IntegrityError as e:
            await self.db.rollback()
            logger.warning(f"Erro ao persistir sinal {match_id}/{market_id}: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_odds_keys(odds: dict) -> dict:
        normalized = {}
        for k, v in odds.items():
            new_key = k
            for prefix in ("Over_", "Under_"):
                if k.startswith(prefix):
                    try:
                        line = float(k[len(prefix):])
                        new_key = f"{prefix}{line:.1f}"
                    except Exception:
                        pass
                    break
            normalized[new_key] = v
        return normalized

    def _is_within_window(self, commence_time: datetime) -> bool:
        now = datetime.now(timezone.utc)
        if commence_time.tzinfo is None:
            commence_time = commence_time.replace(tzinfo=timezone.utc)
        # Futuros dentro do lookahead OU já iniciados (ao vivo vêm do livescores)
        return commence_time <= now + timedelta(hours=settings.lookahead_hours)