"""
LiveAnalyzer — análise estatística ao vivo (substitui Poisson em jogos IN_PLAY/PAUSED).

Filosofia:
  Pré-jogo → Poisson sobre médias históricas (o que DEVE acontecer)
  Ao vivo  → Poisson Dinâmico sobre estado real do jogo (o que AINDA PODE acontecer)

O modelo ao vivo resolve o problema central do Poisson pré-jogo:
  - 0x0 aos 83' → Poisson ainda sugere Over 2.5 se as médias forem altas
  - 3x1 aos 60' → Poisson ainda sugere Under 2.5 se as médias forem baixas

Aqui calculamos λ_residual — o lambda para os minutos que ainda restam —
usando o ritmo atual do jogo (DAPM janelado) como fator de ajuste sobre
as médias históricas da temporada.

Fluxo:
  1. Filtros de inviabilidade física → elimina mercados matematicamente impossíveis
  2. λ_residual = λ_histórico × (minutos_restantes/90) × fator_ritmo
  3. Reconstrói matrix Poisson com λ_residual
  4. Calcula probabilidades residuais para todos os mercados
  5. Retorna probs + matrix pronta para o combo_engine
"""

import math
from typing import Optional
from scipy.stats import poisson
from loguru import logger


# ── Constantes ────────────────────────────────────────────────────────────────

_MAX_GOALS = 13          # limite da matrix (igual ao combo_engine)
_FULL_MATCH = 90.0       # minutos de jogo completo

# Minuto efetivo mínimo para análise ao vivo (evita análise no apito inicial)
_MIN_LIVE_MINUTE = 5

# Fator de suavização do DAPM (evita que um pico de 1 minuto distorça tudo)
# DAPM_5 tem peso maior porque é mais estável que DAPM_1
_DAPM_WEIGHTS = {
    "dapm_1":  0.10,
    "dapm_3":  0.20,
    "dapm_5":  0.40,
    "dapm_10": 0.30,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) and v >= 0 else default
    except (TypeError, ValueError):
        return default


def _minutos_restantes(minute: Optional[int], status: str) -> float:
    """
    Estima minutos restantes com base no minuto atual e período.
    Considera acréscimos padrão de 3min no 1T e 5min no 2T.
    """
    if status == "PAUSED":
        return 45.0 + 5.0  # intervalo: segundo tempo completo + acréscimo

    m = minute or 0

    if m <= 45:
        # Primeiro tempo — considera acréscimo padrão de 3min
        return max(1.0, (45.0 + 3.0) - m + 45.0 + 5.0)
    else:
        # Segundo tempo
        return max(1.0, (90.0 + 5.0) - m)


def _build_matrix(lh: float, la: float) -> dict:
    return {
        (i, j): poisson.pmf(i, lh) * poisson.pmf(j, la)
        for i in range(_MAX_GOALS)
        for j in range(_MAX_GOALS)
    }


def _calc_dapm_ponderado(dapm: dict, side: str) -> Optional[float]:
    """
    Média ponderada do DAPM nas janelas disponíveis.
    side: "home" ou "away"
    """
    total_peso = 0.0
    total_val = 0.0

    for key, peso in _DAPM_WEIGHTS.items():
        val = _safe(dapm.get(f"{side}_{key}"))
        if val > 0:
            total_val += val * peso
            total_peso += peso

    if total_peso == 0:
        return None

    return total_val / total_peso


# ── Filtros de inviabilidade ──────────────────────────────────────────────────

def _mercados_viaveis(
    gols_home: int,
    gols_away: int,
    minute: int,
    minutos_restantes: float,
    corners_home: int,
    corners_away: int,
    cards_pts_atuais: float,
) -> set[str]:
    """
    Retorna conjunto de mercados que AINDA fazem sentido dado o estado do jogo.
    Qualquer mercado fora desse conjunto é descartado antes de qualquer cálculo.
    """
    viaveis = set()
    gols_total = gols_home + gols_away
    corners_total = corners_home + corners_away

    # ── Gols ─────────────────────────────────────────────────────────────────
    # Over N.5: viável se ainda faltam pelo menos N+1 - gols_total gols
    # e o tempo restante é suficiente para isso acontecer (taxa ~2.5 gols/90min)
    taxa_gol_por_minuto = 2.5 / 90.0

    for n in [0.5, 1.5, 2.5, 3.5, 4.5]:
        gols_necessarios = math.ceil(n + 1) - gols_total
        if gols_necessarios <= 0:
            viaveis.add(f"Over_{n}")
            continue
        # Verifica se há tempo razoável para os gols necessários acontecerem
        # Usa 2x a taxa média para não ser muito restritivo
        minutos_necessarios = gols_necessarios / (taxa_gol_por_minuto * 2.0)
        if minutos_restantes >= minutos_necessarios:
            viaveis.add(f"Over_{n}")

    # Under N.5: viável se ainda não passou da linha
    for n in [1.5, 2.5, 3.5, 4.5]:
        if gols_total < math.ceil(n):
            viaveis.add(f"Under_{n}")

    # ── BTTS ─────────────────────────────────────────────────────────────────
    home_marcou = gols_home > 0
    away_marcou = gols_away > 0

    # BTTS Sim: viável se ambos já marcaram OU ainda há tempo para quem não marcou
    if home_marcou and away_marcou:
        viaveis.add("BTTS")
    elif minute <= 75:
        # Ainda há tempo para o time que não marcou fazer o gol
        if not home_marcou or not away_marcou:
            viaveis.add("BTTS")

    # BTTS Não: viável se pelo menos um time ainda pode terminar sem marcar
    if not home_marcou and minutos_restantes < 20:
        viaveis.add("No_BTTS")
    elif not away_marcou and minutos_restantes < 20:
        viaveis.add("No_BTTS")
    elif minute <= 60:
        viaveis.add("No_BTTS")

    # ── Resultado 1X2 ─────────────────────────────────────────────────────────
    # Não é "sempre viável" — depende de quão reversível é o placar atual.
    # Critério: diferença de gols × minutos restantes define o "ponto de não retorno".
    #
    # Tabela de referência (prob empírica de virada no futebol profissional):
    #   diff=1, min>60  → ~8%   → ainda válido
    #   diff=2, min>70  → ~2%   → borderline — descarta
    #   diff=3+, qq min → <1%   → descarta sempre
    #   diff=1, min>80  → ~3%   → descarta (sem tempo)

    diff = abs(gols_home - gols_away)

    # Mercado "Home" (vitória casa)
    if gols_home > gols_away:
        # Já vencendo — sempre viável (pode consolidar ou empatar, resultado já é Win)
        viaveis.add("Home")
    elif diff == 0:
        viaveis.add("Home")   # empatado — viável
    elif diff == 1 and minute < 75:
        viaveis.add("Home")   # 1 gol de diferença, ainda há tempo
    # diff >= 2 ou tarde demais → não adiciona "Home" se o time está perdendo

    # Mercado "Away" (vitória visitante) — mesma lógica espelhada
    if gols_away > gols_home:
        viaveis.add("Away")
    elif diff == 0:
        viaveis.add("Away")
    elif diff == 1 and minute < 75:
        viaveis.add("Away")

    # Mercado "Draw" (empate) — só viável se diferença for 0 ou 1 gol com tempo suficiente
    if diff == 0:
        viaveis.add("Draw")
    elif diff == 1 and minute < 70:
        viaveis.add("Draw")

    # ── Dupla Chance ─────────────────────────────────────────────────────────
    # DC_Home_Draw (1X): viável se Home OU Draw for viável
    if "Home" in viaveis or "Draw" in viaveis:
        viaveis.add("DC_Home_Draw")
    # DC_Draw_Away (X2): viável se Away OU Draw for viável
    if "Away" in viaveis or "Draw" in viaveis:
        viaveis.add("DC_Draw_Away")
    # DC_Home_Away (12): viável se Home E Away forem viáveis (exclui empate certo)
    if "Home" in viaveis and "Away" in viaveis:
        viaveis.add("DC_Home_Away")

    # ── Escanteios ────────────────────────────────────────────────────────────
    # Projeta ritmo de escanteios: ~10.3/jogo em média
    taxa_corner_por_minuto = 10.3 / 90.0
    corners_projetados = corners_total + (minutos_restantes * taxa_corner_por_minuto)

    for n in [8.5, 9.5, 10.5, 11.5]:
        if corners_projetados > n:
            viaveis.add(f"Corners_Over_{n}")
        if corners_projetados <= n + 2:
            viaveis.add(f"Corners_Under_{n}")

    # ── Cartões (Booking Points) ──────────────────────────────────────────────
    # Booking pts: amarelo=10, vermelho=25
    # Projeta restante com taxa ~45pts/jogo em média
    taxa_bpts_por_minuto = 45.0 / 90.0
    bpts_projetados = cards_pts_atuais + (minutos_restantes * taxa_bpts_por_minuto)

    for n in [20.0, 30.0, 40.0, 50.0]:
        if bpts_projetados > n:
            viaveis.add(f"Cards_Over_{n:.0f}")
        if bpts_projetados <= n + 15:
            viaveis.add(f"Cards_Under_{n:.0f}")

    # Mercados de 1T e HT → nunca válidos ao vivo (jogo já começou)
    # (não adicionados → serão descartados automaticamente)

    return viaveis


# ── LiveAnalyzer ──────────────────────────────────────────────────────────────

class LiveAnalyzer:
    """
    Calcula probabilidades residuais para jogos em andamento.

    Recebe o fixture completo (já parseado pelo sokkerpro_client) e retorna
    um dict compatível com o que o orchestrator espera do Poisson pré-jogo:
      - lambda_home, lambda_away (residuais)
      - prob_over_15/25/35, prob_btts, prob_home_win, prob_draw, prob_away_win
      - mat (matrix Poisson residual, para o combo_engine)
      - mercados_viaveis (set de market_ids que ainda fazem sentido)
    """

    def calculate(self, match: dict, full_fixture: dict) -> dict:
        """
        Ponto de entrada principal.

        match: fixture parseado pelo sokkerpro_client._parse_fixture()
        full_fixture: fixture completo do /fixture/{id} (tem medias + dapm)
        """
        home_name = match.get("homeTeam", {}).get("name", "Home")
        away_name = match.get("awayTeam", {}).get("name", "Away")
        status    = match.get("status", "IN_PLAY")
        minute    = match.get("minute") or 0

        # ── Placar atual ──────────────────────────────────────────────────────
        score     = match.get("score", {})
        gols_home = score.get("fullTime", {}).get("home") or 0
        gols_away = score.get("fullTime", {}).get("away") or 0

        # ── Stats ao vivo ─────────────────────────────────────────────────────
        stats         = match.get("statistics", {}) or {}
        corners_home  = int(_safe(stats.get("home_corners"), 0))
        corners_away  = int(_safe(stats.get("away_corners"), 0))
        yellow_home   = int(_safe(stats.get("home_yellow_cards"), 0))
        yellow_away   = int(_safe(stats.get("away_yellow_cards"), 0))
        red_home      = int(_safe(stats.get("home_red_cards"), 0))
        red_away      = int(_safe(stats.get("away_red_cards"), 0))

        # Booking points acumulados
        cards_pts_atuais = (
            (yellow_home + yellow_away) * 10 +
            (red_home + red_away) * 25
        )

        # ── Médias históricas da temporada ────────────────────────────────────
        medias = match.get("medias", {}) or {}
        # Fallback para full_fixture se medias do mini estiver vazio
        if not medias or not medias.get("home_avg_goal"):
            medias = {
                "home_avg_goal":       _safe(full_fixture.get("medias_home_goal"),      1.3),
                "away_avg_goal":       _safe(full_fixture.get("medias_away_goal"),      1.1),
                "home_avg_corners":    _safe(full_fixture.get("medias_home_corners"),   5.2),
                "away_avg_corners":    _safe(full_fixture.get("medias_away_corners"),   5.1),
                "home_avg_dangerous":  _safe(full_fixture.get("medias_home_dangerous_attacks"), 35.0),
                "away_avg_dangerous":  _safe(full_fixture.get("medias_away_dangerous_attacks"), 30.0),
                "home_avg_yellow":     _safe(full_fixture.get("medias_home_yellow_cards"), 1.8),
                "away_avg_yellow":     _safe(full_fixture.get("medias_away_yellow_cards"), 1.8),
                "home_avg_shots_on":   _safe(full_fixture.get("medias_home_shots_on_target"), 4.0),
                "away_avg_shots_on":   _safe(full_fixture.get("medias_away_shots_on_target"), 3.5),
            }

        home_avg_goal    = _safe(medias.get("home_avg_goal"),      1.3)
        away_avg_goal    = _safe(medias.get("away_avg_goal"),      1.1)
        home_avg_corners = _safe(medias.get("home_avg_corners"),   5.2)
        away_avg_corners = _safe(medias.get("away_avg_corners"),   5.1)
        home_avg_da      = _safe(medias.get("home_avg_dangerous"), 35.0)
        away_avg_da      = _safe(medias.get("away_avg_dangerous"), 30.0)
        home_avg_yellow  = _safe(medias.get("home_avg_yellow"),    1.8)
        away_avg_yellow  = _safe(medias.get("away_avg_yellow"),    1.8)

        # ── DAPM ao vivo ──────────────────────────────────────────────────────
        dapm = match.get("dapm", {}) or {}

        dapm_home_pond = _calc_dapm_ponderado(dapm, "home")
        dapm_away_pond = _calc_dapm_ponderado(dapm, "away")

        # ── Minutos restantes ─────────────────────────────────────────────────
        min_restantes = _minutos_restantes(minute, status)
        frac_restante = min_restantes / _FULL_MATCH

        # ── Fator de ritmo atual ──────────────────────────────────────────────
        # Compara DAPM atual ponderado com a média histórica do time
        # fator > 1.0 → time está mais ativo que o normal
        # fator < 1.0 → time está mais passivo que o normal
        # Clamp: 0.4 a 2.5 para não distorcer demais

        def _fator_ritmo(dapm_atual: Optional[float], avg_da: float) -> float:
            if dapm_atual is None or dapm_atual <= 0:
                return 1.0
            # avg_da vem em ataques/jogo, dapm em ataques/minuto — normaliza
            avg_da_por_minuto = avg_da / 90.0
            if avg_da_por_minuto <= 0:
                return 1.0
            fator = dapm_atual / avg_da_por_minuto
            return max(0.4, min(fator, 2.5))

        fator_home = _fator_ritmo(dapm_home_pond, home_avg_da)
        fator_away = _fator_ritmo(dapm_away_pond, away_avg_da)

        # ── Fator de placar ───────────────────────────────────────────────────
        # Times que estão perdendo abrem o jogo (mais gols esperados)
        # Times que estão ganhando por 2+ fecham o jogo
        diff = gols_home - gols_away

        if diff == 0:
            fator_placar_home = 1.05   # jogo aberto — leve estímulo
            fator_placar_away = 1.05
        elif diff == 1:
            fator_placar_home = 0.90   # vencendo — tende a fechar
            fator_placar_away = 1.15   # perdendo — tende a abrir
        elif diff == -1:
            fator_placar_home = 1.15
            fator_placar_away = 0.90
        elif diff >= 2:
            fator_placar_home = 0.75
            fator_placar_away = 1.25
        else:  # diff <= -2
            fator_placar_home = 1.25
            fator_placar_away = 0.75

        # ── λ residual ────────────────────────────────────────────────────────
        # λ_res = média_histórica × fração_do_jogo_restante × fator_ritmo × fator_placar
        lh_res = max(0.05, home_avg_goal * frac_restante * fator_home * fator_placar_home)
        la_res = max(0.05, away_avg_goal * frac_restante * fator_away * fator_placar_away)

        # ── λ residual de escanteios ──────────────────────────────────────────
        lc_home_res = max(0.01, home_avg_corners * frac_restante * fator_home)
        lc_away_res = max(0.01, away_avg_corners * frac_restante * fator_away)
        lambda_corners_res = lc_home_res + lc_away_res

        # ── λ residual de cartões ─────────────────────────────────────────────
        lcard_home_res = max(0.01, home_avg_yellow * frac_restante)
        lcard_away_res = max(0.01, away_avg_yellow * frac_restante)
        lambda_booking_pts_res = (lcard_home_res + lcard_away_res) * 10  # aprox em booking pts

        # ── Matrix Poisson residual ───────────────────────────────────────────
        mat = _build_matrix(lh_res, la_res)

        # ── Probabilidades residuais ──────────────────────────────────────────
        def _over(line: float) -> float:
            return sum(p for (i, j), p in mat.items() if i + j > line)

        def _under(line: float) -> float:
            return sum(p for (i, j), p in mat.items() if i + j < line)

        prob_over_05  = _over(0.5)
        prob_over_15  = _over(1.5)
        prob_over_25  = _over(2.5)
        prob_over_35  = _over(3.5)
        prob_under_25 = _under(2.5)

        # ── Probabilidades residuais de resultado (1X2) ──────────────────────
        # O Poisson residual calcula quem marca MAIS nos minutos restantes,
        # mas ignora o placar atual. Precisamos combinar os dois.
        #
        # P(Home Win final) = soma sobre todos placares residuais (i, j) onde
        #   (gols_home + i) > (gols_away + j)
        #
        # Isso é a única forma correta: 0x5 aos 66' → home precisa de 6+ gols
        # nos 24 minutos restantes → probabilidade ínfima.

        prob_home_win = sum(
            p for (i, j), p in mat.items()
            if (gols_home + i) > (gols_away + j)
        )
        prob_draw = sum(
            p for (i, j), p in mat.items()
            if (gols_home + i) == (gols_away + j)
        )
        prob_away_win = sum(
            p for (i, j), p in mat.items()
            if (gols_home + i) < (gols_away + j)
        )

        prob_btts    = sum(p for (i, j), p in mat.items() if i > 0 and j > 0)
        prob_no_btts = 1.0 - prob_btts

        # Probabilidades de escanteios residuais (Poisson independente)
        lc_total = lc_home_res + lc_away_res
        def _pover_corners(line: float) -> float:
            return 1 - sum(poisson.pmf(k, lc_total) for k in range(int(line) + 1))

        prob_over_85_corners  = _pover_corners(8.5)
        prob_over_95_corners  = _pover_corners(9.5)
        prob_over_105_corners = _pover_corners(10.5)
        prob_under_95_corners = 1 - prob_over_95_corners

        # Probabilidades de cartões residuais
        def _pover_bpts(line: float) -> float:
            # Poisson sobre booking pts residuais
            return 1 - sum(
                poisson.pmf(k, max(0.1, lambda_booking_pts_res))
                for k in range(int(line // 10) + 1)
            )

        prob_over_20_bpts  = _pover_bpts(20.0)
        prob_over_30_bpts  = _pover_bpts(30.0)
        prob_over_40_bpts  = _pover_bpts(40.0)
        prob_over_50_bpts  = _pover_bpts(50.0)
        prob_under_40_bpts = 1 - prob_over_40_bpts

        # ── Filtros de inviabilidade ──────────────────────────────────────────
        mercados_viaveis = _mercados_viaveis(
            gols_home=gols_home,
            gols_away=gols_away,
            minute=minute,
            minutos_restantes=min_restantes,
            corners_home=corners_home,
            corners_away=corners_away,
            cards_pts_atuais=cards_pts_atuais,
        )

        logger.info(
            f"[LiveAnalyzer] {home_name} vs {away_name} | "
            f"{gols_home}x{gols_away} {minute}' | "
            f"λ_res home={lh_res:.2f} away={la_res:.2f} | "
            f"fator_ritmo home={fator_home:.2f} away={fator_away:.2f} | "
            f"min_restantes={min_restantes:.0f} | "
            f"mercados_viaveis={len(mercados_viaveis)}"
        )

        return {
            # Lambdas residuais
            "lambda_home":   lh_res,
            "lambda_away":   la_res,
            "lambda_total":  lh_res + la_res,

            # Probabilidades residuais de gols
            "prob_over_05":  prob_over_05,
            "prob_over_15":  prob_over_15,
            "prob_over_25":  prob_over_25,
            "prob_over_35":  prob_over_35,
            "prob_under_25": prob_under_25,
            "prob_btts":     prob_btts,
            "prob_no_btts":  prob_no_btts,
            "prob_home_win": prob_home_win,
            "prob_draw":     prob_draw,
            "prob_away_win": prob_away_win,

            # Escanteios residuais
            "lambda_corners":        lambda_corners_res,
            "prob_over_85_corners":  prob_over_85_corners,
            "prob_over_95_corners":  prob_over_95_corners,
            "prob_over_105_corners": prob_over_105_corners,
            "prob_under_95_corners": prob_under_95_corners,

            # Cartões residuais
            "lambda_booking_pts":    lambda_booking_pts_res,
            "prob_over_20_bpts":     prob_over_20_bpts,
            "prob_over_30_bpts":     prob_over_30_bpts,
            "prob_over_40_bpts":     prob_over_40_bpts,
            "prob_over_50_bpts":     prob_over_50_bpts,
            "prob_under_40_bpts":    prob_under_40_bpts,

            # Matrix Poisson residual (para o combo_engine)
            "mat": mat,

            # Conjunto de mercados que ainda fazem sentido
            "mercados_viaveis": mercados_viaveis,

            # Contexto para log/debug
            "fator_ritmo_home":  fator_home,
            "fator_ritmo_away":  fator_away,
            "fator_placar_home": fator_placar_home,
            "fator_placar_away": fator_placar_away,
            "minutos_restantes": min_restantes,
            "gols_home":         gols_home,
            "gols_away":         gols_away,
        }


    def cards_live_probability_v4(self, match: dict, referee_stats: dict | None = None) -> dict | None:
        import math

        def _to_float(x, default=0.0) -> float:
            try:
                if x is None:
                    return float(default)
                if isinstance(x, (int, float)):
                    return float(x)
                if isinstance(x, str):
                    s = x.strip().replace(",", ".")
                    if s == "":
                        return float(default)
                    return float(s)
                return float(x)
            except Exception:
                return float(default)

        def _pick_stats(m: dict) -> dict:
            st = m.get("statistics")
            if isinstance(st, dict):
                return st
            d = m.get("data") or {}
            st2 = d.get("statistics")
            if isinstance(st2, dict):
                return st2
            return {}

        def _normalize_referee_stats(rs):
            # RefereeService._from_sokkerpro_fixture sometimes returns a tuple (referee, stats) or similar.
            if rs is None:
                return None
            if isinstance(rs, dict):
                return rs
            if isinstance(rs, (list, tuple)):
                for item in rs[::-1]:
                    if isinstance(item, dict):
                        return item
            return None

        minute = int(_to_float(match.get("minute", (match.get("data") or {}).get("minute", 0)), 0))
        minutes_remaining = max(0, 90 - minute)

        stats = _pick_stats(match)

        home_y = int(_to_float(stats.get("home_yellow_cards", 0), 0))
        away_y = int(_to_float(stats.get("away_yellow_cards", 0), 0))
        home_r = int(_to_float(stats.get("home_red_cards", 0), 0))
        away_r = int(_to_float(stats.get("away_red_cards", 0), 0))

        total_cards = home_y + away_y + home_r + away_r

        # ----- Referee average fallback hierarchy -----
        rs = _normalize_referee_stats(referee_stats)
        avg_cards = 0.0
        fallback_used = "referee"
        yellow_sum = None

        if rs:
            yellow = _to_float(rs.get("yellow_avg"), 0.0)
            red = _to_float(rs.get("red_avg"), 0.0)
            yellowred = _to_float(rs.get("yellowred_avg"), 0.0)
            avg_cards = yellow + red + yellowred

        if avg_cards <= 0.0:
            # fallback team averages (often only yellow cards are available)
            home_avg_y = _to_float(match.get("medias_home_yellow_cards"), 0.0)
            away_avg_y = _to_float(match.get("medias_away_yellow_cards"), 0.0)
            yellow_sum = home_avg_y + away_avg_y

            # Scale yellow → total cards with conservative factor (avoid fake 0.99)
            avg_cards = yellow_sum * 1.15
            fallback_used = "teams_yellow_scaled"

        if avg_cards <= 0.0:
            avg_cards = 4.5  # Brasil default safe fallback
            fallback_used = "default_br_4_5"

        lambda_residual = avg_cards * (minutes_remaining / 90.0)
        prob = 1 - math.exp(-lambda_residual)

        # ----- Context boosts -----
        if total_cards >= 3 and minute < 70:
            prob += 0.10

        if total_cards >= 4:
            prob += 0.15

        fouls_total = int(_to_float(stats.get("home_fouls", 0), 0)) + int(_to_float(stats.get("away_fouls", 0), 0))
        if rs:
            fouls_avg = _to_float(rs.get("fouls_avg"), 0.0)
            if fouls_avg > 0.0 and fouls_total > fouls_avg:
                prob += 0.10

        # clamp (avoid fake 0.99)
        prob = min(max(prob, 0.0), 0.93)

        if prob < 0.80:
            return None

        # Suggested market line (institutional):
        target_line = max(4.5, max(total_cards + 0.5, avg_cards - 0.5))
        suggested_line = math.ceil(target_line * 2) / 2.0

        return {
            "category": "cards",
            "confidence": round(prob, 3),
            "suggestion": f"Over {suggested_line:.1f} cards",
            "meta": {
                "fallback_used": fallback_used,
                "yellow_sum": None if yellow_sum is None else round(yellow_sum, 3),
                "avg_cards": round(avg_cards, 3),
                "current_cards": total_cards,
                "lambda_residual": round(lambda_residual, 3),
                "minutes_remaining": minutes_remaining,
                "model": "poisson_residual_v4",
                "referee_stats_present": bool(rs)
            }
        }




    def goals_live_probability_v2(self, match: dict) -> dict | None:
        import math
        from scipy.stats import poisson

        def _to_float(x, default=0.0) -> float:
            try:
                if x is None:
                    return float(default)
                if isinstance(x, (int, float)):
                    return float(x)
                if isinstance(x, str):
                    s = x.strip().replace(",", ".")
                    if s == "":
                        return float(default)
                    return float(s)
                return float(x)
            except Exception:
                return float(default)

        def _get(m: dict, keys: list[str], default=0.0) -> float:
            for k in keys:
                if k in m and m.get(k) is not None:
                    return _to_float(m.get(k), default)
            return float(default)

        def _pick_stats(m: dict) -> dict:
            st = m.get("statistics")
            if isinstance(st, dict):
                return st
            d = m.get("data") or {}
            st2 = d.get("statistics")
            if isinstance(st2, dict):
                return st2
            return {}

        def _pick_score(m: dict) -> tuple[int, int]:
            # 1) Modelo padrão (score.fullTime)
            score = m.get("score") or {}
            ft = score.get("fullTime") or {}
            if isinstance(ft, dict):
                h = int(_to_float(ft.get("home"), 0))
                a = int(_to_float(ft.get("away"), 0))
                if h >= 0 and a >= 0:
                    return h, a

            # 2) SokkerPRO: scoresLocalTeam / scoresVisitorTeam (top-level)
            h2 = _to_float(m.get("scoresLocalTeam"), None)
            a2 = _to_float(m.get("scoresVisitorTeam"), None)
            if h2 is not None and a2 is not None:
                return int(h2), int(a2)

            # 3) SokkerPRO: pode estar dentro de data
            d = m.get("data") or {}
            h3 = _to_float(d.get("scoresLocalTeam"), None)
            a3 = _to_float(d.get("scoresVisitorTeam"), None)
            if h3 is not None and a3 is not None:
                return int(h3), int(a3)

            # 4) fallback
            return 0, 0

        minute = int(_to_float(match.get("minute", (match.get("data") or {}).get("minute", 0)), 0))
        minutes_remaining = max(0, 90 - minute)
        if minutes_remaining <= 0:
            return None

        gols_home, gols_away = _pick_score(match)
        gols_total = int(gols_home + gols_away)

        stats = _pick_stats(match)

        # --------- Live inputs (aceita stats normalizadas OU chaves SokkerPRO top-level) ---------
        home_sot = _get(stats, ["home_shots_on_target"], 0.0)
        away_sot = _get(stats, ["away_shots_on_target"], 0.0)
        if home_sot == 0.0 and away_sot == 0.0:
            home_sot = _get(match, ["localShotsOnGoal", "localShotsOnTarget"], 0.0)
            away_sot = _get(match, ["visitorShotsOnGoal", "visitorShotsOnTarget"], 0.0)

        shots_on_total = int(home_sot + away_sot)

        home_da = _get(stats, ["home_dangerous_attacks"], 0.0)
        away_da = _get(stats, ["away_dangerous_attacks"], 0.0)
        home_at = _get(stats, ["home_attacks"], 0.0)
        away_at = _get(stats, ["away_attacks"], 0.0)

        if (home_da + away_da) == 0.0 and (home_at + away_at) == 0.0:
            home_da = _get(match, ["localAttacksDangerousAttacks"], 0.0)
            away_da = _get(match, ["visitorAttacksDangerousAttacks"], 0.0)
            home_at = _get(match, ["localAttacksAttacks"], 0.0)
            away_at = _get(match, ["visitorAttacksAttacks"], 0.0)

        dangerous_total = float(home_da + away_da)
        attacks_total = float(home_at + away_at)

        # xG (se existir em qualquer lugar)
        xg_total = (
            _get(stats, ["home_xg"], 0.0) +
            _get(stats, ["away_xg"], 0.0) +
            _get(match, ["localXg", "visitorXg", "xg_total"], 0.0)
        )

        # pressão (fallback): usa local/visitorPressure quando existir
        pressure_proxy = _get(match, ["localPressure"], 0.0) + _get(match, ["visitorPressure"], 0.0)

        # --------- Histórico (melhor esforço) ---------
        home_avg_goal = _get(match, ["medias_home_goal"], 0.0)
        away_avg_goal = _get(match, ["medias_away_goal"], 0.0)
        if home_avg_goal <= 0.0 and away_avg_goal <= 0.0 and isinstance(match.get("medias"), dict):
            home_avg_goal = _to_float((match.get("medias") or {}).get("home_avg_goal"), 0.0)
            away_avg_goal = _to_float((match.get("medias") or {}).get("away_avg_goal"), 0.0)

        if home_avg_goal <= 0.0 and away_avg_goal <= 0.0:
            home_avg_goal, away_avg_goal = 1.25, 1.05

        lambda_base = max(0.2, home_avg_goal + away_avg_goal)

        # --------- Ajuste por pressão atual ---------
        pressure_factor = 1.0
        if minute >= 10:
            da_per_min = dangerous_total / max(1.0, float(minute))
            at_per_min = attacks_total / max(1.0, float(minute))

            pressure_index = (da_per_min * 1.0) + (at_per_min * 0.15)

            # se não tiver ataques, usa proxy de pressão
            if pressure_index <= 0.01 and pressure_proxy > 0.0:
                pressure_index = (pressure_proxy / max(1.0, float(minute))) / 8.0

            pressure_factor = max(0.8, min(1.6, 1.0 + (pressure_index / 12.0)))

        lambda_adj = lambda_base * pressure_factor

        # Boosts institucionais
        if gols_total == 0 and minute >= 60:
            lambda_adj *= 1.15
        if shots_on_total >= 5:
            lambda_adj *= 1.10
        if xg_total >= 1.2:
            lambda_adj *= 1.15

        lambda_residual = max(0.05, lambda_adj * (minutes_remaining / 90.0))

        # ---- Escolha institucional: melhor Over/Under >= 0.80 (linhas realistas) ----
        candidate_lines = [1.5, 2.5, 3.5]
        best = None  # (prob, direction, line, k_threshold)

        for line in candidate_lines:
            # OVER: precisa de K gols adicionais tal que gols_total + K > line
            need = int(math.floor(line - gols_total) + 1)
            need = max(0, need)
            if need <= 0:
                p_over = 1.0
            else:
                p_over = 1.0 - poisson.cdf(need - 1, lambda_residual)

            if p_over >= 0.80:
                cand = (float(p_over), "over", float(line), int(need))
                if best is None or cand[0] > best[0]:
                    best = cand

            # UNDER: precisa que gols_total + K <= line
            allow = int(math.floor(line - gols_total))
            if allow < 0:
                p_under = 0.0
            else:
                p_under = float(poisson.cdf(allow, lambda_residual))

            if p_under >= 0.80:
                cand = (float(p_under), "under", float(line), int(allow))
                if best is None or cand[0] > best[0]:
                    best = cand

        if not best:
            return None

        prob, direction, line, k_thresh = best
        prob = min(prob, 0.93)
        if prob < 0.80:
            return None

        suggestion = f"{'Over' if direction == 'over' else 'Under'} {line:.1f} goals"

        meta = {
            "gols_total": gols_total,
            "minute": minute,
            "minutes_remaining": minutes_remaining,
            "lambda_base": round(lambda_base, 3),
            "pressure_factor": round(pressure_factor, 3),
            "lambda_residual": round(lambda_residual, 3),
            "shots_on_total": shots_on_total,
            "dangerous_total": round(dangerous_total, 1),
            "attacks_total": round(attacks_total, 1),
            "xg_total": round(xg_total, 3),
            "pressure_proxy": round(pressure_proxy, 1),
            "direction": direction,
            "line": float(line),
            "k_threshold": int(k_thresh),
            "model": "poisson_residual_goals_v2",
        }

        return {
            "category": "goals",
            "confidence": round(prob, 3),
            "suggestion": suggestion,
            "meta": meta
        }

    def corners_live_probability_v1(self, match: dict) -> dict | None:
        import math
        from scipy.stats import poisson

        def _to_float(x, default=0.0) -> float:
            try:
                if x is None:
                    return float(default)
                if isinstance(x, (int, float)):
                    return float(x)
                if isinstance(x, str):
                    s = x.strip().replace(",", ".")
                    if s == "":
                        return float(default)
                    return float(s)
                return float(x)
            except Exception:
                return float(default)

        def _get(m: dict, keys: list[str], default=0.0) -> float:
            for k in keys:
                if k in m and m.get(k) is not None:
                    return _to_float(m.get(k), default)
            return float(default)

        def _pick_stats(m: dict) -> dict:
            st = m.get("statistics")
            if isinstance(st, dict):
                return st
            d = m.get("data") or {}
            st2 = d.get("statistics")
            if isinstance(st2, dict):
                return st2
            return {}

        minute = int(_to_float(match.get("minute", (match.get("data") or {}).get("minute", 0)), 0))
        minutes_remaining = max(0, 90 - minute)
        if minutes_remaining <= 0:
            return None

        stats = _pick_stats(match)

        corners_home = int(_get(stats, ["home_corners"], 0.0))
        corners_away = int(_get(stats, ["away_corners"], 0.0))
        if corners_home == 0 and corners_away == 0:
            corners_home = int(_get(match, ["localCorners"], 0.0))
            corners_away = int(_get(match, ["visitorCorners"], 0.0))

        corners_total = corners_home + corners_away

        home_da = _get(stats, ["home_dangerous_attacks"], 0.0)
        away_da = _get(stats, ["away_dangerous_attacks"], 0.0)
        if (home_da + away_da) == 0.0:
            home_da = _get(match, ["localAttacksDangerousAttacks"], 0.0)
            away_da = _get(match, ["visitorAttacksDangerousAttacks"], 0.0)

        dangerous_total = float(home_da + away_da)

        # Médias históricas (SokkerPRO já fornece medias_home/away_corners)
        home_avg_corners = _get(match, ["medias_home_corners"], 0.0)
        away_avg_corners = _get(match, ["medias_away_corners"], 0.0)
        if home_avg_corners <= 0.0 and away_avg_corners <= 0.0 and isinstance(match.get("medias"), dict):
            home_avg_corners = _to_float((match.get("medias") or {}).get("home_avg_corners"), 0.0)
            away_avg_corners = _to_float((match.get("medias") or {}).get("away_avg_corners"), 0.0)

        if home_avg_corners <= 0.0 and away_avg_corners <= 0.0:
            home_avg_corners, away_avg_corners = 5.1, 5.0

        avg_total = max(4.0, home_avg_corners + away_avg_corners)

        # Ritmo atual: corners por minuto vs média
        ritmo_factor = 1.0
        if minute >= 10:
            pace = corners_total / float(minute)
            avg_pace = avg_total / 90.0
            if avg_pace > 0:
                ritmo_factor = max(0.6, min(2.0, pace / avg_pace))

        # Pressão ofensiva: dangerous attacks por minuto
        pressure_factor = 1.0
        if minute >= 10:
            da_per_min = dangerous_total / float(minute)
            pressure_factor = max(0.8, min(1.5, 1.0 + (da_per_min / 15.0)))

        lambda_residual = max(0.05, avg_total * (minutes_remaining / 90.0) * ritmo_factor * pressure_factor)

        # Boosts
        if minute <= 45 and corners_total >= 6:
            lambda_residual *= 1.10
        if dangerous_total >= 60 and minute <= 60:
            lambda_residual *= 1.05

        # ---- Escolha institucional: melhor Over/Under >= 0.80 (linhas realistas) ----
        candidate_lines = [8.5, 9.5, 10.5, 11.5]
        best = None  # (prob, direction, line, k_threshold)

        for line in candidate_lines:
            # OVER: precisa de K corners adicionais tal que corners_total + K > line
            need = int(math.floor(line - corners_total) + 1)
            need = max(0, need)
            if need <= 0:
                p_over = 1.0
            else:
                p_over = 1.0 - poisson.cdf(need - 1, lambda_residual)

            if p_over >= 0.80:
                cand = (float(p_over), "over", float(line), int(need))
                if best is None or cand[0] > best[0]:
                    best = cand

            # UNDER: precisa que corners_total + K <= line
            allow = int(math.floor(line - corners_total))
            if allow < 0:
                p_under = 0.0
            else:
                p_under = float(poisson.cdf(allow, lambda_residual))

            if p_under >= 0.80:
                cand = (float(p_under), "under", float(line), int(allow))
                if best is None or cand[0] > best[0]:
                    best = cand

        if not best:
            return None

        prob, direction, line, k_thresh = best
        prob = min(prob, 0.93)
        if prob < 0.80:
            return None

        suggestion = f"{'Over' if direction == 'over' else 'Under'} {line:.1f} corners"

        meta = {
            "corners_total": corners_total,
            "minute": minute,
            "minutes_remaining": minutes_remaining,
            "avg_total_corners": round(avg_total, 3),
            "ritmo_factor": round(ritmo_factor, 3),
            "pressure_factor": round(pressure_factor, 3),
            "lambda_residual": round(lambda_residual, 3),
            "dangerous_total": round(dangerous_total, 1),
            "direction": direction,
            "line": float(line),
            "k_threshold": int(k_thresh),
            "model": "poisson_residual_corners_v1",
        }

        return {
            "category": "corners",
            "confidence": round(prob, 3),
            "suggestion": suggestion,
            "meta": meta
        }
