"""
ComboEngine — motor de combinações de mercados para geração de sinais.

Gera:
  1) Sinais individuais (single leg) — qualquer mercado com EV+
  2) Combos 2 pernas do mesmo jogo (Over + BTTS, Home + BTTS, etc.)
  3) Combos 3 pernas do mesmo jogo (Over + Home + BTTS, etc.)
  4) Multi-jogo: combos entre jogos diferentes para fechar ODD alvo

Características:
  - Probabilidades correlacionadas calculadas via matrix Poisson
    (evita superprecificação de combos)
  - Targets de odd: 1.5-2.0, 2.0-3.0, 3.0-5.0, 5.0-8.0
  - Filtros por categoria: min_odd, max_odd, min_prob, min_ev
  - BTTS estimado quando ESPN não fornece odd (odd justa * 1.05)
  - Deduplicação: mesmo combo não enviado duas vezes por jogo
"""

import math
from itertools import combinations
from typing import Optional
from scipy.stats import poisson
from loguru import logger


# ── Constantes ────────────────────────────────────────────────────────────────

_MAX_GOALS = 13   # limite da matrix Poisson

# Categorias de odd alvo para combos multi-jogo
ODD_TARGETS = [
    {"label": "ODD ~2",   "min": 1.70, "max": 2.30},
    {"label": "ODD ~3",   "min": 2.50, "max": 3.50},
    {"label": "ODD ~5",   "min": 4.00, "max": 6.00},
    {"label": "ODD ALTA", "min": 6.00, "max": 10.0},
]

# Pares de mercados que são mutuamente exclusivos (não faz sentido combinar)
_MUTEX = [
    {"Home", "Away"}, {"Home", "Draw"}, {"Away", "Draw"},
]

def _mutex(k1: str, k2: str) -> bool:
    return {k1, k2} in _MUTEX

def _over_under_same_line(k1: str, k2: str) -> bool:
    """Over_2.5 + Under_2.5 = sem sentido"""
    if k1.startswith("Over_") and k2.startswith("Under_"):
        return k1.split("_", 1)[1] == k2.split("_", 1)[1]
    if k2.startswith("Over_") and k1.startswith("Under_"):
        return k1.split("_", 1)[1] == k2.split("_", 1)[1]
    return False

def _redundant(k1: str, k2: str) -> bool:
    """Over_1.5 + Over_2.5 = redundante (Over_2.5 já implica Over_1.5 parcialmente)"""
    if k1.startswith("Over_") and k2.startswith("Over_"):
        return True  # mantém só o mais exigente — mas aceita combos pois têm EV diferente
    return False


# ── Matrix Poisson ────────────────────────────────────────────────────────────

def build_matrix(lambda_home: float, lambda_away: float) -> dict:
    return {
        (i, j): poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        for i in range(_MAX_GOALS)
        for j in range(_MAX_GOALS)
    }


def market_single_prob(mat: dict, key: str) -> float:
    """Probabilidade de um mercado individual via matrix."""
    if key == "Home":
        return sum(p for (i, j), p in mat.items() if i > j)
    if key == "Away":
        return sum(p for (i, j), p in mat.items() if i < j)
    if key == "Draw":
        return sum(p for (i, j), p in mat.items() if i == j)
    if key == "BTTS":
        return sum(p for (i, j), p in mat.items() if i > 0 and j > 0)
    if key == "No_BTTS":
        return sum(p for (i, j), p in mat.items() if i == 0 or j == 0)
    # ── Dupla Chance: Home+Draw (1X), Draw+Away (X2), Home+Away (12) ─────────
    if key == "DC_Home_Draw":
        return sum(p for (i, j), p in mat.items() if i >= j)   # casa vence ou empata
    if key == "DC_Draw_Away":
        return sum(p for (i, j), p in mat.items() if i <= j)   # visitante vence ou empata
    if key == "DC_Home_Away":
        return sum(p for (i, j), p in mat.items() if i != j)   # qualquer resultado sem empate
    # 1º Tempo: Over_1T_X.X / Under_1T_X.X
    # A matrix Poisson é de jogo completo — não temos matrix de 1T.
    # Retorna 0 para que o sinal use model_prob injetado diretamente (via sokker_odds).
    if key.startswith("Over_1T_") or key.startswith("Under_1T_"):
        return 0.0
    if key.startswith("HT_"):
        return 0.0
    if key.startswith("Over_"):
        try:
            line = float(key.split("_", 1)[1])
        except ValueError:
            return 0.0
        return sum(p for (i, j), p in mat.items() if i + j > line)
    if key.startswith("Under_"):
        try:
            line = float(key.split("_", 1)[1])
        except ValueError:
            return 0.0
        return sum(p for (i, j), p in mat.items() if i + j < line)
    return 0.0


def _outcome_passes(i: int, j: int, key: str) -> bool:
    if key == "Home":           return i > j
    if key == "Away":           return i < j
    if key == "Draw":           return i == j
    if key == "BTTS":           return i > 0 and j > 0
    if key == "No_BTTS":        return i == 0 or j == 0
    if key == "DC_Home_Draw":   return i >= j
    if key == "DC_Draw_Away":   return i <= j
    if key == "DC_Home_Away":   return i != j
    # 1T e HT sem matrix própria — excluídos de combos
    if key.startswith("Over_1T_") or key.startswith("Under_1T_"):
        return False
    if key.startswith("HT_"):
        return False
    if key.startswith("Over_"):
        try:
            return i + j > float(key.split("_", 1)[1])
        except ValueError:
            return False
    if key.startswith("Under_"):
        try:
            return i + j < float(key.split("_", 1)[1])
        except ValueError:
            return False
    return False


def combo_prob_correlated(mat: dict, keys: list[str]) -> float:
    """Probabilidade real (correlacionada) de todos os mercados ocorrerem juntos."""
    return sum(
        p for (i, j), p in mat.items()
        if all(_outcome_passes(i, j, k) for k in keys)
    )


# ── Odds de BTTS estimadas ────────────────────────────────────────────────────

def estimate_btts_odds(prob_btts: float, margin: float = 1.05) -> float:
    """Estima odd justa de BTTS quando ESPN não fornece."""
    if prob_btts <= 0:
        return 0.0
    return round((1.0 / prob_btts) * margin, 2)


# ── Gerador principal ─────────────────────────────────────────────────────────

class ComboEngine:
    """
    Gera sinais individuais e combos a partir de um ou mais jogos analisados.
    """

    def __init__(self, settings):
        self.s = settings

    # ── Singles ───────────────────────────────────────────────────────────────

    def get_single_signals(
        self,
        match_id: str,
        odds: dict,
        mat: dict,
        btts_prob: float,
    ) -> list[dict]:
        """
        Gera sinais individuais para todos os mercados disponíveis + BTTS estimado.
        Retorna lista de dicts prontos para processar.
        """
        all_odds = dict(odds)

        # BTTS estimado removido — só usamos odds reais de bookmakers
        signals = []
        for key, odd_info in all_odds.items():
            odd = odd_info["odd"] if isinstance(odd_info, dict) else odd_info
            bm  = odd_info.get("bookmaker", "Estimado") if isinstance(odd_info, dict) else "Estimado"

            # ── Rejeita qualquer odd sem bookmaker real ──────────────────────
            if bm == "Estimado" or not bm:
                continue

            if odd <= 0:
                continue

            # Usa model_prob injetado (corners/cards) ou calcula via matrix Poisson
            if isinstance(odd_info, dict) and odd_info.get("model_prob"):
                prob = odd_info["model_prob"]
            else:
                prob = market_single_prob(mat, key)
            if prob <= 0:
                continue

            ev = prob * odd - 1
            cat = self._market_category(key)
            filters = self._get_filters(cat)

            if not self._passes(prob, odd, ev, filters):
                continue

            signals.append({
                "type":      "single",
                "legs":      [key],
                "prob":      round(prob, 4),
                "odd":       round(odd, 3),
                "ev":        round(ev, 4),
                "bookmaker": bm,
                "market_id": key,
                "label":     self._market_label(key),
                "category":  cat,
            })

        return signals

    # ── Combos mesmo jogo ─────────────────────────────────────────────────────

    def get_combo_signals_single_game(
        self,
        match_id: str,
        odds: dict,
        mat: dict,
        btts_prob: float,
        max_legs: int = 3,
    ) -> list[dict]:
        """
        Gera combos de 2 e 3 pernas do mesmo jogo.
        Usa probabilidades correlacionadas via matrix Poisson.
        """
        all_odds = dict(odds)
        # BTTS estimado removido — só usamos odds reais de bookmakers
        # Filtra odds sem bookmaker real antes de gerar combos
        all_odds = {k: v for k, v in all_odds.items()
                    if isinstance(v, dict) and v.get("bookmaker") and v.get("bookmaker") != "Estimado"}

        keys = [k for k, v in all_odds.items() if (v["odd"] if isinstance(v, dict) else v) > 1.0]
        combos = []

        for n_legs in range(2, max_legs + 1):
            for leg_keys in combinations(keys, n_legs):
                # Valida combinação
                valid = True
                for a, b in combinations(leg_keys, 2):
                    if _mutex(a, b) or _over_under_same_line(a, b):
                        valid = False
                        break
                if not valid:
                    continue

                # Calcula probabilidade correlacionada
                prob = combo_prob_correlated(mat, list(leg_keys))
                if prob <= 0:
                    continue

                # Odd do combo = produto das odds individuais
                odd_combo = 1.0
                bookmakers = []
                for k in leg_keys:
                    info = all_odds[k]
                    o    = info["odd"] if isinstance(info, dict) else info
                    bm   = info.get("bookmaker", "Estimado") if isinstance(info, dict) else "Estimado"
                    odd_combo *= o
                    bookmakers.append(bm)

                odd_combo = round(odd_combo, 2)
                ev = prob * odd_combo - 1

                # Filtros para combos
                if not self._passes_combo(prob, odd_combo, ev, n_legs):
                    continue

                label = " + ".join(self._market_label(k) for k in leg_keys)
                combo_id = "COMBO_" + "_".join(sorted(leg_keys))

                combos.append({
                    "type":      f"combo_{n_legs}",
                    "legs":      list(leg_keys),
                    "prob":      round(prob, 4),
                    "odd":       odd_combo,
                    "ev":        round(ev, 4),
                    "bookmaker": " | ".join(set(bookmakers)),
                    "market_id": combo_id,
                    "label":     label,
                    "category":  "combo",
                    "n_legs":    n_legs,
                })

        return combos

    # ── Combos multi-jogo ─────────────────────────────────────────────────────

    def get_multi_game_combos(
        self,
        game_signals: list[dict],
        target_bands: list[dict] = None,
    ) -> list[dict]:
        """
        Combina singles/combos de jogos DIFERENTES para atingir ODD alvo.

        game_signals: lista de dicts com {match_id, match_label, signal}
        Retorna combos multi-jogo ordenados por EV.
        """
        if target_bands is None:
            target_bands = ODD_TARGETS

        if len(game_signals) < 2:
            return []

        results = []

        # Combos de 2 jogos diferentes
        for (g1, g2) in combinations(game_signals, 2):
            s1 = g1["signal"]
            s2 = g2["signal"]

            # Jogos diferentes → probabilidades independentes
            prob_combined = s1["prob"] * s2["prob"]
            odd_combined  = round(s1["odd"] * s2["odd"], 2)
            ev_combined   = prob_combined * odd_combined - 1

            # Verifica se cai em algum target
            target_label = None
            for band in target_bands:
                if band["min"] <= odd_combined <= band["max"]:
                    target_label = band["label"]
                    break

            if target_label is None:
                continue

            if not self._passes_multi(prob_combined, odd_combined, ev_combined):
                continue

            combo_id = f"dupla_{g1['match_id']}_{g2['match_id']}"

            results.append({
                "type":        "multi_2",
                "legs":        [s1["label"], s2["label"]],
                "games":       [g1["match_label"], g2["match_label"]],
                "prob":        round(prob_combined, 4),
                "odd":         odd_combined,
                "ev":          round(ev_combined, 4),
                "bookmaker":   f"{s1['bookmaker']} | {s2['bookmaker']}",
                "market_id":   combo_id,
                "label":       f"{g1['match_label']}: {s1['label']} + {g2['match_label']}: {s2['label']}",
                "category":    "multi",
                "target_band": target_label,
            })

        # Combos de 3 jogos (só para ODD ~5 e ALTA)
        if len(game_signals) >= 3:
            high_bands = [b for b in target_bands if b["min"] >= 3.0]
            for trio in combinations(game_signals, 3):
                g1, g2, g3 = trio
                s1, s2, s3 = g1["signal"], g2["signal"], g3["signal"]

                prob_combined = s1["prob"] * s2["prob"] * s3["prob"]
                odd_combined  = round(s1["odd"] * s2["odd"] * s3["odd"], 2)
                ev_combined   = prob_combined * odd_combined - 1

                target_label = None
                for band in high_bands:
                    if band["min"] <= odd_combined <= band["max"]:
                        target_label = band["label"]
                        break

                if target_label is None:
                    continue

                if not self._passes_multi(prob_combined, odd_combined, ev_combined):
                    continue

                combo_id = f"tripla_{g1['match_id']}_{g2['match_id']}_{g3['match_id']}"
                results.append({
                    "type":        "multi_3",
                    "legs":        [s1["label"], s2["label"], s3["label"]],
                    "games":       [g1["match_label"], g2["match_label"], g3["match_label"]],
                    "prob":        round(prob_combined, 4),
                    "odd":         odd_combined,
                    "ev":          round(ev_combined, 4),
                    "bookmaker":   " | ".join({s1["bookmaker"], s2["bookmaker"], s3["bookmaker"]}),
                    "market_id":   combo_id,
                    "label":       " + ".join(
                        f"{g['match_label'][:15]}: {s['label']}"
                        for g, s in [(g1,s1),(g2,s2),(g3,s3)]
                    ),
                    "category":    "multi",
                    "target_band": target_label,
                })

        return sorted(results, key=lambda x: x["ev"], reverse=True)

    # ── Filtros ───────────────────────────────────────────────────────────────

    def _passes(self, prob: float, odd: float, ev: float, filters: dict) -> bool:
        return (
            prob >= filters["min_prob"] and
            ev   >= filters["min_ev"]   and
            filters["min_odd"] <= odd <= filters["max_odd"]
        )

    def _passes_combo(self, prob: float, odd: float, ev: float, n_legs: int) -> bool:
        """Filtros para combos do mesmo jogo."""
        min_prob = max(0.40, 0.55 - (n_legs - 1) * 0.05)
        return (
            prob >= min_prob  and
            ev   >= 0.05      and
            1.50 <= odd <= 10.0
        )

    def _passes_multi(self, prob: float, odd: float, ev: float) -> bool:
        """Filtros para combos multi-jogo."""
        return (
            prob >= 0.35 and
            ev   >= 0.05 and
            1.50 <= odd <= 10.0
        )

    def _market_category(self, key: str) -> str:
        if key in ("Home", "Away", "Draw"):                     return "result"
        if key in ("BTTS", "No_BTTS"):                          return "btts"
        if key.startswith(("HT_", "Over_1T_", "Under_1T_")):   return "halftime"
        if key.startswith(("Over_", "Under_")):                 return "goals"
        if key.startswith("Corners_"):                          return "corners"
        if key.startswith("Cards_"):                            return "cards"
        return "other"

    def _get_filters(self, cat: str) -> dict:
        s = self.s
        maps = {
            "goals":     {"min_odd": s.goals_min_odd,   "max_odd": s.goals_max_odd,   "min_prob": s.goals_min_prob,   "min_ev": s.goals_min_ev},
            "btts":      {"min_odd": s.btts_min_odd,     "max_odd": s.btts_max_odd,     "min_prob": s.btts_min_prob,     "min_ev": s.btts_min_ev},
            "result":    {"min_odd": s.result_min_odd,   "max_odd": s.result_max_odd,   "min_prob": s.result_min_prob,   "min_ev": s.result_min_ev},
            "corners":   {"min_odd": s.corners_min_odd,  "max_odd": s.corners_max_odd,  "min_prob": s.corners_min_prob,  "min_ev": s.corners_min_ev},
            "cards":     {"min_odd": s.cards_min_odd,    "max_odd": s.cards_max_odd,    "min_prob": s.cards_min_prob,    "min_ev": s.cards_min_ev},
            # 1º tempo usa os mesmos filtros de gols
            "halftime":  {"min_odd": s.goals_min_odd,   "max_odd": s.goals_max_odd,   "min_prob": s.goals_min_prob,   "min_ev": s.goals_min_ev},
        }
        return maps.get(cat, maps["goals"])

    # ── Labels ────────────────────────────────────────────────────────────────

    @staticmethod
    def _market_label(key: str) -> str:
        labels = {
            "Home":         "🏠 Vitória Casa",
            "Away":         "✈️ Vitória Fora",
            "Draw":         "🤝 Empate",
            "BTTS":         "🎯 BTTS Sim",
            "No_BTTS":      "🎯 BTTS Não",
            "Over_1.5":     "⚽ Over 1.5",
            "Over_2.5":     "⚽ Over 2.5",
            "Over_3.5":     "⚽ Over 3.5",
            "Over_4.5":     "⚽ Over 4.5",
            "Under_1.5":    "⚽ Under 1.5",
            "Under_2.5":    "⚽ Under 2.5",
            "Under_3.5":    "⚽ Under 3.5",
            "Under_4.5":    "⚽ Under 4.5",
            "Over_1T_0.5":  "1T Over 0.5",
            "Over_1T_1.5":  "1T Over 1.5",
            "Under_1T_0.5": "1T Under 0.5",
            "Under_1T_1.5": "1T Under 1.5",
            "HT_Home":      "🏠 1T Casa",
            "HT_Draw":      "🤝 1T Empate",
            "HT_Away":      "✈️ 1T Fora",
        }
        if key in labels:
            return labels[key]
        if key.startswith("Corners_Over_"):
            return f"🚩 Esc. Over {key.split('_')[2]}"
        if key.startswith("Corners_Under_"):
            return f"🚩 Esc. Under {key.split('_')[2]}"
        if key.startswith("Cards_Over_"):
            return f"🟨 B.Pts Over {key.split('_')[2]}"
        if key.startswith("Cards_Under_"):
            return f"🟩 B.Pts Under {key.split('_')[2]}"
        if key.startswith("Over_1T_"):
            return f"1T Over {key.split('Over_1T_')[1]}"
        if key.startswith("Under_1T_"):
            return f"1T Under {key.split('Under_1T_')[1]}"
        return key
