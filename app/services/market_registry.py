"""
market_registry.py — Registro e parser de mercados SokkerPRO para Beto v5.

Formato de odds da fixture: "odd#suspenso"  → ex: "2.60#0" = odd 2.60, ativa
                                              ex: "1.44#1" = odd 1.44, suspensa

Mapeamento completo dos mercados disponíveis nas fixtures SokkerPRO.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# ── Categorias de mercado ─────────────────────────────────────────────────────
CATEGORY_GOLS    = "gols"
CATEGORY_CORNERS = "corners"
CATEGORY_CARDS   = "cards"
CATEGORY_RESULT  = "result"
CATEGORY_BTTS    = "btts"

# ── Multiplicadores de stake por categoria (Beto v5 config) ───────────────────
STAKE_MULTIPLIER: dict[str, float] = {
    CATEGORY_GOLS:    1.0,
    CATEGORY_CORNERS: 0.6,
    CATEGORY_CARDS:   0.6,
    CATEGORY_RESULT:  1.0,
    CATEGORY_BTTS:    1.0,
}
STAKE_MULTI: float = 0.8  # multiplicador adicional para múltiplas


# ── Parser de odd ─────────────────────────────────────────────────────────────

def parse_odd(raw) -> tuple[float, bool]:
    """
    Parseia formato SokkerPRO: "2.60#0"
    Retorna (odd_float, ativa).
    """
    if raw is None:
        return 0.0, False
    if isinstance(raw, (int, float)):
        v = float(raw)
        return (v, True) if v > 1.0 else (0.0, False)
    s = str(raw).strip()
    if not s:
        return 0.0, False
    if "#" in s:
        parts = s.split("#", 1)
        try:
            odd = float(parts[0])
            suspensa = parts[1].strip() != "0"
            return (odd, not suspensa) if odd > 1.0 else (0.0, False)
        except ValueError:
            return 0.0, False
    try:
        odd = float(s)
        return (odd, True) if odd > 1.0 else (0.0, False)
    except ValueError:
        return 0.0, False


# ── Definição de mercado ──────────────────────────────────────────────────────

@dataclass
class MarketDef:
    fixture_key:  str
    internal_key: str
    category:     str
    line:         Optional[float] = None
    bookmaker:    str = "BET365"
    label:        str = ""


# ── Registro completo ─────────────────────────────────────────────────────────

MARKET_REGISTRY: list[MarketDef] = [
    # GOLS PRÉ — BET365
    MarketDef("BET365_GOLS_OVER_1_5",   "Over_1.5",   CATEGORY_GOLS, 1.5, "BET365", "⚽ Over 1.5"),
    MarketDef("BET365_GOLS_UNDER_1_5",  "Under_1.5",  CATEGORY_GOLS, 1.5, "BET365", "⚽ Under 1.5"),
    MarketDef("BET365_GOLS_OVER_2_5",   "Over_2.5",   CATEGORY_GOLS, 2.5, "BET365", "⚽ Over 2.5"),
    MarketDef("BET365_GOLS_UNDER_2_5",  "Under_2.5",  CATEGORY_GOLS, 2.5, "BET365", "⚽ Under 2.5"),
    MarketDef("BET365_GOLS_OVER_3_5",   "Over_3.5",   CATEGORY_GOLS, 3.5, "BET365", "⚽ Over 3.5"),
    MarketDef("BET365_GOLS_UNDER_3_5",  "Under_3.5",  CATEGORY_GOLS, 3.5, "BET365", "⚽ Under 3.5"),
    MarketDef("BET365_GOLS_OVER_4_5",   "Over_4.5",   CATEGORY_GOLS, 4.5, "BET365", "⚽ Over 4.5"),
    MarketDef("BET365_GOLS_UNDER_4_5",  "Under_4.5",  CATEGORY_GOLS, 4.5, "BET365", "⚽ Under 4.5"),
    # GOLS LIVE
    MarketDef("BET365_GOLS_OVER_0_5_LIVE",  "Over_0.5_LIVE",  CATEGORY_GOLS, 0.5, "BET365", "⚽ Over 0.5 LIVE"),
    MarketDef("BET365_GOLS_OVER_1_5_LIVE",  "Over_1.5_LIVE",  CATEGORY_GOLS, 1.5, "BET365", "⚽ Over 1.5 LIVE"),
    MarketDef("BET365_GOLS_UNDER_1_5_LIVE", "Under_1.5_LIVE", CATEGORY_GOLS, 1.5, "BET365", "⚽ Under 1.5 LIVE"),
    MarketDef("BET365_GOLS_OVER_2_5_LIVE",  "Over_2.5_LIVE",  CATEGORY_GOLS, 2.5, "BET365", "⚽ Over 2.5 LIVE"),
    MarketDef("BET365_GOLS_UNDER_2_5_LIVE", "Under_2.5_LIVE", CATEGORY_GOLS, 2.5, "BET365", "⚽ Under 2.5 LIVE"),
    MarketDef("BET365_GOLS_OVER_3_5_LIVE",  "Over_3.5_LIVE",  CATEGORY_GOLS, 3.5, "BET365", "⚽ Over 3.5 LIVE"),
    MarketDef("BET365_GOLS_UNDER_3_5_LIVE", "Under_3.5_LIVE", CATEGORY_GOLS, 3.5, "BET365", "⚽ Under 3.5 LIVE"),
    # RESULTADO
    MarketDef("BET365_VENCEDOR_HOME", "Home", CATEGORY_RESULT, None, "BET365", "🏠 Vitória Casa"),
    MarketDef("BET365_VENCEDOR_DRAW", "Draw", CATEGORY_RESULT, None, "BET365", "🤝 Empate"),
    MarketDef("BET365_VENCEDOR_AWAY", "Away", CATEGORY_RESULT, None, "BET365", "✈️ Vitória Fora"),
    # BTTS
    MarketDef("BET365_AMBOS_MARCAM_SIM", "BTTS",    CATEGORY_BTTS, None, "BET365", "🎯 BTTS Sim"),
    MarketDef("BET365_AMBOS_MARCAM_NAO", "No_BTTS", CATEGORY_BTTS, None, "BET365", "🎯 BTTS Não"),
    # ESCANTEIOS PRÉ
    MarketDef("BET365_CANTO_OVER_7",     "Corners_Over_7.0",   CATEGORY_CORNERS, 7.0,  "BET365", "🚩 Esc. Over 7"),
    MarketDef("BET365_CANTO_UNDER_7",    "Corners_Under_7.0",  CATEGORY_CORNERS, 7.0,  "BET365", "🚩 Esc. Under 7"),
    MarketDef("BET365_CANTO_OVER_8",     "Corners_Over_8.0",   CATEGORY_CORNERS, 8.0,  "BET365", "🚩 Esc. Over 8"),
    MarketDef("BET365_CANTO_UNDER_8",    "Corners_Under_8.0",  CATEGORY_CORNERS, 8.0,  "BET365", "🚩 Esc. Under 8"),
    MarketDef("BET365_CANTO_OVER_8_5",   "Corners_Over_8.5",   CATEGORY_CORNERS, 8.5,  "BET365", "🚩 Esc. Over 8.5"),
    MarketDef("BET365_CANTO_UNDER_8_5",  "Corners_Under_8.5",  CATEGORY_CORNERS, 8.5,  "BET365", "🚩 Esc. Under 8.5"),
    MarketDef("BET365_CANTO_OVER_9",     "Corners_Over_9.0",   CATEGORY_CORNERS, 9.0,  "BET365", "🚩 Esc. Over 9"),
    MarketDef("BET365_CANTO_UNDER_9",    "Corners_Under_9.0",  CATEGORY_CORNERS, 9.0,  "BET365", "🚩 Esc. Under 9"),
    MarketDef("BET365_CANTO_OVER_9_5",   "Corners_Over_9.5",   CATEGORY_CORNERS, 9.5,  "BET365", "🚩 Esc. Over 9.5"),
    MarketDef("BET365_CANTO_UNDER_9_5",  "Corners_Under_9.5",  CATEGORY_CORNERS, 9.5,  "BET365", "🚩 Esc. Under 9.5"),
    MarketDef("BET365_CANTO_OVER_10",    "Corners_Over_10.0",  CATEGORY_CORNERS, 10.0, "BET365", "🚩 Esc. Over 10"),
    MarketDef("BET365_CANTO_UNDER_10",   "Corners_Under_10.0", CATEGORY_CORNERS, 10.0, "BET365", "🚩 Esc. Under 10"),
    MarketDef("BET365_CANTO_OVER_10_5",  "Corners_Over_10.5",  CATEGORY_CORNERS, 10.5, "BET365", "🚩 Esc. Over 10.5"),
    MarketDef("BET365_CANTO_UNDER_10_5", "Corners_Under_10.5", CATEGORY_CORNERS, 10.5, "BET365", "🚩 Esc. Under 10.5"),
    MarketDef("BET365_CANTO_OVER_11",    "Corners_Over_11.0",  CATEGORY_CORNERS, 11.0, "BET365", "🚩 Esc. Over 11"),
    MarketDef("BET365_CANTO_OVER_11_5",  "Corners_Over_11.5",  CATEGORY_CORNERS, 11.5, "BET365", "🚩 Esc. Over 11.5"),
    MarketDef("BET365_CANTO_UNDER_11_5", "Corners_Under_11.5", CATEGORY_CORNERS, 11.5, "BET365", "🚩 Esc. Under 11.5"),
    MarketDef("BET365_CANTO_OVER_12_5",  "Corners_Over_12.5",  CATEGORY_CORNERS, 12.5, "BET365", "🚩 Esc. Over 12.5"),
    MarketDef("BET365_CANTO_UNDER_12_5", "Corners_Under_12.5", CATEGORY_CORNERS, 12.5, "BET365", "🚩 Esc. Under 12.5"),
    # ESCANTEIOS LIVE
    MarketDef("BET365_CANTO_OVER_4_5_LIVE",   "Corners_Over_4.5_LIVE",   CATEGORY_CORNERS, 4.5,  "BET365", "🚩 Esc. Over 4.5 LIVE"),
    MarketDef("BET365_CANTO_UNDER_4_5_LIVE",  "Corners_Under_4.5_LIVE",  CATEGORY_CORNERS, 4.5,  "BET365", "🚩 Esc. Under 4.5 LIVE"),
    MarketDef("BET365_CANTO_OVER_8_LIVE",     "Corners_Over_8.0_LIVE",   CATEGORY_CORNERS, 8.0,  "BET365", "🚩 Esc. Over 8 LIVE"),
    MarketDef("BET365_CANTO_UNDER_8_LIVE",    "Corners_Under_8.0_LIVE",  CATEGORY_CORNERS, 8.0,  "BET365", "🚩 Esc. Under 8 LIVE"),
    MarketDef("BET365_CANTO_OVER_9_LIVE",     "Corners_Over_9.0_LIVE",   CATEGORY_CORNERS, 9.0,  "BET365", "🚩 Esc. Over 9 LIVE"),
    MarketDef("BET365_CANTO_UNDER_9_LIVE",    "Corners_Under_9.0_LIVE",  CATEGORY_CORNERS, 9.0,  "BET365", "🚩 Esc. Under 9 LIVE"),
    MarketDef("BET365_CANTO_OVER_10_LIVE",    "Corners_Over_10.0_LIVE",  CATEGORY_CORNERS, 10.0, "BET365", "🚩 Esc. Over 10 LIVE"),
    MarketDef("BET365_CANTO_UNDER_10_LIVE",   "Corners_Under_10.0_LIVE", CATEGORY_CORNERS, 10.0, "BET365", "🚩 Esc. Under 10 LIVE"),
    MarketDef("BET365_CANTO_OVER_10_5_LIVE",  "Corners_Over_10.5_LIVE",  CATEGORY_CORNERS, 10.5, "BET365", "🚩 Esc. Over 10.5 LIVE"),
    MarketDef("BET365_CANTO_UNDER_10_5_LIVE", "Corners_Under_10.5_LIVE", CATEGORY_CORNERS, 10.5, "BET365", "🚩 Esc. Under 10.5 LIVE"),
    MarketDef("BET365_CANTO_OVER_11_5_LIVE",  "Corners_Over_11.5_LIVE",  CATEGORY_CORNERS, 11.5, "BET365", "🚩 Esc. Over 11.5 LIVE"),
    MarketDef("BET365_CANTO_UNDER_11_5_LIVE", "Corners_Under_11.5_LIVE", CATEGORY_CORNERS, 11.5, "BET365", "🚩 Esc. Under 11.5 LIVE"),
    MarketDef("BET365_CANTO_OVER_12_5_LIVE",  "Corners_Over_12.5_LIVE",  CATEGORY_CORNERS, 12.5, "BET365", "🚩 Esc. Over 12.5 LIVE"),
    MarketDef("BET365_CANTO_UNDER_12_5_LIVE", "Corners_Under_12.5_LIVE", CATEGORY_CORNERS, 12.5, "BET365", "🚩 Esc. Under 12.5 LIVE"),
]


def extract_odds_from_fixture(fixture: dict) -> dict:
    """
    Extrai odds da fixture SokkerPRO → dict normalizado:
    { internal_key: {"odd": float, "bookmaker": str, "label": str, "category": str} }
    Mantém a maior odd disponível quando a mesma chave interna aparece em múltiplos BKs.
    """
    result: dict[str, dict] = {}
    for mdef in MARKET_REGISTRY:
        raw = fixture.get(mdef.fixture_key)
        if raw is None:
            continue
        odd, ativa = parse_odd(raw)
        if not ativa or odd <= 1.0:
            continue
        ikey = mdef.internal_key
        existing = result.get(ikey)
        if existing is None or odd > existing["odd"]:
            result[ikey] = {
                "odd":       odd,
                "bookmaker": mdef.bookmaker,
                "label":     mdef.label,
                "category":  mdef.category,
            }
    return result


def get_stake_multiplier(category: str) -> float:
    return STAKE_MULTIPLIER.get(category, 1.0)


def get_category_for_key(internal_key: str) -> str:
    if internal_key.startswith("Corners_"):
        return CATEGORY_CORNERS
    if internal_key.startswith("Cards_"):
        return CATEGORY_CARDS
    if internal_key in ("BTTS", "No_BTTS"):
        return CATEGORY_BTTS
    if internal_key in ("Home", "Away", "Draw"):
        return CATEGORY_RESULT
    return CATEGORY_GOLS
class MarketRegistry:
    @staticmethod
    def all():
        return MARKET_REGISTRY
