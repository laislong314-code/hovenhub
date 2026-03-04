"""
Downloader e importador de dados históricos — football-data.co.uk

CSVs disponíveis gratuitamente com:
  - Resultado final (FTHG, FTAG)
  - Árbitro (Referee)
  - Escanteios (HC, AC)
  - Cartões amarelos (HY, AY)
  - Cartões vermelhos (HR, AR)
  - Booking Points (HBP, ABP) — 10 pts amarelo, 25 pts vermelho
  - Odds históricas Bet365 (B365H, B365D, B365A, B365>2.5, B365<2.5)

URLs dos CSVs:
  Premier League:  https://www.football-data.co.uk/mmz4281/{season}/E0.csv
  La Liga:         https://www.football-data.co.uk/mmz4281/{season}/SP1.csv
  Serie A:         https://www.football-data.co.uk/mmz4281/{season}/I1.csv
  Champions Lg:    não disponível no football-data.co.uk

Seasons: 2122, 2223, 2324, 2425 (formato YYYY)
"""
import os
import httpx
import pandas as pd
from pathlib import Path
from loguru import logger

# Mapeamento liga → código do CSV
LEAGUE_CSV_CODES = {
    "PL":  "E0",   # Premier League
    "PD":  "SP1",  # La Liga
    "SA":  "I1",   # Serie A
    "BL1": "D1",   # Bundesliga
    "FL1": "F1",   # Ligue 1
    "PPL": "P1",   # Primeira Liga
    "DED": "N1",   # Eredivisie
}

# Temporadas disponíveis (formato YYYY = 2 últimos dígitos de cada ano)
SEASONS = ["2122", "2223", "2324", "2425"]

DATA_DIR = Path("data/historical")


async def download_csv(league_code: str, season: str) -> pd.DataFrame | None:
    """
    Baixa CSV histórico de uma liga/temporada.
    Retorna DataFrame ou None se não disponível.
    """
    csv_code = LEAGUE_CSV_CODES.get(league_code)
    if not csv_code:
        logger.warning(f"Liga {league_code} não disponível no football-data.co.uk")
        return None

    url = f"https://www.football-data.co.uk/mmz4281/{season}/{csv_code}.csv"
    local_path = DATA_DIR / f"{league_code}_{season}.csv"

    # Usar cache local se já baixou
    if local_path.exists():
        logger.debug(f"Cache hit: {local_path}")
        return pd.read_csv(local_path, encoding="latin1")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(r.content)
        df = pd.read_csv(local_path, encoding="latin1")
        logger.info(f"✅ Baixado: {league_code} {season} — {len(df)} jogos")
        return df

    except Exception as e:
        logger.warning(f"Erro ao baixar {league_code} {season}: {e}")
        return None


def parse_dataframe(df: pd.DataFrame, league_code: str, season: str) -> list[dict]:
    """
    Converte DataFrame bruto em lista de dicts estruturados.
    Lida com colunas ausentes graciosamente.
    """
    records = []

    for _, row in df.iterrows():
        try:
            # Ignorar linhas sem resultado
            if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                continue

            record = {
                "league_code": league_code,
                "season": season,
                "date": str(row.get("Date", "")),
                "home_team": str(row.get("HomeTeam", "")),
                "away_team": str(row.get("AwayTeam", "")),
                "referee": str(row.get("Referee", "Unknown")),

                # Resultado
                "home_goals": int(row.get("FTHG", 0)),
                "away_goals": int(row.get("FTAG", 0)),
                "total_goals": int(row.get("FTHG", 0)) + int(row.get("FTAG", 0)),
                "result": str(row.get("FTR", "")),  # H, D, A

                # Escanteios
                "home_corners": _safe_int(row.get("HC")),
                "away_corners": _safe_int(row.get("AC")),
                "total_corners": _safe_int(row.get("HC"), 0) + _safe_int(row.get("AC"), 0),

                # Cartões
                "home_yellow": _safe_int(row.get("HY")),
                "away_yellow": _safe_int(row.get("AY")),
                "home_red": _safe_int(row.get("HR")),
                "away_red": _safe_int(row.get("AR")),
                "home_booking_pts": _safe_int(row.get("HBP")),
                "away_booking_pts": _safe_int(row.get("ABP")),
                "total_cards": (
                    _safe_int(row.get("HY"), 0) +
                    _safe_int(row.get("AY"), 0) +
                    _safe_int(row.get("HR"), 0) +
                    _safe_int(row.get("AR"), 0)
                ),
                "total_booking_pts": (
                    _safe_int(row.get("HBP"), 0) +
                    _safe_int(row.get("ABP"), 0)
                ),

                # Odds históricas Bet365
                "b365_home": _safe_float(row.get("B365H")),
                "b365_draw": _safe_float(row.get("B365D")),
                "b365_away": _safe_float(row.get("B365A")),
                "b365_over25": _safe_float(row.get("B365>2.5")),
                "b365_under25": _safe_float(row.get("B365<2.5")),
            }

            records.append(record)

        except Exception as e:
            logger.debug(f"Linha ignorada: {e}")
            continue

    return records


def _safe_int(val, default=None):
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def _safe_float(val, default=None):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


async def download_all_leagues(leagues: list[str] = None, seasons: list[str] = None) -> list[dict]:
    """
    Baixa e processa todos os CSVs das ligas/temporadas especificadas.
    Retorna lista unificada de registros.
    """
    leagues = leagues or list(LEAGUE_CSV_CODES.keys())
    seasons = seasons or SEASONS
    all_records = []

    for league in leagues:
        for season in seasons:
            df = await download_csv(league, season)
            if df is not None and len(df) > 0:
                records = parse_dataframe(df, league, season)
                all_records.extend(records)
                logger.info(f"📊 {league} {season}: {len(records)} jogos processados")

    logger.info(f"✅ Total: {len(all_records)} jogos históricos carregados")
    return all_records
