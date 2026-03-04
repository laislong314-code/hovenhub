"""
Modelos estatísticos para mercados especiais:
  - Escanteios (Over/Under)
  - Cartões (Over/Under booking points)
  - Árbitro (fator de ajuste)
  - BTTS
  - Over/Under 1.5 e 3.5
  - Resultado Final 1X2
"""
import math
from dataclasses import dataclass
from typing import Optional
from scipy.stats import poisson
from loguru import logger


@dataclass
class RefereeProfile:
    name: str
    avg_yellow_per_game: float = 3.5
    avg_red_per_game: float = 0.15
    avg_booking_pts_per_game: float = 38.0
    avg_corners_per_game: float = 10.0
    matches_officiated: int = 0
    strictness_index: float = 1.0


@dataclass
class TeamCornerProfile:
    team_name: str
    avg_corners_for: float = 5.0
    avg_corners_against: float = 5.0
    matches_used: int = 0


@dataclass
class TeamCardProfile:
    team_name: str
    avg_yellow_for: float = 1.8
    avg_yellow_against: float = 1.8
    avg_booking_pts_for: float = 20.0
    avg_booking_pts_against: float = 20.0
    matches_used: int = 0


class ProfileBuilder:

    def build_referee_profiles(
        self,
        records: list[dict],
        min_games: int = 5,
    ) -> dict[str, RefereeProfile]:
        from collections import defaultdict
        stats = defaultdict(lambda: {
            "yellows": [], "reds": [], "bpts": [], "corners": []
        })

        for r in records:
            ref = r.get("referee", "Unknown")
            if not ref or ref in ("Unknown", "nan", ""):
                continue

            total_yellow = (r.get("home_yellow") or 0) + (r.get("away_yellow") or 0)
            total_red = (r.get("home_red") or 0) + (r.get("away_red") or 0)
            total_corners = (r.get("total_corners") or 0)

            # Ignorar jogos sem dados de cartão
            if total_yellow == 0 and total_red == 0:
                continue

            # Calcular booking points: amarelo=10pts, vermelho=25pts
            stored_bpts = r.get("total_booking_pts") or 0
            total_bpts = stored_bpts if stored_bpts > 0 else (total_yellow * 10) + (total_red * 25)

            stats[ref]["yellows"].append(total_yellow)
            stats[ref]["reds"].append(total_red)
            stats[ref]["bpts"].append(total_bpts)
            if total_corners > 0:
                stats[ref]["corners"].append(total_corners)

        # Média global de booking pts para calcular strictness
        all_bpts = [b for s in stats.values() for b in s["bpts"] if b > 0]
        global_avg_bpts = sum(all_bpts) / len(all_bpts) if all_bpts else 38.0

        profiles = {}
        for ref, data in stats.items():
            if len(data["yellows"]) < min_games:
                continue

            avg_yellow = sum(data["yellows"]) / len(data["yellows"])
            avg_red = sum(data["reds"]) / len(data["reds"])
            avg_bpts = sum(data["bpts"]) / len(data["bpts"])
            avg_corners = sum(data["corners"]) / len(data["corners"]) if data["corners"] else 10.0
            strictness = avg_bpts / global_avg_bpts if global_avg_bpts > 0 else 1.0

            profiles[ref] = RefereeProfile(
                name=ref,
                avg_yellow_per_game=round(avg_yellow, 2),
                avg_red_per_game=round(avg_red, 3),
                avg_booking_pts_per_game=round(avg_bpts, 2),
                avg_corners_per_game=round(avg_corners, 2),
                matches_officiated=len(data["yellows"]),
                strictness_index=round(strictness, 3),
            )

        logger.info(f"📋 {len(profiles)} perfis de árbitros construídos")
        return profiles

    def build_corner_profiles(
        self,
        records: list[dict],
        team_name: str,
        last_n: int = 10,
    ) -> TeamCornerProfile:
        team_records = [
            r for r in records
            if r.get("home_team") == team_name or r.get("away_team") == team_name
        ][-last_n:]

        if len(team_records) < 3:
            return TeamCornerProfile(team_name=team_name)

        corners_for, corners_against = [], []
        for r in team_records:
            is_home = r.get("home_team") == team_name
            hc = r.get("home_corners") or 0
            ac = r.get("away_corners") or 0
            if hc == 0 and ac == 0:
                continue
            corners_for.append(hc if is_home else ac)
            corners_against.append(ac if is_home else hc)

        if not corners_for:
            return TeamCornerProfile(team_name=team_name)

        return TeamCornerProfile(
            team_name=team_name,
            avg_corners_for=round(sum(corners_for) / len(corners_for), 2),
            avg_corners_against=round(sum(corners_against) / len(corners_against), 2),
            matches_used=len(corners_for),
        )

    def build_card_profiles(
        self,
        records: list[dict],
        team_name: str,
        last_n: int = 10,
    ) -> TeamCardProfile:
        team_records = [
            r for r in records
            if r.get("home_team") == team_name or r.get("away_team") == team_name
        ][-last_n:]

        if len(team_records) < 3:
            return TeamCardProfile(team_name=team_name)

        yellows_for, bpts_for = [], []
        for r in team_records:
            is_home = r.get("home_team") == team_name
            hy = r.get("home_yellow") or 0
            ay = r.get("away_yellow") or 0
            hbp = r.get("home_booking_pts") or 0
            abp = r.get("away_booking_pts") or 0

            yellows_for.append(hy if is_home else ay)
            # Calcular booking pts se não disponível
            if hbp == 0 and abp == 0:
                hr = r.get("home_red") or 0
                ar = r.get("away_red") or 0
                bpt = (hy * 10 + hr * 25) if is_home else (ay * 10 + ar * 25)
            else:
                bpt = hbp if is_home else abp
            bpts_for.append(bpt)

        return TeamCardProfile(
            team_name=team_name,
            avg_yellow_for=round(sum(yellows_for) / len(yellows_for), 2),
            avg_booking_pts_for=round(sum(bpts_for) / len(bpts_for), 2),
            matches_used=len(yellows_for),
        )


class SpecialMarketsModel:

    GLOBAL_CORNER_AVG = 10.3
    GLOBAL_BPTS_AVG = 38.0

    def prob_over_line(self, lambda_val: float, line: float) -> float:
        k = math.floor(line)
        prob_under_or_equal = sum(poisson.pmf(i, lambda_val) for i in range(k + 1))
        return round(max(0.0, min(1.0, 1.0 - prob_under_or_equal)), 4)

    def calculate_corners(
        self,
        home_profile: TeamCornerProfile,
        away_profile: TeamCornerProfile,
        referee_profile: Optional[RefereeProfile] = None,
        league_avg_corners: float = 10.3,
    ) -> dict:
        half = league_avg_corners / 2
        lambda_home = (home_profile.avg_corners_for / half) * \
                      (away_profile.avg_corners_against / half) * half
        lambda_away = (away_profile.avg_corners_for / half) * \
                      (home_profile.avg_corners_against / half) * half
        lambda_total = lambda_home + lambda_away

        if referee_profile and referee_profile.avg_corners_per_game > 0:
            corner_factor = referee_profile.avg_corners_per_game / self.GLOBAL_CORNER_AVG
            lambda_total *= corner_factor

        lambda_total = max(5.0, min(lambda_total, 20.0))

        return {
            "lambda_home": round(lambda_home, 3),
            "lambda_away": round(lambda_away, 3),
            "lambda_total": round(lambda_total, 3),
            "prob_over_85": self.prob_over_line(lambda_total, 8.5),
            "prob_over_95": self.prob_over_line(lambda_total, 9.5),
            "prob_over_105": self.prob_over_line(lambda_total, 10.5),
            "prob_under_95": round(1 - self.prob_over_line(lambda_total, 9.5), 4),
        }

    def calculate_cards(
        self,
        home_card: TeamCardProfile,
        away_card: TeamCardProfile,
        referee_profile: Optional[RefereeProfile] = None,
        league_avg_bpts: float = 38.0,
    ) -> dict:
        base_lambda = home_card.avg_booking_pts_for + away_card.avg_booking_pts_for
        referee_factor = 1.0

        if referee_profile and referee_profile.strictness_index > 0:
            referee_factor = referee_profile.strictness_index
            base_lambda *= referee_factor

        lambda_bpts = max(10.0, min(base_lambda, 100.0))

        return {
            "lambda_booking_pts": round(lambda_bpts, 2),
            "referee_factor": round(referee_factor, 3),
            "referee_name": referee_profile.name if referee_profile else "Unknown",
            "prob_over_20_bpts": self.prob_over_line(lambda_bpts, 20),
            "prob_over_30_bpts": self.prob_over_line(lambda_bpts, 30),
            "prob_over_40_bpts": self.prob_over_line(lambda_bpts, 40),
            "prob_over_50_bpts": self.prob_over_line(lambda_bpts, 50),
            "prob_under_40_bpts": round(1 - self.prob_over_line(lambda_bpts, 40), 4),
        }

    def calculate_goals_markets(
        self,
        lambda_home: float,
        lambda_away: float,
        # IMPORTANT: keep this aligned with ComboEngine/Matrix ceiling.
        # Using a smaller ceiling (e.g. 10) makes 1X2 probabilities slightly
        # inconsistent with DC probabilities computed via ComboEngine (which
        # uses _MAX_GOALS=13). This inconsistency can inflate/deflate EV and
        # create apparent "instability" in pre-game markets.
        max_goals: int = 13,
    ) -> dict:
        p_home_win = p_draw = p_away_win = 0.0
        p_under_or_equal = {1: 0.0, 2: 0.0, 3: 0.0}
        p_home_scores = 1 - poisson.pmf(0, lambda_home)
        p_away_scores = 1 - poisson.pmf(0, lambda_away)

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                total = i + j
                if total <= 1: p_under_or_equal[1] += p
                if total <= 2: p_under_or_equal[2] += p
                if total <= 3: p_under_or_equal[3] += p
                if i > j: p_home_win += p
                elif i == j: p_draw += p
                else: p_away_win += p

        return {
            "prob_over_15": round(1 - p_under_or_equal[1], 4),
            "prob_under_15": round(p_under_or_equal[1], 4),
            "prob_over_25": round(1 - p_under_or_equal[2], 4),
            "prob_under_25": round(p_under_or_equal[2], 4),
            "prob_over_35": round(1 - p_under_or_equal[3], 4),
            "prob_under_35": round(p_under_or_equal[3], 4),
            "prob_btts": round(p_home_scores * p_away_scores, 4),
            "prob_no_btts": round(1 - (p_home_scores * p_away_scores), 4),
            "prob_home_win": round(p_home_win, 4),
            "prob_draw": round(p_draw, 4),
            "prob_away_win": round(p_away_win, 4),
        }

    def calculate_ev(self, model_prob: float, odd: float) -> float:
        return round((model_prob * odd) - 1.0, 4)
