import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.schemas import MatchState, CustomMetrics
from typing import List


def _get_total_overs(state: MatchState) -> int:
    """Returns total overs for the match type."""
    if state.match_type == "ODI":
        return 50
    elif state.match_type == "T20":
        return 20
    else:
        return 90  # TEST


def calculate_pressure_index(state: MatchState) -> tuple[float, List[str]]:
    """
    Pressure Index (0-100) — how much pressure the batting team is under.

    Formula:
        PI = base_rrr_pressure + wicket_pressure + time_pressure

    Components:
        - base_rrr_pressure : how hard the required run rate is (0-50)
        - wicket_pressure   : how many wickets are lost (0-30)
        - time_pressure     : how late in the innings it is (0-20)
    """
    reasons = []
    total_overs = _get_total_overs(state)

    # ── Component 1: RRR pressure (0–50) ──────────────────────────────
    if state.required_rr and state.current_rr:
        rrr_ratio = state.required_rr / max(state.current_rr, 0.1)
        base_rrr_pressure = min(50, max(0, (rrr_ratio - 1) * 35))
        if rrr_ratio > 1.5:
            reasons.append(
                f"Required RR ({state.required_rr}) is "
                f"{round(rrr_ratio, 1)}x the current RR ({state.current_rr})"
            )
        elif rrr_ratio > 1.2:
            reasons.append(
                f"Required RR ({state.required_rr}) exceeding "
                f"current scoring rate ({state.current_rr})"
            )
    else:
        base_rrr_pressure = 15.0

    # ── Component 2: Wicket pressure (0–30) ───────────────────────────
    wicket_pressure = min(30, state.wickets * 3.5)
    if state.wickets >= 7:
        reasons.append(
            f"Only {10 - state.wickets} wickets remaining — "
            f"batting team in danger"
        )
    elif state.wickets >= 5:
        reasons.append(f"{state.wickets} wickets down — lower order exposed")
    elif state.wickets >= 3:
        reasons.append(
            f"{state.wickets} wickets lost — middle order under pressure"
        )

    # ── Component 3: Time pressure (0–20) ─────────────────────────────
    overs_completed_ratio = state.overs / total_overs
    time_pressure = min(20, overs_completed_ratio * 22)
    if (overs_completed_ratio > 0.8 and
            state.required_rr and state.required_rr > 8):
        reasons.append(
            f"Death overs approaching — "
            f"{round((1 - overs_completed_ratio) * total_overs, 1)} overs left"
        )

    pressure_index = round(
        base_rrr_pressure + wicket_pressure + time_pressure, 1
    )
    pressure_index = min(100, max(0, pressure_index))

    if not reasons:
        reasons.append("Match situation is relatively balanced")

    return pressure_index, reasons


def calculate_momentum_score(state: MatchState) -> float:
    """
    Momentum Score (0-100) — measures batting team's current momentum.
    100 = fully in control, 0 = momentum entirely with bowling team.
    """
    # ── Component 1: Run rate health (0–50) ───────────────────────────
    if state.required_rr and state.required_rr > 0:
        rr_ratio = state.current_rr / max(state.required_rr, 0.1)
        rr_component = min(50, max(0, rr_ratio * 35))
    else:
        par_rate = 5.5 if state.match_type == "ODI" else (
            8.0 if state.match_type == "T20" else 3.0
        )
        rr_ratio = state.current_rr / max(par_rate, 0.1)
        rr_component = min(50, max(0, rr_ratio * 30))

    # ── Component 2: Wicket stability (0–30) ──────────────────────────
    wickets_in_hand = 10 - state.wickets
    wicket_stability = min(30, wickets_in_hand * 3)

    # ── Component 3: Scoring trend proxy (0–20) ───────────────────────
    if state.match_type == "T20":
        scoring_trend = min(20, max(0, (state.current_rr - 6) * 3))
    elif state.match_type == "TEST":
        scoring_trend = min(20, max(0, (state.current_rr - 2) * 5))
    else:
        scoring_trend = min(20, max(0, (state.current_rr - 4) * 4))

    momentum_score = round(
        rr_component + wicket_stability + scoring_trend, 1
    )
    return min(100, max(0, momentum_score))


def calculate_win_probability(state: MatchState) -> float:
    """
    Win probability for batting team (0-100%).
    Handles ODI, T20, and TEST matches.
    """
    # ── TEST match ────────────────────────────────────────────────────
    if state.match_type == "TEST":
        if not state.target:
            return 50.0
        runs_needed = state.target - state.score
        if runs_needed <= 0:
            return 100.0
        wickets_in_hand = 10 - state.wickets
        balls_remaining = max(1, state.balls_remaining)
        achievable = (balls_remaining / 6) * 3.5
        if wickets_in_hand <= 2:
            achievable *= 0.4
        elif wickets_in_hand <= 4:
            achievable *= 0.7
        ratio = achievable / max(runs_needed, 1)
        win_prob = min(92, max(8, (ratio - 0.3) * 80 + 20))
        return round(win_prob, 1)

    # ── First innings — no target ──────────────────────────────────────
    if not state.target or state.target == 0:
        total_overs = _get_total_overs(state)
        if state.current_rr > 0:
            projected = state.current_rr * total_overs
        else:
            projected = 250 if state.match_type == "ODI" else 160
        par = 285 if state.match_type == "ODI" else 175
        diff = projected - par
        win_prob = 50 + (diff * 0.12)
        return round(min(75, max(25, win_prob)), 1)

    # ── Second innings chase ───────────────────────────────────────────
    runs_needed = state.target - state.score
    if runs_needed <= 0:
        return 100.0

    balls_remaining = max(1, state.balls_remaining)
    wickets_in_hand = 10 - state.wickets

    if wickets_in_hand <= 0:
        return 2.0

    # Realistic max RPO — ODI teams rarely sustain above 8.5
    # T20 teams can push to 11-12 in death
    max_rpo = 8.0 if state.match_type == "ODI" else 11.5

    # Wicket factor — losing wickets severely limits scoring
    # Using power 0.8 makes it more punishing than before
    wicket_factor = (wickets_in_hand / 10) ** 0.8

    achievable_runs = (balls_remaining / 6) * max_rpo * wicket_factor

    # How achievable is the target?
    ratio = achievable_runs / max(runs_needed, 1)

    # Sigmoid curve — calibrated so:
    # ratio 0.6 → ~10%, ratio 0.8 → ~25%, ratio 1.0 → ~50%
    # ratio 1.2 → ~68%, ratio 1.5 → ~80%, ratio 2.0 → ~90%
    if ratio >= 2.0:
        win_prob = 90.0
    elif ratio >= 1.5:
        win_prob = 80.0 + (ratio - 1.5) * 20
    elif ratio >= 1.2:
        win_prob = 68.0 + (ratio - 1.2) * 40
    elif ratio >= 1.0:
        win_prob = 50.0 + (ratio - 1.0) * 90
    elif ratio >= 0.8:
        win_prob = 25.0 + (ratio - 0.8) * 125
    elif ratio >= 0.6:
        win_prob = 8.0 + (ratio - 0.6) * 85
    elif ratio >= 0.3:
        win_prob = 3.0 + (ratio - 0.3) * 16.7
    else:
        win_prob = 2.0

    return round(min(95, max(2, win_prob)), 1)



def calculate_collapse_risk(state: MatchState) -> str:
    """
    Collapse risk: Low / Medium / High.
    Based on wickets lost, balls remaining, and required run rate.
    """
    risk_score = 0
    total_overs = _get_total_overs(state)

    # Wickets factor
    if state.wickets >= 7:
        risk_score += 3
    elif state.wickets >= 5:
        risk_score += 2
    elif state.wickets >= 3:
        risk_score += 1

    # Required run rate pressure
    if state.required_rr:
        if state.required_rr > 12:
            risk_score += 3
        elif state.required_rr > 9:
            risk_score += 2
        elif state.required_rr > 7:
            risk_score += 1

    # Late innings
    if state.overs > total_overs * 0.75:
        risk_score += 1

    if risk_score >= 5:
        return "High"
    elif risk_score >= 3:
        return "Medium"
    return "Low"


def compute_all_metrics(state: MatchState) -> CustomMetrics:
    """
    Master function — computes all custom metrics for a match state.
    This is what every agent and the UI calls.
    Always returns a valid CustomMetrics object, never None.
    """
    if not state:
        return CustomMetrics(
            pressure_index=0.0,
            momentum_score=0.0,
            win_probability=0.0,
            collapse_risk="Low",
            pressure_reasons=["No match data available"]
        )

    try:
        pressure_index, pressure_reasons = calculate_pressure_index(state)
        momentum_score = calculate_momentum_score(state)
        win_probability = calculate_win_probability(state)
        collapse_risk = calculate_collapse_risk(state)

        return CustomMetrics(
            pressure_index=pressure_index,
            momentum_score=momentum_score,
            win_probability=win_probability,
            collapse_risk=collapse_risk,
            pressure_reasons=pressure_reasons
        )
    except Exception as e:
        print(f"Metrics calculation error: {e}")
        return CustomMetrics(
            pressure_index=0.0,
            momentum_score=0.0,
            win_probability=0.0,
            collapse_risk="Low",
            pressure_reasons=[f"Metrics calculation error: {str(e)}"]
        )
