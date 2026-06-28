
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import GROQ_API_KEY, LLM_MODEL, GROQ_TEMPERATURE
from backend.schemas import AgentTrace, MatchState as MatchStateModel
from agents.state import GraphState
from tools.metrics import compute_all_metrics


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.4
)


def _convert_match_state(raw, match_id: str) -> MatchStateModel:
    """
    Converts dict to MatchState object if needed.
    Always returns a valid MatchState — never None.
    """
    if isinstance(raw, MatchStateModel):
        return raw

    if isinstance(raw, dict) and raw:
        try:
            return MatchStateModel(**raw)
        except Exception as e:
            print(f"Could not parse match_state dict: {e}")

    # Final fallback
    from tools.cricket_api import cricket_client
    return cricket_client.get_fallback_match_state(match_id)


def _parse_alternate_state(
    original: MatchStateModel,
    scenario_description: str
) -> MatchStateModel:
    """
    Creates a modified MatchState based on the what-if scenario.
    Adjusts score, wickets based on the hypothetical event.
    """
    desc_lower = scenario_description.lower()

    score_delta = 0
    wicket_delta = 0

    if "catch" in desc_lower and (
        "drop" in desc_lower or "not taken" in desc_lower
    ):
        score_delta = 30
        wicket_delta = -1
    elif "run out" in desc_lower and "not" in desc_lower:
        score_delta = 20
        wicket_delta = -1
    elif "six" in desc_lower and "missed" in desc_lower:
        score_delta = -6
    elif "wicket" in desc_lower and "maiden" in desc_lower:
        score_delta = -8
        wicket_delta = 1
    else:
        score_delta = 15
        wicket_delta = 0

    new_score = max(0, original.score + score_delta)
    new_wickets = max(0, min(10, original.wickets + wicket_delta))

    # Recalculate run rates
    total_balls = int(original.overs) * 6 + round(
        (original.overs - int(original.overs)) * 10
    )
    current_rr = round(
        new_score / total_balls * 6, 2
    ) if total_balls > 0 else 0.0

    required_rr = None
    if original.target:
        if original.match_type == "ODI":
            total_overs = 50
        elif original.match_type == "T20":
            total_overs = 20
        else:
            total_overs = 90
        balls_remaining = max(0, (total_overs * 6) - total_balls)
        runs_needed = original.target - new_score
        if balls_remaining > 0:
            required_rr = round(runs_needed / balls_remaining * 6, 2)

    return MatchStateModel(
        match_id=original.match_id,
        team_batting=original.team_batting,
        team_bowling=original.team_bowling,
        score=new_score,
        wickets=new_wickets,
        overs=original.overs,
        target=original.target,
        current_rr=current_rr,
        required_rr=required_rr,
        balls_remaining=original.balls_remaining,
        venue=original.venue,
        match_type=original.match_type
    )


def butterfly_engine(state: GraphState) -> dict:
    """
    Simulates alternate match outcomes based on what-if scenarios.

    Flow:
        1. Parse the hypothetical event from user query
        2. Create alternate match state
        3. Compute metrics for both real and alternate states
        4. Generate narrative of how the match would have unfolded
        5. Show win probability shift
    """
    match_id = state.get("match_id", "fallback_001")
    raw_match_state = state.get("match_state")
    user_query = state.get("user_query", "")
    game_summary = state.get("game_state_summary", "")

    # Always convert to proper MatchState object
    match_state = _convert_match_state(raw_match_state, match_id)

    # ── Step 1: Extract the hypothetical event ────────────────────────
    extraction_prompt = f"""Extract the hypothetical event from this cricket what-if query.

Query: {user_query}

Respond with ONE sentence describing what alternate event to simulate.
Example: "The catch was taken at deep mid-wicket in over 28"
Just the event description, nothing else."""

    try:
        event_response = llm.invoke([
            SystemMessage(content="You extract hypothetical cricket events from what-if questions."),
            HumanMessage(content=extraction_prompt)
        ])
        scenario_description = event_response.content.strip()
    except Exception as e:
        scenario_description = user_query
        print(f"Event extraction error: {e}")

    # ── Step 2: Build alternate match state ───────────────────────────
    alternate_state = _parse_alternate_state(match_state, scenario_description)

    # ── Step 3: Compute metrics for both states ───────────────────────
    original_metrics = compute_all_metrics(match_state)
    alternate_metrics = compute_all_metrics(alternate_state)

    win_prob_shift = round(
        alternate_metrics.win_probability - original_metrics.win_probability, 1
    )
    pressure_shift = round(
        alternate_metrics.pressure_index - original_metrics.pressure_index, 1
    )

    # ── Step 4: Generate narrative ────────────────────────────────────
    narrative_prompt = f"""You are simulating an alternate cricket match timeline.

REAL timeline:
{game_summary}
Score: {match_state.score}/{match_state.wickets} in {match_state.overs} overs
Win probability: {original_metrics.win_probability}%
Pressure Index: {original_metrics.pressure_index}/100

ALTERNATE timeline (if: {scenario_description}):
Score would be: {alternate_state.score}/{alternate_state.wickets} in {alternate_state.overs} overs
Required RR would be: {alternate_state.required_rr}
Win probability would be: {alternate_metrics.win_probability}%
Pressure Index would be: {alternate_metrics.pressure_index}/100

Win probability shift: {'+' if win_prob_shift >= 0 else ''}{win_prob_shift}%
Pressure shift: {'+' if pressure_shift >= 0 else ''}{pressure_shift} points

Write a compelling 3-4 sentence narrative describing:
1. What would have happened differently
2. How the match momentum would have shifted
3. What the likely outcome would be in this alternate timeline

Be specific and dramatic but realistic."""

    try:
        narrative_response = llm.invoke([
            SystemMessage(content="You are a cricket commentator simulating alternate match timelines. Be vivid and analytical."),
            HumanMessage(content=narrative_prompt)
        ])
        narrative = narrative_response.content.strip()
    except Exception as e:
        narrative = (
            f"In this alternate timeline, {scenario_description} "
            f"would have shifted the win probability by {win_prob_shift}%."
        )
        print(f"Narrative generation error: {e}")

    # ── Step 5: Build simulation result ───────────────────────────────
    simulation_result = f"""SCENARIO: {scenario_description}

REAL vs ALTERNATE:
  Score        : {match_state.score}/{match_state.wickets} → {alternate_state.score}/{alternate_state.wickets}
  Required RR  : {match_state.required_rr} → {alternate_state.required_rr}
  Win Prob     : {original_metrics.win_probability}% → {alternate_metrics.win_probability}% ({'+' if win_prob_shift >= 0 else ''}{win_prob_shift}%)
  Pressure     : {original_metrics.pressure_index}/100 → {alternate_metrics.pressure_index}/100

NARRATIVE:
{narrative}"""

    trace = AgentTrace(
        agent_name="Butterfly Engine",
        input_summary=f"Scenario: {scenario_description[:80]}",
        output_summary=(
            f"Win prob shift: {'+' if win_prob_shift >= 0 else ''}{win_prob_shift}% | "
            f"Pressure shift: {'+' if pressure_shift >= 0 else ''}{pressure_shift}"
        ),
        confidence=0.75,
        reasoning=(
            f"Simulated alternate state: "
            f"{alternate_state.score}/{alternate_state.wickets} "
            f"vs real {match_state.score}/{match_state.wickets}"
        )
    )

    return {
        "simulation_result": simulation_result,
        "final_recommendation": simulation_result,
        "final_confidence": 0.75,
        "uncertainty_reason": "Simulation based on statistical averages — actual impact would vary by specific match context",
        "agent_traces": [trace]
    }
