import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import GROQ_API_KEY, LLM_MODEL, GROQ_TEMPERATURE
from backend.schemas import AgentTrace, MatchState as MatchStateModel
from agents.state import GraphState
from tools.metrics import compute_all_metrics
from tools.cricket_api import cricket_client
from tools.retriever import retrieve_similar_scenarios


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=GROQ_TEMPERATURE
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

    return cricket_client.get_fallback_match_state(match_id)


# ── Sub-agent 1: Game State Agent ─────────────────────────────────────────
def game_state_agent(state: GraphState) -> dict:
    """
    Reads the current match state and produces a structured
    natural language summary for the strategy agent.
    """
    match_id = state.get("match_id", "fallback_001")
    raw_match_state = state.get("match_state")

    # Always convert to proper MatchState object
    match_state = _convert_match_state(raw_match_state, match_id)

    # Now safe to compute metrics
    metrics = compute_all_metrics(match_state)

    prompt = f"""Analyze this cricket match situation and provide a concise structured summary:

Match: {match_state.team_batting} vs {match_state.team_bowling}
Score: {match_state.score}/{match_state.wickets} in {match_state.overs} overs
Target: {match_state.target or 'First innings'}
Current RR: {match_state.current_rr}
Required RR: {match_state.required_rr or 'N/A'}
Match Type: {match_state.match_type}
Venue: {match_state.venue}

Custom Metrics:
- Pressure Index: {metrics.pressure_index}/100
- Momentum Score: {metrics.momentum_score}/100
- Win Probability: {metrics.win_probability}%
- Collapse Risk: {metrics.collapse_risk}

Provide a 3-4 sentence game state summary focusing on:
1. Current match situation
2. Key pressure points
3. What the batting/bowling team needs to do
Keep it factual and analytical."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert cricket analyst. Be concise and factual."),
            HumanMessage(content=prompt)
        ])
        summary = response.content.strip()
    except Exception as e:
        summary = (
            f"Match state: {match_state.team_batting} "
            f"{match_state.score}/{match_state.wickets} in "
            f"{match_state.overs} overs. "
            f"Required RR: {match_state.required_rr}."
        )
        print(f"Game state agent error: {e}")

    trace = AgentTrace(
        agent_name="Game State Agent",
        input_summary=(
            f"{match_state.team_batting} "
            f"{match_state.score}/{match_state.wickets} "
            f"in {match_state.overs} ov"
        ),
        output_summary=(
            f"Pressure: {metrics.pressure_index}/100 | "
            f"Win prob: {metrics.win_probability}%"
        ),
        confidence=0.95,
        reasoning=summary
    )

    return {
        "match_state": match_state,
        "metrics": metrics,
        "game_state_summary": summary,
        "agent_traces": [trace]
    }


# ── Sub-agent 2: Strategy Agent ───────────────────────────────────────────
def strategy_agent(state: GraphState) -> dict:
    """
    Takes the game state summary and generates 2-3 strategic options
    for the captain with reasoning for each.
    Now enriched with RAG — pulls similar historical situations.
    """
    game_summary = state.get("game_state_summary", "")
    match_state = state.get("match_state")
    metrics = state.get("metrics")
    user_query = state.get("user_query", "")

    # Convert match_state if it came back as dict
    match_id = state.get("match_id", "fallback_001")
    match_state = _convert_match_state(match_state, match_id)

    # Recompute metrics if missing
    if not metrics:
        metrics = compute_all_metrics(match_state)
    elif isinstance(metrics, dict):
        from backend.schemas import CustomMetrics
        try:
            metrics = CustomMetrics(**metrics)
        except Exception:
            metrics = compute_all_metrics(match_state)

    # ── RAG: fetch historical context ─────────────────────────────────
    historical_context = state.get("historical_context", "")
    if not historical_context and match_state:
        try:
            historical_context = retrieve_similar_scenarios(
                match_state=match_state,
                query=user_query
            )
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            historical_context = ""

    historical_section = ""
    if historical_context:
        historical_section = f"\n{historical_context}\n"

    prompt = f"""You are a cricket strategist advising the captain.

Game situation:
{game_summary}

Captain's question: {user_query}

Pressure Index: {metrics.pressure_index}/100
Collapse Risk: {metrics.collapse_risk}
{historical_section}

Generate exactly 3 strategic options for the captain.
For each option provide:
- ACTION: specific tactical move
- REASON: why this makes sense given the situation
- RISK: what could go wrong

Format each option clearly as OPTION 1, OPTION 2, OPTION 3."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert cricket tactician with 20 years of experience coaching international teams."),
            HumanMessage(content=prompt)
        ])
        strategy = response.content.strip()
    except Exception as e:
        strategy = "Strategy analysis unavailable due to API error."
        print(f"Strategy agent error: {e}")

    trace = AgentTrace(
        agent_name="Strategy Agent",
        input_summary=f"Query: {user_query[:80]}",
        output_summary="Generated 3 strategic options",
        confidence=0.85,
        reasoning=(
            f"Analyzed situation with Pressure Index "
            f"{metrics.pressure_index}/100 and "
            f"Collapse Risk {metrics.collapse_risk}. "
            f"RAG context: {'Yes' if historical_context else 'No'}"
        )
    )

    return {
        "strategy_analysis": strategy,
        "historical_context": historical_context,
        "agent_traces": [trace]
    }


# ── Sub-agent 3: Decision Agent ───────────────────────────────────────────
def decision_agent(state: GraphState) -> dict:
    """
    Scores the strategic options and picks the best one.
    Produces the final recommendation with confidence and
    uncertainty explanation.
    """
    strategy = state.get("strategy_analysis", "")
    game_summary = state.get("game_state_summary", "")
    metrics = state.get("metrics")
    explanation_mode = state.get("explanation_mode", "analyst")

    # Handle metrics as dict or object
    if isinstance(metrics, dict):
        from backend.schemas import CustomMetrics
        try:
            metrics = CustomMetrics(**metrics)
        except Exception:
            from backend.schemas import MatchState as MS
            match_id = state.get("match_id", "fallback_001")
            ms = _convert_match_state(state.get("match_state"), match_id)
            metrics = compute_all_metrics(ms)

    mode_instructions = {
        "simple": "Explain in simple terms a cricket fan would understand. Avoid jargon.",
        "analyst": "Use cricket analytics terminology. Include specific stats and metrics.",
        "coach": "Frame as coaching advice. Focus on execution, player roles, and tactics."
    }
    mode_instruction = mode_instructions.get(
        explanation_mode,
        mode_instructions["analyst"]
    )

    prompt = f"""You are the final decision maker. Review these strategic options and pick the best one.

Game summary:
{game_summary}

Strategic options analyzed:
{strategy}

Win Probability: {metrics.win_probability}%
Pressure Index: {metrics.pressure_index}/100

Task:
1. Pick the BEST option
2. Give a clear recommendation in 2-3 sentences
3. Assign a confidence score (0.0 to 1.0)
4. Explain any uncertainty honestly

Explanation style: {mode_instruction}

Respond in this exact format:
RECOMMENDATION: [your recommendation]
CONFIDENCE: [0.0-1.0]
UNCERTAINTY: [why you are not 100% confident]
REASONING: [detailed reasoning]"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a decisive cricket strategist. Be specific and actionable."),
            HumanMessage(content=prompt)
        ])
        raw = response.content.strip()

        recommendation = ""
        confidence = 0.75
        uncertainty = ""
        reasoning = ""

        for line in raw.split("\n"):
            if line.startswith("RECOMMENDATION:"):
                recommendation = line.replace("RECOMMENDATION:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.75
            elif line.startswith("UNCERTAINTY:"):
                uncertainty = line.replace("UNCERTAINTY:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        if not recommendation:
            recommendation = raw[:200]

    except Exception as e:
        recommendation = "Unable to generate recommendation due to API error."
        confidence = 0.0
        uncertainty = "API error occurred"
        reasoning = str(e)
        print(f"Decision agent error: {e}")

    trace = AgentTrace(
        agent_name="Decision Agent",
        input_summary="Evaluated 3 strategic options",
        output_summary=f"Confidence: {round(confidence * 100)}%",
        confidence=confidence,
        reasoning=reasoning or recommendation
    )

    return {
        "final_recommendation": recommendation,
        "final_confidence": confidence,
        "uncertainty_reason": uncertainty,
        "final_reasoning": reasoning,
        "agent_traces": [trace]
    }
