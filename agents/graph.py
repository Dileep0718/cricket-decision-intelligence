import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langgraph.graph import StateGraph, END
from agents.state import GraphState
from agents.intent_router import intent_router
from agents.captain_brain import game_state_agent, strategy_agent, decision_agent
from agents.critic_agent import critic_agent
from agents.butterfly_engine import butterfly_engine
from agents.prediction_loop import prediction_loop
from backend.schemas import QueryType


def should_retry(state: GraphState) -> str:
    """
    Conditional edge after critic_agent.
    If rejected and retries remain → loop back to strategy_agent.
    If approved → proceed to END.
    """
    if not state.get("critic_approved", False) and state.get("retry_count", 0) < 2:
        return "retry"
    return "done"


def route_by_intent(state: GraphState) -> str:
    """
    Conditional edge after intent_router.
    Routes to the correct agent chain based on query type.
    """
    query_type = state.get("query_type", QueryType.GENERAL)
    if query_type == QueryType.CAPTAIN_BRAIN:
        return "captain"
    elif query_type == QueryType.BUTTERFLY:
        return "butterfly"
    elif query_type == QueryType.PREDICTION:
        return "prediction"
    return "general"


def route_after_game_state(state: GraphState) -> str:
    """
    Conditional edge after game_state_agent.
    Routes to the correct agent based on query type.
    """
    query_type = state.get("query_type", QueryType.GENERAL)
    if query_type == QueryType.BUTTERFLY:
        return "butterfly"
    elif query_type == QueryType.PREDICTION:
        return "prediction"
    return "captain"


def general_response_agent(state: GraphState) -> dict:
    """
    Handles general cricket queries that don't need
    the full captain brain pipeline.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    from backend.config import GROQ_API_KEY, LLM_MODEL, GROQ_TEMPERATURE
    from backend.schemas import AgentTrace

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=GROQ_TEMPERATURE
    )

    match_state = state.get("match_state")
    query = state.get("user_query", "")

    context = ""
    if match_state:
        context = f"""
Current match: {match_state.team_batting} vs {match_state.team_bowling}
Score: {match_state.score}/{match_state.wickets} in {match_state.overs} overs
"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert cricket analyst. Answer questions about the match clearly and concisely."),
            HumanMessage(content=f"{context}\nQuestion: {query}")
        ])
        answer = response.content.strip()
    except Exception as e:
        answer = f"Unable to process query: {e}"

    trace = AgentTrace(
        agent_name="General Agent",
        input_summary=f"Query: {query[:80]}",
        output_summary="General cricket analysis provided",
        confidence=0.8,
        reasoning=answer[:200]
    )

    return {
        "final_recommendation": answer,
        "final_confidence": 0.8,
        "uncertainty_reason": "General query — no tactical decision required",
        "agent_traces": [trace]
    }


def build_graph() -> StateGraph:
    """
    Builds and compiles the full LangGraph state machine.

    Flow:
        intent_router
            ├── captain   → game_state_agent → strategy_agent → decision_agent → critic_agent
            │                                        ↑______________|  (retry loop)
            ├── butterfly → game_state_agent → butterfly_engine
            ├── prediction → game_state_agent → prediction_loop
            └── general   → general_response_agent
    """
    graph = StateGraph(GraphState)

    # ── Register all nodes ────────────────────────────────────────────
    graph.add_node("intent_router", intent_router)
    graph.add_node("game_state_agent", game_state_agent)
    graph.add_node("strategy_agent", strategy_agent)
    graph.add_node("decision_agent", decision_agent)
    graph.add_node("critic_agent", critic_agent)
    graph.add_node("butterfly_engine", butterfly_engine)
    graph.add_node("prediction_loop", prediction_loop)
    graph.add_node("general_response_agent", general_response_agent)

    # ── Entry point ───────────────────────────────────────────────────
    graph.set_entry_point("intent_router")

    # ── Route after intent classification ────────────────────────────
    graph.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "captain": "game_state_agent",
            "butterfly": "game_state_agent",
            "prediction": "game_state_agent",
            "general": "general_response_agent"
        }
    )

    # ── Route after game_state_agent ──────────────────────────────────
    graph.add_conditional_edges(
        "game_state_agent",
        route_after_game_state,
        {
            "butterfly": "butterfly_engine",
            "prediction": "prediction_loop",
            "captain": "strategy_agent"
        }
    )

    # ── Captain Brain chain ───────────────────────────────────────────
    graph.add_edge("strategy_agent", "decision_agent")
    graph.add_edge("decision_agent", "critic_agent")

    # ── Critic retry loop ─────────────────────────────────────────────
    graph.add_conditional_edges(
        "critic_agent",
        should_retry,
        {
            "retry": "strategy_agent",
            "done": END
        }
    )

    # ── All other chains go straight to END ───────────────────────────
    graph.add_edge("butterfly_engine", END)
    graph.add_edge("prediction_loop", END)
    graph.add_edge("general_response_agent", END)

    return graph.compile()


# Single compiled graph instance
cricket_graph = build_graph()
