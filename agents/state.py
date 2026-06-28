from typing import TypedDict, Optional, List, Annotated
import operator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.schemas import (
    MatchState,
    CustomMetrics,
    AgentTrace,
    QueryType,
    ExplanationMode
)


class GraphState(TypedDict):
    """
    The single state object that flows through every agent in LangGraph.

    Each agent receives this state, does its work, and returns
    a dict with only the fields it updated. LangGraph merges
    the updates automatically.

    The Annotated[List, operator.add] on agent_traces means
    each agent APPENDS to the list rather than replacing it —
    so we get the full reasoning chain at the end.
    """

    # ── Input ──────────────────────────────────────────────────────────
    user_query: str
    match_id: str
    explanation_mode: ExplanationMode

    # ── Match data (populated by game_state_agent) ─────────────────────
    match_state: Optional[MatchState]
    metrics: Optional[CustomMetrics]

    # ── Routing (populated by intent_router) ──────────────────────────
    query_type: Optional[QueryType]

    # ── Agent outputs ──────────────────────────────────────────────────
    game_state_summary: Optional[str]       # from game_state_agent
    strategy_analysis: Optional[str]        # from strategy_agent
    historical_context: Optional[str]       # from retriever / RAG
    simulation_result: Optional[str]        # from butterfly_engine
    prediction_result: Optional[str]        # from prediction_loop

    # ── Critic ─────────────────────────────────────────────────────────
    critic_approved: Optional[bool]
    critic_feedback: Optional[str]
    retry_count: int                         # prevents infinite retry loops

    # ── Final output ───────────────────────────────────────────────────
    final_recommendation: Optional[str]
    final_reasoning: Optional[str]
    final_confidence: Optional[float]
    uncertainty_reason: Optional[str]

    # ── Trace — appends from every agent, never replaced ───────────────
    agent_traces: Annotated[List[AgentTrace], operator.add]
