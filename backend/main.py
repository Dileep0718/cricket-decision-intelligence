import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import AnalysisRequest, ExplanationMode
from backend.schemas import MatchState
from backend.config import APP_TITLE
from agents.graph import cricket_graph
from tools.cricket_api import cricket_client
from tools.metrics import compute_all_metrics
from agents.prediction_loop import get_past_predictions, get_system_accuracy

app = FastAPI(
    title=APP_TITLE,
    description="Multi-agent cricket decision intelligence system",
    version="1.0.0"
)

# ── CORS — allows Streamlit frontend to talk to FastAPI ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────
def _safe_traces(result: dict) -> list:
    """Extract agent traces safely regardless of object type."""
    traces = []
    for t in result.get("agent_traces", []):
        if hasattr(t, "model_dump"):
            traces.append(t.model_dump())
        elif isinstance(t, dict):
            traces.append(t)
    return traces


def _safe_metrics(result: dict):
    """Extract metrics safely regardless of object type."""
    m = result.get("metrics")
    if not m:
        return None
    return m.model_dump() if hasattr(m, "model_dump") else m


def _safe_query_type(result: dict) -> str:
    """Extract query type as plain string."""
    qt = result.get("query_type")
    if hasattr(qt, "value"):
        return qt.value
    return qt or "general"


def _build_response(result: dict) -> dict:
    """
    Converts raw graph output into a clean JSON-serializable dict.
    Used by both /analyze and /quick endpoints.
    """
    return {
        "query_type": _safe_query_type(result),
        "recommendation": result.get("final_recommendation") or "No recommendation generated",
        "reasoning": result.get("final_reasoning") or "",
        "confidence": result.get("final_confidence") or 0.0,
        "uncertainty_reason": result.get("uncertainty_reason") or "",
        "agent_traces": _safe_traces(result),
        "metrics": _safe_metrics(result),
        "sources": []
    }


def _build_initial_state(
    query: str,
    match_id: str,
    explanation_mode: str,
    match_state=None
) -> dict:
    """Builds the initial LangGraph state dict."""

    # Convert dict to MatchState object if needed
    if isinstance(match_state, dict) and match_state:
        try:
            match_state = MatchState(**match_state)
        except Exception as e:
            print(f"Could not parse match_state dict: {e}")
            match_state = cricket_client.get_fallback_match_state(match_id)
    elif match_state is None:
        match_state = cricket_client.get_fallback_match_state(match_id)

    return {
        "user_query": query,
        "match_id": match_id,
        "explanation_mode": explanation_mode,
        "match_state": match_state,
        "metrics": None,
        "query_type": None,
        "game_state_summary": None,
        "strategy_analysis": None,
        "historical_context": None,
        "simulation_result": None,
        "prediction_result": None,
        "critic_approved": None,
        "critic_feedback": None,
        "retry_count": 0,
        "final_recommendation": None,
        "final_reasoning": None,
        "final_confidence": None,
        "uncertainty_reason": None,
        "agent_traces": []
    }

# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "app": APP_TITLE}


# ── Live matches ──────────────────────────────────────────────────────────
@app.get("/matches")
def get_matches():
    """
    Returns all live and recent matches.
    Called by Streamlit on page load.
    """
    try:
        matches = cricket_client.get_live_matches()
        return {
            "matches": [m.model_dump() for m in matches],
            "total": len(matches)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Match detail + metrics ────────────────────────────────────────────────
@app.get("/match/{match_id}")
def get_match_detail(match_id: str):
    """
    Returns detailed match state + computed custom metrics.
    Called when user selects a match from the sidebar.
    """
    try:
        match_state = cricket_client.get_match_detail(match_id)
        if not match_state:
            match_state = cricket_client.get_fallback_match_state(match_id)
        metrics = compute_all_metrics(match_state)
        return {
            "match_state": match_state.model_dump(),
            "metrics": metrics.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Main analysis endpoint ────────────────────────────────────────────────
@app.post("/analyze")
def analyze(request: AnalysisRequest):
    """
    Main endpoint — runs the full LangGraph agent pipeline.
    Accepts a user query + match context, returns recommendation
    with full agent trace.
    """
    try:
        initial_state = _build_initial_state(
            query=request.query,
            match_id=request.match_id,
            explanation_mode=request.explanation_mode,
            match_state=request.match_state
        )

        result = cricket_graph.invoke(initial_state)
        return _build_response(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Prediction history ────────────────────────────────────────────────────
@app.get("/predictions/{match_id}")
def get_predictions(match_id: str):
    """
    Returns past predictions for a match + system accuracy.
    Shows the self-correcting loop in action.
    """
    try:
        predictions = get_past_predictions(match_id)
        accuracy = get_system_accuracy()
        return {
            "predictions": predictions,
            "system_accuracy": accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Quick action shortcuts ────────────────────────────────────────────────
@app.get("/quick/{match_id}/{action}")
def quick_action(match_id: str, action: str):
    """
    Handles the quick action buttons in the UI.
    Pre-defined queries mapped to actions.
    """
    action_map = {
        "captain": "What should the captain do in the next over?",
        "butterfly": "What if the last wicket had not fallen?",
        "predict": "Who will win this match and what will be the final score?",
        "pressure": "Explain the current pressure situation in detail"
    }

    query = action_map.get(action, "Analyze the current match situation")

    try:
        match_state = cricket_client.get_match_detail(match_id)
        if not match_state:
            match_state = cricket_client.get_fallback_match_state(match_id)

        initial_state = _build_initial_state(
            query=query,
            match_id=match_id,
            explanation_mode=ExplanationMode.ANALYST,
            match_state=match_state
        )

        result = cricket_graph.invoke(initial_state)
        return _build_response(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
