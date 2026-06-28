import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, Session
from backend.config import GROQ_API_KEY, LLM_MODEL, GROQ_TEMPERATURE, SQLITE_DB_PATH
from backend.schemas import AgentTrace, MatchState as MatchStateModel
from agents.state import GraphState
from tools.metrics import compute_all_metrics


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=GROQ_TEMPERATURE
)

# ── Database setup ────────────────────────────────────────────────────────
Base = declarative_base()


class PredictionRecord(Base):
    """
    Stores every prediction the system makes along with
    the actual outcome once known. This is the self-correcting loop.
    """
    __tablename__ = "predictions"

    id = Column(String, primary_key=True)
    match_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    predicted_winner = Column(String)
    predicted_score = Column(Float)
    predicted_win_probability = Column(Float)
    confidence = Column(Float)
    reasoning = Column(Text)
    actual_winner = Column(String, nullable=True)
    actual_score = Column(Float, nullable=True)
    was_correct = Column(Boolean, nullable=True)
    accuracy_delta = Column(Float, nullable=True)
    match_context = Column(Text)


def get_engine():
    return create_engine(f"sqlite:///{SQLITE_DB_PATH}")


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def save_prediction(
    match_id: str,
    predicted_winner: str,
    predicted_score: float,
    predicted_win_probability: float,
    confidence: float,
    reasoning: str,
    match_context: dict
) -> str:
    """Save a new prediction to SQLite."""
    engine = init_db()
    pred_id = f"{match_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    with Session(engine) as session:
        record = PredictionRecord(
            id=pred_id,
            match_id=match_id,
            predicted_winner=predicted_winner,
            predicted_score=predicted_score,
            predicted_win_probability=predicted_win_probability,
            confidence=confidence,
            reasoning=reasoning,
            match_context=json.dumps(match_context)
        )
        session.add(record)
        session.commit()

    return pred_id


def get_past_predictions(match_id: str) -> list:
    """Retrieve past predictions for a match."""
    try:
        engine = init_db()
        with Session(engine) as session:
            records = session.query(PredictionRecord).filter(
                PredictionRecord.match_id == match_id
            ).order_by(
                PredictionRecord.timestamp.desc()
            ).limit(5).all()
            return [
                {
                    "id": r.id,
                    "timestamp": str(r.timestamp),
                    "predicted_winner": r.predicted_winner,
                    "predicted_win_probability": r.predicted_win_probability,
                    "confidence": r.confidence,
                    "was_correct": r.was_correct,
                    "reasoning": r.reasoning
                }
                for r in records
            ]
    except Exception as e:
        print(f"DB read error: {e}")
        return []


def get_system_accuracy() -> dict:
    """
    Calculate overall system prediction accuracy.
    This is the self-correcting loop metric.
    """
    try:
        engine = init_db()
        with Session(engine) as session:
            total = session.query(PredictionRecord).filter(
                PredictionRecord.was_correct.isnot(None)
            ).count()

            if total == 0:
                return {
                    "total_evaluated": 0,
                    "accuracy": None,
                    "message": "No evaluated predictions yet"
                }

            correct = session.query(PredictionRecord).filter(
                PredictionRecord.was_correct == True
            ).count()

            accuracy = round((correct / total) * 100, 1)
            return {
                "total_evaluated": total,
                "correct": correct,
                "accuracy": accuracy,
                "message": (
                    f"System accuracy: {accuracy}% "
                    f"over {total} evaluated predictions"
                )
            }
    except Exception as e:
        return {"error": str(e), "accuracy": None}


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


# ── Main prediction agent ─────────────────────────────────────────────────
def prediction_loop(state: GraphState) -> dict:
    """
    Generates a match outcome prediction, stores it in SQLite,
    and includes past prediction history for context.
    """
    match_id = state.get("match_id", "fallback_001")
    raw_match_state = state.get("match_state")
    game_summary = state.get("game_state_summary", "")
    user_query = state.get("user_query", "")

    # Always convert to proper MatchState object
    match_state = _convert_match_state(raw_match_state, match_id)
    metrics = compute_all_metrics(match_state)

    # ── Fetch past predictions for this match ─────────────────────────
    past_predictions = get_past_predictions(match_state.match_id)
    past_context = ""
    if past_predictions:
        past_context = "\nPast predictions for this match:\n"
        for p in past_predictions:
            status = "✓ Correct" if p["was_correct"] else (
                "✗ Wrong" if p["was_correct"] is False else "⏳ Pending"
            )
            past_context += (
                f"  - {p['timestamp'][:16]}: "
                f"{p['predicted_winner']} at "
                f"{p['predicted_win_probability']}% [{status}]\n"
            )

    # ── System accuracy context ───────────────────────────────────────
    accuracy_data = get_system_accuracy()
    accuracy_context = ""
    if accuracy_data.get("total_evaluated", 0) > 0:
        accuracy_context = (
            f"\nSystem historical accuracy: "
            f"{accuracy_data['accuracy']}% over "
            f"{accuracy_data['total_evaluated']} predictions."
        )

    # ── Generate prediction ───────────────────────────────────────────
    prediction_prompt = f"""You are a cricket match predictor. Make a specific, data-driven prediction.

Current match situation:
{game_summary}

Live metrics:
- Pressure Index: {metrics.pressure_index}/100
- Momentum Score: {metrics.momentum_score}/100
- Win Probability (formula-based): {metrics.win_probability}%
- Collapse Risk: {metrics.collapse_risk}

Match details:
- {match_state.team_batting}: {match_state.score}/{match_state.wickets} in {match_state.overs} overs
- Target: {match_state.target or 'First innings'}
- Required RR: {match_state.required_rr or 'N/A'}
- Balls remaining: {match_state.balls_remaining}
{past_context}
{accuracy_context}

User question: {user_query}

Make a prediction with:
1. Predicted winner
2. Predicted final score for batting team
3. Win probability percentage
4. Key factors driving this prediction
5. What would change your prediction

Respond in this exact format:
WINNER: [team name]
FINAL_SCORE: [predicted score as integer]
WIN_PROBABILITY: [0.0-1.0]
CONFIDENCE: [0.0-1.0]
KEY_FACTORS: [2-3 key factors]
PREDICTION_REASONING: [detailed reasoning]
WHAT_CHANGES_IT: [what would change this prediction]"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a precise cricket match predictor. Base predictions on data, not gut feel."),
            HumanMessage(content=prediction_prompt)
        ])
        raw = response.content.strip()

        predicted_winner = match_state.team_batting
        predicted_score = float(match_state.score)
        win_probability = metrics.win_probability / 100
        confidence = 0.7
        key_factors = ""
        reasoning = ""
        what_changes = ""

        for line in raw.split("\n"):
            if line.startswith("WINNER:"):
                predicted_winner = line.replace("WINNER:", "").strip()
            elif line.startswith("FINAL_SCORE:"):
                try:
                    predicted_score = float(
                        line.replace("FINAL_SCORE:", "").strip()
                    )
                except ValueError:
                    pass
            elif line.startswith("WIN_PROBABILITY:"):
                try:
                    win_probability = float(
                        line.replace("WIN_PROBABILITY:", "").strip()
                    )
                except ValueError:
                    pass
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(
                        line.replace("CONFIDENCE:", "").strip()
                    )
                except ValueError:
                    pass
            elif line.startswith("KEY_FACTORS:"):
                key_factors = line.replace("KEY_FACTORS:", "").strip()
            elif line.startswith("PREDICTION_REASONING:"):
                reasoning = line.replace("PREDICTION_REASONING:", "").strip()
            elif line.startswith("WHAT_CHANGES_IT:"):
                what_changes = line.replace("WHAT_CHANGES_IT:", "").strip()

    except Exception as e:
        predicted_winner = match_state.team_bowling
        predicted_score = float(match_state.target or 300)
        win_probability = 1 - (metrics.win_probability / 100)
        confidence = 0.5
        key_factors = "API error — using formula-based prediction"
        reasoning = str(e)
        what_changes = "N/A"
        print(f"Prediction agent error: {e}")

    # ── Store prediction in SQLite ────────────────────────────────────
    match_context = {
        "score": match_state.score,
        "wickets": match_state.wickets,
        "overs": match_state.overs,
        "target": match_state.target,
        "required_rr": match_state.required_rr,
        "pressure_index": metrics.pressure_index,
        "momentum_score": metrics.momentum_score
    }

    pred_id = save_prediction(
        match_id=match_state.match_id,
        predicted_winner=predicted_winner,
        predicted_score=predicted_score,
        predicted_win_probability=round(win_probability * 100, 1),
        confidence=confidence,
        reasoning=reasoning,
        match_context=match_context
    )

    # ── Build final output ────────────────────────────────────────────
    prediction_result = f"""PREDICTION STORED (ID: {pred_id})

Winner         : {predicted_winner}
Final score    : {int(predicted_score)}
Win probability: {round(win_probability * 100, 1)}%
Confidence     : {round(confidence * 100)}%

Key factors:
{key_factors}

Reasoning:
{reasoning}

What would change this:
{what_changes}

Past predictions this match: {len(past_predictions)}
System accuracy: {accuracy_data.get('accuracy', 'Not enough data yet')}%"""

    trace = AgentTrace(
        agent_name="Prediction Loop",
        input_summary=(
            f"Predicting: {match_state.team_batting} "
            f"vs {match_state.team_bowling}"
        ),
        output_summary=(
            f"Predicted: {predicted_winner} wins | "
            f"Confidence: {round(confidence * 100)}% | "
            f"ID: {pred_id}"
        ),
        confidence=confidence,
        reasoning=(
            f"Stored prediction {pred_id}. "
            f"{accuracy_data.get('message', '')}"
        )
    )

    return {
        "prediction_result": prediction_result,
        "final_recommendation": prediction_result,
        "final_confidence": confidence,
        "uncertainty_reason": what_changes,
        "agent_traces": [trace]
    }
