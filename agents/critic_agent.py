import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import GROQ_API_KEY, LLM_MODEL
from backend.schemas import AgentTrace
from agents.state import GraphState


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.1
)

MAX_RETRIES = 2

CRITIC_SYSTEM_PROMPT = """You are a critical reviewer of cricket strategy recommendations.
Your job is to validate whether a recommendation is:
1. Specific enough to be actionable
2. Grounded in the match situation provided
3. Not contradicting itself
4. Realistic given the match context

Be strict but fair. Approve good recommendations. Reject vague or contradictory ones."""


def critic_agent(state: GraphState) -> dict:
    """
    Validates the final recommendation from decision_agent.

    - If approved: sets critic_approved=True, pipeline continues
    - If rejected and retries remain: sets critic_approved=False
      with feedback, pipeline loops back
    - If max retries hit: approves with lowered confidence
      to prevent infinite loops
    """
    recommendation = state.get("final_recommendation", "")
    reasoning = state.get("final_reasoning", "")
    confidence = state.get("final_confidence", 0.75)
    metrics = state.get("metrics")
    retry_count = state.get("retry_count", 0)
    game_summary = state.get("game_state_summary", "")

    # Safety valve — never loop more than MAX_RETRIES times
    if retry_count >= MAX_RETRIES:
        trace = AgentTrace(
            agent_name="Critic Agent",
            input_summary=f"Retry limit reached ({retry_count}/{MAX_RETRIES})",
            output_summary="Approved with reduced confidence (retry limit)",
            confidence=max(0.4, confidence - 0.2),
            reasoning="Max retries reached. Approving with reduced confidence to prevent infinite loop."
        )
        return {
            "critic_approved": True,
            "final_confidence": max(0.4, confidence - 0.2),
            "uncertainty_reason": f"Confidence reduced — recommendation required {retry_count} retries",
            "agent_traces": [trace]
        }

    prompt = f"""Review this cricket strategy recommendation:

Game situation:
{game_summary}

Recommendation: {recommendation}
Reasoning: {reasoning}
Confidence claimed: {round(confidence * 100)}%
Pressure Index: {metrics.pressure_index if metrics else 'N/A'}/100

Evaluate strictly:
1. Is this recommendation SPECIFIC and ACTIONABLE? (not vague like "play better")
2. Does it directly address the match situation?
3. Is the confidence claim reasonable?
4. Any logical contradictions?

Respond in this exact format:
VERDICT: APPROVE or REJECT
FEEDBACK: [specific feedback — what is good or what needs improvement]
ADJUSTED_CONFIDENCE: [0.0-1.0, your adjusted confidence score]"""

    try:
        response = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        raw = response.content.strip()

        # Parse response
        verdict = "APPROVE"
        feedback = ""
        adjusted_confidence = confidence

        for line in raw.split("\n"):
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip().upper()
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()
            elif line.startswith("ADJUSTED_CONFIDENCE:"):
                try:
                    adjusted_confidence = float(
                        line.replace("ADJUSTED_CONFIDENCE:", "").strip()
                    )
                except ValueError:
                    adjusted_confidence = confidence

        approved = verdict == "APPROVE"

    except Exception as e:
        print(f"Critic agent error: {e}")
        approved = True
        feedback = "Critic unavailable — auto approved"
        adjusted_confidence = confidence * 0.9

    trace = AgentTrace(
        agent_name="Critic Agent",
        input_summary=f"Reviewing: {recommendation[:80]}",
        output_summary=f"{'Approved' if approved else 'Rejected'} | Confidence: {round(adjusted_confidence * 100)}%",
        confidence=adjusted_confidence,
        reasoning=feedback
    )

    return {
        "critic_approved": approved,
        "critic_feedback": feedback,
        "final_confidence": adjusted_confidence,
        "retry_count": retry_count + (0 if approved else 1),
        "agent_traces": [trace]
    }
