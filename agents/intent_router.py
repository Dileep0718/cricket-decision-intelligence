import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import GROQ_API_KEY, LLM_MODEL, GROQ_TEMPERATURE
from backend.schemas import QueryType, AgentTrace
from agents.state import GraphState


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL,
    temperature=0.1  # low temp for classification — we want consistent routing
)


ROUTING_SYSTEM_PROMPT = """You are a cricket query classifier. 
Classify the user query into exactly one of these categories:

- captain_brain: questions about what the captain should do, 
  tactics, field placement, bowling changes, batting strategy
- butterfly: "what if" questions, hypothetical scenarios, 
  alternate match outcomes, dropped catches, missed decisions
- prediction: questions about final score prediction, 
  win probability, match outcome forecasting
- general: anything else about the match, stats, commentary, 
  player performance

Respond with ONLY the category name, nothing else.
Examples:
"What should the captain do next over?" -> captain_brain
"What if that catch had been taken?" -> butterfly
"Who will win this match?" -> prediction
"How is Kohli batting today?" -> general
"""


def intent_router(state: GraphState) -> dict:
    """
    Classifies the user query and sets query_type in state.
    Always the first node to run in the graph.
    """
    query = state["user_query"]

    try:
        response = llm.invoke([
            SystemMessage(content=ROUTING_SYSTEM_PROMPT),
            HumanMessage(content=query)
        ])

        raw = response.content.strip().lower()

        # Map to enum — default to general if unrecognised
        type_map = {
            "captain_brain": QueryType.CAPTAIN_BRAIN,
            "butterfly": QueryType.BUTTERFLY,
            "prediction": QueryType.PREDICTION,
            "general": QueryType.GENERAL
        }
        query_type = type_map.get(raw, QueryType.GENERAL)

    except Exception as e:
        print(f"Intent router error: {e}")
        query_type = QueryType.GENERAL

    trace = AgentTrace(
        agent_name="Intent Router",
        input_summary=f"Query: {query[:100]}",
        output_summary=f"Classified as: {query_type.value}",
        confidence=0.95,
        reasoning=f"Routed '{query[:60]}...' to {query_type.value} agent chain"
    )

    return {
        "query_type": query_type,
        "agent_traces": [trace]
    }
