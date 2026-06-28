import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.seed_matches import get_collection
from backend.schemas import MatchState


def retrieve_similar_scenarios(
    match_state: MatchState,
    query: str,
    n_results: int = 3
) -> str:
    """
    Retrieves historically similar match scenarios from ChromaDB.
    Returns a formatted string ready to inject into agent prompts.

    This is what makes your agents say:
    "In 3 similar historical situations, the team that brought
    in a spinner won 67% of the time."
    """
    try:
        collection = get_collection()

        if collection.count() == 0:
            return ""

        # Build a rich search query combining match state + user query
        search_query = (
            f"{query} "
            f"Match type: {match_state.match_type}. "
            f"Score: {match_state.score}/{match_state.wickets} "
            f"in {match_state.overs} overs. "
            f"Required RR: {match_state.required_rr or 'N/A'}. "
            f"Pressure situation with {10 - match_state.wickets} wickets in hand."
        )

        # Query ChromaDB
        results = collection.query(
            query_texts=[search_query],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results["documents"][0]:
            return ""

        # Format results for agent consumption
        formatted = "RELEVANT HISTORICAL PRECEDENTS:\n"
        formatted += "-" * 40 + "\n"

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        wins_for_recommended = 0
        total_similar = len(docs)

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            similarity = round((1 - dist) * 100, 1)
            result_label = {
                "batting_team_won": "Batting team WON",
                "bowling_team_won": "Bowling team WON",
                "draw": "Match DRAWN"
            }.get(meta.get("result", ""), "Unknown")

            formatted += (
                f"\nPrecedent {i+1} "
                f"(Similarity: {similarity}% | {meta.get('match_type', '')})\n"
                f"Situation : {doc.split('Decision:')[0].replace('Situation:', '').strip()}\n"
                f"Decision  : {meta.get('decision', '')}\n"
                f"Outcome   : {meta.get('outcome', '')}\n"
                f"Result    : {result_label}\n"
            )

            if meta.get("result") == "bowling_team_won":
                wins_for_recommended += 1

        # Add win rate summary
        if total_similar > 0:
            win_rate = round((wins_for_recommended / total_similar) * 100)
            formatted += (
                f"\n{'─' * 40}\n"
                f"Summary: In {total_similar} similar situations, "
                f"bowling team won {win_rate}% of the time.\n"
            )

        return formatted

    except Exception as e:
        print(f"Retriever error: {e}")
        return ""


def retrieve_for_query(query: str, n_results: int = 3) -> str:
    """
    Simple text-only retrieval without match state context.
    Used for general queries.
    """
    try:
        collection = get_collection()
        if collection.count() == 0:
            return ""

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas"]
        )

        if not results or not results["documents"][0]:
            return ""

        formatted = "RELEVANT HISTORICAL PRECEDENTS:\n"
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            formatted += (
                f"\nPrecedent {i+1} ({meta.get('match_type', '')}):\n"
                f"{doc[:200]}...\n"
                f"Result: {meta.get('result', 'Unknown')}\n"
            )

        return formatted

    except Exception as e:
        print(f"Retriever error: {e}")
        return ""
