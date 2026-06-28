import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import chromadb
from chromadb.utils import embedding_functions
from backend.config import VECTOR_STORE_PATH

# ── Historical match scenarios ────────────────────────────────────────────
# These are real-world inspired match situations with known outcomes.
# Each scenario has a situation description, what decision was made,
# and what the outcome was. This grounds agent reasoning in real precedent.

HISTORICAL_SCENARIOS = [
    {
        "id": "hist_001",
        "situation": "ODI chase: Team needs 180 from 20 overs, 6 wickets in hand, RRR 9.0. Pitch dry and turning. New batsman at crease.",
        "decision": "Captain brought in left-arm spinner immediately. Set attacking field with 2 slips.",
        "outcome": "Wicket in next over. Batting team collapsed from 220/4 to 240 all out. Bowling team won by 35 runs.",
        "match_type": "ODI",
        "pressure_index": 78,
        "result": "bowling_team_won",
        "tags": ["spin_attack", "dry_pitch", "death_overs", "new_batsman"]
    },
    {
        "id": "hist_002",
        "situation": "T20 chase: Team needs 54 from 4 overs, 4 wickets in hand, RRR 13.5. Dew on ground, pacers ineffective.",
        "decision": "Captain persisted with pace bowlers despite dew. Did not bring in spinner.",
        "outcome": "Batting team scored 58 in 4 overs. Won by 3 wickets with 2 balls to spare.",
        "match_type": "T20",
        "pressure_index": 85,
        "result": "batting_team_won",
        "tags": ["dew_factor", "death_overs", "pace_bowling", "t20_chase"]
    },
    {
        "id": "hist_003",
        "situation": "ODI first innings: Team at 240/5 in 42 overs. RR 5.71. Big hitter at crease, 8 overs left.",
        "decision": "Captain promoted pinch hitter at number 7 to maximise powerplay hitting.",
        "outcome": "Pinch hitter scored 34 off 18 balls. Final score 298. Won match by 42 runs.",
        "match_type": "ODI",
        "pressure_index": 45,
        "result": "batting_team_won",
        "tags": ["pinch_hitter", "death_overs", "first_innings", "acceleration"]
    },
    {
        "id": "hist_004",
        "situation": "Test match: Team needs 280 in final innings, 7 wickets in hand, 60 overs remaining. Pitch deteriorating.",
        "decision": "Captain sent nightwatchman to protect top order batsman.",
        "outcome": "Nightwatchman survived 8 overs. Main batsman came in fresh next day. Team drew match.",
        "match_type": "TEST",
        "pressure_index": 65,
        "result": "draw",
        "tags": ["nightwatchman", "test_match", "pitch_deterioration", "defensive"]
    },
    {
        "id": "hist_005",
        "situation": "T20: Batting team 120/2 in 15 overs. RR 8.0. Required RR 9.5. Set batsman on 68.",
        "decision": "Captain decided to keep set batsman and rotate strike rather than go for big shots early.",
        "outcome": "Partnership of 45 in 4 overs. Won with 2 balls to spare.",
        "match_type": "T20",
        "pressure_index": 55,
        "result": "batting_team_won",
        "tags": ["partnership", "set_batsman", "controlled_aggression", "t20"]
    },
    {
        "id": "hist_006",
        "situation": "ODI: Bowling team defending 245. Opposition 180/3 in 36 overs. RRR 8.25. Dangerous batsman on 72.",
        "decision": "Captain set defensive field — 7 on boundary. No attacking field.",
        "outcome": "Batsman found gaps easily. Scored 40 off next 4 overs. Target reached with 8 balls to spare.",
        "match_type": "ODI",
        "pressure_index": 60,
        "result": "batting_team_won",
        "tags": ["defensive_field", "dangerous_batsman", "wrong_tactics", "death_overs"]
    },
    {
        "id": "hist_007",
        "situation": "ODI: Team chasing 310. 220/5 in 38 overs. RRR 11.2. Two new batsmen. Spinner bowling well.",
        "decision": "Captain sent most experienced remaining batsman next despite him being number 8.",
        "outcome": "Batsman scored 35 off 22 balls. Gave team a chance but fell short by 18 runs.",
        "match_type": "ODI",
        "pressure_index": 88,
        "result": "bowling_team_won",
        "tags": ["promoted_batsman", "high_rrr", "experienced_player", "chase"]
    },
    {
        "id": "hist_008",
        "situation": "T20: Defending 165. Opposition 90/1 in 11 overs. RRR 7.5. Two set batsmen. Flat pitch.",
        "decision": "Captain made double bowling change — brought in two spinners together.",
        "outcome": "Two wickets in 2 overs. Opposition lost momentum. Defended target by 12 runs.",
        "match_type": "T20",
        "pressure_index": 50,
        "result": "bowling_team_won",
        "tags": ["double_bowling_change", "spin_attack", "flat_pitch", "pressure_moment"]
    },
    {
        "id": "hist_009",
        "situation": "ODI: 195/2 in 30 overs. Target 290. RRR 9.5. Set batsman on 89. Pitch getting slower.",
        "decision": "Captain changed batting order — promoted aggressive player ahead of technically correct player.",
        "outcome": "Aggressive player hit 3 sixes in 2 overs. Changed match momentum. Won by 4 wickets.",
        "match_type": "ODI",
        "pressure_index": 70,
        "result": "batting_team_won",
        "tags": ["batting_order_change", "aggressive_player", "slow_pitch", "momentum_shift"]
    },
    {
        "id": "hist_010",
        "situation": "Test: First innings 280 all out. Opposition 180/3 end of day 2. Pitch expected to deteriorate day 3.",
        "decision": "Captain used second new ball immediately at start of day 3 despite spinners bowling well.",
        "outcome": "New ball skidded on, took 3 wickets in first session. Opposition bowled out for 240.",
        "match_type": "TEST",
        "pressure_index": 55,
        "result": "bowling_team_won",
        "tags": ["new_ball", "day3_pitch", "test_match", "bowling_strategy"]
    },
    {
        "id": "hist_011",
        "situation": "T20 World Cup: Chase of 185. 140/4 in 16 overs. RRR 15. 2 specialist batsmen remaining.",
        "decision": "Last recognized batsman played conservatively trying to stay in rather than go for it.",
        "outcome": "Run rate became impossible. Lost by 22 runs despite wickets in hand.",
        "match_type": "T20",
        "pressure_index": 92,
        "result": "bowling_team_won",
        "tags": ["conservative_batting", "high_rrr", "wickets_in_hand", "wrong_approach"]
    },
    {
        "id": "hist_012",
        "situation": "ODI: Defending 220. Opposition 150/4 in 32 overs. New batsman, weak vs off-spin. RRR 8.8.",
        "decision": "Captain brought in off-spinner immediately to exploit weakness. Set close-in field.",
        "outcome": "New batsman stumped second ball. Opposition collapsed. Won by 35 runs.",
        "match_type": "ODI",
        "pressure_index": 65,
        "result": "bowling_team_won",
        "tags": ["exploit_weakness", "off_spin", "new_batsman", "close_field"]
    },
    {
        "id": "hist_013",
        "situation": "Test: Chasing 380 in 4th innings. 200/3 at tea on day 4. 55 overs remaining. Pitch crumbling.",
        "decision": "Captain went for aggressive chase rather than playing for draw.",
        "outcome": "Lost 7 wickets in final session trying to score quick runs. Lost by 85 runs.",
        "match_type": "TEST",
        "pressure_index": 75,
        "result": "bowling_team_won",
        "tags": ["aggressive_chase", "crumbling_pitch", "test_match", "tactical_error"]
    },
    {
        "id": "hist_014",
        "situation": "T20: 95/3 in 12 overs. Target 160. RRR 8.1. Set batsman on 45. Fresh pace bowler coming on.",
        "decision": "Captain rotated strike aggressively, used powerplay fielding restrictions smartly.",
        "outcome": "Scored 65 in last 8 overs. Won by 3 wickets off last ball.",
        "match_type": "T20",
        "pressure_index": 58,
        "result": "batting_team_won",
        "tags": ["strike_rotation", "powerplay_usage", "t20_tactics", "close_win"]
    },
    {
        "id": "hist_015",
        "situation": "ODI: 280/4 in 44 overs. On track for 320+. Star batsman retired hurt. RR 6.36.",
        "decision": "Captain promoted number 8 batsman known for big hitting in final overs.",
        "outcome": "Hit 28 off 14 balls. Final score 318. Won comfortably by 40 runs.",
        "match_type": "ODI",
        "pressure_index": 35,
        "result": "batting_team_won",
        "tags": ["promoted_hitter", "injury_setback", "final_overs", "recovery"]
    },
    {
        "id": "hist_016",
        "situation": "T20: Defending 145 on slow pitch. Opposition 80/2 in 10 overs. RRR 6.5. Spinners bowling well.",
        "decision": "Captain kept spinners on for 3 consecutive overs despite batter trying to slog.",
        "outcome": "Both set batters dismissed trying to attack spinners. Won by 18 runs.",
        "match_type": "T20",
        "pressure_index": 48,
        "result": "bowling_team_won",
        "tags": ["slow_pitch", "spinners", "consecutive_overs", "patience"]
    },
    {
        "id": "hist_017",
        "situation": "ODI: Chase 295. 180/6 in 35 overs. RRR 11.5. Tail-enders remaining. One big hitter at 8.",
        "decision": "Captain sent in big hitter at 7 to go for broke rather than defend.",
        "outcome": "Big hitter hit 42 off 24 but got out. Last 3 wickets fell cheaply. Lost by 48 runs.",
        "match_type": "ODI",
        "pressure_index": 90,
        "result": "bowling_team_won",
        "tags": ["tail_end_batting", "big_hitter", "high_pressure", "gamble"]
    },
    {
        "id": "hist_018",
        "situation": "Test: Lead of 180 in 2nd innings. 280/4 declared. Set opposition 261 to win on deteriorating pitch.",
        "decision": "Captain attacked with spinners from both ends immediately on day 5.",
        "outcome": "Opposition bowled out for 198. Won by 62 runs.",
        "match_type": "TEST",
        "pressure_index": 40,
        "result": "bowling_team_won",
        "tags": ["spin_attack", "day5_pitch", "declaration", "test_win"]
    },
    {
        "id": "hist_019",
        "situation": "T20: Last over, need 18 to win. 2 wickets in hand. Specialist hitter on strike.",
        "decision": "Specialist hitter went for big shots from ball 1 rather than rotating strike.",
        "outcome": "Hit 6, 4, 6 in first 3 balls. Won off 5th ball.",
        "match_type": "T20",
        "pressure_index": 95,
        "result": "batting_team_won",
        "tags": ["last_over", "big_hitting", "high_pressure", "aggressive_approach"]
    },
    {
        "id": "hist_020",
        "situation": "ODI: Defending 230. Opposition 170/2 in 35 overs. RRR 6.0. Two set batsmen. 15 overs left.",
        "decision": "Captain brought back opening bowler for 2nd spell on hunch despite expensive first spell.",
        "outcome": "Bowler got 2 wickets in 2 overs. Opposition collapsed. Won by 22 runs.",
        "match_type": "ODI",
        "pressure_index": 55,
        "result": "bowling_team_won",
        "tags": ["bowling_change", "second_spell", "set_batsmen", "instinct_decision"]
    }
]


def _get_client():
    """
    Returns the right ChromaDB client based on environment.
    Streamlit Cloud has a read-only filesystem so we use
    in-memory client there. Render and local use persistent.
    """
    import os
    # Streamlit Cloud sets this env var
    is_streamlit_cloud = os.getenv("STREAMLIT_SHARING_MODE") or \
                         os.getenv("IS_STREAMLIT_CLOUD") or \
                         not os.access(os.path.dirname(VECTOR_STORE_PATH), os.W_OK)
    if is_streamlit_cloud:
        print("Using in-memory ChromaDB (Streamlit Cloud)")
        return chromadb.EphemeralClient()
    else:
        print("Using persistent ChromaDB")
        return chromadb.PersistentClient(path=VECTOR_STORE_PATH)


def seed_vector_store():
    """Loads all historical scenarios into ChromaDB."""
    print("Initializing ChromaDB...")
    client = _get_client()
    ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="cricket_scenarios",
        embedding_function=ef,
        metadata={"description": "Historical cricket match scenarios and decisions"}
    )

    existing = collection.count()
    if existing >= len(HISTORICAL_SCENARIOS):
        print(f"Vector store already seeded with {existing} scenarios. Skipping.")
        return collection

    print(f"Seeding {len(HISTORICAL_SCENARIOS)} historical scenarios...")

    documents = []
    metadatas = []
    ids = []

    for scenario in HISTORICAL_SCENARIOS:
        doc = (
            f"Situation: {scenario['situation']} "
            f"Decision: {scenario['decision']} "
            f"Outcome: {scenario['outcome']}"
        )
        documents.append(doc)
        metadatas.append({
            "match_type": scenario["match_type"],
            "pressure_index": scenario["pressure_index"],
            "result": scenario["result"],
            "tags": ",".join(scenario["tags"]),
            "decision": scenario["decision"],
            "outcome": scenario["outcome"]
        })
        ids.append(scenario["id"])

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully seeded {len(HISTORICAL_SCENARIOS)} scenarios.")
    print(f"Collection now has {collection.count()} documents.")
    return collection


def get_collection():
    """Returns the ChromaDB collection. Used by retriever."""
    client = _get_client()
    ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="cricket_scenarios",
        embedding_function=ef
    )

    # Re-seed if empty (happens with ephemeral client on each startup)
    if collection.count() == 0:
        print("Collection empty — re-seeding...")
        seed_vector_store()
        collection = client.get_or_create_collection(
            name="cricket_scenarios",
            embedding_function=ef
        )

    return collection


if __name__ == "__main__":
    seed_vector_store()
    print("Done. Vector store ready.")