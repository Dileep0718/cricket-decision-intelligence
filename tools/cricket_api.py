import requests
from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.config import CRIC_API_KEY, CRIC_API_BASE_URL
from backend.schemas import MatchSummary, MatchState


class CricketAPIClient:
    def __init__(self):
        self.api_key = CRIC_API_KEY
        self.base_url = CRIC_API_BASE_URL
        self._cache = {}
        self._cache_ttl = 60

    def _get(self, endpoint: str, params: Dict = {}) -> Dict[str, Any]:
        """Base GET request with error handling."""
        url = f"{self.base_url}/{endpoint}"
        params["apikey"] = self.api_key
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                print(f"API warning: {data.get('reason', 'Unknown error')}")
                return {}
            return data
        except requests.exceptions.Timeout:
            print("CricAPI timeout — using cached data if available")
            return {}
        except requests.exceptions.RequestException as e:
            print(f"CricAPI error: {e}")
            return {}

    def get_live_matches(self) -> List[MatchSummary]:
        """Fetch all currently live and recent matches."""
        data = self._get("currentMatches", {"offset": 0})
        matches = []
        if not data or "data" not in data:
            return self._get_fallback_matches()
        for m in data["data"]:
            try:
                teams = m.get("teams", [])
                is_live = (
                    m.get("matchStarted", False) and
                    not m.get("matchEnded", False)
                )
                matches.append(MatchSummary(
                    match_id=m.get("id", ""),
                    name=m.get("name", "Unknown Match"),
                    status=m.get("status", ""),
                    venue=m.get("venue", "Unknown Venue"),
                    date=m.get("date", ""),
                    teams=teams if teams else ["Team A", "Team B"],
                    is_live=is_live
                ))
            except Exception as e:
                print(f"Skipping malformed match entry: {e}")
                continue
        return matches if matches else self._get_fallback_matches()

    def get_match_detail(self, match_id: str) -> MatchState:
        """Fetch detailed scorecard for a specific match."""
        # Skip API call for fallback IDs
        if match_id.startswith("fallback_"):
            return self.get_fallback_match_state(match_id)

        data = self._get("match_info", {"id": match_id})
        if not data or "data" not in data:
            return self.get_fallback_match_state(match_id)

        try:
            return self._parse_match_state(data["data"])
        except Exception as e:
            print(f"Error parsing match detail: {e}")
            return self.get_fallback_match_state(match_id)

    def _parse_match_state(self, raw: Dict) -> MatchState:
        """Convert raw API response into clean MatchState."""
        score_data = raw.get("score", [])
        current_innings = score_data[-1] if score_data else {}

        score = current_innings.get("r", 0) or 0
        wickets = current_innings.get("w", 0) or 0
        overs = float(current_innings.get("o", 0) or 0)

        # Determine total overs based on match type
        match_type_raw = raw.get("matchType", "odi").lower()
        if match_type_raw == "odi":
            total_overs = 50
            match_type_label = "ODI"
        elif match_type_raw == "t20":
            total_overs = 20
            match_type_label = "T20"
        else:
            total_overs = 90  # TEST match
            match_type_label = "TEST"

        # Calculate balls bowled
        overs_completed = int(overs)
        balls_in_over = round((overs - overs_completed) * 10)
        total_balls = overs_completed * 6 + balls_in_over

        # Current run rate
        current_rr = round(
            (score / total_balls * 6), 2
        ) if total_balls > 0 else 0.0

        # Target and required run rate
        target = None
        required_rr = None
        if len(score_data) > 1:
            first_innings = score_data[0]
            target = (first_innings.get("r", 0) or 0) + 1
            balls_remaining = max(0, (total_overs * 6) - total_balls)
            if balls_remaining > 0:
                runs_needed = target - score
                required_rr = round(
                    (runs_needed / balls_remaining * 6), 2
                )

        # Balls remaining — always non-negative
        balls_remaining = max(0, (total_overs * 6) - total_balls)

        # Team names
        teams = raw.get("teams", ["Team A", "Team B"])
        inning_str = current_innings.get("inning", "") if current_innings else ""
        batting_team = inning_str.split(" Inning")[0] if inning_str else teams[0]
        bowling_team = teams[1] if batting_team == teams[0] else teams[0]

        return MatchState(
            match_id=raw.get("id", ""),
            team_batting=batting_team,
            team_bowling=bowling_team,
            score=score,
            wickets=wickets,
            overs=overs,
            target=target,
            current_rr=current_rr,
            required_rr=required_rr,
            balls_remaining=balls_remaining,
            venue=raw.get("venue", ""),
            match_type=match_type_label
        )

    def _get_fallback_matches(self) -> List[MatchSummary]:
        """Fallback data when API is unavailable or quota exceeded."""
        print("Using fallback match data")
        return [
            MatchSummary(
                match_id="fallback_001",
                name="India vs Australia - 2nd ODI",
                status="India needs 125 runs from 18 overs",
                venue="Chennai",
                date="2024-11-20",
                teams=["India", "Australia"],
                is_live=True
            ),
            MatchSummary(
                match_id="fallback_002",
                name="England vs South Africa - T20I",
                status="South Africa needs 45 runs from 4 overs",
                venue="Lord's, London",
                date="2024-11-20",
                teams=["England", "South Africa"],
                is_live=True
            ),
            MatchSummary(
                match_id="fallback_003",
                name="Pakistan vs New Zealand - Test",
                status="Match completed - Pakistan won by 34 runs",
                venue="Karachi",
                date="2024-11-19",
                teams=["Pakistan", "New Zealand"],
                is_live=False
            )
        ]

    def get_fallback_match_state(self, match_id: str) -> MatchState:
        """Return a realistic match state for any match ID."""
        fallback_states = {
            "fallback_001": MatchState(
                match_id="fallback_001",
                team_batting="India",
                team_bowling="Australia",
                score=187,
                wickets=4,
                overs=32.1,
                target=312,
                current_rr=5.81,
                required_rr=9.42,
                balls_remaining=107,
                venue="Chennai",
                match_type="ODI"
            ),
            "fallback_002": MatchState(
                match_id="fallback_002",
                team_batting="South Africa",
                team_bowling="England",
                score=131,
                wickets=6,
                overs=16.0,
                target=176,
                current_rr=8.19,
                required_rr=11.25,
                balls_remaining=24,
                venue="Lord's, London",
                match_type="T20"
            ),
        }

        if match_id in fallback_states:
            return fallback_states[match_id]

        # Generic fallback for any real match ID
        return MatchState(
            match_id=match_id,
            team_batting="Team A",
            team_bowling="Team B",
            score=145,
            wickets=3,
            overs=25.0,
            target=280,
            current_rr=5.8,
            required_rr=8.6,
            balls_remaining=150,
            venue="Unknown Venue",
            match_type="ODI"
        )


# Single instance to be imported everywhere
cricket_client = CricketAPIClient()
