from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class QueryType(str, Enum):
    CAPTAIN_BRAIN = "captain_brain"
    BUTTERFLY = "butterfly"
    PREDICTION = "prediction"
    GENERAL = "general"


class ExplanationMode(str, Enum):
    SIMPLE = "simple"
    ANALYST = "analyst"
    COACH = "coach"


class MatchSummary(BaseModel):
    match_id: str
    name: str
    status: str
    venue: str
    date: str
    teams: List[str]
    is_live: bool


class BattingStats(BaseModel):
    batsman_name: str
    runs: int
    balls: int
    strike_rate: float
    fours: int
    sixes: int


class BowlingStats(BaseModel):
    bowler_name: str
    overs: float
    maidens: int
    runs: int
    wickets: int
    economy: float


class MatchState(BaseModel):
    match_id: str
    team_batting: str
    team_bowling: str
    score: int = 0
    wickets: int = 0
    overs: float = 0.0
    target: Optional[int] = None
    current_rr: float = 0.0
    required_rr: Optional[float] = None
    balls_remaining: int = 0
    batting_stats: List[BattingStats] = []
    bowling_stats: List[BowlingStats] = []
    venue: str = ""
    match_type: str = "ODI"


class CustomMetrics(BaseModel):
    pressure_index: float = Field(..., ge=0, le=100)
    momentum_score: float = Field(..., ge=0, le=100)
    win_probability: float = Field(..., ge=0, le=100)
    collapse_risk: str  # "Low", "Medium", "High"
    pressure_reasons: List[str] = []


class AgentTrace(BaseModel):
    agent_name: str
    input_summary: str
    output_summary: str
    confidence: float
    reasoning: str


class AnalysisRequest(BaseModel):
    match_id: str
    query: str
    explanation_mode: ExplanationMode = ExplanationMode.ANALYST
    match_state: Optional[MatchState] = None


class AnalysisResponse(BaseModel):
    query_type: QueryType
    recommendation: str
    reasoning: str
    confidence: float
    uncertainty_reason: str
    agent_traces: List[AgentTrace]
    metrics: Optional[CustomMetrics] = None
    sources: List[str] = []


