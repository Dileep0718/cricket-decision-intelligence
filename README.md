# 🏏 Cricket Decision Intelligence System

> A production-grade multi-agent AI system for real-time cricket tactical analysis, scenario simulation, and self-correcting match prediction — built with LangGraph, RAG, ChromaDB, FastAPI, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-green)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What This Is

This is **not** a cricket score tracker or a chatbot.

It is a **decision intelligence engine** — a system that reasons about cricket matches the way an expert analyst would, using a multi-agent architecture where specialized AI agents collaborate, validate each other's reasoning, and produce explainable, confidence-scored decisions grounded in historical precedent.

**Live Demo:** [cricket-decision-intelligence.streamlit.app](https://cricket-decision-intelligence.streamlit.app)  
**API Docs:** [cricket-ai-backend.onrender.com/docs](https://cricket-ai-backend.onrender.com/docs)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│              Streamlit (Match Selector + Chat UI)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│                       FASTAPI BACKEND                           │
│         /matches  /match/{id}  /analyze  /predictions          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  LANGGRAPH ORCHESTRATION LAYER                  │
│                                                                 │
│   ┌─────────────┐                                               │
│   │Intent Router│ ──── classifies query type ────────────────┐  │
│   └─────────────┘                                            │  │
│         │                                                    │  │
│    ┌────▼──────────────────────────────────────────────┐    │  │
│    │              Game State Agent                     │    │  │
│    │  Reads match state → computes custom metrics      │    │  │
│    └────┬──────────────────┬──────────────────┬────────┘    │  │
│         │                  │                  │             │  │
│    ┌────▼────┐        ┌────▼──────┐    ┌──────▼──────┐     │  │
│    │Captain  │        │Butterfly  │    │ Prediction  │     │  │
│    │Brain    │        │Engine     │    │ Loop        │     │  │
│    │         │        │           │    │             │     │  │
│    │Strategy │        │Scenario   │    │Self-correct │     │  │
│    │Agent    │        │Simulation │    │+ SQLite log │     │  │
│    │   +RAG  │        │           │    │             │     │  │
│    │Decision │        └────┬──────┘    └──────┬──────┘     │  │
│    │Agent    │             │                  │            │  │
│    └────┬────┘             │                  │            │  │
│         │                  └────────┬─────────┘            │  │
│    ┌────▼──────────────────────────▼────────────────────┐  │  │
│    │                  Critic Agent                      │  │  │
│    │  Validates reasoning → approves or requests retry  │  │  │
│    └────────────────────────┬───────────────────────────┘  │  │
│                             │                               │  │
│    ┌────────────────────────▼───────────────────────────┐  │  │
│    │              General Response Agent                │◄─┘  │
│    └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
         │                          │
┌────────▼────────┐      ┌──────────▼──────────┐
│   ChromaDB      │      │   CricAPI (Live)     │
│  Vector Store   │      │   Match Data         │
│  20 historical  │      │   Scores + Stats     │
│  scenarios      │      └─────────────────────┘
└─────────────────┘
```

---

## 🔄 LangGraph Agent Flow

```
                    User Query
                        │
                        ▼
              ┌─────────────────┐
              │  Intent Router  │
              │  (LLM-based     │
              │  classifier)    │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      captain      butterfly    prediction
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
            ┌──────────────────┐
            │ Game State Agent │◄── CricAPI + Custom Metrics
            └────────┬─────────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
    Strategy    Butterfly   Prediction
    Agent       Engine      Loop
    (+ RAG)     (What-if    (SQLite
                Simulator)   Storage)
          │
          ▼
    Decision Agent
    (Picks best option)
          │
          ▼
    Critic Agent ──── REJECT ────► Strategy Agent (retry)
          │
        APPROVE
          │
          ▼
    Final Response
    (Recommendation + Confidence + Uncertainty + Full Trace)
```

---

## ✨ Core Features

### ⚔️ Captain Brain — Tactical Decision Engine
A 3-agent chain (Game State → Strategy → Decision) backed by a Critic validator. Generates captain recommendations with:
- 3 strategic options with action/reason/risk breakdown
- RAG-grounded reasoning from 20 historical match situations
- Confidence score + honest uncertainty explanation
- Full agent reasoning trace visible in UI

### 🦋 Butterfly Effect Engine — Scenario Simulator
Simulates alternate match timelines for what-if questions:
- "What if that catch had been taken?"
- Produces alternate scorecard, win probability shift, pressure delta
- Vivid narrative of how the match would have unfolded differently

### 🔮 Self-Correcting Prediction Loop
Generates and stores match outcome predictions in SQLite:
- Tracks its own accuracy over time
- Calibrates confidence based on historical performance
- Shows prediction history per match in UI

### 📊 Custom Metrics Engine (Own Formulas)

| Metric | Formula | Range |
|--------|---------|-------|
| **Pressure Index** | RRR pressure (0-50) + Wicket pressure (0-30) + Time pressure (0-20) | 0-100 |
| **Momentum Score** | RR health (0-50) + Wicket stability (0-30) + Scoring trend (0-20) | 0-100 |
| **Win Probability** | Resource-based model (balls × wickets factor vs runs needed) | 0-100% |
| **Collapse Risk** | Wickets + RRR + Innings stage weighted scoring | Low/Med/High |

### 🧠 RAG — Historical Precedent Retrieval
ChromaDB vector store with 20 curated historical match scenarios. Agents retrieve similar situations and ground recommendations in real precedents — "In 3 similar ODI chase situations with RRR > 9, teams that brought in a spinner won 67% of the time."

### 🔍 Visible Agent Reasoning Chain
Every decision is fully traceable in the UI — see exactly which agent ran, what it received, what it reasoned, and what confidence it assigned. Makes the agentic architecture tangible.

---

## 📊 Evaluation Results

```
Overall Accuracy   : 90%     ████████████████████░░  
Routing Accuracy   : 100%    ████████████████████████
Avg Confidence     : 83%     ████████████████████░░░
Avg Keyword Match  : 67%     ████████████████░░░░░░░
Avg Response Time  : 4.7s    
```

Evaluation suite: 10 scenarios covering ODI chase, T20 death overs, what-if simulation, Test match tactics, and general queries.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Agent Orchestration | LangGraph 0.2.28 | State machine, agent routing, retry loops |
| LLM | Groq LLaMA 3.3 70B | Fast inference, free tier |
| LLM Framework | LangChain 0.3.7 | Tool wrappers, prompt management |
| Vector DB | ChromaDB 0.5.18 | Historical scenario retrieval (RAG) |
| Prediction Storage | SQLite + SQLAlchemy | Self-correcting prediction loop |
| API Layer | FastAPI 0.115.4 | REST endpoints, CORS, validation |
| UI | Streamlit 1.40.0 | Live match selector, agent trace display |
| Cricket Data | CricAPI | Live scores, match details |
| Data Validation | Pydantic 2.9.2 | Type-safe state passing between agents |

---

## 📁 Project Structure

```
cricket-decision-intelligence/
│
├── agents/                     # Core agent logic
│   ├── graph.py                # LangGraph state machine (entry point)
│   ├── state.py                # GraphState — single object flowing through all agents
│   ├── intent_router.py        # Classifies query → routes to correct chain
│   ├── captain_brain.py        # 3-agent tactical decision chain
│   ├── butterfly_engine.py     # What-if scenario simulation
│   ├── prediction_loop.py      # Self-correcting match prediction
│   └── critic_agent.py         # Validates agent output, triggers retries
│
├── tools/                      # What agents call
│   ├── cricket_api.py          # CricAPI wrapper with fallback data
│   ├── metrics.py              # Custom metrics formulas (Pressure Index etc.)
│   └── retriever.py            # ChromaDB RAG queries
│
├── backend/                    # API layer
│   ├── main.py                 # FastAPI app, all endpoints
│   ├── schemas.py              # Pydantic models (MatchState, GraphState etc.)
│   └── config.py               # Environment config, validation
│
├── frontend/                   # UI layer
│   ├── app.py                  # Streamlit main app
│   └── components/
│       ├── match_view.py       # Live match selector sidebar
│       ├── metrics_view.py     # Custom metrics display
│       └── agent_trace.py      # Agent reasoning chain display
│
├── data/
│   ├── seed_matches.py         # Loads 20 historical scenarios into ChromaDB
│   └── vector_store/           # ChromaDB persistent storage
│
├── evals/
│   ├── scenarios.json          # 10 test scenarios with expected outputs
│   ├── run_evals.py            # Evaluation runner with scoring
│   └── eval_results.json       # Latest evaluation results
│
├── streamlit_app.py            # Streamlit Cloud entry point
├── Procfile                    # Render deployment config
├── requirements.txt
└── .env.example
```

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cricket-decision-intelligence.git
cd cricket-decision-intelligence

# 2. Create virtual environment (Python 3.11 recommended)
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 5. Seed the vector store (one time)
python data/seed_matches.py

# 6. Start the backend (Terminal 1)
uvicorn backend.main:app --reload --port 8000

# 7. Start the frontend (Terminal 2)
streamlit run frontend/app.py
```

Open `http://localhost:8501` — select a live match and start asking questions.

---

## 🔑 Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key        # Free at console.groq.com
CRIC_API_KEY=your_cricapi_key         # Free at cricapi.com (100 req/day)
LLM_MODEL=llama-3.3-70b-versatile     # Groq model
GROQ_TEMPERATURE=0.3                  # LLM temperature
APP_TITLE=Cricket Decision Intelligence System
DEBUG=False
```

---

## 🧪 Running Evaluations

```bash
python evals/run_evals.py
```

Runs 10 scenarios through the full agent pipeline and scores:
- Routing accuracy (did the right agent chain run?)
- Keyword match rate (did the recommendation address the scenario?)
- Confidence threshold (was confidence above minimum?)
- Agent trace completeness (did all agents run?)

---

## 🎯 Key Design Decisions

**Why LangGraph over plain LangChain?**  
LangGraph provides a proper state machine with typed state (`GraphState`), conditional edges, and retry loops. The Critic → Strategy retry loop and query-type-based routing are only possible because of LangGraph's graph structure.

**Why a Critic Agent?**  
Most agent projects skip validation. The Critic reads every recommendation and either approves it, requests a retry with specific feedback, or reduces confidence. This demonstrates real multi-agent coordination rather than sequential LLM calls.

**Why custom metrics instead of standard stats?**  
Standard cricket stats (run rate, strike rate) are well-known. Custom metrics like Pressure Index and Momentum Score are defensible, formula-based, and unique to this system — giving interviewers something concrete to probe.

**Why an eval suite?**  
Portfolio projects that claim "it works" without measurement are common. A scored eval suite with 90% accuracy on 10 diverse scenarios turns a demo into a measurable system.

---

## 🌐 Deployment

| Service | Platform | URL |
|---------|----------|-----|
| Frontend | Streamlit Cloud | `yourname-cricket-decision-intelligence.streamlit.app` |
| Backend API | Render (free tier) | `cricket-ai-backend.onrender.com` |

> **Note:** Render free tier spins down after 15 minutes of inactivity. Open the backend URL 1-2 minutes before a demo to wake it up.

---

## 📝 Resume Line

> Built a multi-agent cricket decision intelligence system using LangGraph that routes queries through a 5-agent reasoning chain, retrieves historical precedents via RAG (ChromaDB), generates self-correcting predictions stored in SQLite, and achieves 90% accuracy on a custom evaluation suite with 100% intent routing accuracy.

---

## 🔮 Roadmap

- [ ] Fix win probability formula calibration
- [ ] Add player-level historical data to RAG
- [ ] WebSocket support for real-time score updates
- [ ] LangSmith integration for production tracing
- [ ] Expand eval suite to 50 scenarios

---

## 📄 License

MIT License — free to use, modify, and distribute.
