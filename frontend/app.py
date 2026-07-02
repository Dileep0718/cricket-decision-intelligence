import streamlit as st
import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from frontend.components.match_view import render_match_sidebar, render_match_header
from frontend.components.metrics_view import render_metrics
from frontend.components.agent_trace import render_agent_trace, render_recommendation

API_BASE = os.getenv("API_BASE_URL","https://cricket-ai-backend-g4zr.onrender.com")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Decision Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar — always dark with white text */
    div[data-testid="stSidebarContent"] {
        background-color: #1a1a2e !important;
    }
    div[data-testid="stSidebarContent"] * {
        color: #ffffff !important;
    }
    /* Sidebar buttons — dark background, white text, always visible */
    div[data-testid="stSidebarContent"] .stButton > button {
        background-color: #2d2d4e !important;
        color: #ffffff !important;
        border: 1px solid #4a4a6a !important;
        border-radius: 8px !important;
        text-align: left !important;
    }
    div[data-testid="stSidebarContent"] .stButton > button:hover {
        background-color: #3d3d6e !important;
        border-color: #6a6a9a !important;
    }
    /* Metrics cards — explicit light background so they show in dark theme */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        color: #1a1a2e !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
    }
    div[data-testid="stMetric"] label {
        color: #444444 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1a1a2e !important;
    }
    /* General button styling */
    .stButton > button {
        border-radius: 8px;
    }
    /* Expander border */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def fetch_match_detail(match_id: str) -> dict:
    """Fetch match state + metrics from backend."""
    try:
        resp = requests.get(f"{API_BASE}/match/{match_id}", timeout=10)
        return resp.json()
    except Exception as e:
        st.error(f"Could not load match detail: {e}")
        return {}


def run_analysis(
    match_id: str,
    query: str,
    explanation_mode: str,
    match_state: dict
) -> dict:
    """Call the /analyze endpoint and return result."""
    try:
        payload = {
            "match_id": match_id,
            "query": query,
            "explanation_mode": explanation_mode,
            "match_state": match_state
        }
        resp = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            timeout=60
        )
        if resp.status_code != 200:
            st.error(f"Backend error {resp.status_code}: {resp.text[:200]}")
            return {}
        return resp.json()
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return {}


def main():
    # ── Sidebar: match selector ───────────────────────────────────────
    selected_match = render_match_sidebar()

    if not selected_match:
        st.title("🏏 Cricket Decision Intelligence System")
        st.info("Select a match from the sidebar to begin analysis.")
        return

    match_id = selected_match.get("match_id")

    # ── Fetch match detail + metrics ──────────────────────────────────
    match_detail = fetch_match_detail(match_id)
    if not match_detail:
        st.error("Could not load match details.")
        return

    match_state = match_detail.get("match_state", {})
    metrics = match_detail.get("metrics", {})

    # ── Match header ──────────────────────────────────────────────────
    render_match_header(selected_match, match_detail)

    # ── Metrics row ───────────────────────────────────────────────────
    render_metrics(metrics)

    # ── Explanation mode ──────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col2:
        explanation_mode = st.selectbox(
            "Explanation Mode",
            ["analyst", "simple", "coach"],
            help="analyst: deep stats | simple: beginner friendly | coach: tactical"
        )

    # ── Quick action buttons ──────────────────────────────────────────
    st.markdown("### ⚡ Quick Analysis")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)

    quick_query = None
    with qcol1:
        if st.button("⚔️ Captain's Next Move", use_container_width=True):
            quick_query = "What should the captain do in the next over?"
    with qcol2:
        if st.button("🦋 What If?", use_container_width=True):
            quick_query = "What if the last wicket had not fallen?"
    with qcol3:
        if st.button("🔮 Predict Outcome", use_container_width=True):
            quick_query = "Who will win this match and what will be the final score?"
    with qcol4:
        if st.button("📊 Pressure Breakdown", use_container_width=True):
            quick_query = "Explain the current pressure situation in detail"

    # ── Custom query input ────────────────────────────────────────────
    st.markdown("### 💬 Ask Anything")
    user_query = st.text_input(
        label="Ask anything about this match",
        placeholder="e.g. What should the captain do next over? / What if that catch was taken?",
        label_visibility="collapsed"
    )

    analyze_btn = st.button("🧠 Analyse", type="primary", use_container_width=False)

    # ── Trigger analysis ──────────────────────────────────────────────
    final_query = quick_query or (user_query if analyze_btn else None)

    if final_query:
        with st.spinner(f"Running agent pipeline for: *{final_query}*"):
            result = run_analysis(
                match_id=match_id,
                query=final_query,
                explanation_mode=explanation_mode,
                match_state=match_state
            )

        if result:

            recommendation = result.get("recommendation", "")
            confidence = result.get("confidence", 0.0)

            if not recommendation or recommendation == "No recommendation generated":
                st.error("Agent returned empty recommendation. Check debug panel above.")
            else:
                st.markdown("---")
                render_recommendation(result)
                render_agent_trace(result.get("agent_traces", []))

                # Prediction history
                if result.get("query_type") == "prediction":
                    with st.expander("📈 Prediction History & System Accuracy"):
                        try:
                            pred_resp = requests.get(
                                f"{API_BASE}/predictions/{match_id}",
                                timeout=10
                            )
                            pred_data = pred_resp.json()
                            accuracy = pred_data.get("system_accuracy", {})
                            preds = pred_data.get("predictions", [])

                            if accuracy.get("total_evaluated", 0) > 0:
                                st.metric(
                                    "System Accuracy",
                                    f"{accuracy.get('accuracy')}%",
                                    f"Over {accuracy.get('total_evaluated')} evaluated predictions"
                                )
                            else:
                                st.info("No evaluated predictions yet — system accuracy builds over time")

                            if preds:
                                st.markdown("**Recent predictions:**")
                                for p in preds:
                                    status = "⏳ Pending"
                                    if p.get("was_correct") is True:
                                        status = "✅ Correct"
                                    elif p.get("was_correct") is False:
                                        status = "❌ Wrong"
                                    st.markdown(
                                        f"- `{p['timestamp'][:16]}` → "
                                        f"**{p['predicted_winner']}** "
                                        f"at {p['predicted_win_probability']}% "
                                        f"confidence | {status}"
                                    )
                        except Exception as e:
                            st.warning(f"Could not load prediction history: {e}")
        else:
            st.error("No response from backend. Is the FastAPI server running?")


if __name__ == "__main__":
    main()


