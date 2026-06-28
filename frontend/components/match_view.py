import streamlit as st
import requests

API_BASE = "http://localhost:8000"


def render_match_sidebar() -> dict | None:
    """
    Renders live/recent matches in the sidebar.
    Returns the selected match dict or None.
    """
    st.sidebar.title("🏏 Cricket AI")
    st.sidebar.markdown("---")

    with st.sidebar:
        if st.button("🔄 Refresh Matches", use_container_width=True):
            st.cache_data.clear()

    matches = _fetch_matches()
    if not matches:
        st.sidebar.error("Could not load matches.")
        return None

    st.sidebar.markdown(f"**{len(matches)} matches found**")
    st.sidebar.markdown("---")

    # Separate live and completed
    live = [m for m in matches if m.get("is_live")]
    completed = [m for m in matches if not m.get("is_live")]

    selected = None

    if live:
        st.sidebar.markdown("### 🔴 Live")
        for m in live[:5]:
            label = f"{m['name'][:35]}..." if len(m['name']) > 35 else m['name']
            if st.sidebar.button(label, key=f"match_{m['match_id']}", use_container_width=True):
                st.session_state.selected_match = m
            if st.session_state.get("selected_match", {}).get("match_id") == m["match_id"]:
                selected = m

    if completed:
        st.sidebar.markdown("### ✅ Recent")
        for m in completed[:5]:
            label = f"{m['name'][:35]}..." if len(m['name']) > 35 else m['name']
            if st.sidebar.button(label, key=f"match_{m['match_id']}", use_container_width=True):
                st.session_state.selected_match = m
            if st.session_state.get("selected_match", {}).get("match_id") == m["match_id"]:
                selected = m

    # Auto-select first live match on first load
    if not st.session_state.get("selected_match") and matches:
        first = live[0] if live else matches[0]
        st.session_state.selected_match = first

    return st.session_state.get("selected_match")


def render_match_header(match: dict, match_detail: dict):
    """Renders the match title and score header."""
    state = match_detail.get("match_state", {})

    st.markdown(f"## 🏟️ {match.get('name', 'Match')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**{state.get('team_batting', 'Batting')}**")
        st.markdown(
            f"### {state.get('score', 0)}/{state.get('wickets', 0)}"
        )
        st.caption(f"{state.get('overs', 0)} overs")
    with col2:
        st.markdown("**vs**")
        if state.get("target"):
            st.markdown(f"Target: **{state.get('target')}**")
        st.caption(state.get("match_type", "ODI"))
    with col3:
        st.markdown(f"**{state.get('team_bowling', 'Bowling')}**")
        if state.get("required_rr"):
            st.markdown(f"RRR: **{state.get('required_rr')}**")
        st.caption(state.get("venue", ""))

    st.markdown("---")


@st.cache_data(ttl=60)
def _fetch_matches():
    """Cached match fetch — refreshes every 60 seconds."""
    try:
        resp = requests.get(f"{API_BASE}/matches", timeout=10)
        return resp.json().get("matches", [])
    except Exception:
        return []
