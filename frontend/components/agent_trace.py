import streamlit as st


AGENT_COLORS = {
    "Intent Router": "🔵",
    "Game State Agent": "🟢",
    "Strategy Agent": "🟣",
    "Decision Agent": "🟠",
    "Critic Agent": "🔴",
    "Butterfly Engine": "🦋",
    "Prediction Loop": "🔮",
    "General Agent": "⚪"
}


def render_agent_trace(traces: list):
    """
    Renders the full agent reasoning chain.
    This is the key differentiator — makes the agentic
    architecture visible to interviewers.
    """
    if not traces:
        return

    st.markdown("### 🧠 Agent Reasoning Chain")
    st.caption("Every decision is traceable — see exactly which agent did what")

    for i, trace in enumerate(traces):
        agent_name = trace.get("agent_name", "Unknown")
        emoji = AGENT_COLORS.get(agent_name, "⚙️")
        confidence = trace.get("confidence", 0)
        conf_pct = round(confidence * 100)

        with st.expander(
            f"{emoji} {agent_name} — {trace.get('output_summary', '')} | Confidence: {conf_pct}%",
            expanded=(i == len(traces) - 1)  # expand last agent by default
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input**")
                st.info(trace.get("input_summary", ""))
            with col2:
                st.markdown("**Output**")
                st.success(trace.get("output_summary", ""))

            st.markdown("**Reasoning**")
            st.markdown(trace.get("reasoning", ""))

            # Confidence bar
            st.markdown(f"**Confidence: {conf_pct}%**")
            st.progress(confidence)

    st.markdown("---")


def render_recommendation(result: dict):
    """
    Renders the final recommendation box prominently.
    """
    if not result:
        return

    recommendation = result.get("recommendation", "")
    confidence = result.get("confidence", 0)
    uncertainty = result.get("uncertainty_reason", "")
    query_type = result.get("query_type", "general")

    type_labels = {
        "captain_brain": "⚔️ Captain Brain Decision",
        "butterfly": "🦋 Butterfly Effect Simulation",
        "prediction": "🔮 Match Prediction",
        "general": "💬 Analysis"
    }
    label = type_labels.get(query_type, "💬 Analysis")

    st.markdown(f"### {label}")

    st.success(recommendation)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Confidence: {round(confidence * 100)}%**")
        st.progress(confidence)
    with col2:
        if uncertainty:
            st.warning(f"⚠️ {uncertainty}")
