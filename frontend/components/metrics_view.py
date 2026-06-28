import streamlit as st


def render_metrics(metrics: dict):
    """
    Renders the four custom metric cards.
    Color coded by severity.
    """
    if not metrics:
        return

    st.markdown("### 📊 Live Intelligence Metrics")

    col1, col2, col3, col4 = st.columns(4)

    pressure = metrics.get("pressure_index", 0)
    momentum = metrics.get("momentum_score", 0)
    win_prob = metrics.get("win_probability", 0)
    collapse = metrics.get("collapse_risk", "Low")

    with col1:
        color = "🔴" if pressure > 70 else "🟡" if pressure > 40 else "🟢"
        st.metric(
            label=f"{color} Pressure Index",
            value=f"{pressure}/100",
            help="Custom metric: how much pressure the batting team is under. Formula: RRR pressure + wicket pressure + time pressure"
        )

    with col2:
        color = "🟢" if momentum > 60 else "🟡" if momentum > 35 else "🔴"
        st.metric(
            label=f"{color} Momentum Score",
            value=f"{momentum}/100",
            help="Custom metric: batting team's current momentum. Higher = batting team in control"
        )

    with col3:
        color = "🟢" if win_prob > 50 else "🟡" if win_prob > 25 else "🔴"
        st.metric(
            label=f"{color} Win Probability",
            value=f"{win_prob}%",
            help="Estimated win probability for batting team based on resources remaining"
        )

    with col4:
        color = "🔴" if collapse == "High" else "🟡" if collapse == "Medium" else "🟢"
        st.metric(
            label=f"{color} Collapse Risk",
            value=collapse,
            help="Risk of batting collapse based on wickets, required RR, and match stage"
        )

    # Pressure reasons
    reasons = metrics.get("pressure_reasons", [])
    if reasons:
        with st.expander("📋 Pressure breakdown"):
            for r in reasons:
                st.markdown(f"- {r}")

    st.markdown("---")
