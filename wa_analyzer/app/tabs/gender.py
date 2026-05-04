from __future__ import annotations


import plotly.express as px
import streamlit as st

from wa_analyzer.app.state import AppContext
from wa_analyzer.src.analyzer import WhatsappAnalyzer


@st.fragment
def render(ctx: AppContext) -> None:
    full_analyzer = ctx.full_analyzer
    df_base = ctx.df_base
    exclude_family_gender = ctx.exclude_family_gender
    exclude_family_global = ctx.exclude_family_global
    family_list = list(ctx.family_list)

    st.header("Demographics")
    # Use full_analyzer to ensure Reply Time calc has 'Me' messages
    # analyze_by_gender() handles 'from_me=0' internally for volume.
    gender_analyzer = full_analyzer

    if exclude_family_gender and not exclude_family_global and family_list:
        # Apply family filter to base for gender analysis
        # Note: df_base includes Me. gender_stats needs Me for reply time.
        gender_df_source = df_base[~df_base["chat_name"].isin(family_list)]
        gender_analyzer = WhatsappAnalyzer(gender_df_source)

    with st.spinner("Analyzing demographics..."):
        gender_counts = gender_analyzer.analyze_by_gender()
        gender_stats = gender_analyzer.calculate_gender_stats()

    c1, c2 = st.columns([1, 2])
    with c1:
        fig_pie = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Messages by Gender",
            color=gender_counts.index,
            color_discrete_map={
                "male": "#636EFA",
                "female": "#EF553B",
                "unknown": "gray",
            },
        )
        st.plotly_chart(fig_pie, width="stretch")
    with c2:
        st.subheader("Deep Dive Metrics")
        if not gender_stats.empty:
            st.dataframe(gender_stats, width="stretch")
            metrics = ["count", "avg_wpm", "media_pct", "avg_reply_time"]
            metric_choice = st.selectbox("Select Metric", metrics)
            fig_comp = px.bar(
                gender_stats,
                x="gender",
                y=metric_choice,
                color="gender",
                title=f"Comparison: {metric_choice}",
                color_discrete_map={
                    "male": "#636EFA",
                    "female": "#EF553B",
                    "unknown": "gray",
                },
            )
            st.plotly_chart(fig_comp, width="stretch")
