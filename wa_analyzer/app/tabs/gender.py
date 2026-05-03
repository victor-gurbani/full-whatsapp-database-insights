from __future__ import annotations

import re
import threading

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt
from streamlit.runtime.scriptrunner import add_script_run_ctx
from wordcloud import WordCloud

from wa_analyzer.app.db_loaders import (
    _coerce_bar_chart_data,
    _format_backup_horizon,
    _format_inbox_gender,
    build_unanswered_chats,
    build_unread_chats_for_context,
    load_backup_message_horizon,
    load_group_receipt_events,
    load_jid_raw_lookup,
    load_lid_jid_map,
    load_vcf_contact_lookup,
)
from wa_analyzer.app.filters import is_number
from wa_analyzer.app.privacy import _anon_hash, _anon_hash_cut, _anon_random, av
from wa_analyzer.app.race_video import build_rolling_counts, render_contact_race_video
from wa_analyzer.app.state import AppContext
from wa_analyzer.app.ui_helpers import get_correlation_text
from wa_analyzer.src.analyzer import WhatsappAnalyzer
from wa_analyzer.src.chat_viewer import (
    export_chat_html_standalone,
    export_chat_json,
    export_chat_txt,
    generate_chat_html,
)


@st.fragment
def render(ctx: AppContext) -> None:
    analyzer = ctx.analyzer
    full_analyzer = ctx.full_analyzer
    df_raw = ctx.df_raw
    df_base = ctx.df_base
    df_group_base = ctx.df_group_base
    filtered_df = ctx.filtered_df
    msgstore_path = ctx.msgstore_path
    wa_path = ctx.wa_path
    vcf_path = ctx.vcf_path
    msgstore_file_signature = ctx.msgstore_file_signature
    me_display = ctx.me_display
    _anon_key = ctx._anon_key
    _anon_numbers = ctx._anon_numbers
    exclude_groups = ctx.exclude_groups
    exclude_me = ctx.exclude_me
    exclude_non_contacts = ctx.exclude_non_contacts
    exclude_family_gender = ctx.exclude_family_gender
    exclude_family_global = ctx.exclude_family_global
    exclude_family_behavior = ctx.exclude_family_behavior
    family_list = list(ctx.family_list)
    use_medians = ctx.use_medians
    use_longer_stats = ctx.use_longer_stats
    reply_threshold_hours = ctx.reply_threshold_hours
    min_word_len = ctx.min_word_len
    exclude_emails = ctx.exclude_emails

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
