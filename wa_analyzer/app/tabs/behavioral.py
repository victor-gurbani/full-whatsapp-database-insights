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

    st.header("Behavioral Analysis")

    ghost_thresh = 432000 if use_longer_stats else 86400
    init_thresh = 172800 if use_longer_stats else 21600

    col_g, col_i = st.columns(2)

    with col_g:
        st.subheader("👻 Top Ghosters (Left you on read)")
        st.caption(
            f"Threshold: {ghost_thresh / 3600:.1f} hours silence after your last msg."
        )
        st.info(
            "ℹ️ 'End of Data' Logic: If a conversation extends to the very end of your message history (e.g. yesterday), it is NOT counted as ghosting."
        )

        bhv_exclude = family_list if exclude_family_behavior else None

        with st.spinner("Computing ghosting stats..."):
            ghosts = full_analyzer.get_ghosting_stats(
                ghost_thresh, exclude_list=bhv_exclude
            )
        if not ghosts.empty:
            fig_ghost = px.bar(
                ghosts,
                x="count",
                y="contact_name",
                orientation="h",
                color="gender",
                title="Unanswered Threads Count",
                color_discrete_map={
                    "male": "#636EFA",
                    "female": "#EF553B",
                    "unknown": "gray",
                },
            )
            fig_ghost.update_layout(
                yaxis={"categoryorder": "total ascending", "type": "category"}
            )
            st.plotly_chart(fig_ghost, width="stretch")
        else:
            st.write("No ghosting detected!")

        st.divider()
        st.subheader("😶 People I Ignore (Me → Them)")
        with st.spinner("Computing ignore stats..."):
            ignored = full_analyzer.get_left_on_read_stats(
                ghost_thresh, exclude_list=bhv_exclude
            )
        if not ignored.empty:
            # ignored is a pivot table with columns like 'True Ghost', 'Left on Delivered', 'Total Ignored'
            # We can plot 'Total Ignored' or stack the types. Stacked is better.
            # Reset index to get contact_name as column
            ignored_reset = ignored.reset_index()
            cols_to_plot = [
                c
                for c in ignored.columns
                if c in ["True Ghost 👻", "Left on Delivered 📨"]
            ]

            # Check if ignored is empty (it might have gender but no counts if all 0, but filtering logic in analyzer handles non-empty)

            # Check for gender column (added in latest update)
            color_arg = "gender" if "gender" in ignored_reset.columns else None
            color_map = (
                {"male": "#636EFA", "female": "#EF553B", "unknown": "gray"}
                if color_arg
                else None
            )

            fig_ignore = px.bar(
                ignored_reset,
                x=cols_to_plot,
                y="contact_name",
                orientation="h",
                title="Ignored Threads Count",
                color=color_arg,
                color_discrete_map=color_map,
            )
            fig_ignore.update_layout(
                yaxis={"categoryorder": "total ascending", "type": "category"},
                xaxis_title="Count",
            )
            st.plotly_chart(fig_ignore, width="stretch")
        else:
            st.write("You reply to everyone 😇")

    with col_i:
        st.subheader("👋 Conversation Initiators")
        st.caption(f"Threshold: {init_thresh / 3600:.1f} hours silence.")
        with st.spinner("Computing initiation stats..."):
            initiations = full_analyzer.get_initiation_stats(
                init_thresh, exclude_list=bhv_exclude
            )
        if not initiations.empty:
            # Overall summary stats
            total_me = initiations["Me"].sum()
            total_them = initiations["Them"].sum()
            total_all = total_me + total_them
            pct_me = (total_me / total_all * 100) if total_all > 0 else 0
            pct_them = (total_them / total_all * 100) if total_all > 0 else 0

            init_c1, init_c2 = st.columns(2)
            init_c1.metric(
                "Started by Me",
                f"{pct_me:.1f}%",
                help=f"{av(int(total_me), _anon_numbers):,} conversations",
            )
            init_c2.metric(
                "Started by Them",
                f"{pct_them:.1f}%",
                help=f"{av(int(total_them), _anon_numbers):,} conversations",
            )

            fig_init = px.bar(
                initiations[["Me", "Them"]],
                barmode="group",
                title="Initiations: Me vs Them",
            )
            st.plotly_chart(fig_init, width="stretch")
        else:
            st.write("Not enough data.")

        # First Message Stats (Who broke the ice)
        st.divider()
        st.subheader("🧊 First Message (Who Broke the Ice)")
        st.caption("Who sent the very first message in each chat.")

        first_me, first_them, total_chats = full_analyzer.get_first_message_stats(
            exclude_list=bhv_exclude
        )

        if total_chats > 0:
            pct_first_me = first_me / total_chats * 100
            pct_first_them = first_them / total_chats * 100

            first_c1, first_c2 = st.columns(2)
            first_c1.metric(
                "I Started",
                f"{pct_first_me:.1f}%",
                help=f"{av(first_me, _anon_numbers):,} of {av(total_chats, _anon_numbers):,} chats",
            )
            first_c2.metric(
                "They Started",
                f"{pct_first_them:.1f}%",
                help=f"{av(first_them, _anon_numbers):,} of {av(total_chats, _anon_numbers):,} chats",
            )
        else:
            st.write("No chat data available.")

    st.divider()
    st.subheader("⏱️ Reply Time Rankings (Avg Minutes)")

    rc1, rc2 = st.columns(2)
    min_msgs_input = rc1.number_input("Min Messages", min_value=5, value=25, step=5)
    top_30_only = rc2.checkbox("Rank Only Top 30 Contacts", value=True)

    st.caption(f"Delays > {reply_threshold_hours}h ignored.")

    # We need the FULL transaction history (including ME) to calculate reply times.
    # 'analyzer' uses filtered_df which might exclude 'Me'.

    reply_stats = full_analyzer.get_reply_time_ranking(
        min_messages=min_msgs_input,
        max_delay_seconds=reply_threshold_hours * 3600,
        exclude_list=bhv_exclude,
    )

    if top_30_only and not reply_stats.empty:
        # Get Top 30 names
        top_30 = analyzer.get_top_talkers(30)["contact_name"].tolist()
        reply_stats = reply_stats[reply_stats["contact_name"].isin(top_30)]

    if not reply_stats.empty:
        rt_col1, rt_col2 = st.columns(2)

        # Colors
        color_map = {"male": "#636EFA", "female": "#EF553B", "unknown": "gray"}

        with rt_col1:
            st.write("**Who replies to me the FASTEST?**")
            fastest_them = reply_stats.nsmallest(8, "their_avg")
            fig_ft = px.bar(
                fastest_them,
                x="their_avg",
                y="contact_name",
                orientation="h",
                color="gender",
                color_discrete_map=color_map,
                title="Lowest Avg Reply Time (Them)",
            )
            fig_ft.update_layout(
                yaxis={"categoryorder": "total descending", "type": "category"},
                xaxis_title="Minutes",
            )
            st.plotly_chart(fig_ft, width="stretch")

            st.write("**Who replies to me the SLOWEST?**")
            slowest_them = reply_stats.nlargest(8, "their_avg")
            fig_st = px.bar(
                slowest_them,
                x="their_avg",
                y="contact_name",
                orientation="h",
                color="gender",
                color_discrete_map=color_map,
                title="Highest Avg Reply Time (Them)",
            )
            fig_st.update_layout(
                yaxis={"categoryorder": "total ascending", "type": "category"},
                xaxis_title="Minutes",
            )
            st.plotly_chart(fig_st, width="stretch")

        with rt_col2:
            st.write("**Who do I reply to the FASTEST?**")
            fastest_me = reply_stats.nsmallest(8, "my_avg")
            fig_fm = px.bar(
                fastest_me,
                x="my_avg",
                y="contact_name",
                orientation="h",
                color="gender",
                color_discrete_map=color_map,
                title="My Lowest Avg Reply Time",
            )
            fig_fm.update_layout(
                yaxis={"categoryorder": "total descending", "type": "category"},
                xaxis_title="Minutes",
            )
            st.plotly_chart(fig_fm, width="stretch")

            st.write("**Who do I reply to the SLOWEST?**")
            slowest_me = reply_stats.nlargest(8, "my_avg")
            fig_sm = px.bar(
                slowest_me,
                x="my_avg",
                y="contact_name",
                orientation="h",
                color="gender",
                color_discrete_map=color_map,
                title="My Highest Avg Reply Time",
            )
            fig_sm.update_layout(
                yaxis={"categoryorder": "total ascending", "type": "category"},
                xaxis_title="Minutes",
            )
            st.plotly_chart(fig_sm, width="stretch")
    else:
        st.info(
            "Not enough conversation data to calculate reply times (need >25 messages)."
        )

    st.divider()
    st.subheader("✍️ Write Time Rankings (Avg Minutes)")
    st.caption("Read receipt -> Send reply. Shows actual typing/composition time.")
    st.write(
        "*(Note: Requires 'Read Receipts' to be enabled on both ends for accurate data)*"
    )

    write_stats = full_analyzer.get_write_time_ranking(
        min_messages=min_msgs_input,
        max_delay_seconds=10800,
        exclude_list=bhv_exclude,
    )

    if top_30_only and not write_stats.empty:
        # Use same top_30 logic
        top_30 = analyzer.get_top_talkers(30)["contact_name"].tolist()
        write_stats = write_stats[write_stats["contact_name"].isin(top_30)]

    if not write_stats.empty:
        wt_col1, wt_col2 = st.columns(2)

        # Colors
        color_map = {"male": "#636EFA", "female": "#EF553B", "unknown": "gray"}

        with wt_col1:
            st.write("**Who takes the SHORTEST to write a reply to me?**")
            fastest_wt_them = write_stats.nsmallest(8, "their_avg")
            fig_fwt = px.bar(
                fastest_wt_them,
                x="their_avg",
                y="contact_name",
                orientation="h",
                color="gender",
                color_discrete_map=color_map,
                title="Lowest Avg Write Time (Them)",
            )
            fig_fwt.update_layout(
                yaxis={"categoryorder": "total descending", "type": "category"},
                xaxis_title="Minutes",
            )
            st.plotly_chart(fig_fwt, width="stretch")

            st.write("**Who takes the LONGEST to write a reply to me?**")
            slowest_wt_them = write_stats.nlargest(8, "their_avg")
            fig_swt = px.bar(
                slowest_wt_them,
                x="their_avg",
                y="contact_name",
                orientation="h",
                color="gender",
                color_discrete_map=color_map,
                title="Highest Avg Write Time (Them)",
            )
            fig_swt.update_layout(
                yaxis={"categoryorder": "total ascending", "type": "category"},
                xaxis_title="Minutes",
            )
            st.plotly_chart(fig_swt, width="stretch")

        with wt_col2:
            st.write("**Who do I take the SHORTEST to write back to?**")
            fastest_wt_me = write_stats.nsmallest(8, "my_avg")
            fig_fwm = px.bar(
                fastest_wt_me,
                x="my_avg",
                y="contact_name",
                orientation="h",
                color="gender",
                color_discrete_map=color_map,
                title="My Lowest Avg Write Time",
            )
            fig_fwm.update_layout(
                yaxis={"categoryorder": "total descending", "type": "category"},
                xaxis_title="Minutes",
            )
            st.plotly_chart(fig_fwm, width="stretch")

            st.write("**Who do I take the LONGEST to write back to?**")
            slowest_wt_me = write_stats.nlargest(8, "my_avg")
            fig_swm = px.bar(
                slowest_wt_me,
                x="my_avg",
                y="contact_name",
                orientation="h",
                color="gender",
                color_discrete_map=color_map,
                title="My Highest Avg Write Time",
            )
            fig_swm.update_layout(
                yaxis={"categoryorder": "total ascending", "type": "category"},
                xaxis_title="Minutes",
            )
            st.plotly_chart(fig_swm, width="stretch")
    else:
        st.info("No write time history found (Receipts might be disabled).")
