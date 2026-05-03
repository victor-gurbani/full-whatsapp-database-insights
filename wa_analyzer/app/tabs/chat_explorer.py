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

    st.header("Chat Explorer & Deep Dive")
    contacts = sorted(df_base["chat_name"].dropna().unique().astype(str))
    selected_contact = st.selectbox(
        "Select Contact", contacts, key="chat_explorer_contact"
    )

    if selected_contact:
        sub_df = df_base[df_base["chat_name"] == selected_contact].copy()
        st.write(f"### Analysis: **{selected_contact}**")
        total_msgs = len(sub_df)
        me_rows = sub_df[sub_df["from_me"] == 1]
        them_rows = sub_df[sub_df["from_me"] == 0]

        me_count = len(me_rows)
        them_count = total_msgs - me_count
        me_pct = (me_count / total_msgs * 100) if total_msgs > 0 else 0
        them_pct = (them_count / total_msgs * 100) if total_msgs > 0 else 0

        # Word Counts
        me_words = (
            me_rows["text_data"]
            .fillna("")
            .astype(str)
            .apply(lambda x: len(x.split()))
            .sum()
        )
        them_words = (
            them_rows["text_data"]
            .fillna("")
            .astype(str)
            .apply(lambda x: len(x.split()))
            .sum()
        )
        total_words = me_words + them_words
        me_word_pct = (me_words / total_words * 100) if total_words > 0 else 0
        them_word_pct = (them_words / total_words * 100) if total_words > 0 else 0

        st.markdown(f"""
        **Messages**: {av(total_msgs, _anon_numbers)} total
        - **Me**: {av(me_count, _anon_numbers)} ({me_pct:.1f}%)
        - **Them**: {av(them_count, _anon_numbers)} ({them_pct:.1f}%)

        **Words**: {av(total_words, _anon_numbers)} total
        - **Me**: {av(me_words, _anon_numbers):,} ({me_word_pct:.1f}%)
        - **Them**: {av(them_words, _anon_numbers):,} ({them_word_pct:.1f}%)
        """)

        # --- Media Pie Chart ---
        st.write("### Message Composition")

        # Categorize
        def categorize_msg(row):
            mime = str(row.get("mime_type", ""))
            if pd.isna(mime) or mime == "" or mime == "None":
                return "Text"
            if "image/webp" in mime:
                return "Sticker"
            if "image" in mime:
                return "Image"
            if "video" in mime:
                return "Video"
            if "audio" in mime:
                return "Audio"
            return "Other"

        sub_df["type_category"] = sub_df.apply(categorize_msg, axis=1)

        # Pie Chart Controls
        hide_text = st.checkbox(
            "Hide 'Text' messages (Focus on Media)", value=False
        )

        # Helper to generate pie data for a subset
        def get_pie_data(df_source, hide_text_flag):
            if df_source.empty:
                return pd.DataFrame()
            p_data = df_source["type_category"].value_counts().reset_index()
            p_data.columns = ["Type", "Count"]
            if hide_text_flag:
                p_data = p_data[p_data["Type"] != "Text"]
            return p_data

        pie_me = get_pie_data(sub_df[sub_df["from_me"] == 1], hide_text)
        pie_them = get_pie_data(sub_df[sub_df["from_me"] == 0], hide_text)

        p1, p2 = st.columns(2)

        color_map = {
            "Text": "lightgray",
            "Image": "#636EFA",
            "Video": "#EF553B",
            "Audio": "#00CC96",
            "Sticker": "#AB63FA",
        }

        with p1:
            if not pie_me.empty:
                fig_pie_me = px.pie(
                    pie_me,
                    values="Count",
                    names="Type",
                    title="Me",
                    color="Type",
                    color_discrete_map=color_map,
                )
                st.plotly_chart(fig_pie_me, width="stretch")
            else:
                st.info("No data (Me)")

        with p2:
            if not pie_them.empty:
                fig_pie_them = px.pie(
                    pie_them,
                    values="Count",
                    names="Type",
                    title="Them",
                    color="Type",
                    color_discrete_map=color_map,
                )
                st.plotly_chart(fig_pie_them, width="stretch")
            else:
                st.info("No data (Them)")

        chat_analyzer = WhatsappAnalyzer(sub_df)
        my_reply, their_reply = chat_analyzer.calculate_chat_reply_times()

        col_s1, col_s2 = st.columns(2)
        col_s1.metric("My Avg Reply Time", f"{av(my_reply, _anon_numbers):.1f} min")
        col_s2.metric(
            "Their Avg Reply Time", f"{av(their_reply, _anon_numbers):.1f} min"
        )

        st.caption(
            "ℹ️ **Calculation Method**: Time elapsed between a received message and your first subsequent reply (and vice versa). Does not account for 'Read' time, only delivery/sent timestamps."
        )

        # Avg Write Time (Read -> Reply)
        my_write, their_write = chat_analyzer.calculate_chat_write_times()

        col_w1, col_w2 = st.columns(2)

        w_help = "Time between READING the message (Blue Tick) and SENDING the reply. Replies over 240 minutes are ignored."

        # Diagnostics for N/A
        if "read_at" in sub_df.columns:
            has_receipts = sub_df["read_at"].notnull().sum() > 0
            has_incoming_receipts = (
                sub_df[sub_df["from_me"] == 0]["read_at"].notnull().sum() > 0
            )
            has_outgoing_receipts = (
                sub_df[sub_df["from_me"] == 1]["read_at"].notnull().sum() > 0
            )
        else:
            has_receipts = False
            has_incoming_receipts = False
            has_outgoing_receipts = False
        if my_write is not None:
            col_w1.metric(
                "My Avg Write Time",
                f"{av(my_write, _anon_numbers):.1f} min",
                help=w_help,
            )
        else:
            reason = (
                "Database missing 'Read' timestamps for incoming messages."
                if not has_incoming_receipts
                else "Timestamps exist but no direct reply sequence found."
            )
            col_w1.metric(
                "My Avg Write Time", "N/A", help=f"Cannot calculate: {reason}"
            )

        if their_write is not None:
            col_w2.metric(
                "Their Avg Write Time",
                f"{av(their_write, _anon_numbers):.1f} min",
                help=w_help,
            )
        else:
            reason = (
                "Contact has Read Receipts DISABLED."
                if not has_outgoing_receipts
                else "Read Receipts available but no direct reply sequence found (or system messages interrupted flow)."
            )
            col_w2.metric(
                "Their Avg Write Time", "N/A", help=f"Cannot calculate: {reason}"
            )

        # Reply Time Over Time
        st.write("### Reply Time Over Time")
        rt_over_time = chat_analyzer.calculate_reply_time_over_time(
            max_minutes=1440, freq="ME"
        )
        if not rt_over_time.empty:
            fig_rt = px.line(
                rt_over_time,
                x=rt_over_time.index,
                y=rt_over_time.columns,
                markers=True,
                labels={"value": "Minutes", "index": "Month", "variable": "Sender"},
                title="Avg Reply Time Over Time",
            )
            st.plotly_chart(fig_rt, width="stretch")
        else:
            st.caption("Not enough data to calculate reply time over time.")

        # Write Time Over Time
        st.write("### Write Time Over Time")
        wt_over_time = chat_analyzer.calculate_write_time_over_time(
            max_minutes=240, freq="ME"
        )
        if not wt_over_time.empty:
            fig_wt = px.line(
                wt_over_time,
                x=wt_over_time.index,
                y=wt_over_time.columns,
                markers=True,
                labels={"value": "Minutes", "index": "Month", "variable": "Sender"},
                title="Avg Write Time Over Time",
            )
            st.plotly_chart(fig_wt, width="stretch")
        else:
            st.caption("No write-time data available for this chat.")

        # Weekly Reply Time Stability
        st.write("### Weekly Reply Time Stability (Median & IQR)")

        smooth_data = st.checkbox(
            "Smooth data (3-week rolling average)",
            value=False,
            key="smooth_reply_stability",
        )

        stability_tab1, stability_tab2 = st.tabs(
            ["Combined View", "Split View (Me vs Them)"]
        )

        weekly_stability_raw = chat_analyzer.calculate_weekly_reply_time_stability(
            max_minutes=1440
        )

        if not weekly_stability_raw.empty:
            if smooth_data:
                # Apply 3-week rolling window, using min_periods=1 to avoid dropping edge cases
                weekly_stability = weekly_stability_raw.rolling(
                    window=3, min_periods=1
                ).mean()
            else:
                weekly_stability = weekly_stability_raw.copy()
        else:
            weekly_stability = weekly_stability_raw

        with stability_tab1:
            st.caption(
                "Combined view shows median reply time and IQR (spread) for both parties over weeks."
            )
            if not weekly_stability.empty:
                fig_stability = go.Figure()

                if "Me_Median" in weekly_stability.columns:
                    fig_stability.add_trace(
                        go.Scatter(
                            x=weekly_stability.index,
                            y=weekly_stability["Me_Median"],
                            mode="lines+markers",
                            name="Me (Median)",
                            line=dict(color="#636EFA", width=2),
                            marker=dict(size=5),
                        )
                    )
                    if (
                        "Me_Q25" in weekly_stability.columns
                        and "Me_Q75" in weekly_stability.columns
                    ):
                        me_q25 = weekly_stability["Me_Q25"].fillna(0)
                        me_q75 = weekly_stability["Me_Q75"].fillna(0)
                        fig_stability.add_trace(
                            go.Scatter(
                                x=weekly_stability.index.tolist()
                                + weekly_stability.index[::-1].tolist(),
                                y=me_q75.tolist() + me_q25.tolist()[::-1],
                                fill="toself",
                                fillcolor="rgba(99, 110, 250, 0.2)",
                                line=dict(color="rgba(255,255,255,0)"),
                                showlegend=False,
                                name="Me (IQR)",
                            )
                        )

                if "Them_Median" in weekly_stability.columns:
                    fig_stability.add_trace(
                        go.Scatter(
                            x=weekly_stability.index,
                            y=weekly_stability["Them_Median"],
                            mode="lines+markers",
                            name="Them (Median)",
                            line=dict(color="#EF553B", width=2),
                            marker=dict(size=5),
                        )
                    )
                    if (
                        "Them_Q25" in weekly_stability.columns
                        and "Them_Q75" in weekly_stability.columns
                    ):
                        them_q25 = weekly_stability["Them_Q25"].fillna(0)
                        them_q75 = weekly_stability["Them_Q75"].fillna(0)
                        fig_stability.add_trace(
                            go.Scatter(
                                x=weekly_stability.index.tolist()
                                + weekly_stability.index[::-1].tolist(),
                                y=them_q75.tolist() + them_q25.tolist()[::-1],
                                fill="toself",
                                fillcolor="rgba(239, 85, 59, 0.2)",
                                line=dict(color="rgba(255,255,255,0)"),
                                showlegend=False,
                                name="Them (IQR)",
                            )
                        )

                fig_stability.update_layout(
                    title="Weekly Reply Time Stability (Median & IQR)",
                    xaxis_title="Week",
                    yaxis_title="Minutes",
                    yaxis=dict(rangemode="tozero"),
                    hovermode="x unified",
                    template="plotly_white",
                )
                st.plotly_chart(fig_stability, width="stretch")
            else:
                st.caption(
                    "Not enough data to calculate weekly reply time stability."
                )

        with stability_tab2:
            st.caption(
                "Split view: Compare Me vs Them for both median (main line) and IQR (shaded area)."
            )

            if not weekly_stability.empty:
                me_cols = [
                    c for c in weekly_stability.columns if c.startswith("Me_")
                ]
                them_cols = [
                    c for c in weekly_stability.columns if c.startswith("Them_")
                ]

                if me_cols:
                    fig_me = px.line(
                        weekly_stability,
                        x=weekly_stability.index,
                        y=me_cols,
                        markers=True,
                        labels={
                            "value": "Minutes",
                            "index": "Week",
                            "variable": "Metric",
                        },
                        title="My Reply Time Stability (Median, IQR, MAD)",
                    )
                    fig_me.update_xaxes(title_text="Week")
                    fig_me.update_yaxes(title_text="Minutes", rangemode="tozero")
                    st.plotly_chart(fig_me, width="stretch")

                if them_cols:
                    fig_them = px.line(
                        weekly_stability,
                        x=weekly_stability.index,
                        y=them_cols,
                        markers=True,
                        labels={
                            "value": "Minutes",
                            "index": "Week",
                            "variable": "Metric",
                        },
                        title="Their Reply Time Stability (Median, IQR, MAD)",
                    )
                    fig_them.update_xaxes(title_text="Week")
                    fig_them.update_yaxes(title_text="Minutes", rangemode="tozero")
                    st.plotly_chart(fig_them, width="stretch")
            else:
                st.caption(
                    "Not enough data to calculate weekly reply time stability."
                )

        # Debug Expander (Temporary for troubleshooting)
        if their_write is None:
            st.warning("⚠️ Write Time is N/A - Check Debug Info below")
            with st.expander("Why N/A? (Debug Info)", expanded=True):
                st.write(f"Target Chat: '{selected_contact}'")
                if "read_at" not in sub_df.columns:
                    st.info(
                        "No 'read_at' column in this dataset, so read receipts-based stats can't be calculated."
                    )
                else:
                    st.write(
                        f"Incoming Receipts Data Points: {sub_df[sub_df['from_me'] == 0]['read_at'].notnull().sum()}"
                    )
                    st.write(
                        f"Outgoing Receipts Data Points: {sub_df[sub_df['from_me'] == 1]['read_at'].notnull().sum()}"
                    )

                    st.write("Checking Logic...")
                    try:
                        # Quick check on raw data
                        dbg_df = sub_df.sort_values("timestamp").copy()
                        dbg_df["prev_read"] = dbg_df["read_at"].shift(1)
                        dbg_df["prev_from"] = dbg_df["from_me"].shift(1)
                        valid_raw = dbg_df[
                            (dbg_df["from_me"] == 0)
                            & (dbg_df["prev_from"] == 1)
                            & (dbg_df["prev_read"].notnull())
                        ]
                        st.write(f"Valid Raw Pairs: {len(valid_raw)}")
                        if not valid_raw.empty:
                            diffs = (
                                valid_raw["timestamp"] - valid_raw["prev_read"]
                            ).dt.total_seconds()
                            st.write("Raw Seconds Stats:")
                            st.write(diffs.describe())
                    except Exception as e:
                        st.write(f"Debug Error: {e}")

        # --- Advanced Chat Stats --- (Use chat_analyzer for specific context)
        dist_them, _ = chat_analyzer.get_advanced_reply_stats(reply_to=0)
        dist_me, _ = chat_analyzer.get_advanced_reply_stats(reply_to=1)

        st.write("### Response Time Analysis")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.write("**Their Speed (Them → Me)**")
            if dist_them is not None and selected_contact in dist_them.index:
                row = dist_them.loc[selected_contact]
                fig_dist = px.bar(
                    x=row.index,
                    y=row.values,
                    labels={"x": "Time", "y": "Count"},
                    title=f"{selected_contact}'s Speed",
                )
                st.plotly_chart(fig_dist, width="stretch")
            else:
                st.caption("No data")

        with col_d2:
            st.write("**My Speed (Me → Them)**")
            if dist_me is not None and selected_contact in dist_me.index:
                row_me = dist_me.loc[selected_contact]
                fig_dist_me = px.bar(
                    x=row_me.index,
                    y=row_me.values,
                    labels={"x": "Time", "y": "Count"},
                    title="My Speed",
                )
                fig_dist_me.update_traces(marker_color="#EF553B")
                st.plotly_chart(fig_dist_me, width="stretch")

        # Ghosting Control
        st.divider()
        gh_hours = st.slider(
            "Ghosting Threshold (Hours)",
            1,
            72,
            24,
            key="chat_explorer_ghost_thresh",
        )

        # Use FULL Analyzer for specific chat logic (needs Me + Them)
        # Create a dedicated analyzer for this chat using BASE data (includes Me)
        # df_base might contain all chats, so we filter by contact first logic?
        # get_true_ghosting_stats is global but returns per contact.
        # We can use full_analyzer_tab6 (if available globablly? No, it was local to Tab 6).
        # Let's instantiate a specific one or use a global 'full_analyzer' if I make it available.
        full_chat_df = df_base[df_base["chat_name"] == selected_contact]
        full_single_analyzer = WhatsappAnalyzer(full_chat_df)

        true_ghosts = full_single_analyzer.get_true_ghosting_stats(
            threshold_hours=gh_hours
        )

        if not true_ghosts.empty:
            # true_ghosts index is contact_name. Since we filtered to one contact, it should be there.
            # But get_true_ghosting_stats groups by Chat Name. If contact_name is 'You' for my messages?
            # No, get_true_ghosting_stats uses 'chat_name' column.

            # Check if selected_contact is in index
            if selected_contact in true_ghosts.index:
                st.write(f"**Ghosting Stats (> {gh_hours}h)**")
                g_row = true_ghosts.loc[selected_contact]
                cols_g = st.columns(3)
                cols_g[0].metric(
                    "True Ghosts 👻",
                    av(int(g_row.get("True Ghost 👻", 0)), _anon_numbers),
                    help="Read but ignored",
                )
                cols_g[1].metric(
                    "Left on Delivered 📨",
                    av(int(g_row.get("Left on Delivered 📨", 0)), _anon_numbers),
                    help="Never read",
                )
            else:
                st.info("No ghosting detected with current threshold.")
        else:
            st.info("No ghosting detected with current threshold.")

        st.subheader("Behavioral Timeline")
        # Get behavioral timeline data
        ghost_thresh = 432000 if use_longer_stats else 86400
        init_thresh = 172800 if use_longer_stats else 21600

        # Use the FULL SINGLE ANALYZER
        beh_timeline = full_single_analyzer.get_behavioral_timeline(
            ghost_thresh, init_thresh
        )

        if not beh_timeline.empty:
            b1, b2 = st.columns(2)
            with b1:
                st.caption("Ghosting Over Time")
                # Ghosted by Them vs Ghosted by Me
                fig_g = px.bar(
                    beh_timeline,
                    x=beh_timeline.index,
                    y=["Ghosted by Them", "Ghosted by Me"],
                    title="Ghosting Incidents",
                    barmode="group",
                )
                st.plotly_chart(fig_g, width="stretch")

            with b2:
                st.caption("Initiations Over Time")
                fig_i = px.bar(
                    beh_timeline,
                    x=beh_timeline.index,
                    y=["Initiated by Me", "Initiated by Them"],
                    title="Conversation Initiations",
                    barmode="group",
                )
                st.plotly_chart(fig_i, width="stretch")
        else:
            st.info("No behavioral timeline data available for this selection.")

        st.subheader("Activity Analysis")

        # Controls for Activity Logic
        c_act1, c_act2 = st.columns([2, 1])
        with c_act1:
            act_view = st.radio(
                "View Mode",
                ["Combined", "Split (Me vs Them)", "Only Me", "Only Them"],
                horizontal=True,
                key="chat_act_view",
            )
        with c_act2:
            chat_show_lines = st.checkbox(
                "Show as Lines", value=False, key="chat_show_lines"
            )

        plot_func_chat = px.line if chat_show_lines else px.area

        # Monthly Activity
        monthly_split_arg = "sender" if act_view != "Combined" else None
        monthly_chat = chat_analyzer.get_monthly_activity(
            split_by=monthly_split_arg
        )

        # Filter if needed
        if act_view == "Only Me" and "Me" in monthly_chat.columns:
            monthly_chat = monthly_chat[["Me"]]
        elif act_view == "Only Them" and "Them" in monthly_chat.columns:
            monthly_chat = monthly_chat[["Them"]]

        if monthly_chat.empty:
            st.info(f"No monthly data available for view: {act_view}")
        else:
            if isinstance(monthly_chat, pd.Series) and not monthly_chat.name:
                monthly_chat.name = "Messages"
            fig_chat_time = plot_func_chat(
                monthly_chat, title=f"Message Volume ({act_view})"
            )
            st.plotly_chart(fig_chat_time, width="stretch")

        # Hourly Activity
        hourly_split_arg = "sender" if act_view != "Combined" else None
        hourly_chat = chat_analyzer.get_hourly_activity(split_by=hourly_split_arg)

        if isinstance(hourly_chat, pd.DataFrame):
            if act_view == "Only Me" and "Me" in hourly_chat.columns:
                hourly_chat = hourly_chat[["Me"]]
            elif act_view == "Only Them" and "Them" in hourly_chat.columns:
                hourly_chat = hourly_chat[["Them"]]

        if hourly_chat.empty:
            st.info(f"No hourly data available for view: {act_view}")
        else:
            if isinstance(hourly_chat, pd.Series) and not hourly_chat.name:
                hourly_chat.name = "Messages"
            fig_chat_hour = px.line(
                hourly_chat, markers=True, title=f"Hourly Activity ({act_view})"
            )
            st.plotly_chart(fig_chat_hour, width="stretch")

        st.subheader("Word Usage Comparison")
        if st.button("Generate Comparative Word Clouds"):
            col_wc1, col_wc2 = st.columns(2)
            with col_wc1:
                st.caption("My Words")
                my_text = chat_analyzer.get_wordcloud_text(
                    filter_from_me=True,
                    min_word_length=min_word_len,
                    exclude_emails=exclude_emails,
                )
                if my_text:
                    wc1 = WordCloud(
                        width=400, height=300, background_color="white"
                    ).generate(my_text)
                    plt.figure(figsize=(5, 4))
                    plt.imshow(wc1)
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.write("No data.")

            with col_wc2:
                st.caption(f"{selected_contact}'s Words")
                their_text = chat_analyzer.get_wordcloud_text(
                    filter_from_me=False,
                    min_word_length=min_word_len,
                    exclude_emails=exclude_emails,
                )
                if their_text:
                    wc2 = WordCloud(
                        width=400, height=300, background_color="white"
                    ).generate(their_text)
                    plt.figure(figsize=(5, 4))
                    plt.imshow(wc2)
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.write("No data.")

        st.write("Recent Messages:")
        cols_to_show = ["timestamp", "from_me", "text_data"]
        if "mime_type" in sub_df.columns:
            cols_to_show.append("mime_type")
        st.dataframe(
            sub_df[cols_to_show].sort_values("timestamp", ascending=False).head(10)
        )
