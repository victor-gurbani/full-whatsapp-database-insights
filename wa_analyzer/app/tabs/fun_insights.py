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

    st.header("🎪 Fun & Insights")

    col_fun_ctrl1, col_fun_ctrl2 = st.columns(2)
    top_n_filter = col_fun_ctrl1.selectbox(
        "Filter Rank to Top Contacts:",
        [50, 100, 200, "All"],
        index=1,
        help="Analyze only the most active contacts to reduce noise.",
    )
    top_n_val = None if top_n_filter == "All" else int(top_n_filter)

    # Calculate Stats
    # Pass exclude_groups from sidebar
    # Calculate Stats using FULL Data (Context needed for Double Text, Streaks, Killers)
    # But we must respect exclude_me for the DISPLAY.
    full_analyzer_tab6 = full_analyzer

    ex_groups = exclude_groups if "exclude_groups" in locals() else False

    with st.spinner("Crunching fun stats..."):
        beh_scorecard = full_analyzer_tab6.get_behavioral_scorecard(
            exclude_groups=True
        )
        fun_stats = full_analyzer_tab6.get_fun_stats(
            top_n=top_n_val, exclude_groups=True
        )
        streaks = full_analyzer_tab6.get_streak_stats(exclude_groups=ex_groups)
        killers = full_analyzer_tab6.get_conversation_killers(
            exclude_groups=ex_groups
        )

    # New Stats
    # Pass exclude_me (global) to filter "Me" from top reactors
    reaction_stats = full_analyzer_tab6.get_reaction_stats(
        exclude_groups=ex_groups, exclude_me=exclude_me
    )
    emoji_stats = full_analyzer_tab6.get_emoji_stats(
        top_n=top_n_val, exclude_groups=ex_groups
    )
    mention_stats = full_analyzer_tab6.get_mention_stats(
        top_n=top_n_val, exclude_groups=ex_groups
    )
    history_stats = full_analyzer_tab6.get_historical_stats(
        exclude_groups=ex_groups
    )

    # Post-Calculation Filter: Remove "You" / "Me" if exclude_me is True
    if exclude_me:
        # Identify 'Me' name (usually 'You', 'Me', 'Myself' or me_display)
        # Parser ensures outgoing is mapped to "You".
        me_names = ["You", "Me", "Myself"]
        if "me_display" in locals():
            me_names.append(str(me_display))  # if resolved

        # Filter Index (usually contact_name)
        if not beh_scorecard.empty:
            beh_scorecard = beh_scorecard[~beh_scorecard.index.isin(me_names)]
        if not fun_stats.empty:
            fun_stats = fun_stats[~fun_stats.index.isin(me_names)]
        if not streaks.empty:
            streaks = streaks[~streaks.index.isin(me_names)]
        if not killers.empty:
            killers = killers[~killers.index.isin(me_names)]

        # Reaction stats is a dict
        if reaction_stats and "top_reactors" in reaction_stats:
            reaction_stats["top_reactors"] = reaction_stats["top_reactors"][
                ~reaction_stats["top_reactors"].index.isin(me_names)
            ]

            # Fix: Apply "Exclude Non-Contacts" filter to Reactors too
            # Because the reactor might be a non-contact even if the message thread is valid.
            if exclude_non_contacts:
                # Filter out names that are just numbers/invalid
                is_valid_contact = pd.Series(
                    reaction_stats["top_reactors"].index,
                    index=reaction_stats["top_reactors"].index,
                ).apply(lambda x: bool(re.search("[a-zA-Z]", str(x))))
                reaction_stats["top_reactors"] = reaction_stats["top_reactors"][
                    is_valid_contact
                ]

        # Emoji stats is dict
        # per_contact...
        if emoji_stats and "per_contact" in emoji_stats:
            # per_contact has 'contact_name' column
            emoji_stats["per_contact"] = emoji_stats["per_contact"][
                ~emoji_stats["per_contact"]["contact_name"].isin(me_names)
            ]

        # Mention stats is dict
        # who_mentions_me (index is sender)
        if mention_stats and "who_mentions_me" in mention_stats:
            mention_stats["who_mentions_me"] = mention_stats["who_mentions_me"][
                ~mention_stats["who_mentions_me"].index.isin(me_names)
            ]
        # i_mention (index is target) - keep? Yes, it's who I mention.

    # Pre-calc combined media for Gallery Curator
    if "image_media" in fun_stats.columns and "video_media" in fun_stats.columns:
        fun_stats["gallery_count"] = (
            fun_stats["image_media"] + fun_stats["video_media"]
        )
    else:
        fun_stats["gallery_count"] = fun_stats["media"]  # Fallback

    # 1. Hall of Fame
    st.subheader("🏆 Hall of Fame")

    # New Feature: Percentage Mode
    use_pct = st.checkbox(
        "Calculate based of % of total message (Density)",
        value=False,
        help="Normalizes metrics by the number of messages sent by each person.",
    )

    # Pre-calculate Message Counts for Normalization
    # Use full_analyzer data (df_base)
    # We need a series of contact_name -> message_count
    if use_pct:
        # We already have df_base.
        msg_counts = df_base["contact_name"].value_counts()

    hof_1, hof_2, hof_3 = st.columns(3)

    # Helper to get top user with optional normalization
    def get_top(df, col, exclude_list=[]):
        if df.empty or col not in df.columns:
            return "N/A", 0

        # If % mode, we need to normalize 'col' by message count
        # This requires 'df' to have an index of contact_name matching msg_counts
        target_series = df[col]

        if (
            use_pct and "pct" not in col
        ):  # Don't normalize columns that are already % (like night_owl_pct)
            # Align data
            # We need to ensure we match indices
            aligned_counts = msg_counts.reindex(target_series.index).fillna(
                1
            )  # avoid div/0
            target_series = (target_series / aligned_counts) * 100

        sorted_series = target_series.sort_values(ascending=False)
        if sorted_series.empty:
            return "N/A", 0

        top_name = sorted_series.index[0]
        top_val = sorted_series.iloc[0]
        return top_name, top_val

    with hof_1:
        name, val = get_top(beh_scorecard, "night_owl_pct")
        st.metric(
            "🦉 The Night Owl", name, f"{av(val, _anon_numbers):.1f}% Night Msgs"
        )

    with hof_2:
        name, val = get_top(beh_scorecard, "early_bird_pct")
        st.metric(
            "☀️ The Early Bird", name, f"{av(val, _anon_numbers):.1f}% Morning Msgs"
        )

    with hof_3:
        name, val = get_top(fun_stats, "laughs")
        if use_pct:
            st.metric(
                "😂 The Comedian", name, f"{av(val, _anon_numbers):.1f}% of msgs"
            )
        else:
            st.metric(
                "😂 The Comedian", name, f"{av(int(val), _anon_numbers)} Laughs"
            )

    st.divider()

    hof_4, hof_5, hof_6 = st.columns(3)
    with hof_4:
        name, val = get_top(fun_stats, "deleted")
        if use_pct:
            st.metric(
                "🗑️ The Deleter", name, f"{av(val, _anon_numbers):.1f}% of msgs"
            )
        else:
            st.metric(
                "🗑️ The Deleter", name, f"{av(int(val), _anon_numbers)} Retracted"
            )

    with hof_5:
        if not streaks.empty:
            # new get_streak_stats returns DataFrame with 'streak', 'start_date', 'end_date'
            # Check format just in case
            if isinstance(streaks, pd.DataFrame):
                top_row = streaks.iloc[0]
                name = top_row.name  # index is contact_name
                val = top_row["streak"]
                s_date = top_row.get("start_date", "?")
                e_date = top_row.get("end_date", "?")
                tooltip_txt = f"Longest Streak: {s_date} to {e_date}"
            else:
                # Legacy fallback if something weird happens
                name = streaks.idxmax()
                val = streaks.max()
                tooltip_txt = "Date range unavailable"
        else:
            name, val = "N/A", 0
            tooltip_txt = ""

        st.metric(
            "🔥 Streak Master",
            name,
            f"{av(val, _anon_numbers)} Days",
            help=tooltip_txt,
        )

    with hof_6:  # (Was hof_5 in original code, fixing index)
        # Killers is a Series
        if not killers.empty:
            target = killers
            if use_pct:
                aligned_counts = msg_counts.reindex(target.index).fillna(1)
                target = (target / aligned_counts) * 100
                target = target.sort_values(ascending=False)

            name = target.index[0]
            val = target.iloc[0]

            if use_pct:
                st.metric(
                    "🤐 Conversation Killer",
                    name,
                    f"{av(val, _anon_numbers):.1f}% Kill Rate",
                )
            else:
                st.metric(
                    "🤐 Conversation Killer",
                    name,
                    f"{av(val, _anon_numbers)} Silences",
                )
        else:
            # name, val = "N/A", 0 # Variable leak if block skipped? No.
            st.metric("🤐 Conversation Killer", "N/A", "0")

    st.divider()

    hof_7, hof_8, hof_9 = st.columns(3)
    with hof_7:
        name, val = get_top(fun_stats, "audio_media")
        if use_pct:
            st.metric(
                "🎙️ The Podcaster", name, f"{av(val, _anon_numbers):.1f}% of msgs"
            )
        else:
            st.metric(
                "🎙️ The Podcaster",
                name,
                f"{av(int(val), _anon_numbers)} Voice Notes",
            )

    with hof_8:
        name, val = get_top(fun_stats, "gallery_count")
        if use_pct:
            st.metric(
                "🖼️ Gallery Curator", name, f"{av(val, _anon_numbers):.1f}% of msgs"
            )
        else:
            st.metric(
                "🖼️ Gallery Curator",
                name,
                f"{av(int(val), _anon_numbers)} Pics/Vids",
            )

    with hof_9:
        if reaction_stats and not reaction_stats["top_reactors"].empty:
            target = reaction_stats["top_reactors"]  # This is a Series
            if use_pct:
                aligned_counts = msg_counts.reindex(target.index).fillna(1)
                target = (target / aligned_counts) * 100
                target = target.sort_values(ascending=False)

            name = target.index[0]
            val = target.iloc[0]

            if use_pct:
                st.metric(
                    "😍 Reaction addict",
                    name,
                    f"{av(val, _anon_numbers):.1f}% Rate",
                )
            else:
                st.metric(
                    "😍 Reaction addict",
                    name,
                    f"{av(val, _anon_numbers)} Reactions",
                )
        else:
            st.metric("😍 Reaction addict", "N/A", "0")

    st.divider()

    # --- NEW SECTIONS ---

    # 1. Emoji Analysis
    if emoji_stats and not emoji_stats["per_contact"].empty:
        st.subheader("❤️ The Emoji Fanatic")
        # Show Top 5 Emojis for Top 5 Users
        # emoji_stats['per_contact'] is a DF with contact, emoji, count
        # Pivot to allow nice display? Or just list?
        # Let's show a dataframe of "User | Top 5 Emojis"

        # Group by contact, join emojis
        # Group by contact, join emojis
        # --- UI IMPROVEMENT: Default Top 10 (Most Active) + Search ---
        # Get top talkers for default selection
        top_active = full_analyzer_tab6.get_top_talkers(n=10, metric="messages")
        default_selection = (
            top_active["contact_name"].tolist() if not top_active.empty else []
        )

        all_contacts_emoji = (
            emoji_stats["per_contact"]["contact_name"].unique().tolist()
        )

        # Intersect top active with emoji contacts to ensure validity
        default_emoji_sel = [
            c for c in default_selection if c in all_contacts_emoji
        ]
        if not default_emoji_sel:
            default_emoji_sel = all_contacts_emoji[:10]

        sel_emoji_contacts = st.multiselect(
            "Select Contacts", all_contacts_emoji, default=default_emoji_sel
        )

        if sel_emoji_contacts:
            filtered_emoji = emoji_stats["per_contact"][
                emoji_stats["per_contact"]["contact_name"].isin(sel_emoji_contacts)
            ]
            top_emo_disp = (
                filtered_emoji.groupby("contact_name")["emoji"]
                .apply(lambda x: " ".join(x))
                .reset_index(name="Top Emojis")
            )
            st.dataframe(top_emo_disp.set_index("contact_name"), width="stretch")
        else:
            st.write("No contacts selected.")

    # 2. Reaction Deep Dive
    if reaction_stats:
        st.subheader("😍 Reaction Insights")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.write("**Most Reacted Messages**")
            st.dataframe(
                reaction_stats["most_reacted"][
                    ["chat_contact", "preview", "count"]
                ].head(5)
            )
        with col_r2:
            st.write("**Top Global Emojis**")
            # Top emojis as bar chart
            top_em = _coerce_bar_chart_data(reaction_stats["top_emojis"])
            if top_em is not None:
                st.bar_chart(top_em)
            else:
                st.info("No emoji reaction data to chart.")

    # 3. Mentions
    if mention_stats:
        st.subheader("📢 Mentions (@Tags)")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("**Who Mentions Me Most?**")
            who_mentions_me = _coerce_bar_chart_data(
                mention_stats["who_mentions_me"]
            )
            if who_mentions_me is not None:
                st.bar_chart(who_mentions_me)
            else:
                st.info("No incoming mention data to chart.")
        with col_m2:
            st.write("**Who Do I Mention Most?**")
            i_mention = _coerce_bar_chart_data(mention_stats["i_mention"])
            if i_mention is not None:
                st.bar_chart(i_mention)
            else:
                st.info("No outgoing mention data to chart.")

    # 4. Historical Deep Dive
    if history_stats:
        st.subheader("📜 Historical Deep Dive")

        # Velocity
        st.write("**⚡ Message Velocity (Max Words Per Minute)**")
        st.caption("Highest WPM achieved in a single minute of conversation.")

        # --- UI IMPROVEMENT: Default Top 10 + Search ---
        velocity_df = history_stats["velocity_wpm"]  # Series
        all_vel_contacts = velocity_df.index.tolist()

        # Reuse default_selection (Top 10 Active) from above if available, else recalc or slice
        if "default_selection" in locals():
            default_vel = [c for c in default_selection if c in all_vel_contacts]
            if not default_vel:
                default_vel = all_vel_contacts[:10]
        else:
            default_vel = all_vel_contacts[:10]

        sel_vel_contacts = st.multiselect(
            "Select Contacts for Velocity", all_vel_contacts, default=default_vel
        )

        if sel_vel_contacts:
            filtered_vel = velocity_df[velocity_df.index.isin(sel_vel_contacts)]
            filtered_vel = _coerce_bar_chart_data(filtered_vel)
            if filtered_vel is not None:
                st.bar_chart(filtered_vel, horizontal=True)
            else:
                st.info("No velocity data to chart for this selection.")
        else:
            st.write("No contacts selected.")

        # First Message
        # First Message
        st.write("**🕰️ How it Started (First Messages)**")
        first_msgs = history_stats["first_msgs"].reset_index()
        # Select contact
        sel_contact_hist = st.selectbox(
            "Select Contact to see First Message:", first_msgs["chat_name"].unique()
        )
        if sel_contact_hist:
            row = first_msgs[first_msgs["chat_name"] == sel_contact_hist].iloc[0]
            st.markdown(f"**Date:** {row['timestamp']}")
            st.markdown(f'**Message:** *"{row["text_data"]}"*')

    st.divider()
    col_lod1, col_lod2 = st.columns(2)

    # Filter Checkbox
    ghost_filter = st.checkbox(
        "Hide contacts with 0 'True Ghost' value (Only show confirmed ghosts)",
        value=False,
    )

    with col_lod1:
        st.write("**People I Ignore** (Me → Them)")
        my_ignore_stats = full_analyzer_tab6.get_left_on_read_stats()

        if not my_ignore_stats.empty:
            cols_to_plot = [
                c
                for c in my_ignore_stats.columns
                if c in ["True Ghost 👻", "Left on Delivered 📨"]
            ]
            data_to_plot = my_ignore_stats[cols_to_plot]

            if ghost_filter and "True Ghost 👻" in data_to_plot.columns:
                data_to_plot = data_to_plot[data_to_plot["True Ghost 👻"] > 0]

            data_to_plot = _coerce_bar_chart_data(data_to_plot.head(15))
            if data_to_plot is not None:
                st.bar_chart(data_to_plot, horizontal=True)
            else:
                st.info("No ghosts found with current filter.")
        else:
            st.info("You're a saint! 😇")

    with col_lod2:
        st.write("**People Who Ignore Me** (Them → Me)")
        them_ignore_stats = full_analyzer_tab6.get_true_ghosting_stats(
            threshold_hours=24
        )  # ghosting = them ignoring me

        if not them_ignore_stats.empty:
            cols_to_plot = [
                c
                for c in them_ignore_stats.columns
                if c in ["True Ghost 👻", "Left on Delivered 📨"]
            ]
            data_to_plot = them_ignore_stats[cols_to_plot]

            if ghost_filter and "True Ghost 👻" in data_to_plot.columns:
                data_to_plot = data_to_plot[data_to_plot["True Ghost 👻"] > 0]

            data_to_plot = _coerce_bar_chart_data(data_to_plot.head(15))
            if data_to_plot is not None:
                st.bar_chart(data_to_plot, horizontal=True)
            else:
                st.info("No ghosts found with current filter.")
        else:
            st.info("Everyone loves you! 💖")

    st.divider()

    # 2. Charts
    c_fun1, c_fun2 = st.columns(2)

    with c_fun1:
        st.subheader("🎭 Double Text Ratio")
        st.caption(
            "Percentage of your turns that are double-texts (continuing without reply after >20m)."
        )
        if not beh_scorecard.empty:
            dt_df = beh_scorecard.sort_values(
                "double_text_ratio", ascending=False
            ).head(15)
            fig_dt = px.bar(
                dt_df,
                x="double_text_ratio",
                y=dt_df.index,
                orientation="h",
                title="Highest Double Text Ratio",
                color="gender",
                color_discrete_map={
                    "male": "#636EFA",
                    "female": "#EF553B",
                    "unknown": "gray",
                },
            )
            fig_dt.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Double Text %",
            )
            st.plotly_chart(fig_dt, width="stretch")

    with c_fun2:
        st.subheader("🌵 Dry Texter Index")
        st.caption("Average words per message.")
        if not fun_stats.empty:
            # Filter out 0 value (Media only or empty text)
            mask_dry = fun_stats["avg_word_len"] > 0
            dry_df = (
                fun_stats[mask_dry]
                .sort_values("avg_word_len", ascending=True)
                .head(15)
            )
            fig_dry = px.bar(
                dry_df,
                x="avg_word_len",
                y=dry_df.index,
                orientation="h",
                title="Shortest Responses (Dryest)",
                color="gender",
                color_discrete_map={
                    "male": "#636EFA",
                    "female": "#EF553B",
                    "unknown": "gray",
                },
            )
            fig_dry.update_layout(
                yaxis={"categoryorder": "total descending"},
                xaxis_title="Avg Words/Msg",
            )
            st.plotly_chart(fig_dry, width="stretch")
