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

    st.header("📱 WhatsApp Chat Viewer")
    st.write("Preview your chats exactly as they look in WhatsApp.")

    if "data" in st.session_state and not df_raw.empty:
        contacts_list = sorted(df_raw["chat_name"].dropna().unique().astype(str))

        selected_preview_chat = st.selectbox(
            "Select Chat to Preview", contacts_list, key="chat_viewer_select"
        )

        if selected_preview_chat:
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                flip_sides = st.toggle(
                    "Flip Sides (I am them, they are me)",
                    value=False,
                    help="Swaps the left/right and color orientation of the messages. Useful if you want to view the conversation from the other person's perspective.",
                )
                if flip_sides:
                    my_name = st.text_input(
                        "My Name (for flipped view)",
                        value="Me",
                        help="Name to display for your messages when viewed from the other person's perspective.",
                        key="chat_viewer_my_name",
                    )
                else:
                    my_name = "Me"
            with col_t2:
                adv_replies = st.toggle(
                    "Advanced Replies View",
                    value=False,
                    help="Highlights any message that received a reply (in Orange) and marks the messages directly above/below it (in Blue) for extra context.",
                )
                if adv_replies:
                    col_num1, col_num2 = st.columns(2)
                    with col_num1:
                        adv_msgs_above = st.number_input(
                            "Messages above",
                            min_value=0,
                            max_value=10,
                            value=2,
                            help="Number of messages preceding the replied message to highlight.",
                        )
                    with col_num2:
                        adv_msgs_below = st.number_input(
                            "Messages below",
                            min_value=0,
                            max_value=10,
                            value=1,
                            help="Number of messages following the replied message to highlight.",
                        )
                else:
                    adv_msgs_above = 2
                    adv_msgs_below = 1
            with col_t3:
                context_view = st.toggle(
                    "Context/Unreplied View",
                    value=False,
                    help="Highlights consecutive messages (in Pink) that were implicitly replied to. This happens when someone responds directly without explicitly quoting.",
                )
                if context_view:
                    unreplied_chunk_size = st.number_input(
                        "Unreplied Chunk Size",
                        min_value=1,
                        max_value=100,
                        value=1,
                        help="Search for chunks of X consecutive unreplied messages (Magnifying Glass button).",
                    )
                    unreplied_other_only = st.checkbox(
                        "Target Other Person Only",
                        value=False,
                        help="If checked, the magnifying glass (unreplied) will only target unreplied messages sent by the other person.",
                    )
                else:
                    unreplied_chunk_size = 1
                    unreplied_other_only = False

                collapse_replies = st.toggle(
                    "Collapse Replied-To Messages",
                    value=False,
                    help="Groups consecutive messages that were explicitly replied to. You can click to expand them in chunks of 5.",
                )

            st.markdown("<br>", unsafe_allow_html=True)
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                enable_tooltips = st.toggle(
                    "Enable Tooltip Timestamps",
                    value=True,
                    help="Hover over a message to see the exact date and time.",
                )
            with col_f2:
                enable_minimap = st.toggle(
                    "Show Chat Minimap",
                    value=True,
                    help="Shows a VS Code style minimap of the chat history on the right.",
                )
            with col_f3:
                enable_sentiment = st.toggle(
                    "Minimap Sentiment",
                    value=False,
                    help="Highlights positive (yellow) and negative (red) messages in the minimap.",
                )

            st.markdown("<br>", unsafe_allow_html=True)
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                enable_virtualization = st.toggle(
                    "Enable JS Virtualization (Fast)",
                    value=True,
                    help="Significantly speeds up the browser by only rendering messages currently visible on screen. Recommended for large chats.",
                )
                if enable_virtualization:
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        virt_initial = st.number_input(
                            "Initial Messages (Chunk)",
                            min_value=50,
                            max_value=2000,
                            value=300,
                            step=50,
                            help="How many messages to load instantly.",
                        )
                    with col_v2:
                        virt_chunk = st.number_input(
                            "Scroll Load (Chunk)",
                            min_value=50,
                            max_value=500,
                            value=120,
                            step=10,
                            help="How many messages to load when scrolling up.",
                        )
                else:
                    virt_initial = 300
                    virt_chunk = 120
            with col_p2:
                max_messages_limit = st.number_input(
                    "Max Messages to Load",
                    min_value=100,
                    max_value=1000000,
                    value=10000,
                    step=1000,
                    help="Limits the total number of messages loaded into the viewer to prevent memory crashes.",
                )

            st.markdown(
                """
                <div style="font-size:0.85rem; padding: 10px; background-color: rgba(255,255,255,0.05); border-radius: 5px; margin-bottom: 10px; line-height: 1.8;">
                    <span style="display:inline-block; margin-right: 15px;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom;"><circle cx="12" cy="12" r="10" fill="#dcf8c6"></circle></svg> <b>Green</b>: Me</span>
                    <span style="display:inline-block; margin-right: 15px;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom;"><circle cx="12" cy="12" r="10" fill="#ffffff"></circle></svg> <b>White</b>: Them</span>
                    <span style="display:inline-block; margin-right: 15px;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ff9800" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom;"><circle cx="12" cy="12" r="10" fill="#fff3e0"></circle></svg> <b>Orange Highlight</b>: Explicitly Replied To</span>
                    <span style="display:inline-block; margin-right: 15px;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#03a9f4" stroke-width="2" stroke-dasharray="4" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom;"><circle cx="12" cy="12" r="10" fill="none"></circle></svg> <b>Blue Dashed</b>: Surrounding Context</span>
                    <span style="display:inline-block; margin-right: 15px;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#e91e63" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom;"><circle cx="12" cy="12" r="10" fill="#fce4ec"></circle></svg> <b>Pink Highlight</b>: Implicit Reply Context</span>
                    <span style="display:inline-block;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="gray" stroke-width="2" stroke-dasharray="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom;"><circle cx="12" cy="12" r="10" fill="transparent"></circle></svg> <b>Unmarked</b>: Unreplied / Ignored</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            chat_df = df_raw[df_raw["chat_name"] == selected_preview_chat].copy()
            chat_df = chat_df.sort_values("timestamp").reset_index(drop=True)
            chat_df = chat_df.tail(int(max_messages_limit)).reset_index(drop=True)

            st.subheader("📥 Export Chat")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                json_data = export_chat_json(chat_df)
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name=f"{selected_preview_chat}_export.json",
                    mime="application/json",
                )
            with col_dl2:
                txt_data = export_chat_txt(
                    chat_df, flip_sides=flip_sides, my_name=my_name
                )
                st.download_button(
                    label="Download as TXT",
                    data=txt_data,
                    file_name=f"{selected_preview_chat}_export.txt",
                    mime="text/plain",
                )
            with col_dl3:
                html_standalone = export_chat_html_standalone(
                    chat_df=chat_df,
                    flip_sides=flip_sides,
                    adv_replies=adv_replies,
                    msgs_above=adv_msgs_above,
                    msgs_below=adv_msgs_below,
                    context_view=context_view,
                    unreplied_chunk_size=unreplied_chunk_size,
                    unreplied_other_only=unreplied_other_only,
                    collapse_replies=collapse_replies,
                    virtualization=enable_virtualization,
                    virt_initial=virt_initial,
                    virt_chunk=virt_chunk,
                    my_name=my_name,
                    enable_tooltips=enable_tooltips,
                    enable_minimap=enable_minimap,
                    enable_sentiment=enable_sentiment,
                )
                st.download_button(
                    label="Download as Standalone HTML",
                    data=html_standalone,
                    file_name=f"{selected_preview_chat}_viewer.html",
                    mime="text/html",
                )

            st.divider()
            st.subheader("💬 Chat Preview")

            with st.spinner("Rendering chat..."):
                html_output = generate_chat_html(
                    chat_df=chat_df,
                    flip_sides=flip_sides,
                    adv_replies=adv_replies,
                    msgs_above=adv_msgs_above,
                    msgs_below=adv_msgs_below,
                    context_view=context_view,
                    unreplied_chunk_size=unreplied_chunk_size,
                    unreplied_other_only=unreplied_other_only,
                    collapse_replies=collapse_replies,
                    virtualization=enable_virtualization,
                    virt_initial=virt_initial,
                    virt_chunk=virt_chunk,
                    my_name=my_name,
                    enable_tooltips=enable_tooltips,
                    enable_minimap=enable_minimap,
                    enable_sentiment=enable_sentiment,
                )

            st.components.v1.html(html_output, height=620, scrolling=False)
    else:
        st.info("Load data to preview chats.")
