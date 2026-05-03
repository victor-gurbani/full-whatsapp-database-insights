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

    st.header("Word Cloud")

    wc_source = st.radio(
        "Source", ["All Messages", "Sent by Me", "Received by Me"], horizontal=True
    )

    if st.button("Generate Word Cloud"):
        with st.spinner("Generating..."):
            filter_me = None
            if wc_source == "Sent by Me":
                filter_me = True
            elif wc_source == "Received by Me":
                filter_me = False

            text = full_analyzer.get_wordcloud_text(
                filter_from_me=filter_me,
                min_word_length=min_word_len,
                exclude_emails=exclude_emails,
            )
            if text:
                wc = WordCloud(
                    width=800, height=400, background_color="white"
                ).generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.warning("Not enough text data.")
