"""Sidebar rendering and context construction."""

from __future__ import annotations

import datetime
import json
import os

import pandas as pd
import streamlit as st

from wa_analyzer.app.db_loaders import (
    _format_backup_horizon,
    get_file_signature,
    load_archived_chat_ids,
    load_backup_message_horizon,
)
from wa_analyzer.app.filters import apply_global_filters
from wa_analyzer.app.privacy import (
    _anon_hash,
    _anon_hash_cut,
    _anon_random,
    apply_anon_to_df,
    av,
    build_anon_map,
    collect_nested_identity_values,
)
from wa_analyzer.app.state import (
    AnonymizationState,
    AppContext,
    DataPaths,
    FilterState,
    SidebarState,
)
from wa_analyzer.src.analyzer import WhatsappAnalyzer
from wa_analyzer.src.parser import WhatsappParser


# --- Config Management Functions ---
def load_config():
    uploaded_file = st.session_state.get("config_uploader")
    if uploaded_file is not None:
        try:
            config = json.load(uploaded_file)
            # Update session state
            for key, value in config.items():
                if key == "cfg_date_range":
                    # Convert iso strings back to date objects
                    st.session_state[key] = [
                        datetime.date.fromisoformat(d) for d in value
                    ]
                else:
                    st.session_state[key] = value
            # st.rerun() not needed in callback
        except Exception as e:
            st.error(f"Error loading config: {e}")


def get_config_json():
    # Gather keys
    config = {}
    keys_to_save = [
        "cfg_msgstore",
        "cfg_wa",
        "cfg_vcf",
        "cfg_date_range",
        "cfg_ex_groups",
        "cfg_ex_archived",
        "cfg_ex_chan",
        "cfg_fam_list",
        "cfg_ex_fam_glob",
        "cfg_ex_non_con",
        "cfg_ex_fam_gend",
        "cfg_long_stats",
        "cfg_reply_thresh",
        "cfg_min_word_len",
        "cfg_ex_emails",
        "cfg_anon_mode",
        "cfg_anon_numbers",
    ]
    for k in keys_to_save:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, (list, tuple)) and k == "cfg_date_range":
                # Serialize dates
                config[k] = [d.isoformat() for d in val]
            else:
                config[k] = val
    return json.dumps(config, indent=2, sort_keys=True)


def render_sidebar() -> AppContext | None:
    # Sidebar
    st.sidebar.header("Data Sources")

    # --- Config Import/Export UI (Top of Sidebar for Visibility) ---
    with st.sidebar.expander("💾 Configuration Manager", expanded=False):
        st.file_uploader(
            "Import Config", type=["json"], key="config_uploader", on_change=load_config
        )

        st.divider()

        # Export
        # We rely on current session state. Note: Widgets must have keys assigned below.
        json_str = get_config_json()
        st.download_button(
            label="Download Configuration",
            data=json_str,
            file_name="wa_analyzer_config.json",
            mime="application/json",
        )

    base_dir = os.getcwd()
    default_msgstore = os.path.join(base_dir, "msgstore.db")
    default_wa = os.path.join(base_dir, "wa.db")
    default_vcf = os.path.join(base_dir, "contacts.vcf")

    with st.sidebar.expander("📂 Data Source", expanded="data" not in st.session_state):
        msgstore_path = st.text_input(
            "Msgstore Path",
            value=default_msgstore if os.path.exists(default_msgstore) else "",
            key="cfg_msgstore",
        )
        wa_path = st.text_input(
            "WA DB Path",
            value=default_wa if os.path.exists(default_wa) else "",
            key="cfg_wa",
        )
        vcf_path = st.text_input(
            "VCF Path",
            value=default_vcf if os.path.exists(default_vcf) else "",
            key="cfg_vcf",
        )

        if st.button("Load Data"):
            with st.spinner("Parsing databases..."):
                parser = WhatsappParser(msgstore_path, wa_path, vcf_path)
                df = parser.get_merged_data()

                if df.empty:
                    st.error("Failed to parse data or no messages found.")
                else:
                    st.session_state["data"] = df
                    st.success(f"Loaded {len(df)} messages!")

    if "data" in st.session_state:
        df_raw = st.session_state["data"]
        msgstore_file_signature = get_file_signature(msgstore_path)

        # --- Apply Anonymisation (before any filtering/display) ---
        _anon_mode_label = st.session_state.get("cfg_anon_mode", "Off")
        _anon_mode_map = {
            "Off": "off",
            "Hash All": "hash",
            "Hash + Cut (8 chars)": "hash_cut",
            "Fully Randomised": "random",
        }
        _anon_key = _anon_mode_map.get(_anon_mode_label, "off")
        _anon_numbers = (
            st.session_state.get("cfg_anon_numbers", False) and _anon_key != "off"
        )
        _anon_map = {}
        if _anon_key != "off":
            _all_names = set()
            for _col in ("contact_name", "chat_name", "subject"):
                if _col in df_raw.columns:
                    _all_names.update(df_raw[_col].dropna().unique())
            _all_names.update(collect_nested_identity_values(df_raw))
            _anon_map = build_anon_map(frozenset(_all_names), _anon_key)
            # Only re-anonymize when mode changes or data was reloaded
            _anon_cache_key = f"_anon_df_{_anon_key}"
            if _anon_cache_key not in st.session_state or st.session_state.get(
                "_anon_data_ver"
            ) != id(st.session_state["data"]):
                st.session_state[_anon_cache_key] = apply_anon_to_df(
                    df_raw.copy(), _anon_map
                )
                st.session_state["_anon_data_ver"] = id(st.session_state["data"])
            df_raw = st.session_state[_anon_cache_key]

        with st.sidebar.expander("📅 Date Range", expanded=True):
            min_date = df_raw["timestamp"].min().date()
            max_date = df_raw["timestamp"].max().date()

            st.caption("Quick Date Filters")
            cols_q = st.columns(5)
            labels = ["3M", "6M", "1Y", "3Y", "10Y"]
            offsets = [3, 6, 12, 36, 120]

            def _set_date_range(start, end):
                st.session_state["cfg_date_range"] = [start, end]

            for i, label in enumerate(labels):
                new_start = max_date - pd.DateOffset(months=offsets[i])
                new_start = new_start.date()
                if new_start < min_date:
                    new_start = min_date
                cols_q[i].button(
                    label,
                    on_click=_set_date_range,
                    args=(new_start, max_date),
                )

            st.button(
                "Reset Date",
                on_click=_set_date_range,
                args=(min_date, max_date),
            )

            date_range = st.date_input(
                "Date Range",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date,
                key="cfg_date_range",
            )

            n_sent = len(df_raw[df_raw["from_me"] == 1])
            n_recv = len(df_raw[df_raw["from_me"] == 0])
            st.caption(
                f"**{av(n_sent, _anon_numbers):,}** sent · **{av(n_recv, _anon_numbers):,}** received"
            )
            backup_message_horizon = load_backup_message_horizon(
                msgstore_path, msgstore_file_signature
            )
            if pd.notna(backup_message_horizon):
                st.caption(
                    f"Backup message horizon: **{_format_backup_horizon(backup_message_horizon)}**"
                )

        with st.sidebar.expander("🔒 Anonymisation", expanded=False):
            anon_mode = st.selectbox(
                "Anonymisation Mode",
                ["Off", "Hash All", "Hash + Cut (8 chars)", "Fully Randomised"],
                key="cfg_anon_mode",
                help="Anonymise contact names/labels before processing. All stats and charts update normally.",
            )
            if anon_mode != "Off":
                st.checkbox(
                    "Also anonymise numeric values",
                    value=False,
                    key="cfg_anon_numbers",
                    help="Replace absolute numbers (message counts, reply times, etc.) with randomised values of similar magnitude.",
                )

        with st.sidebar.expander("🔍 Filters", expanded=True):
            exclude_groups = st.checkbox("Exclude Groups", value=False, key="cfg_ex_groups")

            exclude_archived = st.checkbox(
                "Exclude Archived Chats",
                value=False,
                key="cfg_ex_archived",
                help="Removes chats currently marked archived in msgstore.db from all tabs and stats.",
            )

            exclude_low_participation = st.checkbox(
                "Exclude Low-Participation Groups",
                value=True,
                key="cfg_ex_low_part",
                help="Excludes groups (>4 members) where you sent less than 10% of the messages.",
            )

            exclude_me = st.checkbox(
                "Exclude 'Me/You' from Charts",
                value=True,
                key="cfg_ex_me",
                help="Remove your own sent messages from Activity and Top Talkers charts.",
            )

            exclude_channels = st.checkbox(
                "Exclude Channels / Announcements",
                value=True,
                help="Removes WhatsApp Channels (@newsletter) and Status Broadcasts",
                key="cfg_ex_chan",
            )

        with st.sidebar.expander("👥 Contact Management", expanded=True):
            all_contacts = sorted(df_raw["contact_name"].unique().astype(str))

            default_fam = []
            if "cfg_fam_list" not in st.session_state:
                for candidate in ["You", "Me", "Myself", "Tú", "Yo"]:
                    match = next(
                        (c for c in all_contacts if candidate.lower() == c.lower()), None
                    )
                    if match:
                        default_fam.append(match)

            family_list = st.multiselect(
                "Select Family / Close Contacts",
                all_contacts,
                default=default_fam if "cfg_fam_list" not in st.session_state else None,
                key="cfg_fam_list",
            )

            exclude_family_global = st.checkbox(
                "Exclude Family from ALL Stats", value=False, key="cfg_ex_fam_glob"
            )
            exclude_non_contacts = st.checkbox(
                "Exclude Non-Contacts from ALL Stats", value=False, key="cfg_ex_non_con"
            )
            exclude_family_gender = st.checkbox(
                "Exclude Family from GENDER Stats Only", value=False, key="cfg_ex_fam_gend"
            )
            exclude_family_behavior = st.checkbox(
                "Exclude Family from BEHAVIORAL Stats", value=False, key="cfg_ex_fam_beh"
            )

        with st.sidebar.expander("⚙️ Behavioral Config", expanded=False):
            use_medians = st.checkbox(
                "Use Median for Stats",
                value=False,
                help="Switch between Average and Median for all statistical metrics (Reply times, write times, word counts).",
                key="cfg_use_median",
            )

            def _on_longer_stats_toggle():
                if st.session_state["cfg_long_stats"]:
                    st.session_state["cfg_reply_thresh"] = 48
                else:
                    st.session_state["cfg_reply_thresh"] = 12

            use_longer_stats = st.checkbox(
                "Use Longer Time Stats",
                value=False,
                help="Ghosting: 5 days (vs 24h), Initiation: 2 days (vs 6h), Reply threshold: 48h (vs 12h)",
                key="cfg_long_stats",
                on_change=_on_longer_stats_toggle,
            )

            reply_threshold_hours = st.slider(
                "Max Reply Delay (Hours)",
                min_value=1,
                max_value=120,
                value=12,
                help="Messages after this delay are considered new conversations, not replies.",
                key="cfg_reply_thresh",
            )

            min_word_len = st.number_input(
                "Min Word Length (Word Cloud)",
                min_value=1,
                max_value=20,
                value=4,
                key="cfg_min_word_len",
            )

            exclude_emails = st.checkbox(
                "Exclude Emails from Word Cloud", value=False, key="cfg_ex_emails"
            )

        # Apply Global Filters
        date_start = date_range[0] if len(date_range) == 2 else None
        date_end = date_range[1] if len(date_range) == 2 else None
        archived_chat_ids = load_archived_chat_ids(msgstore_path, msgstore_file_signature)
        df_base, df_group_base = apply_global_filters(
            df_raw,
            date_start,
            date_end,
            exclude_groups,
            exclude_archived,
            archived_chat_ids,
            exclude_low_participation,
            exclude_channels,
            exclude_family_global,
            tuple(family_list) if family_list else (),
            exclude_non_contacts,
        )

        # 2. View Filters (Used for General Stats where Me skews it)
        filtered_df = df_base.copy()
        if exclude_me:
            filtered_df = filtered_df[filtered_df["from_me"] == 0]

        # Update Identity Info with Phone Number attempt
        me_jid = None
        me_rows = df_raw[df_raw["from_me"] == 1]
        if not me_rows.empty:
            possible = me_rows["sender_string"].dropna().unique()
            if len(possible) > 0:
                me_jid = possible[0]

        me_display = me_jid.split("@")[0] if me_jid else "Unknown"
        if _anon_key != "off":
            _anon_fn = {
                "hash": _anon_hash,
                "hash_cut": _anon_hash_cut,
                "random": _anon_random,
            }[_anon_key]
            me_display = _anon_fn(me_display)

        st.sidebar.markdown(f"**User**: {me_display}")

        analyzer = WhatsappAnalyzer(filtered_df, use_medians=use_medians)
        full_analyzer = WhatsappAnalyzer(df_base, use_medians=use_medians)
        app_context = AppContext(
            raw_df=st.session_state["data"],
            display_df=df_raw,
            base_df=df_base,
            group_base_df=df_group_base,
            filtered_df=filtered_df,
            analyzer=analyzer,
            full_analyzer=full_analyzer,
            sidebar=SidebarState(
                paths=DataPaths(msgstore=msgstore_path, wa=wa_path, vcf=vcf_path),
                filters=FilterState(
                    date_start=date_start,
                    date_end=date_end,
                    exclude_groups=exclude_groups,
                    exclude_archived=exclude_archived,
                    exclude_low_participation=exclude_low_participation,
                    exclude_channels=exclude_channels,
                    exclude_me=exclude_me,
                    exclude_family_global=exclude_family_global,
                    exclude_non_contacts=exclude_non_contacts,
                    exclude_family_gender=exclude_family_gender,
                    exclude_family_behavior=exclude_family_behavior,
                    family_list=tuple(family_list) if family_list else (),
                ),
                anonymization=AnonymizationState(
                    mode_label=_anon_mode_label,
                    mode_key=_anon_key,
                    anonymize_numbers=_anon_numbers,
                    mapping=_anon_map,
                ),
                use_medians=use_medians,
                use_longer_stats=use_longer_stats,
                reply_threshold_hours=reply_threshold_hours,
                min_word_len=min_word_len,
                exclude_emails=exclude_emails,
            ),
            msgstore_file_signature=msgstore_file_signature,
            me_display=me_display,
        )
        return app_context


    return None
