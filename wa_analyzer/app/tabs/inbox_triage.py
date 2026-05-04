from __future__ import annotations


import pandas as pd
import plotly.express as px
import streamlit as st

from wa_analyzer.app.db_loaders import (
    _format_backup_horizon,
    _format_inbox_gender,
    build_unanswered_chats,
    build_unread_chats_for_context,
    load_backup_message_horizon,
)
from wa_analyzer.app.privacy import _anon_hash, _anon_hash_cut, _anon_random, av
from wa_analyzer.app.state import AppContext


@st.fragment
def render(ctx: AppContext) -> None:
    df_base = ctx.df_base
    msgstore_path = ctx.msgstore_path
    msgstore_file_signature = ctx.msgstore_file_signature
    _anon_key = ctx._anon_key
    _anon_numbers = ctx._anon_numbers

    st.header("📥 Inbox Triage")
    st.caption(
        "Unread counts come from WhatsApp chat counters. Needs-reply chats are trailing incoming bursts since your last sent message."
    )

    def _prepare_inbox_display(source_df):
        display_df = source_df.copy()
        if _anon_key == "off" or display_df.empty or "number" not in display_df:
            return display_df

        anon_fn = {
            "hash": _anon_hash,
            "hash_cut": _anon_hash_cut,
            "random": _anon_random,
        }[_anon_key]

        def _anon_number(value):
            if value is None or pd.isna(value) or str(value).strip() == "":
                return value
            return anon_fn(str(value))

        display_df["number"] = display_df["number"].map(_anon_number)
        return display_df

    def _add_inbox_split_label(source_df, split_by):
        split_df = source_df.copy()
        if split_df.empty:
            split_df["split_label"] = pd.Series(dtype=str)
            return split_df

        if split_by == "Gender":
            gender_values = (
                split_df["gender"]
                if "gender" in split_df.columns
                else pd.Series("unknown", index=split_df.index)
            )
            split_df["split_label"] = gender_values.fillna("unknown").map(
                _format_inbox_gender
            )
        else:
            type_values = (
                split_df["type"]
                if "type" in split_df.columns
                else pd.Series("Unknown", index=split_df.index)
            )
            split_df["split_label"] = type_values.fillna("Unknown")
        return split_df

    with st.spinner("Reading inbox state..."):
        backup_latest = load_backup_message_horizon(
            msgstore_path, msgstore_file_signature
        )
        unread_df = build_unread_chats_for_context(
            df_base, msgstore_path, msgstore_file_signature
        )
        unanswered_df = build_unanswered_chats(df_base, backup_latest)

    if unread_df.empty and unanswered_df.empty:
        st.warning("No unread or needs-reply data could be loaded from msgstore.")
        return

    if pd.notna(backup_latest):
        st.caption(
            f"Backup message horizon: {_format_backup_horizon(backup_latest)}"
        )

    overlap_ids = (
        set(unread_df["chat_id"]).intersection(unanswered_df["chat_id"])
        if not unread_df.empty and not unanswered_df.empty
        else set()
    )
    if not unread_df.empty:
        unread_df = unread_df.copy()
        unread_df["needs_reply"] = unread_df["chat_id"].isin(overlap_ids)
    if not unanswered_df.empty:
        unanswered_df = unanswered_df.copy()
        unanswered_df["is_unread"] = unanswered_df["chat_id"].isin(overlap_ids)

    unread_messages_total = (
        int(unread_df["unread_messages"].sum()) if not unread_df.empty else 0
    )
    direct_unread_total = (
        int(unread_df.loc[unread_df["type"] == "Individual", "unread_messages"].sum())
        if not unread_df.empty
        else 0
    )
    manual_unread_total = (
        int((unread_df["unread_messages_raw"] < 0).sum())
        if not unread_df.empty
        else 0
    )
    (
        int(unanswered_df["words_since_reply"].sum())
        if not unanswered_df.empty
        else 0
    )
    unanswered_messages_total = (
        int(unanswered_df["messages_since_reply"].sum())
        if not unanswered_df.empty
        else 0
    )
    question_chats = (
        int(unanswered_df["has_question"].sum()) if not unanswered_df.empty else 0
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Unread Chats", av(len(unread_df), _anon_numbers))
    k2.metric("Unread Messages", f"{av(unread_messages_total, _anon_numbers):,}")
    k3.metric("Direct Unread", f"{av(direct_unread_total, _anon_numbers):,}")
    k4.metric("Marked Unread", av(manual_unread_total, _anon_numbers))
    k5.metric("Needs Reply", av(len(unanswered_df), _anon_numbers))
    k6.metric("Unread + Reply", av(len(overlap_ids), _anon_numbers))

    unread_tab, unanswered_tab, stats_tab = st.tabs(
        ["Unread Chats", "Needs Reply", "Stats"]
    )

    with unread_tab:
        st.subheader("Unread Chats")
        if unread_df.empty:
            st.info("No unread chats found in the current msgstore counters.")
        else:
            sort_choice = st.selectbox(
                "Sort unread chats by",
                ["Last message (newest)", "Unread messages", "Chat"],
                key="inbox_unread_sort",
            )
            unread_split = st.selectbox(
                "Break unread charts by",
                ["Type", "Gender"],
                key="inbox_unread_split",
            )
            unread_sorted = unread_df.copy()
            if sort_choice == "Unread messages":
                unread_sorted = unread_sorted.sort_values(
                    ["unread_messages", "unread_rows", "last_message_at"],
                    ascending=[False, False, False],
                )
            elif sort_choice == "Chat":
                unread_sorted = unread_sorted.sort_values("chat")
            else:
                unread_sorted = unread_sorted.sort_values(
                    "last_message_at", ascending=False
                )

            unread_display = _prepare_inbox_display(unread_sorted)
            unread_display["unread_count_display"] = unread_display.apply(
                lambda row: "Marked unread"
                if row["unread_messages_raw"] < 0
                else str(int(row["unread_messages"])),
                axis=1,
            )
            unread_display["last_message_at"] = pd.to_datetime(
                unread_display["last_message_at"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M")
            unread_display = unread_display.rename(
                columns={
                    "chat": "Chat",
                    "type": "Type",
                    "gender": "Gender",
                    "contact_name": "Contact",
                    "number": "Number",
                    "unread_count_display": "Unread",
                    "unread_rows": "Unread rows",
                    "unread_reactions": "Unread reactions",
                    "unread_comments": "Unread comments",
                    "unread_missed_calls": "Unread missed calls",
                    "unread_state": "Unread status",
                    "last_message_at": "Last message",
                    "needs_reply": "Needs reply",
                    "archived": "Archived",
                }
            )
            unread_display["Gender"] = unread_display["Gender"].map(
                _format_inbox_gender
            )
            st.dataframe(
                unread_display[
                    [
                        "Chat",
                        "Type",
                        "Gender",
                        "Contact",
                        "Number",
                        "Unread",
                        "Unread status",
                        "Unread rows",
                        "Unread reactions",
                        "Unread comments",
                        "Unread missed calls",
                        "Needs reply",
                        "Last message",
                        "Archived",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

            chart_l, chart_r = st.columns(2)
            unread_chart_df = _add_inbox_split_label(unread_df, unread_split)
            split_name = "gender" if unread_split == "Gender" else "type"
            with chart_l:
                unread_by_split = (
                    unread_chart_df.groupby("split_label", as_index=False)
                    .size()
                    .rename(columns={"size": "chats"})
                )
                fig = px.pie(
                    unread_by_split,
                    names="split_label",
                    values="chats",
                    title=f"Unread chats by {split_name}",
                )
                st.plotly_chart(fig, width="stretch")

            with chart_r:
                unread_messages_by_split = unread_chart_df.groupby(
                    "split_label", as_index=False
                )["unread_messages"].sum()
                fig = px.pie(
                    unread_messages_by_split,
                    names="split_label",
                    values="unread_messages",
                    title=f"Unread message count by {split_name}",
                )
                st.plotly_chart(fig, width="stretch")

            top_unread = unread_display[
                pd.to_numeric(unread_display["Unread"], errors="coerce").fillna(0)
                > 0
            ].copy()
            top_unread["Unread numeric"] = pd.to_numeric(
                top_unread["Unread"], errors="coerce"
            ).fillna(0)
            top_unread = top_unread.nlargest(20, "Unread numeric")
            top_unread["Gender"] = top_unread.get("Gender", "unknown").map(
                _format_inbox_gender
            )
            color_col = "Gender" if unread_split == "Gender" else "Type"
            fig = px.bar(
                top_unread,
                x="Unread numeric",
                y="Chat",
                color=color_col,
                orientation="h",
                title="Top unread chats",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

    with unanswered_tab:
        st.subheader("Other Chats That Haven't Been Answered")
        st.caption(
            "Included when the latest incoming burst has 5+ messages, more than 30 words, or at least one question mark."
        )

        if unanswered_df.empty:
            st.info("No chats currently match the needs-reply rules.")
        else:
            sort_choice = st.selectbox(
                "Sort needs-reply chats by",
                ["Attention score", "Last message (newest)", "Words", "Messages"],
                key="inbox_unanswered_sort",
            )
            unanswered_sorted = unanswered_df.copy()
            if sort_choice == "Last message (newest)":
                unanswered_sorted = unanswered_sorted.sort_values(
                    "latest_message_at", ascending=False
                )
            elif sort_choice == "Words":
                unanswered_sorted = unanswered_sorted.sort_values(
                    "words_since_reply", ascending=False
                )
            elif sort_choice == "Messages":
                unanswered_sorted = unanswered_sorted.sort_values(
                    "messages_since_reply", ascending=False
                )
            else:
                unanswered_sorted = unanswered_sorted.sort_values(
                    ["attention_score", "latest_message_at"],
                    ascending=[False, False],
                )

            unanswered_display = _prepare_inbox_display(unanswered_sorted)
            table_df = unanswered_display.copy()
            table_df["waiting_hours_in_backup"] = table_df[
                "waiting_hours_in_backup"
            ].round(1)
            table_df["latest_message_at"] = pd.to_datetime(
                table_df["latest_message_at"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M")
            table_df = table_df.rename(
                columns={
                    "chat": "Chat",
                    "type": "Type",
                    "gender": "Gender",
                    "contact_name": "Contact",
                    "number": "Number",
                    "messages_since_reply": "Messages",
                    "words_since_reply": "Words",
                    "question_marks": "?",
                    "waiting_hours_in_backup": "Hours waiting",
                    "attention_score": "Attention score",
                    "latest_message_at": "Latest message",
                    "reason": "Why included",
                    "preview": "Preview",
                    "is_unread": "Unread too",
                }
            )
            table_df["Gender"] = table_df["Gender"].map(_format_inbox_gender)
            st.dataframe(
                table_df[
                    [
                        "Chat",
                        "Type",
                        "Gender",
                        "Contact",
                        "Number",
                        "Messages",
                        "Words",
                        "?",
                        "Hours waiting",
                        "Attention score",
                        "Latest message",
                        "Unread too",
                        "Why included",
                        "Preview",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

    with stats_tab:
        st.subheader("Inbox Pressure Stats")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "Unanswered Messages",
            f"{av(unanswered_messages_total, _anon_numbers):,}",
        )
        s2.metric("Question Chats", av(question_chats, _anon_numbers))
        stale_chats = (
            int((unanswered_df["waiting_hours_in_backup"] >= 24).sum())
            if not unanswered_df.empty
            else 0
        )
        s3.metric("Waiting 24h+", av(stale_chats, _anon_numbers))
        overlap = (
            len(set(unread_df["chat_id"]).intersection(unanswered_df["chat_id"]))
            if not unread_df.empty and not unanswered_df.empty
            else 0
        )
        s4.metric("Unread + Needs Reply", av(overlap, _anon_numbers))

        if unanswered_df.empty:
            st.info("Needs-reply charts will appear once matching chats exist.")
            return

        unanswered_display = _prepare_inbox_display(unanswered_df)
        unanswered_split = st.selectbox(
            "Break needs-reply charts by",
            ["Type", "Gender"],
            key="inbox_unanswered_split",
        )
        unanswered_chart_df = _add_inbox_split_label(
            unanswered_df, unanswered_split
        )
        unanswered_split_name = (
            "gender" if unanswered_split == "Gender" else "type"
        )

        if overlap:
            st.subheader("Unread Chats That Also Need Reply")
            overlap_df = unread_df[unread_df["chat_id"].isin(overlap_ids)].merge(
                unanswered_df[
                    [
                        "chat_id",
                        "messages_since_reply",
                        "words_since_reply",
                        "question_marks",
                        "attention_score",
                        "reason",
                    ]
                ],
                on="chat_id",
                how="left",
            )
            overlap_display = _prepare_inbox_display(
                overlap_df.sort_values("last_message_at", ascending=False)
            )
            overlap_display["unread_count_display"] = overlap_display.apply(
                lambda row: "Marked unread"
                if row["unread_messages_raw"] < 0
                else str(int(row["unread_messages"])),
                axis=1,
            )
            overlap_display["last_message_at"] = pd.to_datetime(
                overlap_display["last_message_at"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M")
            overlap_display = overlap_display.rename(
                columns={
                    "chat": "Chat",
                    "type": "Type",
                    "gender": "Gender",
                    "number": "Number",
                    "unread_count_display": "Unread",
                    "messages_since_reply": "Needs-reply messages",
                    "words_since_reply": "Needs-reply words",
                    "question_marks": "?",
                    "attention_score": "Attention score",
                    "last_message_at": "Last message",
                    "reason": "Why included",
                }
            )
            overlap_display["Gender"] = overlap_display["Gender"].map(
                _format_inbox_gender
            )
            st.dataframe(
                overlap_display[
                    [
                        "Chat",
                        "Type",
                        "Gender",
                        "Number",
                        "Unread",
                        "Needs-reply messages",
                        "Needs-reply words",
                        "?",
                        "Attention score",
                        "Last message",
                        "Why included",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        chart_a, chart_b = st.columns(2)
        with chart_a:
            by_split = (
                unanswered_chart_df.groupby("split_label", as_index=False)
                .size()
                .rename(columns={"size": "chats"})
            )
            fig = px.pie(
                by_split,
                names="split_label",
                values="chats",
                title=f"Needs-reply chats by {unanswered_split_name}",
            )
            st.plotly_chart(fig, width="stretch")

        with chart_b:
            words_by_split = unanswered_chart_df.groupby(
                "split_label", as_index=False
            )["words_since_reply"].sum()
            fig = px.pie(
                words_by_split,
                names="split_label",
                values="words_since_reply",
                title=f"Unanswered words by {unanswered_split_name}",
            )
            st.plotly_chart(fig, width="stretch")

        scatter_df = _add_inbox_split_label(unanswered_display, unanswered_split)
        fig = px.scatter(
            scatter_df,
            x="messages_since_reply",
            y="words_since_reply",
            size="attention_score",
            color="split_label",
            hover_name="chat",
            hover_data={
                "question_marks": True,
                "waiting_hours_in_backup": ":.1f",
                "attention_score": ":.1f",
            },
            title="Unanswered burst size vs word load",
            labels={
                "messages_since_reply": "Messages since your last reply",
                "words_since_reply": "Words since your last reply",
            },
        )
        st.plotly_chart(fig, width="stretch")

        top_pressure = _add_inbox_split_label(
            unanswered_display.nlargest(20, "attention_score"), unanswered_split
        )
        fig = px.bar(
            top_pressure,
            x="attention_score",
            y="chat",
            color="split_label",
            orientation="h",
            title="Highest attention score",
            labels={"attention_score": "Attention score", "chat": "Chat"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Attention score blends unanswered words, message count, question marks, and waiting time within the backup window."
        )
