from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from wa_analyzer.app.db_loaders import (
    load_group_receipt_events,
    load_jid_raw_lookup,
    load_lid_jid_map,
    load_vcf_contact_lookup,
)
from wa_analyzer.app.privacy import _anon_hash, _anon_hash_cut, _anon_random, av
from wa_analyzer.app.state import AppContext


@st.fragment
def render(ctx: AppContext) -> None:
    df_group_base = ctx.df_group_base
    msgstore_path = ctx.msgstore_path
    vcf_path = ctx.vcf_path
    msgstore_file_signature = ctx.msgstore_file_signature
    _anon_key = ctx._anon_key
    _anon_numbers = ctx._anon_numbers

    st.header("👥 Group Explorer & Member Comparison")
    st.caption(
        "Deep dive into group dynamics, member activity, reply speed, and read behavior."
    )

    if "is_group" not in df_group_base.columns:
        st.info("Group metadata is unavailable in this dataset.")
    else:
        with st.spinner("Loading group data..."):
            groups_df = df_group_base[df_group_base["is_group"] == True].copy()
            groups_df = groups_df[groups_df["chat_name"].notnull()]
            lid_to_pn_map = load_lid_jid_map(
                msgstore_path, msgstore_file_signature
            )
            jid_raw_lookup = load_jid_raw_lookup(
                msgstore_path, msgstore_file_signature
            )
            vcf_contact_lookup = load_vcf_contact_lookup(vcf_path)
            lids_by_pn = {}
            for lid_id, pn_id in lid_to_pn_map.items():
                lids_by_pn.setdefault(pn_id, []).append(lid_id)

        if groups_df.empty:
            st.info("No groups found with current filters.")
        else:
            group_options = sorted(
                groups_df["chat_name"].astype(str).unique().tolist()
            )
            selected_group = st.selectbox(
                "Select Group", group_options, key="grp_selected_group"
            )

            if selected_group:
                gdf = (
                    groups_df[groups_df["chat_name"] == selected_group]
                    .copy()
                    .sort_values("timestamp")
                )

                if gdf.empty:
                    st.info("No data for this group with current filters.")
                else:
                    if "read_at" not in gdf.columns:
                        gdf["read_at"] = pd.NaT

                    # Canonicalize participants with LID -> PN mapping to avoid split identities
                    gdf["sender_jid_num"] = pd.to_numeric(
                        gdf["sender_jid_row_id"], errors="coerce"
                    )
                    gdf["canonical_sender_id"] = gdf["sender_jid_num"]
                    if lid_to_pn_map:
                        mapped_ids = (
                            gdf["sender_jid_num"].astype("Int64").map(lid_to_pn_map)
                        )
                        gdf["canonical_sender_id"] = mapped_ids.fillna(
                            gdf["sender_jid_num"]
                        )

                    gdf.loc[gdf["from_me"] == 1, "canonical_sender_id"] = -1

                    # Pick best display label per canonical ID.
                    def _label_has_letters(x):
                        return bool(re.search(r"[A-Za-zÀ-ÿ]", str(x)))

                    inbound = gdf[gdf["from_me"] == 0][
                        ["canonical_sender_id", "contact_name", "sender_string"]
                    ].copy()
                    inbound["candidate_name"] = (
                        inbound["contact_name"].fillna("").astype(str).str.strip()
                    )
                    inbound.loc[
                        inbound["candidate_name"].eq(""), "candidate_name"
                    ] = inbound["sender_string"].fillna("").astype(str)
                    inbound["has_letters"] = inbound["candidate_name"].apply(
                        _label_has_letters
                    )
                    inbound["is_unknown"] = (
                        inbound["candidate_name"]
                        .str.lower()
                        .isin(["", "unknown", "none", "nan"])
                    )

                    label_counts = (
                        inbound.groupby(
                            [
                                "canonical_sender_id",
                                "candidate_name",
                                "has_letters",
                                "is_unknown",
                            ]
                        )
                        .size()
                        .reset_index(name="count")
                        .sort_values(
                            [
                                "canonical_sender_id",
                                "has_letters",
                                "is_unknown",
                                "count",
                            ],
                            ascending=[True, False, True, False],
                        )
                    )
                    preferred_labels = (
                        label_counts.drop_duplicates(
                            subset=["canonical_sender_id"], keep="first"
                        )
                        .set_index("canonical_sender_id")["candidate_name"]
                        .to_dict()
                    )

                    # Improve unresolved labels:
                    # 1) try global non-numeric name evidence across dataset for mapped ids
                    # 2) fallback to mapped phone JID user part
                    canonical_ids = [
                        int(x)
                        for x in gdf["canonical_sender_id"].dropna().unique()
                        if int(x) != -1
                    ]
                    for cid in canonical_ids:
                        current = str(preferred_labels.get(cid, "") or "").strip()
                        needs_improve = (
                            (not current)
                            or (current.lower() in {"unknown", "none", "nan"})
                            or current.isdigit()
                        )
                        if not needs_improve:
                            continue

                        related_ids = [cid] + lids_by_pn.get(cid, [])
                        global_rows = df_group_base[
                            (df_group_base["from_me"] == 0)
                            & (
                                pd.to_numeric(
                                    df_group_base["sender_jid_row_id"],
                                    errors="coerce",
                                ).isin(related_ids)
                            )
                        ].copy()

                        if not global_rows.empty:
                            cand = (
                                global_rows["contact_name"]
                                .fillna("")
                                .astype(str)
                                .str.strip()
                            )
                            cand = cand[
                                cand.ne("")
                                & (
                                    ~cand.str.lower().isin(
                                        ["unknown", "none", "nan"]
                                    )
                                )
                                & (cand.str.contains(r"[A-Za-zÀ-ÿ]", regex=True))
                            ]
                            if not cand.empty:
                                best = cand.value_counts().index[0]
                                preferred_labels[cid] = best
                                continue

                        pn_raw = jid_raw_lookup.get(cid, "")
                        if pn_raw:
                            phone_part = str(pn_raw).split("@")[0]
                            digits = "".join(filter(str.isdigit, phone_part))
                            if digits in vcf_contact_lookup:
                                preferred_labels[cid] = vcf_contact_lookup[digits]
                            elif (
                                len(digits) > 9
                                and digits[-9:] in vcf_contact_lookup
                            ):
                                preferred_labels[cid] = vcf_contact_lookup[
                                    digits[-9:]
                                ]
                            else:
                                preferred_labels[cid] = phone_part

                    gdf["sender_label"] = gdf["canonical_sender_id"].map(
                        preferred_labels
                    )
                    gdf.loc[gdf["from_me"] == 1, "sender_label"] = "You"
                    missing_label = gdf["sender_label"].isna()
                    gdf.loc[missing_label, "sender_label"] = (
                        gdf.loc[missing_label, "contact_name"]
                        .fillna(gdf.loc[missing_label, "sender_string"])
                        .fillna("Unknown")
                        .astype(str)
                    )
                    gdf["sender_label"] = gdf["sender_label"].str.replace(
                        r"@lid$", "", regex=True
                    )
                    gdf["sender_label"] = gdf["sender_label"].str.replace(
                        r"@s\\.whatsapp\\.net$", "", regex=True
                    )
                    gdf["sender_label"] = gdf["sender_label"].astype(str)
                    if _anon_key != "off":
                        _anon_fn_grp = {
                            "hash": _anon_hash,
                            "hash_cut": _anon_hash_cut,
                            "random": _anon_random,
                        }[_anon_key]
                        gdf["sender_label"] = gdf["sender_label"].map(
                            lambda v: v if v == "You" else _anon_fn_grp(str(v))
                        )
                    gdf["word_count"] = (
                        gdf["text_data"]
                        .fillna("")
                        .astype(str)
                        .str.split()
                        .str.len()
                    )

                    participants = sorted(
                        gdf[gdf["sender_label"] != "You"]["sender_label"]
                        .unique()
                        .tolist()
                    )
                    active_days = gdf["timestamp"].dt.date.nunique()
                    total_msgs = len(gdf)
                    total_words = int(gdf["word_count"].sum())
                    my_msgs = int((gdf["sender_label"] == "You").sum())
                    my_share = (my_msgs / total_msgs * 100) if total_msgs > 0 else 0

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric(
                        "Group Messages", f"{av(total_msgs, _anon_numbers):,}"
                    )
                    m2.metric("Participants", f"{len(participants):,}")
                    m3.metric("Active Days", f"{av(active_days, _anon_numbers):,}")
                    m4.metric("My Messages", f"{av(my_msgs, _anon_numbers):,}")
                    m5.metric("My Share", f"{my_share:.1f}%")

                    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
                    top_members_n = cfg1.slider(
                        "Top Members", 3, 25, 10, key="grp_top_members_n"
                    )
                    grp_reply_thresh_h = cfg2.slider(
                        "Max Reply Delay (hours)",
                        1,
                        72,
                        12,
                        key="grp_reply_thresh_h",
                    )
                    grp_show_lines = cfg3.checkbox(
                        "Show as Lines", value=True, key="grp_show_lines"
                    )
                    grp_show_cumulative = cfg4.checkbox(
                        "Show Cumulative", value=False, key="grp_show_cumulative"
                    )

                    # ---- Member Rankings ----
                    member_msg = (
                        gdf["sender_label"].value_counts().rename("Messages")
                    )
                    member_words = (
                        gdf.groupby("sender_label")["word_count"]
                        .sum()
                        .rename("Words")
                    )
                    avg_words = (
                        gdf.groupby("sender_label")["word_count"]
                        .mean()
                        .rename("Avg Words/Msg")
                    )

                    # Reply speed: member replies after someone else in the same group.
                    gdf["prev_sender"] = gdf["sender_label"].shift(1)
                    gdf["prev_timestamp"] = gdf["timestamp"].shift(1)
                    gdf["reply_seconds"] = (
                        gdf["timestamp"] - gdf["prev_timestamp"]
                    ).dt.total_seconds()

                    reply_mask = (
                        (gdf["sender_label"] != gdf["prev_sender"])
                        & (gdf["reply_seconds"] >= 0)
                        & (gdf["reply_seconds"] <= grp_reply_thresh_h * 3600)
                    )
                    reply_df = gdf[reply_mask].copy()
                    reply_avg = (
                        reply_df.groupby("sender_label")["reply_seconds"].mean()
                        / 60
                    ).rename("Avg Reply (min)")
                    reply_events = (
                        reply_df.groupby("sender_label")
                        .size()
                        .rename("Reply Events")
                    )

                    # How long I take to read each member's messages.
                    read_delay_df = gdf[
                        (gdf["sender_label"] != "You") & gdf["read_at"].notnull()
                    ].copy()
                    read_delay_df["my_read_seconds"] = (
                        read_delay_df["read_at"] - read_delay_df["timestamp"]
                    ).dt.total_seconds()
                    read_delay_df.loc[
                        read_delay_df["my_read_seconds"].between(-60, 0),
                        "my_read_seconds",
                    ] = 0
                    read_delay_df = read_delay_df[
                        (read_delay_df["my_read_seconds"] >= 0)
                        & (read_delay_df["my_read_seconds"] <= 7 * 24 * 3600)
                    ]
                    my_read_avg = (
                        read_delay_df.groupby("sender_label")[
                            "my_read_seconds"
                        ].mean()
                        / 60
                    ).rename("Avg My Read (min)")

                    # How long each member takes to read MY group messages (per-recipient receipts).
                    group_jids = gdf["raw_string"].dropna().astype(str)
                    group_jid = group_jids.iloc[0] if not group_jids.empty else None
                    receipt_events = load_group_receipt_events(
                        msgstore_path, group_jid, msgstore_file_signature
                    )

                    member_read_avg = pd.Series(dtype="float64")
                    member_read_events = pd.Series(dtype="int64")
                    if not receipt_events.empty:
                        jid_to_name = (
                            gdf[["sender_string", "sender_label"]]
                            .dropna(subset=["sender_string", "sender_label"])
                            .drop_duplicates("sender_string")
                            .set_index("sender_string")["sender_label"]
                            .to_dict()
                        )

                        receipt_events = receipt_events.copy()
                        receipt_events["member"] = receipt_events["reader_jid"].map(
                            jid_to_name
                        )
                        missing_mask = receipt_events["member"].isna()
                        _fallback_labels = (
                            receipt_events.loc[missing_mask, "reader_jid"]
                            .astype(str)
                            .str.split("@")
                            .str[0]
                        )
                        if _anon_key != "off":
                            _anon_fn_rcpt = {
                                "hash": _anon_hash,
                                "hash_cut": _anon_hash_cut,
                                "random": _anon_random,
                            }[_anon_key]
                            _fallback_labels = _fallback_labels.map(
                                lambda v: _anon_fn_rcpt(str(v))
                            )
                        receipt_events.loc[missing_mask, "member"] = (
                            _fallback_labels
                        )

                        receipt_events["their_read_seconds"] = (
                            receipt_events["read_timestamp"]
                            - receipt_events["msg_timestamp"]
                        ).dt.total_seconds()
                        receipt_events.loc[
                            receipt_events["their_read_seconds"].between(-60, 0),
                            "their_read_seconds",
                        ] = 0
                        receipt_events = receipt_events[
                            (receipt_events["their_read_seconds"] >= 0)
                            & (
                                receipt_events["their_read_seconds"]
                                <= 7 * 24 * 3600
                            )
                        ]
                        receipt_events = receipt_events[
                            receipt_events["member"].astype(str) != "You"
                        ]

                        member_read_avg = (
                            receipt_events.groupby("member")[
                                "their_read_seconds"
                            ].mean()
                            / 60
                        ).rename("Avg Their Read (min)")
                        member_read_events = (
                            receipt_events.groupby("member")
                            .size()
                            .rename("Read Events")
                        )

                    rankings = pd.DataFrame({"Messages": member_msg})
                    rankings = rankings.join(member_words, how="left")
                    rankings = rankings.join(avg_words, how="left")
                    rankings = rankings.join(reply_avg, how="left")
                    rankings = rankings.join(reply_events, how="left")
                    rankings = rankings.join(my_read_avg, how="left")
                    if not member_read_avg.empty:
                        rankings = rankings.join(member_read_avg, how="left")
                        rankings = rankings.join(member_read_events, how="left")
                    else:
                        rankings["Avg Their Read (min)"] = np.nan
                        rankings["Read Events"] = 0

                    rankings["Message Share %"] = (
                        rankings["Messages"] / total_msgs * 100
                    ).round(2)
                    rankings["Word Share %"] = (
                        rankings["Words"] / max(total_words, 1) * 100
                    ).round(2)
                    rankings = rankings.fillna(
                        {
                            "Words": 0,
                            "Avg Words/Msg": 0,
                            "Avg Reply (min)": np.nan,
                            "Reply Events": 0,
                            "Avg My Read (min)": np.nan,
                            "Avg Their Read (min)": np.nan,
                            "Read Events": 0,
                        }
                    )
                    rankings = rankings.sort_values("Messages", ascending=False)
                    rankings.index = rankings.index.map(str)
                    rankings.index.name = "sender_label"

                    unresolved_ids = [
                        m for m in rankings.index.tolist() if str(m).isdigit()
                    ]
                    if unresolved_ids:
                        st.caption(
                            "Some members appear as numeric IDs (WhatsApp LID/contact-mapping limitation in backup data)."
                        )

                    st.subheader("🏅 Member Rankings")
                    st.caption(
                        "Includes message/word share, reply speed, and read-time metrics."
                    )
                    st.dataframe(rankings, width="stretch")

                    # ---- Activity Comparison Over Time ----
                    st.subheader("📈 Member Activity Over Time")
                    recent_cutoff = gdf["timestamp"].max() - pd.DateOffset(months=6)
                    recent_activity = gdf[gdf["timestamp"] >= recent_cutoff][
                        "sender_label"
                    ].value_counts()
                    recent_default = recent_activity.head(
                        top_members_n
                    ).index.tolist()
                    if "You" in rankings.index and "You" not in recent_default:
                        recent_default = ["You"] + recent_default
                    all_members = rankings.index.tolist()

                    selected_timeline_members = st.multiselect(
                        "Members to Plot",
                        all_members,
                        default=recent_default[
                            : min(len(recent_default), len(all_members))
                        ],
                        key="grp_timeline_members",
                    )

                    if not selected_timeline_members:
                        selected_timeline_members = rankings.head(
                            top_members_n
                        ).index.tolist()

                    timeline_df = gdf[
                        gdf["sender_label"].isin(selected_timeline_members)
                    ].copy()
                    timeline = (
                        timeline_df.set_index("timestamp")
                        .groupby("sender_label")
                        .resample("ME", include_groups=False)
                        .size()
                        .unstack(level=0)
                        .fillna(0)
                    )
                    if grp_show_cumulative:
                        timeline = timeline.cumsum()

                    if not timeline.empty:
                        plot_func_group = px.line if grp_show_lines else px.area
                        fig_group_timeline = plot_func_group(
                            timeline,
                            x=timeline.index,
                            y=timeline.columns,
                            title=f"Monthly Volume by Member ({'Cumulative' if grp_show_cumulative else 'Monthly'})",
                            labels={
                                "value": "Messages",
                                "index": "Month",
                                "variable": "Member",
                            },
                        )
                        st.plotly_chart(fig_group_timeline, width="stretch")
                    else:
                        st.info(
                            "Not enough data to plot member activity over time."
                        )

                    # ---- Hourly Comparison ----
                    st.subheader("🕒 Hourly Member Activity")
                    hourly_df = gdf[
                        gdf["sender_label"].isin(selected_timeline_members)
                    ].copy()
                    hourly = (
                        hourly_df.groupby(
                            [hourly_df["timestamp"].dt.hour, "sender_label"]
                        )
                        .size()
                        .unstack(fill_value=0)
                    )
                    if grp_show_cumulative and not hourly.empty:
                        hourly = hourly.cumsum()

                    if not hourly.empty:
                        fig_hourly_group = px.line(
                            hourly,
                            x=hourly.index,
                            y=hourly.columns,
                            markers=True,
                            title=f"Hourly Activity by Member ({'Cumulative' if grp_show_cumulative else 'Per Hour'})",
                            labels={
                                "value": "Messages",
                                "index": "Hour",
                                "variable": "Member",
                            },
                        )
                        st.plotly_chart(fig_hourly_group, width="stretch")
                    else:
                        st.info("Not enough data to plot hourly member activity.")

                    # ---- Reply + Read Rankings ----
                    rr1, rr2 = st.columns(2)

                    with rr1:
                        st.write("**⚡ Fastest Repliers**")
                        if (
                            "Avg Reply (min)" in rankings.columns
                            and rankings["Avg Reply (min)"].notna().any()
                        ):
                            rep_rank = (
                                rankings.dropna(subset=["Avg Reply (min)"])
                                .sort_values("Avg Reply (min)")
                                .head(10)
                            )
                            fig_rep_fast = px.bar(
                                rep_rank.reset_index(),
                                x="Avg Reply (min)",
                                y="sender_label",
                                orientation="h",
                                title="Lowest Avg Reply Delay",
                                labels={"sender_label": "Member"},
                            )
                            rep_order = rep_rank.index.map(str).tolist()[::-1]
                            fig_rep_fast.update_layout(
                                yaxis={
                                    "type": "category",
                                    "categoryorder": "array",
                                    "categoryarray": rep_order,
                                }
                            )
                            st.plotly_chart(fig_rep_fast, width="stretch")
                        else:
                            st.caption("Not enough reply events.")

                    with rr2:
                        st.write("**👀 Fastest Readers of My Messages**")
                        if (
                            "Avg Their Read (min)" in rankings.columns
                            and rankings["Avg Their Read (min)"].notna().any()
                        ):
                            read_rank = (
                                rankings.dropna(subset=["Avg Their Read (min)"])
                                .sort_values("Avg Their Read (min)")
                                .head(10)
                            )
                            fig_read_fast = px.bar(
                                read_rank.reset_index(),
                                x="Avg Their Read (min)",
                                y="sender_label",
                                orientation="h",
                                title="Lowest Avg Read Delay (Per Member)",
                                labels={"sender_label": "Member"},
                            )
                            read_order = read_rank.index.map(str).tolist()[::-1]
                            fig_read_fast.update_layout(
                                yaxis={
                                    "type": "category",
                                    "categoryorder": "array",
                                    "categoryarray": read_order,
                                }
                            )
                            st.plotly_chart(fig_read_fast, width="stretch")
                        else:
                            st.caption(
                                "No per-member read receipts found for this group."
                            )

                    # ---- Response Time Distribution (Multi-Member) ----
                    st.subheader("⏱️ Response Time Distribution")
                    focus_members = rankings.index.tolist()
                    if focus_members and not reply_df.empty:
                        default_dist_members = focus_members[
                            : min(5, len(focus_members))
                        ]
                        selected_dist_members = st.multiselect(
                            "Members for Distribution",
                            focus_members,
                            default=default_dist_members,
                            key="grp_dist_members",
                        )

                        if selected_dist_members:
                            dist_df = reply_df[
                                reply_df["sender_label"].isin(selected_dist_members)
                            ].copy()
                            if not dist_df.empty:
                                max_seconds = grp_reply_thresh_h * 3600
                                segments = [
                                    (60, "<1m"),
                                    (300, "1-5m"),
                                    (900, "5-15m"),
                                    (3600, "15m-1h"),
                                    (14400, "1h-4h"),
                                    (28800, "4h-8h"),
                                    (max_seconds, f">8h to {grp_reply_thresh_h}h"),
                                ]
                                bins = [0]
                                labels = []
                                for edge, label in segments:
                                    edge = min(edge, max_seconds)
                                    if edge > bins[-1]:
                                        bins.append(edge)
                                        labels.append(label)

                                dist_df["bucket"] = pd.cut(
                                    dist_df["reply_seconds"],
                                    bins=bins,
                                    labels=labels,
                                    include_lowest=True,
                                    duplicates="drop",
                                )

                                dist_counts = (
                                    dist_df.groupby(
                                        ["sender_label", "bucket"], observed=False
                                    )
                                    .size()
                                    .reset_index(name="count")
                                )
                                dist_counts["sender_label"] = dist_counts[
                                    "sender_label"
                                ].astype(str)
                                fig_dist = px.bar(
                                    dist_counts,
                                    x="sender_label",
                                    y="count",
                                    color="bucket",
                                    barmode="stack",
                                    title="Reply Delay Buckets by Member",
                                    labels={
                                        "sender_label": "Member",
                                        "count": "Replies",
                                        "bucket": "Delay Bucket",
                                    },
                                )
                                fig_dist.update_layout(xaxis={"type": "category"})
                                st.plotly_chart(fig_dist, width="stretch")

                                summary = (
                                    dist_df.groupby("sender_label")["reply_seconds"]
                                    .mean()
                                    .div(60)
                                    .rename("Avg Reply (min)")
                                    .to_frame()
                                )
                                summary["Group Avg (min)"] = (
                                    reply_df["reply_seconds"].mean() / 60
                                )
                                summary["Delta vs Group (min)"] = (
                                    summary["Avg Reply (min)"]
                                    - summary["Group Avg (min)"]
                                )
                                st.dataframe(
                                    summary.sort_values("Avg Reply (min)"),
                                    width="stretch",
                                )
                            else:
                                st.caption(
                                    "Not enough reply events for selected members."
                                )
                        else:
                            st.caption("Select at least one member.")
                    else:
                        st.caption(
                            "Not enough reply events for distribution analysis."
                        )

                    # ---- Composition Comparison ----
                    st.subheader("🧩 Message Composition by Member")
                    members_for_comp = rankings.index.tolist()
                    default_members = members_for_comp[
                        : min(6, len(members_for_comp))
                    ]
                    selected_comp_members = st.multiselect(
                        "Members for Composition Chart",
                        members_for_comp,
                        default=default_members,
                        key="grp_comp_members",
                    )

                    def categorize_group_msg(row):
                        mime = str(row.get("mime_type", ""))
                        if pd.isna(mime) or mime in ["", "None"]:
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

                    gdf["type_category"] = gdf.apply(categorize_group_msg, axis=1)

                    if selected_comp_members:
                        comp_df = (
                            gdf[gdf["sender_label"].isin(selected_comp_members)]
                            .groupby(["sender_label", "type_category"])
                            .size()
                            .reset_index(name="count")
                        )
                        if not comp_df.empty:
                            fig_comp = px.bar(
                                comp_df,
                                x="sender_label",
                                y="count",
                                color="type_category",
                                barmode="stack",
                                title="Message Type Mix by Member",
                                labels={
                                    "sender_label": "Member",
                                    "count": "Messages",
                                    "type_category": "Type",
                                },
                            )
                            st.plotly_chart(fig_comp, width="stretch")
                        else:
                            st.caption(
                                "No composition data for the selected members."
                            )
                    else:
                        st.caption("Select at least one member.")

                    # ---- Word Cloud Comparison ----
                    st.subheader("📝 Member Word Cloud Comparison")
                    all_member_choices = rankings.index.tolist()
                    if len(all_member_choices) >= 1:
                        wc1, wc2 = st.columns(2)
                        member_a = wc1.selectbox(
                            "Member A", all_member_choices, key="grp_wc_member_a"
                        )
                        member_b = wc2.selectbox(
                            "Member B",
                            all_member_choices,
                            index=min(1, len(all_member_choices) - 1),
                            key="grp_wc_member_b",
                        )

                        if st.button(
                            "Generate Group Word Clouds", key="grp_generate_wc"
                        ):
                            wc_col1, wc_col2 = st.columns(2)
                            for col, member in [
                                (wc_col1, member_a),
                                (wc_col2, member_b),
                            ]:
                                member_text = " ".join(
                                    gdf[gdf["sender_label"] == member]["text_data"]
                                    .dropna()
                                    .astype(str)
                                    .tolist()
                                )
                                with col:
                                    st.caption(member)
                                    if member_text.strip():
                                        wc = WordCloud(
                                            width=450,
                                            height=300,
                                            background_color="white",
                                        ).generate(member_text)
                                        plt.figure(figsize=(5, 4))
                                        plt.imshow(wc, interpolation="bilinear")
                                        plt.axis("off")
                                        st.pyplot(plt)
                                    else:
                                        st.caption("No text data.")

                    # ---- Recent Messages ----
                    st.subheader("💬 Recent Group Messages")
                    msg_cols = ["timestamp", "sender_label", "text_data"]
                    if "mime_type" in gdf.columns:
                        msg_cols.append("mime_type")
                    st.dataframe(
                        gdf[msg_cols]
                        .sort_values("timestamp", ascending=False)
                        .head(30),
                        width="stretch",
                    )
