"""Filtering helpers for the Streamlit app."""

import re

import pandas as pd
import streamlit as st


def is_number(name):
    return not bool(re.search("[a-zA-Z]", str(name)))

@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: id})
def apply_global_filters(
    _df_raw,
    date_start,
    date_end,
    exclude_groups,
    exclude_archived,
    archived_chat_ids,
    exclude_low_participation,
    exclude_channels,
    exclude_family_global,
    family_tuple,
    exclude_non_contacts,
):
    df_base = _df_raw.copy()

    if "message_type" in df_base.columns:
        df_base = df_base[df_base["message_type"] != 7]

    if date_start is not None and date_end is not None:
        mask = (df_base["timestamp"].dt.date >= date_start) & (
            df_base["timestamp"].dt.date <= date_end
        )
        df_base = df_base.loc[mask]

    df_group_base = df_base.copy()
    if "is_group" not in df_group_base.columns:
        if "raw_string" in df_group_base.columns:
            df_group_base["is_group"] = (
                df_group_base["raw_string"].astype(str).str.endswith("@g.us")
            )
        else:
            df_group_base["is_group"] = False

    if exclude_groups and "raw_string" in df_base.columns:
        is_group = df_base["raw_string"].astype(str).str.endswith("@g.us")
        df_base = df_base[~is_group]

    if exclude_archived and archived_chat_ids and "chat_row_id" in df_base.columns:
        df_base = df_base[~df_base["chat_row_id"].isin(archived_chat_ids)]
    if (
        exclude_archived
        and archived_chat_ids
        and "chat_row_id" in df_group_base.columns
    ):
        df_group_base = df_group_base[
            ~df_group_base["chat_row_id"].isin(archived_chat_ids)
        ]

    if exclude_low_participation and "raw_string" in df_group_base.columns:
        is_group_mask = (
            df_group_base["raw_string"].astype(str).str.endswith("@g.us")
        )
        group_chats = df_group_base[is_group_mask]

        groups_to_exclude = set()
        for chat_id, group_df in group_chats.groupby("chat_name"):
            unique_senders = group_df["sender_string"].nunique()
            if unique_senders > 4:
                total_messages = len(group_df)
                my_messages = group_df["from_me"].sum()
                participation_ratio = (
                    my_messages / total_messages if total_messages > 0 else 0
                )
                if participation_ratio < 0.1:
                    groups_to_exclude.add(chat_id)

        if groups_to_exclude:
            df_base = df_base[~df_base["chat_name"].isin(groups_to_exclude)]
            df_group_base = df_group_base[
                ~df_group_base["chat_name"].isin(groups_to_exclude)
            ]

    if exclude_channels and "raw_string" in df_base.columns:
        is_channel = (
            df_base["raw_string"].astype(str).str.endswith("@newsletter")
            | (df_base["raw_string"] == "status@broadcast")
            | (df_base["raw_string"] == "0@s.whatsapp.net")
        )
        df_base = df_base[~is_channel]
    if exclude_channels and "raw_string" in df_group_base.columns:
        is_channel_group = (
            df_group_base["raw_string"].astype(str).str.endswith("@newsletter")
            | (df_group_base["raw_string"] == "status@broadcast")
            | (df_group_base["raw_string"] == "0@s.whatsapp.net")
        )
        df_group_base = df_group_base[~is_channel_group]

    family_list_inner = list(family_tuple) if family_tuple else []
    if exclude_family_global and family_list_inner:
        df_base = df_base[~df_base["chat_name"].isin(family_list_inner)]
        df_group_base = df_group_base[
            ~df_group_base["chat_name"].isin(family_list_inner)
        ]
    if exclude_non_contacts:
        if "contact_name" in df_base.columns:
            mask_nums = df_base["contact_name"].apply(
                lambda name: not bool(re.search("[a-zA-Z]", str(name)))
            )
            df_base = df_base[~mask_nums]
        if "contact_name" in df_group_base.columns:
            group_mask_nums = df_group_base["contact_name"].apply(
                lambda name: not bool(re.search("[a-zA-Z]", str(name)))
            )
            df_group_base = df_group_base[~group_mask_nums]

    return df_base, df_group_base
