"""Cached database and inbox helper loaders for the Streamlit app."""

import hashlib
import os
import re
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

from wa_analyzer.src.utils import Utils, get_user_timezone, to_user_datetime
from wa_analyzer.src.parser import WhatsappParser


def get_file_signature(path):
    """Return a stable cache-busting signature for files that may be replaced in place."""
    if not path or not os.path.exists(path):
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def load_group_receipt_events(msgstore_path, group_jid, file_signature=None):
    """
    Load per-recipient read events for outgoing messages in a specific group.
    Returns one earliest read event per (message_row_id, reader_jid).
    """
    cols = ["message_row_id", "reader_jid", "msg_timestamp", "read_timestamp"]

    if not msgstore_path or not group_jid or not os.path.exists(msgstore_path):
        return pd.DataFrame(columns=cols)

    # Newer backups usually store per-member reads in receipt_user.
    query_receipt_user = """
    SELECT
        m._id AS message_row_id,
        m.timestamp AS message_ts,
        jr.raw_string AS reader_jid,
        ru.read_timestamp AS read_ts,
        ru.played_timestamp AS played_ts
    FROM receipt_user ru
    JOIN message m ON ru.message_row_id = m._id
    JOIN chat c ON m.chat_row_id = c._id
    JOIN jid jg ON c.jid_row_id = jg._id
    JOIN jid jr ON ru.receipt_user_jid_row_id = jr._id
    WHERE jg.raw_string = ?
      AND m.from_me = 1
      AND (ru.read_timestamp > 0 OR ru.played_timestamp > 0)
      AND jr.raw_string IS NOT NULL
      AND TRIM(jr.raw_string) <> ''
    """

    # Legacy fallback in old backups.
    query_receipts = """
    SELECT
        m._id AS message_row_id,
        m.timestamp AS message_ts,
        r.remote_resource AS reader_jid,
        r.read_device_timestamp AS read_ts,
        r.played_device_timestamp AS played_ts
    FROM receipts r
    JOIN message m ON r.key_id = m.key_id
    JOIN chat c ON m.chat_row_id = c._id
    JOIN jid j ON c.jid_row_id = j._id
    WHERE j.raw_string = ?
      AND m.from_me = 1
      AND (r.read_device_timestamp > 0 OR r.played_device_timestamp > 0)
      AND r.remote_resource IS NOT NULL
      AND TRIM(r.remote_resource) <> ''
    """

    conn = None
    try:
        conn = sqlite3.connect(msgstore_path)
        raw_user = pd.read_sql_query(query_receipt_user, conn, params=[group_jid])
        raw_legacy = pd.read_sql_query(query_receipts, conn, params=[group_jid])
        to_concat = [df for df in [raw_user, raw_legacy] if not df.empty]
        raw = (
            pd.concat(to_concat, ignore_index=True)
            if to_concat
            else pd.DataFrame(columns=cols)
        )
    except Exception:
        return pd.DataFrame(columns=cols)
    finally:
        if conn is not None:
            conn.close()

    if raw.empty:
        return pd.DataFrame(columns=cols)

    raw["read_ts"] = pd.to_numeric(raw["read_ts"], errors="coerce").where(
        lambda s: s > 0
    )
    raw["played_ts"] = pd.to_numeric(raw["played_ts"], errors="coerce").where(
        lambda s: s > 0
    )
    raw["event_ts"] = raw["read_ts"].fillna(raw["played_ts"])
    raw = raw[raw["event_ts"].notna()].copy()

    if raw.empty:
        return pd.DataFrame(columns=cols)

    raw["msg_timestamp"] = to_user_datetime(
        pd.to_numeric(raw["message_ts"], errors="coerce"), unit="ms"
    )
    raw["read_timestamp"] = to_user_datetime(raw["event_ts"], unit="ms")
    raw = raw.dropna(subset=["msg_timestamp", "read_timestamp", "reader_jid"])

    if raw.empty:
        return pd.DataFrame(columns=cols)

    # First read event per person per message.
    raw = raw.sort_values("read_timestamp").drop_duplicates(
        subset=["message_row_id", "reader_jid"], keep="first"
    )
    return raw[cols]


@st.cache_data(show_spinner=False)
def load_lid_jid_map(msgstore_path, file_signature=None):
    """
    Load LID -> PN JID row-id mappings from msgstore jid_map.
    Returns dict {lid_row_id: jid_row_id}.
    """
    if not msgstore_path or not os.path.exists(msgstore_path):
        return {}

    conn = None
    try:
        conn = sqlite3.connect(msgstore_path)
        df_map = pd.read_sql_query(
            "SELECT lid_row_id, jid_row_id, COALESCE(sort_id, 0) AS sort_id FROM jid_map",
            conn,
        )
    except Exception:
        return {}
    finally:
        if conn is not None:
            conn.close()

    if df_map.empty:
        return {}

    df_map["lid_row_id"] = pd.to_numeric(df_map["lid_row_id"], errors="coerce")
    df_map["jid_row_id"] = pd.to_numeric(df_map["jid_row_id"], errors="coerce")
    df_map["sort_id"] = pd.to_numeric(df_map["sort_id"], errors="coerce").fillna(0)
    df_map = df_map.dropna(subset=["lid_row_id", "jid_row_id"])

    if df_map.empty:
        return {}

    # Keep the newest mapping when duplicates exist.
    df_map = df_map.sort_values("sort_id").drop_duplicates(
        subset=["lid_row_id"], keep="last"
    )
    return {
        int(r.lid_row_id): int(r.jid_row_id) for r in df_map.itertuples(index=False)
    }


@st.cache_data(show_spinner=False)
def load_jid_raw_lookup(msgstore_path, file_signature=None):
    """
    Load jid row-id -> raw_string for fallback labeling.
    """
    if not msgstore_path or not os.path.exists(msgstore_path):
        return {}

    conn = None
    try:
        conn = sqlite3.connect(msgstore_path)
        df_jid = pd.read_sql_query("SELECT _id AS jid_id, raw_string FROM jid", conn)
    except Exception:
        return {}
    finally:
        if conn is not None:
            conn.close()

    if df_jid.empty:
        return {}

    df_jid["jid_id"] = pd.to_numeric(df_jid["jid_id"], errors="coerce")
    df_jid = df_jid.dropna(subset=["jid_id"])
    return {
        int(r.jid_id): (str(r.raw_string) if pd.notna(r.raw_string) else "")
        for r in df_jid.itertuples(index=False)
    }


@st.cache_data(show_spinner=False)
def load_vcf_contact_lookup(vcf_path):
    """
    Load VCF contact map using parser's VCF logic.
    Returns dict: normalized digits -> display name.
    """
    if not vcf_path or not os.path.exists(vcf_path):
        return {}
    try:
        parser = WhatsappParser("", "", vcf_path)
        contacts = parser.parse_vcf()
        return contacts if isinstance(contacts, dict) else {}
    except Exception:
        return {}


def _count_words_text(value):
    """Count user-visible word tokens in a message body."""
    if value is None or pd.isna(value):
        return 0
    return len(re.findall(r"\b\w+\b", str(value), flags=re.UNICODE))


def _contains_question(value):
    if value is None or pd.isna(value):
        return False
    return "?" in str(value)


def _format_backup_horizon(value):
    if value is None or pd.isna(value):
        return ""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(get_user_timezone()).tz_localize(None)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _coerce_bar_chart_data(data):
    if data is None:
        return None
    if isinstance(data, pd.Series):
        numeric = pd.to_numeric(data, errors="coerce").dropna()
        return numeric if not numeric.empty else None
    if isinstance(data, pd.DataFrame):
        numeric = data.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return numeric if not numeric.empty else None
    return data


@st.cache_data(show_spinner=False)
def load_backup_message_horizon(msgstore_path, file_signature=None):
    """Return the latest message timestamp available in the backup."""
    if not msgstore_path or not os.path.exists(msgstore_path):
        return pd.NaT

    conn = None
    try:
        conn = sqlite3.connect(msgstore_path)
        value = conn.execute("SELECT MAX(timestamp) FROM message").fetchone()[0]
    except Exception:
        return pd.NaT
    finally:
        if conn is not None:
            conn.close()

    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value) or value <= 0:
        return pd.NaT
    return to_user_datetime(value, unit="ms")


@st.cache_data(show_spinner=False)
def load_unread_chat_counters(msgstore_path, file_signature=None):
    """Load WhatsApp unread counters from chat metadata."""
    empty_unread_cols = [
        "chat_id",
        "unread_messages",
        "unread_messages_raw",
        "unread_rows",
        "unread_reactions",
        "unread_comments",
        "unread_missed_calls",
        "unread_state",
        "last_message_at",
        "archived",
        "hidden",
    ]

    if not msgstore_path or not os.path.exists(msgstore_path):
        return pd.DataFrame(columns=empty_unread_cols)

    chat_query = """
    SELECT
        c._id AS chat_id,
        c.sort_timestamp,
        COALESCE(c.unseen_message_count, 0) AS unread_messages,
        COALESCE(c.unseen_row_count, 0) AS unread_rows,
        COALESCE(c.unseen_missed_calls_count, 0) AS unread_missed_calls,
        COALESCE(c.unseen_message_reaction_count, 0) AS unread_reactions,
        COALESCE(c.unseen_comment_message_count, 0) AS unread_comments,
        COALESCE(c.archived, 0) AS archived,
        COALESCE(c.hidden, 0) AS hidden
    FROM chat c
    WHERE COALESCE(c.unseen_message_count, 0) != 0
       OR COALESCE(c.unseen_row_count, 0) != 0
       OR COALESCE(c.unseen_missed_calls_count, 0) != 0
       OR COALESCE(c.unseen_message_reaction_count, 0) != 0
       OR COALESCE(c.unseen_comment_message_count, 0) != 0
    """

    conn = None
    try:
        conn = sqlite3.connect(msgstore_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        unread_df = pd.read_sql_query(chat_query, conn)
    except Exception:
        return pd.DataFrame(columns=empty_unread_cols)
    finally:
        if conn is not None:
            conn.close()

    if unread_df.empty:
        return pd.DataFrame(columns=empty_unread_cols)

    unread_df["unread_messages_raw"] = unread_df["unread_messages"]
    unread_df["unread_state"] = np.select(
        [
            unread_df["unread_messages_raw"] < 0,
            unread_df["unread_messages_raw"] > 0,
        ],
        ["Marked unread", "Unread messages"],
        default="Unread activity",
    )
    unread_df["unread_messages"] = unread_df["unread_messages"].clip(lower=0)
    unread_df["last_message_at"] = to_user_datetime(
        pd.to_numeric(unread_df["sort_timestamp"], errors="coerce"),
        unit="ms",
    )
    return unread_df[empty_unread_cols]


@st.cache_data(show_spinner=False)
def load_archived_chat_ids(msgstore_path, file_signature=None):
    """Load chat row ids that WhatsApp currently marks as archived."""
    if not msgstore_path or not os.path.exists(msgstore_path):
        return frozenset()

    conn = None
    try:
        conn = sqlite3.connect(msgstore_path)
        rows = conn.execute(
            "SELECT _id FROM chat WHERE COALESCE(archived, 0) != 0"
        ).fetchall()
    except Exception:
        return frozenset()
    finally:
        if conn is not None:
            conn.close()

    return frozenset(int(row[0]) for row in rows if row and row[0] is not None)


def _format_inbox_gender(value):
    value = str(value).strip().lower() if value is not None else "unknown"
    labels = {
        "male": "Male",
        "female": "Female",
        "unknown": "Unknown",
        "group": "Group",
        "newsletter": "Newsletter",
    }
    return labels.get(value, value.title() if value else "Unknown")


def build_chat_label_frame(df_context):
    cols = [
        "chat_id",
        "type",
        "gender",
        "chat",
        "contact_name",
        "number",
        "last_message_at",
    ]
    if (
        df_context is None
        or df_context.empty
        or "chat_row_id" not in df_context.columns
    ):
        return pd.DataFrame(columns=cols)

    df = df_context.copy()
    df["chat_id"] = pd.to_numeric(df["chat_row_id"], errors="coerce")
    df = df.dropna(subset=["chat_id"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    def _first_valid(series, default=""):
        valid = series.dropna()
        valid = valid[valid.astype(str).str.strip() != ""]
        return valid.iloc[0] if not valid.empty else default

    utils = Utils()
    rows = []
    for chat_id, chat_df in df.groupby("chat_id", sort=False):
        raw_string = str(_first_valid(chat_df.get("raw_string", pd.Series(dtype=str))))
        chat_name = str(
            _first_valid(chat_df.get("chat_name", pd.Series(dtype=str)), raw_string)
        )

        if raw_string.endswith("@g.us"):
            chat_type = "Group"
            chat_gender = "group"
            number = ""
            contact_name = ""
        elif raw_string.endswith("@newsletter") or raw_string == "status@broadcast":
            chat_type = "Newsletter"
            chat_gender = "newsletter"
            number = ""
            contact_name = ""
        else:
            chat_type = "Individual"
            number = raw_string.split("@", 1)[0] if "@" in raw_string else raw_string
            contact_name = chat_name
            if "gender" in chat_df.columns:
                from_me_values = (
                    pd.to_numeric(chat_df["from_me"], errors="coerce").fillna(0)
                    if "from_me" in chat_df.columns
                    else pd.Series(0, index=chat_df.index)
                )
                incoming = chat_df[from_me_values == 0]
                gender_source = incoming if not incoming.empty else chat_df
                chat_gender = str(
                    _first_valid(gender_source.get("gender", pd.Series(dtype=str)))
                ).lower()
                if chat_gender not in {"male", "female", "unknown"}:
                    chat_gender = utils.guess_gender(contact_name)
            else:
                chat_gender = utils.guess_gender(contact_name)

        rows.append(
            {
                "chat_id": int(chat_id),
                "type": chat_type,
                "gender": chat_gender,
                "chat": chat_name,
                "contact_name": contact_name if chat_type == "Individual" else "",
                "number": number,
                "last_message_at": chat_df["timestamp"].max()
                if "timestamp" in chat_df.columns
                else pd.NaT,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def build_unanswered_chats(df_context, backup_latest_ts):
    empty_cols = [
        "chat_id",
        "type",
        "gender",
        "chat",
        "contact_name",
        "number",
        "messages_since_reply",
        "words_since_reply",
        "question_marks",
        "has_question",
        "non_text_messages",
        "latest_message_at",
        "waiting_hours_in_backup",
        "attention_score",
        "reason",
        "preview",
    ]
    if (
        df_context is None
        or df_context.empty
        or "chat_row_id" not in df_context.columns
    ):
        return pd.DataFrame(columns=empty_cols)

    df = df_context.copy()
    df["chat_id"] = pd.to_numeric(df["chat_row_id"], errors="coerce")
    df = df.dropna(subset=["chat_id"])
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    if "message_type" in df.columns:
        df = df[df["message_type"] != 7]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    df["from_me"] = pd.to_numeric(df["from_me"], errors="coerce").fillna(0)
    df["word_count"] = df["text_data"].apply(_count_words_text)
    df["has_question"] = df["text_data"].apply(_contains_question)
    if "message_row_id" not in df.columns:
        df["message_row_id"] = np.arange(len(df))

    chat_meta = build_chat_label_frame(df).set_index("chat_id")
    horizon = backup_latest_ts if pd.notna(backup_latest_ts) else df["timestamp"].max()
    unanswered_rows = []

    for chat_id, chat_msgs in df.groupby("chat_id", sort=False):
        if chat_id not in chat_meta.index:
            continue

        chat_msgs = chat_msgs.sort_values(
            ["timestamp", "message_row_id"], kind="stable"
        ).reset_index(drop=True)
        if chat_msgs.empty or int(chat_msgs.iloc[-1]["from_me"]) == 1:
            continue

        sent_mask = chat_msgs["from_me"] == 1
        if sent_mask.any():
            last_sent_pos = int(np.flatnonzero(sent_mask.to_numpy())[-1])
            trailing = chat_msgs.iloc[last_sent_pos + 1 :].copy()
        else:
            trailing = chat_msgs.copy()

        trailing = trailing[trailing["from_me"] == 0]
        if trailing.empty:
            continue

        message_count = int(len(trailing))
        word_count = int(trailing["word_count"].sum())
        question_marks = int(
            trailing["text_data"].fillna("").astype(str).str.count(r"\?").sum()
        )
        has_question = question_marks > 0

        reasons = []
        if message_count >= 5:
            reasons.append("5+ messages")
        if word_count > 30:
            reasons.append(">30 words")
        if has_question:
            reasons.append("question")
        if not reasons:
            continue

        meta = chat_meta.loc[chat_id]
        latest_message_at = trailing["timestamp"].max()
        waiting_hours = np.nan
        if pd.notna(horizon) and pd.notna(latest_message_at):
            waiting_hours = max(
                0.0, (pd.Timestamp(horizon) - latest_message_at).total_seconds() / 3600
            )

        text_parts = [
            str(v).strip()
            for v in trailing["text_data"].dropna().tail(3).tolist()
            if str(v).strip()
        ]
        preview = " / ".join(text_parts)
        if len(preview) > 220:
            preview = preview[:217] + "..."

        non_text_messages = int((trailing["word_count"] == 0).sum())
        attention_score = (
            word_count
            + message_count * 8
            + question_marks * 20
            + min(waiting_hours if pd.notna(waiting_hours) else 0, 72) * 0.25
        )

        unanswered_rows.append(
            {
                "chat_id": int(chat_id),
                "type": meta["type"],
                "gender": meta["gender"],
                "chat": meta["chat"],
                "contact_name": meta["contact_name"],
                "number": meta["number"],
                "messages_since_reply": message_count,
                "words_since_reply": word_count,
                "question_marks": question_marks,
                "has_question": has_question,
                "non_text_messages": non_text_messages,
                "latest_message_at": latest_message_at,
                "waiting_hours_in_backup": waiting_hours,
                "attention_score": round(attention_score, 1),
                "reason": ", ".join(reasons),
                "preview": preview,
            }
        )

    unanswered_df = pd.DataFrame(unanswered_rows, columns=empty_cols)
    if not unanswered_df.empty:
        unanswered_df = unanswered_df.sort_values(
            ["attention_score", "latest_message_at"], ascending=[False, False]
        )
    return unanswered_df


def build_unread_chats_for_context(df_context, msgstore_path, file_signature=None):
    labels = build_chat_label_frame(df_context)
    counters = load_unread_chat_counters(msgstore_path, file_signature)
    if labels.empty or counters.empty:
        return pd.DataFrame(
            columns=[
                "chat_id",
                "type",
                "gender",
                "chat",
                "contact_name",
                "number",
                "unread_messages",
                "unread_messages_raw",
                "unread_rows",
                "unread_reactions",
                "unread_comments",
                "unread_missed_calls",
                "unread_state",
                "last_message_at",
                "archived",
                "hidden",
            ]
        )

    unread_df = counters.merge(labels, on="chat_id", how="inner", suffixes=("", "_ctx"))
    unread_df["last_message_at"] = unread_df["last_message_at_ctx"].fillna(
        unread_df["last_message_at"]
    )
    unread_df = unread_df.drop(columns=["last_message_at_ctx"], errors="ignore")
    return unread_df.sort_values("last_message_at", ascending=False)
