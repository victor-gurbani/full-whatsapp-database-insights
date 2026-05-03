import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = [
    "Helvetica",
    "Arial",
    "Apple Color Emoji",
    "Segoe UI Emoji",
    "Noto Color Emoji",
    "DejaVu Sans",
]

import os
import sqlite3
import tempfile
import hashlib
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
from wa_analyzer.src.parser import WhatsappParser
from wa_analyzer.src.analyzer import WhatsappAnalyzer
from wa_analyzer.src.utils import Utils
from wa_analyzer.src.chat_viewer import (
    generate_chat_html,
    export_chat_json,
    export_chat_txt,
    export_chat_html_standalone,
)
from wordcloud import WordCloud

# Page Config
st.set_page_config(page_title="WhatsApp Analytics", layout="wide", page_icon="💬")

# Title
st.title("💬 WhatsApp Interactive Analyzer")

if st.session_state.get("race_video_ready_notification") and not st.session_state.get(
    "race_video_generating"
):
    with st.container():
        st.success(
            "🎬 **Your Bar Chart Race Video is ready!** Go to the 'Activity & Top Users' tab to view and download it."
        )
        if st.button("Dismiss Notification"):
            st.session_state["race_video_ready_notification"] = False
            st.rerun()

import json
import datetime
import re
import numpy as np
import string
import random as _random

USER_TIMEZONE = "Europe/Madrid"


# --- Anonymisation Helpers ---
def _anon_hash(name):
    """Full SHA-256 hex hash of a name."""
    return hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest()


def _anon_hash_cut(name):
    """SHA-256 hex hash truncated to 8 characters."""
    return hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest()[:8]


def _anon_random(name):
    """Deterministic random string: 6-10 alphanumeric chars + 2-4 digits. Seeded from name for stability."""
    seed = int(hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest(), 16)
    rng = _random.Random(seed)
    alpha_len = rng.randint(6, 10)
    digit_len = rng.randint(2, 4)
    letters = "".join(rng.choices(string.ascii_letters, k=alpha_len))
    digits = "".join(rng.choices(string.digits, k=digit_len))
    return letters + digits


@st.cache_data(show_spinner=False)
def build_anon_map(unique_names, mode):
    """
    Build a stable mapping {original_name → anonymised_name}.
    'You' / 'Me' are never anonymised.
    mode: 'off' | 'hash' | 'hash_cut' | 'random'
    """
    if mode == "off":
        return {}
    fn = {"hash": _anon_hash, "hash_cut": _anon_hash_cut, "random": _anon_random}[mode]
    me_names = {"you", "me", "myself", "yo", "tú"}
    return {
        n: (n if str(n).strip().lower() in me_names else fn(str(n)))
        for n in unique_names
        if pd.notna(n)
    }


def apply_anon_to_df(df, anon_map, cols=("contact_name", "chat_name", "subject")):
    """Apply anonymisation mapping to identity columns in-place and return df."""
    if not anon_map:
        return df
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(lambda v: anon_map.get(v, v))
    return df


def av(value, anon_numbers=False, rng_seed=None):
    """
    Anonymise Value: if anon_numbers is True, replace numeric value with a random
    value of similar magnitude (±50%). If False, return as-is.
    For display only — does not affect underlying computations.
    """
    if not anon_numbers:
        return value
    if value is None or (isinstance(value, float) and (pd.isna(value) or value == 0)):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if v == 0:
        return value
    seed = int(abs(v * 1000)) if rng_seed is None else rng_seed
    rng = _random.Random(seed)
    factor = rng.uniform(0.5, 1.5)
    result = v * factor
    # Preserve integer type if original was int-like
    if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
        return int(round(result))
    return result


# --- Helper: Calculate Correlation Matrix Text ---
def get_correlation_text(df):
    """
    Calculate Pearson correlations between columns of a DataFrame.
    Returns a formatted string showing correlation pairs.
    """
    if df is None or df.empty or len(df.columns) < 2:
        return None

    # Drop rows with NaN to get valid correlation
    df_clean = df.dropna()
    if len(df_clean) < 3:  # Need at least 3 data points
        return None

    cols = df.columns.tolist()
    correlations = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = df_clean[cols[i]].corr(df_clean[cols[j]])
            if pd.notna(corr):
                # Format: positive = move together, negative = opposite
                direction = "↗↗" if corr > 0.5 else "↗↘" if corr < -0.5 else "→"
                correlations.append(
                    f"**{cols[i]}** vs **{cols[j]}**: r={corr:.2f} {direction}"
                )

    return " | ".join(correlations) if correlations else None


def ease_in_out_cubic(alpha):
    """
    Smooth nonlinear interpolation for polished overtakes.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha < 0.5:
        return 4.0 * alpha**3
    return 1.0 - ((-2.0 * alpha + 2.0) ** 3) / 2.0


def build_distinct_color_map(names):
    """
    Build a stable, high-contrast color map so top bars remain visually distinct.
    """
    names = [str(n) for n in names]
    if not names:
        return {}

    # Hash-sort names so color assignment stays stable across runs and pool changes.
    ordered = sorted(names, key=lambda n: hashlib.sha1(n.encode("utf-8")).hexdigest())
    golden = 0.618033988749895
    hue = 0.13
    sat_cycle = [0.90, 0.78, 0.86]
    val_cycle = [0.96, 0.84]

    cmap = {}
    for i, name in enumerate(ordered):
        hue = (hue + golden) % 1.0
        sat = sat_cycle[i % len(sat_cycle)]
        val = val_cycle[(i // len(sat_cycle)) % len(val_cycle)]
        cmap[name] = mcolors.hsv_to_rgb((hue, sat, val))

    return cmap


def build_rolling_counts(
    df_input,
    count_mode="their_only",
    average_my_messages=False,
    top_candidates=120,
    time_bin="D",
    window_offset=None,
):
    """
    Build rolling message counts per contact using configurable counting mode and time window.
    window_offset: pd.DateOffset object (e.g., pd.DateOffset(months=3), defaults to 3 months)
    count_mode:
    - 'their_only': incoming messages only.
    - 'chat_total': incoming + my messages mapped to counterpart chat.
    """
    required_cols = {"timestamp", "contact_name"}
    if (
        df_input is None
        or df_input.empty
        or not required_cols.issubset(set(df_input.columns))
    ):
        return pd.DataFrame()

    cols = ["timestamp", "contact_name"]
    if "from_me" in df_input.columns:
        cols.append("from_me")
    if "chat_name" in df_input.columns:
        cols.append("chat_name")

    work = df_input[cols].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp", "contact_name"])
    work["contact_name"] = work["contact_name"].astype(str).str.strip()
    work = work[work["contact_name"] != ""]

    if count_mode not in {"their_only", "chat_total"}:
        count_mode = "their_only"

    if count_mode == "their_only":
        if "from_me" in work.columns:
            work = work[work["from_me"] == 0]
        work["entity_name"] = work["contact_name"]
        work["weight"] = 1.0
    else:
        if "from_me" in work.columns:
            if "chat_name" in work.columns:
                work["entity_name"] = np.where(
                    work["from_me"] == 1,
                    work["chat_name"].fillna(""),
                    work["contact_name"].fillna(""),
                )
            else:
                work["entity_name"] = work["contact_name"]
        else:
            work["entity_name"] = work["contact_name"]

        work["entity_name"] = work["entity_name"].astype(str).str.strip()
        work = work[work["entity_name"] != ""]
        work = work[~work["entity_name"].str.lower().isin(["you", "me", "myself"])]
        work["weight"] = 1.0

        # Optional: dilute my outgoing influence by the number of people spoken with that day.
        if average_my_messages and "from_me" in work.columns:
            outgoing_mask = work["from_me"] == 1
            if outgoing_mask.any():
                work["active_day"] = work["timestamp"].dt.floor("D")
                day_people = (
                    work.loc[outgoing_mask, ["active_day", "entity_name"]]
                    .drop_duplicates()
                    .groupby("active_day")
                    .size()
                    .astype(float)
                )
                divisors = (
                    work.loc[outgoing_mask, "active_day"]
                    .map(day_people)
                    .replace(0, np.nan)
                    .fillna(1.0)
                )
                work.loc[outgoing_mask, "weight"] = 1.0 / divisors

    if work.empty:
        return pd.DataFrame()

    try:
        work["bucket"] = work["timestamp"].dt.floor(time_bin)
    except Exception:
        time_bin = "D"
        work["bucket"] = work["timestamp"].dt.floor(time_bin)

    daily_counts = (
        work.groupby(["bucket", "entity_name"])["weight"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    if daily_counts.empty:
        return pd.DataFrame()

    # Keep a broad candidate pool for performance while preserving likely overtakes.
    if top_candidates and daily_counts.shape[1] > int(top_candidates):
        keep_cols = (
            daily_counts.sum(axis=0)
            .sort_values(ascending=False)
            .head(int(top_candidates))
            .index
        )
        daily_counts = daily_counts[keep_cols]

    full_days = pd.date_range(
        daily_counts.index.min(), daily_counts.index.max(), freq=time_bin
    )
    daily_counts = daily_counts.reindex(full_days, fill_value=0)

    # Exact calendar rolling window per day.
    if window_offset is None:
        window_offset = pd.DateOffset(months=3)
    indexer = pd.api.indexers.VariableOffsetWindowIndexer(
        index=daily_counts.index, offset=window_offset
    )
    rolling_counts = daily_counts.rolling(window=indexer, min_periods=1).sum()
    rolling_counts = rolling_counts.loc[:, rolling_counts.max(axis=0) > 0]
    rolling_counts.index.name = "date"
    return rolling_counts.astype(float)


def render_contact_race_video(
    rolling_counts,
    top_k=10,
    fps=15,
    seconds_per_month=2.0,
    width=1280,
    height=720,
    window_offset=None,
    window_label="3-month",
):
    """
    Render a dynamic top-N bar chart race video from rolling contact counts.
    Returns dict with bytes, mime, filename, frame_count.
    """
    if rolling_counts is None or rolling_counts.empty:
        raise ValueError("No rolling data to render.")

    data = rolling_counts.copy().sort_index()
    if data.shape[1] == 0:
        raise ValueError("No contacts available after filtering.")

    values = data.to_numpy(dtype=float)
    names = data.columns.astype(str).to_numpy()
    n_steps, n_contacts = values.shape

    # Stable rank snapshots used for smooth vertical motion between days.
    ranks = np.empty_like(values)
    for i in range(n_steps):
        order = np.argsort(-values[i], kind="mergesort")
        ranks[i, order] = np.arange(n_contacts)

    top_k = max(1, min(int(top_k), n_contacts))
    day_top_max = np.partition(values, -top_k, axis=1)[:, -top_k:].max(axis=1)
    global_max = max(float(day_top_max.max()), 1.0)

    # Drive animation by elapsed calendar time, so FPS changes smoothness only (not speed).
    elapsed_days = (
        (data.index - data.index[0]).total_seconds() / (24 * 60 * 60)
    ).to_numpy(dtype=float)
    total_elapsed_days = float(elapsed_days[-1]) if n_steps > 1 else 0.0
    days_per_second = 30.4375 / max(seconds_per_month, 1e-9)
    duration_seconds = total_elapsed_days / max(days_per_second, 1e-9)
    min_duration_seconds = 8.0
    hold_tail_seconds = 0.8
    base_duration_seconds = max(duration_seconds, min_duration_seconds)
    total_frames = max(
        1, int(np.ceil((base_duration_seconds + hold_tail_seconds) * fps)) + 1
    )

    if n_steps > 1:
        step_days_arr = np.diff(elapsed_days)
        positive = step_days_arr[step_days_arr > 0]
        step_days = float(np.median(positive)) if positive.size else 1.0
    else:
        step_days = 1.0

    color_lookup = build_distinct_color_map(names)
    resolution_scale = float(
        np.clip(np.sqrt((width * height) / (1280 * 720)), 0.9, 1.8)
    )
    frame_dt = 1.0 / max(fps, 1e-9)
    smoothing_tau_seconds = 0.35
    smooth_alpha = 1.0 - np.exp(-frame_dt / max(smoothing_tau_seconds, 1e-9))

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#05070f")
    motion_state = {"vals": None, "pos": None, "xmax": None}

    def style_axes(x_lim):
        ax.set_facecolor("#0b1220")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(
            axis="x",
            color="#334155",
            alpha=0.32,
            linewidth=max(1.0, 1.0 * resolution_scale),
        )
        ax.tick_params(axis="x", colors="#94a3b8", labelsize=10 * resolution_scale)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlim(0, max(x_lim, 1))
        ax.set_ylim(top_k - 0.35, -0.65)
        ax.set_yticks([])
        ax.set_xlabel(
            f"Messages in rolling {window_label} window",
            color="#cbd5e1",
            fontsize=11 * resolution_scale,
        )

    def draw_frame(frame_idx):
        elapsed_seconds = min(frame_idx / max(fps, 1e-9), base_duration_seconds)
        elapsed = min(elapsed_seconds * days_per_second, total_elapsed_days)
        if n_steps <= 1:
            i0 = i1 = 0
            eased = 0.0
        else:
            i1 = int(np.searchsorted(elapsed_days, elapsed, side="right"))
            i1 = min(max(i1, 1), n_steps - 1)
            i0 = i1 - 1
            span = max(elapsed_days[i1] - elapsed_days[i0], 1e-9)
            raw_alpha = (elapsed - elapsed_days[i0]) / span
            eased = ease_in_out_cubic(raw_alpha)

        frame_values = values[i0] * (1.0 - eased) + values[i1] * eased
        frame_pos = ranks[i0] * (1.0 - eased) + ranks[i1] * eased
        if motion_state["vals"] is None:
            motion_state["vals"] = frame_values.copy()
            motion_state["pos"] = frame_pos.copy()
        else:
            motion_state["vals"] = motion_state["vals"] + smooth_alpha * (
                frame_values - motion_state["vals"]
            )
            motion_state["pos"] = motion_state["pos"] + smooth_alpha * (
                frame_pos - motion_state["pos"]
            )

        # Show only the leaders plus a few near-threshold bars to make entries smoother.
        keep_n = min(top_k + 3, n_contacts)
        visible = np.argsort(motion_state["pos"])[:keep_n]
        visible = visible[np.argsort(motion_state["pos"][visible])]
        draw_ids = visible[:top_k]

        bar_vals = motion_state["vals"][draw_ids]
        bar_pos = motion_state["pos"][draw_ids]
        bar_names = names[draw_ids]

        day_max = day_top_max[i0] * (1.0 - eased) + day_top_max[i1] * eased
        x_max = max(day_max * 1.18, global_max * 0.2, 1.0)
        if motion_state["xmax"] is None:
            motion_state["xmax"] = x_max
        else:
            motion_state["xmax"] = motion_state["xmax"] + smooth_alpha * (
                x_max - motion_state["xmax"]
            )

        ax.cla()
        style_axes(motion_state["xmax"])

        ax.barh(
            bar_pos,
            bar_vals,
            height=0.78,
            color=[color_lookup[n] for n in bar_names],
            edgecolor="none",
            alpha=0.95,
        )

        label_pad = 0.012 * x_max
        for y, x, name in zip(bar_pos, bar_vals, bar_names):
            ax.text(
                x + label_pad,
                y,
                f"{name}  {int(round(x)):,}",
                va="center",
                ha="left",
                color="white",
                fontsize=10 * resolution_scale,
                fontweight="bold",
            )

        current_day = data.index[0] + pd.to_timedelta(elapsed, unit="D")
        window_start = current_day - window_offset
        range_fmt = "%b %d, %Y" if step_days >= 1 else "%b %d, %Y %H:%M"

        ax.text(
            0.01,
            1.06,
            f"Top 10 Contacts • {window_label.title()} Rolling Message Race",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=17 * resolution_scale,
            fontweight="bold",
        )
        ax.text(
            0.01,
            1.00,
            current_day.strftime("%b %d, %Y %H:%M" if step_days < 1 else "%b %d, %Y"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="#22d3ee",
            fontsize=13 * resolution_scale,
            fontweight="bold",
        )
        ax.text(
            0.99,
            1.03,
            f"Window: {window_start.strftime(range_fmt)} to {current_day.strftime(range_fmt)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#e2e8f0",
            fontsize=10 * resolution_scale,
            bbox=dict(
                facecolor="#0f172a",
                edgecolor="none",
                alpha=0.82,
                boxstyle=f"round,pad={0.32 * resolution_scale:.2f}",
            ),
        )
        ax.text(
            0.99,
            0.97,
            f"{fps} fps • 1 month ≈ {seconds_per_month:.1f}s",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#94a3b8",
            fontsize=9 * resolution_scale,
        )

    anim = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=total_frames,
        interval=1000 / fps,
        repeat=False,
        blit=False,
    )

    tmp_mp4 = None
    tmp_gif = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            tmp_mp4 = handle.name

        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264", bitrate=3200, extra_args=["-pix_fmt", "yuv420p"]
        )
        anim.save(tmp_mp4, writer=writer, dpi=100)

        with open(tmp_mp4, "rb") as f:
            return {
                "bytes": f.read(),
                "mime": "video/mp4",
                "filename": f"contact_race_{window_label.replace(' ', '_')}.mp4",
                "frame_count": total_frames,
            }
    except Exception as mp4_err:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as handle:
            tmp_gif = handle.name

        try:
            gif_writer = animation.PillowWriter(fps=min(15, fps))
            anim.save(tmp_gif, writer=gif_writer, dpi=100)
            with open(tmp_gif, "rb") as f:
                return {
                    "bytes": f.read(),
                    "mime": "image/gif",
                    "filename": f"contact_race_{window_label.replace(' ', '_')}.gif",
                    "frame_count": total_frames,
                    "fallback_reason": str(mp4_err),
                }
        except Exception as gif_err:
            raise RuntimeError(
                f"Video export failed (MP4 and GIF). MP4 error: {mp4_err}; GIF error: {gif_err}"
            )
    finally:
        plt.close(fig)
        if tmp_mp4 and os.path.exists(tmp_mp4):
            os.remove(tmp_mp4)
        if tmp_gif and os.path.exists(tmp_gif):
            os.remove(tmp_gif)


# ... (imports)


# --- Config Management Functions ---
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


@st.cache_data(show_spinner=False)
def load_group_receipt_events(msgstore_path, group_jid):
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

    raw["msg_timestamp"] = _to_user_datetime(
        pd.to_numeric(raw["message_ts"], errors="coerce"), unit="ms"
    )
    raw["read_timestamp"] = _to_user_datetime(raw["event_ts"], unit="ms")
    raw = raw.dropna(subset=["msg_timestamp", "read_timestamp", "reader_jid"])

    if raw.empty:
        return pd.DataFrame(columns=cols)

    # First read event per person per message.
    raw = raw.sort_values("read_timestamp").drop_duplicates(
        subset=["message_row_id", "reader_jid"], keep="first"
    )
    return raw[cols]


@st.cache_data(show_spinner=False)
def load_lid_jid_map(msgstore_path):
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
def load_jid_raw_lookup(msgstore_path):
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
        ts = ts.tz_convert(USER_TIMEZONE).tz_localize(None)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _to_user_datetime(values, unit="ms"):
    """Convert WhatsApp UTC millisecond timestamps to the user's local wall time."""
    converted = pd.to_datetime(values, unit=unit, errors="coerce", utc=True)
    if isinstance(converted, pd.Series):
        return converted.dt.tz_convert(USER_TIMEZONE).dt.tz_localize(None)
    if pd.isna(converted):
        return converted
    return converted.tz_convert(USER_TIMEZONE).tz_localize(None)


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
def load_backup_message_horizon(msgstore_path):
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
    return _to_user_datetime(value, unit="ms")


@st.cache_data(show_spinner=False)
def load_unread_chat_counters(msgstore_path):
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
    unread_df["last_message_at"] = _to_user_datetime(
        pd.to_numeric(unread_df["sort_timestamp"], errors="coerce"),
        unit="ms",
    )
    return unread_df[empty_unread_cols]


@st.cache_data(show_spinner=False)
def load_archived_chat_ids(msgstore_path):
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


def build_unread_chats_for_context(df_context, msgstore_path):
    labels = build_chat_label_frame(df_context)
    counters = load_unread_chat_counters(msgstore_path)
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
        backup_message_horizon = load_backup_message_horizon(msgstore_path)
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
    def is_number(name):
        return not bool(re.search("[a-zA-Z]", str(name)))

    @st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: id})
    def _apply_global_filters(
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

        if exclude_low_participation and "raw_string" in df_base.columns:
            is_group_mask = df_base["raw_string"].astype(str).str.endswith("@g.us")
            group_chats = df_base[is_group_mask]

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
            mask_nums = df_base["contact_name"].apply(
                lambda name: not bool(re.search("[a-zA-Z]", str(name)))
            )
            df_base = df_base[~mask_nums]

        return df_base, df_group_base

    date_start = date_range[0] if len(date_range) == 2 else None
    date_end = date_range[1] if len(date_range) == 2 else None
    archived_chat_ids = load_archived_chat_ids(msgstore_path)
    df_base, df_group_base = _apply_global_filters(
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

    @st.fragment
    def _frag_kpi():
        total_msgs_raw = len(df_base)
        sent_raw = df_base[df_base["from_me"] == 1].shape[0]
        received_raw = total_msgs_raw - sent_raw
        unique_contacts_raw = df_base["contact_name"].nunique()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", f"{av(total_msgs_raw, _anon_numbers):,}")
        col2.metric("Sent", f"{av(sent_raw, _anon_numbers):,}")
        col3.metric("Received", f"{av(received_raw, _anon_numbers):,}")
        col4.metric("Unique Contacts", av(unique_contacts_raw, _anon_numbers))

    _frag_kpi()

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
        [
            "📊 Activity & Top Users",
            "🔥 Behavioral Patterns",
            "👫 Gender Insights",
            "📝 Word Cloud",
            "🔍 Chat Explorer",
            "👥 Group Explorer",
            "🎪 Fun & Insights",
            "🗺️ Map",
            "📱 Chat Viewer",
            "📥 Inbox Triage",
        ]
    )

    @st.fragment
    def _frag_activity():
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Top Talkers")

            # Filter by Category
            cat_filter = st.selectbox(
                "Filter by Category",
                [
                    "All",
                    "Only Female",
                    "Only Male",
                    "Only Unknown",
                    "Only Groups",
                    "Only Non-Contacts",
                ],
            )

            rank_by_words = st.checkbox("Rank by Total Words", value=False)
            metric_arg = "words" if rank_by_words else "messages"

            # Fetch a large number first to ensure we fill the top 20 after filtering
            top_talkers_df = analyzer.get_top_talkers(
                1000, metric=metric_arg, exclude_me=exclude_me
            )

            if cat_filter == "Only Female":
                top_talkers_df = top_talkers_df[top_talkers_df["gender"] == "female"]
            elif cat_filter == "Only Male":
                top_talkers_df = top_talkers_df[top_talkers_df["gender"] == "male"]
            elif cat_filter == "Only Unknown":
                top_talkers_df = top_talkers_df[top_talkers_df["gender"] == "unknown"]
            elif cat_filter == "Only Groups":
                top_talkers_df = top_talkers_df[top_talkers_df["is_group"] == True]
            elif cat_filter == "Only Non-Contacts":
                mask = top_talkers_df["contact_name"].apply(is_number)
                top_talkers_df = top_talkers_df[mask]

            # Slice top 20 after filtering
            top_talkers_final = top_talkers_df.head(20)

            fig_bar = px.bar(
                top_talkers_final,
                x="count",
                y="contact_name",
                orientation="h",
                color="gender",
                title=f"Most Active Contacts ({cat_filter})",
                color_discrete_map={
                    "male": "#636EFA",
                    "female": "#EF553B",
                    "unknown": "gray",
                },
            )
            fig_bar.update_layout(
                yaxis={"categoryorder": "total ascending", "type": "category"},
                height=600,
            )
            st.plotly_chart(fig_bar, width="stretch")

        with col_r:
            st.subheader("Hourly Activity")
            split_opt = st.selectbox(
                "Split by:",
                ["None", "Gender", "Type (Group/Indiv)"],
                key="hourly_split",
            )

            split_arg = None
            if split_opt == "Gender":
                split_arg = "gender"
            elif split_opt.startswith("Type"):
                split_arg = "group"

            hourly = analyzer.get_hourly_activity(
                split_by=split_arg, exclude_me=exclude_me
            )

            if split_arg:
                fig_line = px.line(
                    hourly,
                    x=hourly.index,
                    y=hourly.columns,
                    markers=True,
                    labels={"value": "Count", "timestamp": "Hour"},
                    title=f"Activity by Hour (Split by {split_opt})",
                )
                st.plotly_chart(fig_line, width="stretch")
                # Show correlation
                corr_text = get_correlation_text(hourly)
                if corr_text:
                    st.caption(f"📊 Correlation: {corr_text}")
            else:
                fig_line = px.line(
                    x=hourly.index,
                    y=hourly.values,
                    markers=True,
                    labels={"x": "Hour of Day", "y": "Message Count"},
                    title="Activity by Hour",
                )
                st.plotly_chart(fig_line, width="stretch")

        st.subheader("Message Volume Over Time")
        show_as_lines = st.checkbox("Show as Lines (Easier Comparison)", value=False)
        plot_func = px.line if show_as_lines else px.area

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            split_opt_m = st.selectbox(
                "Split Total by:",
                ["None", "Gender", "Type (Group/Indiv)"],
                key="monthly_split",
            )

            split_arg_m = None
            if split_opt_m == "Gender":
                split_arg_m = "gender"
            elif split_opt_m.startswith("Type"):
                split_arg_m = "group"

            monthly = analyzer.get_monthly_activity(
                split_by=split_arg_m, exclude_me=exclude_me
            )

            if split_arg_m:
                color_map = (
                    {"male": "#636EFA", "female": "#EF553B", "unknown": "gray"}
                    if split_arg_m == "gender"
                    else None
                )
                fig_time = plot_func(
                    monthly,
                    title=f"Total Volume (Split by {split_opt_m})",
                    color_discrete_map=color_map,
                )
                st.plotly_chart(fig_time, width="stretch")
                # Show correlation
                corr_text = get_correlation_text(monthly)
                if corr_text:
                    st.caption(f"📊 Correlation: {corr_text}")
            else:
                fig_time = plot_func(
                    x=monthly.index, y=monthly.values, title="Total Volume"
                )
                st.plotly_chart(fig_time, width="stretch")

        with col_t2:
            st.write(f"**Top Contacts Volume ({cat_filter})**")
            # Use the already filtered top_talkers_df
            top_contacts_list = top_talkers_final["contact_name"].head(10).tolist()
            if top_contacts_list:
                monthly_contacts = analyzer.get_activity_over_time_by_contact(
                    top_contacts_list
                )
                fig_contacts = plot_func(
                    monthly_contacts, title=f"Top 10 Contacts Activity ({cat_filter})"
                )
                st.plotly_chart(fig_contacts, width="stretch")
            else:
                st.info("No contacts match filter.")

        # --- Message Dispersion Over Time ---
        st.subheader("📊 Message Dispersion Over Time")
        st.caption(
            "**High dispersion** = Messages spread evenly across many chats. **Low dispersion** = Focused on specific people."
        )

        disp_col1, disp_col2 = st.columns(2)
        with disp_col1:
            split_opt_disp = st.selectbox(
                "Split Dispersion by:",
                ["None", "Gender", "Type (Group/Indiv)"],
                key="dispersion_split",
            )
        with disp_col2:
            smooth_dispersion = st.checkbox(
                "Smooth (3-month avg)",
                value=False,
                key="dispersion_smooth",
                help="Apply 3-month rolling average to reduce volatility from low-activity months",
            )

        split_arg_disp = None
        if split_opt_disp == "Gender":
            split_arg_disp = "gender"
        elif split_opt_disp.startswith("Type"):
            split_arg_disp = "group"

        dispersion = analyzer.get_message_dispersion_over_time(
            split_by=split_arg_disp, exclude_me=exclude_me
        )

        # Apply smoothing if enabled
        if smooth_dispersion:
            if isinstance(dispersion, pd.Series):
                dispersion = dispersion.rolling(
                    window=3, min_periods=1, center=True
                ).mean()
            elif isinstance(dispersion, pd.DataFrame):
                dispersion = dispersion.rolling(
                    window=3, min_periods=1, center=True
                ).mean()

        if isinstance(dispersion, pd.Series):
            # Single series (no split)
            if not dispersion.empty and dispersion.notna().any():
                fig_disp = px.line(
                    x=dispersion.index,
                    y=dispersion.values,
                    markers=True,
                    labels={"x": "Month", "y": "Dispersion (%)"},
                    title="Message Dispersion Over Time",
                )
                fig_disp.update_traces(
                    connectgaps=False
                )  # Skip gaps instead of connecting
                fig_disp.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig_disp, width="stretch")
            else:
                st.info("Not enough data to calculate dispersion.")
        elif isinstance(dispersion, pd.DataFrame) and not dispersion.empty:
            # DataFrame with multiple columns (split by gender/group)
            color_map = None
            if split_arg_disp == "gender":
                color_map = {"male": "#636EFA", "female": "#EF553B", "unknown": "gray"}

            fig_disp = px.line(
                dispersion,
                x=dispersion.index,
                y=dispersion.columns,
                markers=True,
                labels={
                    "value": "Dispersion (%)",
                    "index": "Month",
                    "variable": split_opt_disp,
                },
                title=f"Message Dispersion Over Time (Split by {split_opt_disp})",
                color_discrete_map=color_map,
            )
            fig_disp.update_traces(connectgaps=False)  # Skip gaps instead of connecting
            fig_disp.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_disp, width="stretch")
            # Show correlation
            corr_text = get_correlation_text(dispersion)
            if corr_text:
                st.caption(f"📊 Correlation: {corr_text}")

        else:
            st.info("Not enough data to calculate dispersion.")

        st.divider()
        st.subheader("🎬 Rolling Contact Race Video")
        st.caption(
            "Dynamic top-10 bar chart race with adjustable rolling window and smooth nonlinear overtaking. "
            "Pacing is fixed to 1 month ≈ 2 seconds."
        )

        # Window Duration Options
        window_options = {
            "1 day": pd.DateOffset(days=1),
            "3 days": pd.DateOffset(days=3),
            "1 week": pd.DateOffset(weeks=1),
            "2 weeks": pd.DateOffset(weeks=2),
            "1 month": pd.DateOffset(months=1),
            "2 months": pd.DateOffset(months=2),
            "3 months": pd.DateOffset(months=3),
            "6 months": pd.DateOffset(months=6),
            "9 months": pd.DateOffset(months=9),
            "12 months": pd.DateOffset(months=12),
        }

        window_label = st.select_slider(
            "Rolling Window Duration",
            options=list(window_options.keys()),
            value="3 months",
            key="race_window_duration",
            help="Select the rolling window size for the message race animation",
        )
        window_offset = window_options[window_label]

        race_c1, race_c2, race_c3 = st.columns(3)
        race_60fps = race_c1.checkbox(
            "60 FPS high precision",
            value=False,
            key="race_60fps",
            help="Uses 60 fps + 6-hour sampling for smoother overtakes at the same overall speed.",
        )
        include_chat_sum = race_c2.checkbox(
            "Include my msgs in each chat total",
            value=False,
            key="race_include_chat_sum",
            help="Off: only their incoming messages. On: each person's total = their messages + mine in that chat.",
        )
        average_my_msgs = race_c3.checkbox(
            "Average my msgs by people spoken",
            value=False,
            key="race_average_my_msgs",
            disabled=not include_chat_sum,
            help="When enabled, my outgoing messages are weighted by 1 / people I spoke with that day.",
        )

        race_c4, race_c5 = st.columns(2)
        candidate_pool = race_c4.slider(
            "Candidate contact pool",
            min_value=20,
            max_value=300,
            value=120,
            step=10,
            key="race_candidate_pool",
            help="Larger pools capture more late overtakes but render slower.",
        )
        quality = race_c5.selectbox(
            "Video quality",
            ["Preview (960x540)", "HD (1280x720)", "Full HD (1920x1080)"],
            index=1,
            key="race_video_quality",
        )
        seconds_per_month = st.slider(
            "Seconds per month",
            min_value=2.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            key="race_seconds_per_month",
            help="Higher values slow down the race. 4.0 is smoother and easier to read than 2.0.",
        )

        if st.button("Generate Bar Chart Race Video", key="race_video_generate"):
            st.session_state["race_video_generating"] = True
            st.session_state.pop("race_video_payload", None)
            st.session_state.pop("race_video_error", None)

            def _generate_video():
                try:
                    fps_value = 60 if race_60fps else 15
                    bucket_freq = "6h" if race_60fps else "D"
                    count_mode = "chat_total" if include_chat_sum else "their_only"
                    rolling_counts = build_rolling_counts(
                        df_base,
                        count_mode=count_mode,
                        average_my_messages=(average_my_msgs and include_chat_sum),
                        top_candidates=candidate_pool,
                        time_bin=bucket_freq,
                        window_offset=window_offset,
                    )

                    if rolling_counts.empty:
                        st.session_state["race_video_error"] = (
                            "No usable activity found for the race video with current filters."
                        )
                    else:
                        dims = {
                            "Preview (960x540)": (960, 540),
                            "HD (1280x720)": (1280, 720),
                            "Full HD (1920x1080)": (1920, 1080),
                        }
                        width, height = dims[quality]
                        payload = render_contact_race_video(
                            rolling_counts=rolling_counts,
                            top_k=10,
                            fps=fps_value,
                            seconds_per_month=seconds_per_month,
                            width=width,
                            height=height,
                            window_offset=window_offset,
                            window_label=window_label,
                        )
                        months_span = max(
                            (
                                rolling_counts.index.max() - rolling_counts.index.min()
                            ).total_seconds()
                            / (86400 * 30.4375),
                            0,
                        )
                        date_fmt = (
                            "%Y-%m-%d %H:%M" if bucket_freq != "D" else "%Y-%m-%d"
                        )
                        payload["date_start"] = rolling_counts.index.min().strftime(
                            date_fmt
                        )
                        payload["date_end"] = rolling_counts.index.max().strftime(
                            date_fmt
                        )
                        payload["contacts_count"] = int(rolling_counts.shape[1])
                        payload["seconds_per_month"] = float(seconds_per_month)
                        payload["approx_seconds"] = (
                            max(months_span * float(seconds_per_month), 8.0) + 0.8
                        )
                        payload["quality"] = quality
                        payload["fps"] = fps_value
                        payload["bucket_freq"] = bucket_freq
                        payload["precision_label"] = (
                            "6h sampling + time interpolation"
                            if race_60fps
                            else "Standard daily steps"
                        )
                        payload["count_mode_label"] = (
                            "Their messages only"
                            if count_mode == "their_only"
                            else "Chat total (theirs + mine)"
                        )
                        payload["avg_my_msgs"] = bool(
                            average_my_msgs and include_chat_sum
                        )
                        st.session_state["race_video_payload"] = payload
                except Exception as e:
                    st.session_state["race_video_error"] = (
                        f"Error generating video: {e}"
                    )
                finally:
                    st.session_state["race_video_generating"] = False
                    st.session_state["race_video_ready_notification"] = True

            thread = threading.Thread(target=_generate_video)
            add_script_run_ctx(thread)
            thread.start()

        if st.session_state.get("race_video_generating"):

            @st.fragment(run_every="2s")
            def _poll_video_status():
                if st.session_state.get("race_video_generating"):
                    st.info(
                        "⏳ Rendering race video in the background... You can continue using the rest of the app."
                    )
                elif st.session_state.get("race_video_ready_notification"):
                    st.rerun()

            _poll_video_status()

        race_video_error = st.session_state.get("race_video_error")
        if race_video_error:
            st.error(race_video_error)

        race_payload = st.session_state.get("race_video_payload")
        if race_payload and not st.session_state.get("race_video_generating"):
            # Intentionally leaving the notification active until the user dismisses it manually at the top.
            if race_payload.get("mime") == "video/mp4":
                st.video(race_payload["bytes"])
            else:
                st.image(race_payload["bytes"])
                if race_payload.get("fallback_reason"):
                    st.caption(
                        f"MP4 not available, used GIF fallback: {race_payload['fallback_reason']}"
                    )

            st.download_button(
                "Download Race Video",
                data=race_payload["bytes"],
                file_name=race_payload.get("filename", "contact_race_3month.mp4"),
                mime=race_payload.get("mime", "video/mp4"),
                key="race_video_download",
            )
            st.caption(
                f"Data range: {race_payload.get('date_start')} to {race_payload.get('date_end')} • "
                f"Contacts considered: {race_payload.get('contacts_count', 0)} • "
                f"Mode: {race_payload.get('count_mode_label')} • "
                f"FPS: {race_payload.get('fps', 15)} ({race_payload.get('precision_label', 'Standard')}) • "
                f"Speed: 1 month ≈ {race_payload.get('seconds_per_month', 4.0):.1f}s • "
                f"Avg my msgs: {'On' if race_payload.get('avg_my_msgs') else 'Off'} • "
                f"Frames: {race_payload.get('frame_count', 0):,} • "
                f"Approx length: {race_payload.get('approx_seconds', 0):.1f}s"
            )

    with tab1:
        _frag_activity()

    @st.fragment
    def _frag_behavioral():
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

    with tab2:
        _frag_behavioral()

    @st.fragment
    def _frag_gender():
        st.header("Demographics")
        # Use full_analyzer to ensure Reply Time calc has 'Me' messages
        # analyze_by_gender() handles 'from_me=0' internally for volume.
        gender_analyzer = full_analyzer

        if exclude_family_gender and not exclude_family_global and family_list:
            # Apply family filter to base for gender analysis
            # Note: df_base includes Me. gender_stats needs Me for reply time.
            gender_df_source = df_base[~df_base["chat_name"].isin(family_list)]
            gender_analyzer = WhatsappAnalyzer(gender_df_source)

        with st.spinner("Analyzing demographics..."):
            gender_counts = gender_analyzer.analyze_by_gender()
            gender_stats = gender_analyzer.calculate_gender_stats()

        c1, c2 = st.columns([1, 2])
        with c1:
            fig_pie = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Messages by Gender",
                color=gender_counts.index,
                color_discrete_map={
                    "male": "#636EFA",
                    "female": "#EF553B",
                    "unknown": "gray",
                },
            )
            st.plotly_chart(fig_pie, width="stretch")
        with c2:
            st.subheader("Deep Dive Metrics")
            if not gender_stats.empty:
                st.dataframe(gender_stats, width="stretch")
                metrics = ["count", "avg_wpm", "media_pct", "avg_reply_time"]
                metric_choice = st.selectbox("Select Metric", metrics)
                fig_comp = px.bar(
                    gender_stats,
                    x="gender",
                    y=metric_choice,
                    color="gender",
                    title=f"Comparison: {metric_choice}",
                    color_discrete_map={
                        "male": "#636EFA",
                        "female": "#EF553B",
                        "unknown": "gray",
                    },
                )
                st.plotly_chart(fig_comp, width="stretch")

    with tab3:
        _frag_gender()

    @st.fragment
    def _frag_wordcloud():
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

    with tab4:
        _frag_wordcloud()

    @st.fragment
    def _frag_chat_explorer():
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

    with tab5:
        _frag_chat_explorer()

    @st.fragment
    def _frag_group_explorer():
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
                lid_to_pn_map = load_lid_jid_map(msgstore_path)
                jid_raw_lookup = load_jid_raw_lookup(msgstore_path)
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
                            msgstore_path, group_jid
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

    with tab6:
        _frag_group_explorer()

    @st.fragment
    def _frag_fun_insights():
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

    with tab7:
        _frag_fun_insights()

    @st.fragment
    def _frag_map():
        st.header("🗺️ Location Map")
        with st.spinner("Loading location data..."):
            loc_data = analyzer.get_location_data()
        if not loc_data.empty:
            st.map(loc_data, latitude="latitude", longitude="longitude")
            st.dataframe(loc_data[["contact_name", "timestamp", "place_name"]])
        else:
            st.info("No location data found in this backup.")

    with tab8:
        _frag_map()

    @st.fragment
    def _frag_chat_viewer():
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

    with tab9:
        _frag_chat_viewer()

    @st.fragment
    def _frag_inbox_triage():
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
            backup_latest = load_backup_message_horizon(msgstore_path)
            unread_df = build_unread_chats_for_context(df_base, msgstore_path)
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
        unanswered_words_total = (
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

    with tab10:
        _frag_inbox_triage()

else:
    st.info("👈 Please enter file paths.")
