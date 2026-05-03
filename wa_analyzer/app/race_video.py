"""Rolling contact race video helpers."""

import hashlib
import os
import tempfile

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


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
