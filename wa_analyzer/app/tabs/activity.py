from __future__ import annotations

import threading

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from wa_analyzer.app.filters import is_number
from wa_analyzer.app.race_video import build_rolling_counts, render_contact_race_video
from wa_analyzer.app.state import AppContext
from wa_analyzer.app.ui_helpers import get_correlation_text


@st.fragment
def render(ctx: AppContext) -> None:
    analyzer = ctx.analyzer
    df_base = ctx.df_base
    exclude_me = ctx.exclude_me

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
