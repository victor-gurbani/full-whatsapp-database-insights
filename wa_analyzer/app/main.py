"""Streamlit application entrypoint."""

from importlib import import_module


TAB_SPECS = (
    ("activity", "📊 Activity & Top Users", "wa_analyzer.app.tabs.activity"),
    ("behavioral", "🔥 Behavioral Patterns", "wa_analyzer.app.tabs.behavioral"),
    ("gender", "👫 Gender Insights", "wa_analyzer.app.tabs.gender"),
    ("wordcloud", "📝 Word Cloud", "wa_analyzer.app.tabs.wordcloud"),
    ("chat_explorer", "🔍 Chat Explorer", "wa_analyzer.app.tabs.chat_explorer"),
    ("group_explorer", "👥 Group Explorer", "wa_analyzer.app.tabs.group_explorer"),
    ("fun_insights", "🎪 Fun & Insights", "wa_analyzer.app.tabs.fun_insights"),
    ("map_view", "🗺️ Map", "wa_analyzer.app.tabs.map_view"),
    ("chat_viewer", "📱 Chat Viewer", "wa_analyzer.app.tabs.chat_viewer"),
    ("inbox_triage", "📥 Inbox Triage", "wa_analyzer.app.tabs.inbox_triage"),
)


def run_app() -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = [
        "Helvetica",
        "Arial",
        "Apple Color Emoji",
        "Segoe UI Emoji",
        "Noto Color Emoji",
        "DejaVu Sans",
    ]

    from wa_analyzer.app.privacy import av
    from wa_analyzer.app.sidebar import render_sidebar

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

    ctx = render_sidebar()
    if ctx is not None:
        @st.fragment
        def _frag_kpi():
            df_base = ctx.df_base
            anon_numbers = ctx._anon_numbers
            total_msgs_raw = len(df_base)
            sent_raw = df_base[df_base["from_me"] == 1].shape[0]
            received_raw = total_msgs_raw - sent_raw
            unique_contacts_raw = df_base["contact_name"].nunique()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Messages", f"{av(total_msgs_raw, anon_numbers):,}")
            col2.metric("Sent", f"{av(sent_raw, anon_numbers):,}")
            col3.metric("Received", f"{av(received_raw, anon_numbers):,}")
            col4.metric("Unique Contacts", av(unique_contacts_raw, anon_numbers))

        _frag_kpi()

        # --- Lazy active tab ---
        tab_labels = [label for _, label, _ in TAB_SPECS]
        active_label = st.segmented_control(
            "Analysis section",
            tab_labels,
            default=tab_labels[0],
            key="active_analysis_tab",
            label_visibility="collapsed",
        )
        if active_label is None:
            active_label = tab_labels[0]

        module_name = next(
            module for _, label, module in TAB_SPECS if label == active_label
        )
        active_tab = import_module(module_name)
        active_tab.render(ctx)

    else:
        st.info("👈 Please enter file paths.")
