"""Streamlit application entrypoint."""


def run_app() -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    from wa_analyzer.app.privacy import av
    from wa_analyzer.app.sidebar import render_sidebar
    from wa_analyzer.app.tabs import (
        activity,
        behavioral,
        chat_explorer,
        chat_viewer,
        fun_insights,
        gender,
        group_explorer,
        inbox_triage,
        map_view,
        wordcloud,
    )

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


        with tab1:
            activity.render(ctx)


        with tab2:
            behavioral.render(ctx)


        with tab3:
            gender.render(ctx)


        with tab4:
            wordcloud.render(ctx)


        with tab5:
            chat_explorer.render(ctx)


        with tab6:
            group_explorer.render(ctx)


        with tab7:
            fun_insights.render(ctx)


        with tab8:
            map_view.render(ctx)


        with tab9:
            chat_viewer.render(ctx)


        with tab10:
            inbox_triage.render(ctx)

    else:
        st.info("👈 Please enter file paths.")
