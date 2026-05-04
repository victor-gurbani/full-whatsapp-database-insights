from __future__ import annotations


import streamlit as st
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from wa_analyzer.app.state import AppContext


@st.fragment
def render(ctx: AppContext) -> None:
    full_analyzer = ctx.full_analyzer
    min_word_len = ctx.min_word_len
    exclude_emails = ctx.exclude_emails

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
