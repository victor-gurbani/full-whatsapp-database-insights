from __future__ import annotations


import streamlit as st

from wa_analyzer.app.state import AppContext


@st.fragment
def render(ctx: AppContext) -> None:
    analyzer = ctx.analyzer

    st.header("🗺️ Location Map")
    with st.spinner("Loading location data..."):
        loc_data = analyzer.get_location_data()
    if not loc_data.empty:
        st.map(loc_data, latitude="latitude", longitude="longitude")
        st.dataframe(loc_data[["contact_name", "timestamp", "place_name"]])
    else:
        st.info("No location data found in this backup.")
