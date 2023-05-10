# Author: Mirela Cazzolato
# Date: May 2023
# Goal: Generate and visualize features from t-graphs
# =======================================================================

import streamlit as st

from window_dashboard import launch_w_dashboard

# Change page width and configure on load
st.set_page_config(
                   layout="wide",
                   page_icon="🔎",
                   page_title="TgraphSpot App",
                   initial_sidebar_state="auto"
)

with st.sidebar:
    st.write(
        """
        # 🔎 TgraphSpot App
        Fast and Effective Anomaly Detection for tGraphs
        """
    )
    
launch_w_dashboard()
