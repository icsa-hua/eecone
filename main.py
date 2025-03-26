import pandas as pd
import streamlit as st
from reports import *
from static.styles import apply_css
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="'H' is deprecated and will be removed in a future version, please use 'h' instead.")


st.set_page_config(page_title="Drift Detection", page_icon="static/icsa_logo.png", layout="wide")
apply_css(sidebar_width=440)
reports: dict[str, Report] = {
    '🔍 Data Preview': DataPreviewReport(),
    '📊 Data Quality': DataQualityReport(),
    '👁️ Data Drift': DataDriftReport(),
    '🎯 Target Drift': TargetDriftReport(),
    '⏳ Lifetime Exploration': LifetimeExplorationReport(),
}


def main():

    with st.sidebar:
        st.text("")
        icon, title = st.columns([0.3, 0.7])
        icon.markdown("<a href='https://icsa.hua.gr/'><img src='app/static/icsa_logo.png' style='width:95%;'></a>", unsafe_allow_html=True)
        title.markdown("<h1>Intelligent Computer Systems & Applications Drift Detection</h1>", unsafe_allow_html=True)
        st.divider()
        
        with st.form("my_form"):
            explore_item = st.selectbox('Report', list(reports.keys()))
            ref_data = st.file_uploader("Reference Data")
            cur_data = st.file_uploader("Current Data")
            submitted = st.form_submit_button("Create Report")

    if submitted:
        with st.spinner("Wait for it...", show_time=True):
            reports[explore_item].create_report(ref_data, cur_data)

    print("Done!")


if __name__ == "__main__":
    main()



