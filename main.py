import pandas as pd
import streamlit as st
from reports import *
from static.styles import apply_css
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="'H' is deprecated and will be removed in a future version, please use 'h' instead.")


st.set_page_config(page_title="Drift Detection", page_icon="static/icsa_logo.png", layout="wide")
apply_css(sidebar_width=440)

_DATA_TYPE_SETTINGS = {
    "Tabular": {
        "file_uploader_settings": {"accept_multiple_files": False, "type": ["csv"]}
    },
    "Image": {
        "file_uploader_settings": {"accept_multiple_files": True, "type": ["png"]}
    },
    "Text": {
        "file_uploader_settings": {"accept_multiple_files": False, "type": ["txt"]}
    }
}


def main():

    with st.sidebar:
        st.text("")
        icon, title = st.columns([0.3, 0.7])
        icon.markdown("<a href='https://icsa.hua.gr/'><img src='app/static/icsa_logo.png' style='width:95%;'></a>", unsafe_allow_html=True)
        title.markdown("<h1>Intelligent Computer Systems & Applications Drift Detection</h1>", unsafe_allow_html=True)
        st.divider()
        
        data_type = st.radio("Data Type", ["Tabular", "Image", "Text"], horizontal=True)
        refe_data = st.file_uploader("Reference Data", **_DATA_TYPE_SETTINGS[data_type]["file_uploader_settings"])
        test_data = st.file_uploader("Test Data", **_DATA_TYPE_SETTINGS[data_type]["file_uploader_settings"])

    if refe_data and test_data:
        pass
    if not refe_data:
        st.warning("Please upload reference data.", icon="🚨")
    if not test_data:
        st.warning("Please upload test data.", icon="🚨")

    print("Done!")


if __name__ == "__main__":
    main()



