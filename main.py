import streamlit as st
from importlib import import_module
from drift_reports.tabular_drift_report import TabularDriftReport
from drift_reports.image_drift_report import ImageDriftReport
from drift_reports.text_drift_report import TextDriftReport
from drift_reports.base_drift_report import Report
from static.styles import apply_css
import warnings

from use_case import Usecase
warnings.filterwarnings("ignore", category=FutureWarning, message="'H' is deprecated and will be removed in a future version, please use 'h' instead.")


st.set_page_config(page_title="Drift Detection", page_icon="static/icsa_logo.png", layout="wide")
apply_css(sidebar_width=440)

_DATA_TYPE_SETTINGS = {
    "Tabular": {
        "report": TabularDriftReport,
        "file_uploader_settings": {"accept_multiple_files": False, "type": ["csv"]}
    },
    "Image": {
        "report": ImageDriftReport,
        "file_uploader_settings": {"accept_multiple_files": True, "type": ["png"]}
    },
    "Text": {
        "report": TextDriftReport,
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
        
        use_case = st.selectbox("Use Case", ["Custom", "Eecone", "IncomePrediction", "Archimedes"], index=0)
        if use_case == "Custom":
            data_type = st.radio("Data Type", ["Tabular", "Image", "Text"], horizontal=True)
            data_ref = st.file_uploader("Reference Data", **_DATA_TYPE_SETTINGS[data_type]["file_uploader_settings"])
            data_test = st.file_uploader("Test Data", **_DATA_TYPE_SETTINGS[data_type]["file_uploader_settings"])

        drift_method_placeholder = st.empty()
        dt_features_placeholder = st.empty()
        cat_features_placeholder = st.empty()
        ref_slider_placeholder = st.empty()
        test_slider_placeholder = st.empty()

    if use_case == "Custom":
        report = init_report(data_type, data_ref, data_test)
    else:
        module = import_module("use_case")
        use_case_class: Usecase = getattr(module, f'{use_case}Case')()
        report = use_case_class.get_report(ref_slider_placeholder, test_slider_placeholder)
    if report:
        report.generate_streamlit_report(drift_method_placeholder, dt_features_placeholder, cat_features_placeholder)

    print("Done!")


# @st.cache_resource
def init_report(data_type, data_ref, data_test) -> Report:
    if data_ref and data_test:
        print("Initializing report...")
        return _DATA_TYPE_SETTINGS[data_type]["report"](data_ref, data_test)
    if not data_ref:
        print("Please upload reference data to initialize report.")
        st.warning("Please upload reference data.", icon="🚨")
    if not data_test:
        print("Please upload test data to initialize report.")
        st.warning("Please upload test data.", icon="🚨")



if __name__ == "__main__":
    main()



