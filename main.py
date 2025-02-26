import pandas as pd
import streamlit as st
from reports import EeconeReport, DataPreviewReport, EvidentlyReport, LifetimeExplorationReport
from static.styles import apply_css
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="'H' is deprecated and will be removed in a future version, please use 'h' instead.")


st.set_page_config(page_title="EECONE", page_icon="static/eecone_icon.png", layout="wide")
apply_css(sidebar_width=420)
reports: dict[str, EeconeReport] = {
    # '🔍 Data Preview': DataPreviewReport(),
    '📊 Data Quality': EvidentlyReport(DataQualityPreset()),
    '👁️ Data Drift': EvidentlyReport(DataDriftPreset()),
    '🎯 Target Drift': EvidentlyReport(TargetDriftPreset()),
    '⏳ Lifetime Exploration': LifetimeExplorationReport(),
}


def main():
    if st.session_state.get("data") is None:
        df = get_dummy_data()
    else:
        df = st.session_state["data"]

    with st.sidebar:
        st.text("")
        _, icon, title = st.columns([0.01, 0.4, 0.63])
        icon.markdown("<a href='https://www.eecone.com/eecone/home/'><img src='app/static/eecone_icon.png' style='width:95%;'></a>", unsafe_allow_html=True)
        title.markdown("<h1 style='color: #2EAA53;'>Eco-design Tool Advanced Reliability Lifetime Exploration</h1>", unsafe_allow_html=True)

        if st.button(st.session_state.get('csv_file_name', 'dummy_data.csv'), icon='✏️'):
            upload_data()
        
        with st.form("my_form"):
            explore_item = st.selectbox('Report', list(reports.keys()))
            st.selectbox('Datetime column', df.columns, key="datetime_column")
            st.date_input('Reference Period', key="ref_period")
            st.date_input("Current Period", key="cur_period")
            submitted = st.form_submit_button("Create Report")

    if submitted or explore_item == "📊 Data Quality":
        df_ref = df[(df[st.session_state['datetime_column']] >= pd.to_datetime(st.session_state['ref_period'][0])) & (df["timestamp"] <= pd.to_datetime(st.session_state['ref_period'][1]))]
        df_prod = df[(df[st.session_state['datetime_column']] >= pd.to_datetime(st.session_state['cur_period'][0])) & (df["timestamp"] <= pd.to_datetime(st.session_state['cur_period'][1]))]
        with st.spinner("Wait for it...", show_time=True):
            reports[explore_item].create_report(df_ref, df_prod)

    print("Done!")



def get_dummy_data():
    df = pd.read_csv("data/dummy_data.csv", delimiter=",")
    df.convert_dtypes()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['target'] = df['mill_kw']
    st.session_state["data"] = df
    st.session_state['datetime_column'] = "timestamp"
    st.session_state['ref_period'] = (df["timestamp"].min(), df["timestamp"].min() + pd.Timedelta(days=1))
    st.session_state['cur_period'] = (df["timestamp"].max() - pd.Timedelta(days=1), df["timestamp"].max())

    return df


@st.dialog("Upload your data")
def upload_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, delimiter=",")
        df.convert_dtypes()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        st.session_state['csv_file_name'] = uploaded_file.name
        st.session_state["data"] = df
        st.session_state['datetime_column'] = try_to_find_datetime_column(df)
        st.session_state['ref_period'] = (df["timestamp"].min(), df["timestamp"].min() + pd.Timedelta(days=1))
        st.session_state['cur_period'] = (df["timestamp"].max() - pd.Timedelta(days=1), df["timestamp"].max())
        st.toast("Data uploaded successfully!", icon='🎉')
        st.rerun()


def try_to_find_datetime_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            print(f"Found datetime column: {col}")
            return col
    return df.columns[1]


if __name__ == "__main__":
    main()



