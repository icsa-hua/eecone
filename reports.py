from abc import ABC, abstractmethod
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from evidently.report import Report
from evidently.metric_preset.metric_preset import MetricPreset


class EeconeReport(ABC):
    @abstractmethod
    def create_report(self, df_ref, df_prod):
        pass


class EvidentlyReport(EeconeReport):
    def __init__(self, report: MetricPreset):
        self.report = report
    
    def create_report(self, df_ref, df_prod):
        report = Report([self.report])
        report.run(reference_data=df_ref, current_data=df_prod)
        components.html(report._repr_html_(), height=920, scrolling=True)


class DataPreviewReport(EeconeReport):
    def create_report(self, df_ref: pd.DataFrame, df_prod: pd.DataFrame):
        st.write("Reference data:")
        st.write(df_ref.head(100))
        st.write("Current data:")
        st.write(df_prod.head(100))


class LifetimeExplorationReport(EeconeReport):
    def create_report(self, df_ref, df_prod):
        pass

        