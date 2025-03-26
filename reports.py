from abc import ABC, abstractmethod
import pandas as pd
import streamlit as st


class Report(ABC):
    @abstractmethod
    def create_report(self, df_ref, df_prod):
        pass


class DataPreviewReport(Report):
    def create_report(self, df_ref: pd.DataFrame, df_prod: pd.DataFrame):
        st.write("Reference data:")
        st.write(df_ref.head(100))
        st.write("Current data:")
        st.write(df_prod.head(100))


class DataQualityReport(Report):
    def create_report(self, df_ref, df_prod):
        st.write('Not implemented yet!')


class DataDriftReport(Report):
    def create_report(self, df_ref, df_prod):
        st.write('Not implemented yet!')


class TargetDriftReport(Report):
    def create_report(self, df_ref, df_prod):
        st.write('Not implemented yet!')


class LifetimeExplorationReport(Report):
    def create_report(self, df_ref, df_prod):
        st.write('Not implemented yet!')

        