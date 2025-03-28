import streamlit as st
import pandas as pd
from drift_reports.base_drift_report import Report

class TabularDriftReport(Report):

    PD_NUMERICAL_KEYS = ["number"]
    PD_CATEGORICAL_KEYS = ["category", "object", "string"]
    PD_DATETIME_KEYS = ["datetime"]
    
    def __init__(self, refe_data, test_data):
        self.df_ref = pd.read_csv(refe_data)
        self.df_ref = self.df_ref.convert_dtypes()
        self.df_test = pd.read_csv(test_data)
        self.df_test = self.df_test.convert_dtypes()
        # if st.session_state.get("schema") is None:
        #     st.session_state.schema = self.df_test.dtypes
        # else:            
        #     try:
        #         self.df_ref = self.df_ref.astype(st.session_state.schema)
        #         self.df_test = self.df_test.astype(st.session_state.schema)
        #     except Exception as e:
        #         print('Exception on data types: ', e)
        #         st.toast("Could not cast data to schema! ", icon="🚨")
        #         st.session_state.schema = self.df_test.dtypes  


    def generate_report(self):
        
        tab_data_quality, tab_data_drift, tab_target_drift, tab_data_schema = st.tabs(["📊 Data Quality", "👁️ Data Drift", "🎯 Target Drift", "🔍 Data Schema",])

        with tab_data_quality:
            st.write("##### Data Summary")
            comparison_df = self.compare_dataframes()
            st.dataframe(comparison_df)

            for df_col in self.df_test.columns:
                st_col1, st_col2, st_col3 = st.columns([0.3, 0.3, 0.3])
                with st_col1:
                    st.write(f"##### {df_col}")
                    st.write(self.st_data_type(self.df_test, df_col))
                                            
                with st_col2:
                    st.write("comparison_feature")
                with st_col3:
                    st.write('distribution')
                st.divider()
            print(self.df_test.dtypes) 

        with tab_data_drift:
            st.write("To be implemented")

        with tab_target_drift:
            st.write("To be implemented")

        with tab_data_schema:
            st.write("To be implemented")
    

    def compare_dataframes(self):
        ref_summary = self.summarize_dataframe(self.df_ref)
        test_summary = self.summarize_dataframe(self.df_test)

        return pd.DataFrame([ref_summary, test_summary], index=["Reference", "Test"]).T


    @staticmethod
    def summarize_dataframe(df: pd.DataFrame):
        return {
            "Number of observations": df.shape[0],
            "Number of features": df.shape[1],
            "Missing cells": df.isna().sum().sum(),
            "Constant features": len([col for col in df.columns if df[col].nunique() == 1]),          
        }
    

    @staticmethod
    def st_data_type(df: pd.DataFrame, df_col: str):
        if df_col in df.select_dtypes(include=TabularDriftReport.PD_NUMERICAL_KEYS).columns:
            return "Numerical"
        elif df_col in df.select_dtypes(include=TabularDriftReport.PD_CATEGORICAL_KEYS).columns:
            return "Categorical"
        elif df_col in df.select_dtypes(include=TabularDriftReport.PD_DATETIME_KEYS).columns:
            return "Datetime"
        else:
            return None
        

    
