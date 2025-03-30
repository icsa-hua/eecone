from dataclasses import dataclass
import pandas as pd
import plotly.graph_objects as go
from drift_reports.base_drift_report import Report
from alibi_detect.cd import ChiSquareDrift, TabularDrift
import numpy as np


@dataclass
class DriftDetectionResult:
    '''TODO: Class to hold the results of drift detection'''
    pass

class TabularDriftReport(Report):

    PD_NUMERICAL_KEYS = ["number"]
    PD_CATEGORICAL_KEYS = ["category", "object", "string"]
    PD_DATETIME_KEYS = ["datetime"]
    
    def __init__(self, refe_data, test_data, dt_columns=None):
        self.df_ref = pd.read_csv(refe_data)
        self.df_ref = self.df_ref.convert_dtypes()
        self.df_test = pd.read_csv(test_data)
        self.df_test = self.df_test.convert_dtypes()
        self.schema = self.df_test.dtypes
        self.common_features = self.df_ref.columns.intersection(self.df_test.columns)
        self.common_features = [col for col in self.common_features if self.df_ref[col].dtype == self.df_test[col].dtype]
        if dt_columns:
            self.cast_dt_columns(dt_columns)
    

    def detect_drift(self, method="Statistical", threshold=0.05):
        if method == "Statistical":
            categorical_features = self.df_ref.select_dtypes(include=self.PD_CATEGORICAL_KEYS).columns
            # categories_per_feature = {col: None for col in categorical_features}
            categories_per_feature = {col: self.df_ref[col].nunique() for col in categorical_features}
            X_ref = self.df_ref[self.common_features].to_numpy()
            X_test = self.df_test[self.common_features].to_numpy()
            drift_detector = TabularDrift(X_ref, p_val=.05, categories_per_feature=categories_per_feature)
            preds = drift_detector.predict(X_test)

            return preds['data']


    def cast_dt_columns(self, dt_columns):
        for col in dt_columns:
            self.df_ref[col] = pd.to_datetime(self.df_ref[col])
            self.df_test[col] = pd.to_datetime(self.df_test[col])


    def compare_dataframes(self):
        ref_summary = self.summarize_dataframe(self.df_ref)
        test_summary = self.summarize_dataframe(self.df_test)

        return pd.DataFrame([ref_summary, test_summary], index=["Reference", "Test"]).T
    

    def compare_feature_stats(self, feature_name):
        ref_stats = self.df_ref[feature_name].describe()
        test_stats = self.df_test[feature_name].describe()

        return pd.DataFrame([ref_stats, test_stats], index=["Reference", "Test"]).T
    

    def compare_hist_distibutions(self, feature_name, is_cumulative=False):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.df_ref[feature_name], name='Reference', opacity=0.5, cumulative_enabled=is_cumulative))
        fig.add_trace(go.Histogram(x=self.df_test[feature_name], name='Test', opacity=0.5, cumulative_enabled=is_cumulative))
        fig.update_layout(
            barmode='overlay',
            height=300,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        return fig
    

    def compare_ecdf_distributions(self, feature_name):
        # Calculate ECDF for reference data
        x_ref = np.sort(self.df_ref[feature_name])
        y_ref = np.arange(1, len(x_ref)+1) / len(x_ref)        
        # Calculate ECDF for test data
        x_test = np.sort(self.df_test[feature_name])
        y_test = np.arange(1, len(x_test)+1) / len(x_test)        
        # Combine all x points to evaluate ECDFs at same positions
        all_x = np.sort(np.concatenate([x_ref, x_test]))
        y_ref_interp = np.interp(all_x, x_ref, y_ref)
        y_test_interp = np.interp(all_x, x_test, y_test)        
        # Calculate maximum absolute difference (Kolmogorov-Smirnov D)
        diffs = np.abs(y_ref_interp - y_test_interp)
        max_diff = np.max(diffs)
        max_diff_idx = np.argmax(diffs)
        max_diff_x = all_x[max_diff_idx]
        
        fig = go.Figure()
        # Add ECDF traces
        fig.add_trace(go.Scatter(x=x_ref, y=y_ref, name='Reference', line=dict(width=2)))
        fig.add_trace(go.Scatter(x=x_test, y=y_test, name='Test', line=dict(width=2)))        
        # Add maximum difference indicator
        fig.add_trace(go.Scatter(x=[max_diff_x, max_diff_x],
            y=[y_ref_interp[max_diff_idx], y_test_interp[max_diff_idx]],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name=f'Max abs diff (D={max_diff:.3f})'
        ))
        
        fig.update_layout(
            height=310,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig


    def generate_streamlit_report(self, drift_method_placeholder, dt_features_placeholder):
            import streamlit as st

            with drift_method_placeholder:
                drift_method = st.selectbox("Drift Detection Method", ["Statistical", "MMD"], index=0)

            with dt_features_placeholder:
                dt_features = st.multiselect('Datetime features', self.common_features)
                self.cast_dt_columns(dt_features)
                        
            st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Tabular Drift Detection Report</h2>", unsafe_allow_html=True)

            with st.spinner("Wait for it...", show_time=True):
                result = self.detect_drift(method=drift_method)
                overall_comparison = self.compare_dataframes()

            st_col1, st_col2 = st.columns([0.2, 0.8])
            with st_col1:
                st.write("##### Overall")
                drift_info = pd.DataFrame([
                    ['Drift Method:', drift_method],
                    ['Threshold:', result['threshold']],
                    ['Drift:', "Yes" if result['is_drift'] else "No"]])
                styled_df = drift_info.style.hide(axis="index").hide(axis="columns").map(self.apply_status_style).set_properties(**{'border': 'none', 'font-size': '11pt'}).set_table_styles([dict(selector="td", props=[("padding-right", "40px"), ("padding-bottom", "5px")])]).to_html()
                st.markdown(styled_df, unsafe_allow_html=True)

            with st_col2:                                
                st.dataframe(overall_comparison)
            st.divider()

            for idx, df_col in enumerate(self.common_features):
                feature_p_val = result['p_val'][idx]
                feature_distance = result['distance'][idx]
                feature_is_drift = "Yes" if feature_p_val < result['threshold'] else "No"
                st_col1, st_col2, st_col3 = st.columns([0.2, 0.4, 0.4])
                with st_col1:
                    st.write(f"##### {df_col}")
                    feature_info = pd.DataFrame([
                        ['Data type:', self.get_simplified_type(self.df_test, df_col)],
                        ['Distance:', feature_distance],
                        ['p-value:', feature_p_val],
                        ['Drift:', feature_is_drift]])
                    styled_df = feature_info.style.hide(axis="index").hide(axis="columns").map(self.apply_status_style).set_properties(**{'border': 'none', 'font-size': '11pt'}).set_table_styles([dict(selector="td", props=[("padding-right", "40px"), ("padding-bottom", "5px")])]).to_html()
                    st.markdown(styled_df, unsafe_allow_html=True)

                with st_col2:
                    feature_comparison = self.compare_feature_stats(df_col)
                    st.dataframe(feature_comparison)
                    
                with st_col3:
                    if self.get_simplified_type(self.df_test, df_col) == "Numerical":
                        tab_hist, tab_ecdf = st.tabs(["Histogram", "ECDF"])
                        with tab_hist:
                            st.plotly_chart(self.compare_hist_distibutions(df_col), config={'displayModeBar': False}, use_container_width=True)
                        with tab_ecdf:
                            st.plotly_chart(self.compare_ecdf_distributions(df_col), config={'displayModeBar': False}, use_container_width=True)
                        

                st.divider()


    @staticmethod
    def summarize_dataframe(df: pd.DataFrame):
        return {
            "Number of observations": df.shape[0],
            "Number of features": df.shape[1],
            "Missing cells": df.isna().sum().sum(),
            "Constant features": len([col for col in df.columns if df[col].nunique() == 1]),          
        }
    

    @staticmethod
    def get_simplified_type(df: pd.DataFrame, df_col: str):
        if df_col in df.select_dtypes(include=TabularDriftReport.PD_NUMERICAL_KEYS).columns:
            return "Numerical"
        elif df_col in df.select_dtypes(include=TabularDriftReport.PD_CATEGORICAL_KEYS).columns:
            return "Categorical"
        elif df_col in df.select_dtypes(include=TabularDriftReport.PD_DATETIME_KEYS).columns:
            return "Datetime"
        else:
            return None
        
    @staticmethod
    def apply_status_style(val):
        if val == 'No':
            return 'color: green'
        elif val == 'Yes':
            return 'color: red'
        
