import streamlit as st
import numpy as np
import cv2
from drift_reports.base_drift_report import Report
import scipy.stats as stats
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots


class ImageDriftReport(Report):
    def __init__(self, ref_images, test_images):
        self.ref_images = ref_images
        self.test_images = test_images      


    def generate_streamlit_report(self, *args):
        import streamlit as st    
        st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Image Drift Detection Report</h2>", unsafe_allow_html=True)
        images = self.ref_images + self.test_images
        ref_start, ref_end = st.session_state.ref_slider
        test_start, test_end = st.session_state.test_slider

        ref_images = images[ref_start:ref_end]
        test_images = images[test_start:test_end]
        mrri_values = [self.compute_color_relative_intensity(img, 'red') for img in images]
        ref_mrri_values = [self.compute_color_relative_intensity(img, 'red') for img in ref_images]
        test_mrri_values = [self.compute_color_relative_intensity(img, 'red') for img in test_images]
        ks_stat, p_value = self.perform_ks_test(ref_mrri_values, test_mrri_values)
        print(f"KS Statistic: {ks_stat}, P-Value: {p_value}")

        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.text('Drift Detection')
            drift_info = pd.DataFrame([
                    ['Method:',  "KS Test on MRRI"],
                    ['KS Statistic:', ks_stat],
                    ['P-Value:', p_value],
                    ['Threshold:', 0.05],
                    ['Result:', "Drift Detected!" if p_value < 0.05 else "No Drift Detected"]])
            styled_df = drift_info.style.hide(axis="index").hide(axis="columns").map(self.apply_status_style).set_properties(**{'border': 'none', 'font-size': '11pt'}).set_table_styles([dict(selector="td", props=[("padding-right", "40px"), ("padding-bottom", "5px")])]).to_html()
            st.markdown(styled_df, unsafe_allow_html=True)
        with col2:
            st.plotly_chart(self.plot_mrri_over_time(mrri_values, ref_start, ref_end, test_start, test_end), config={'displayModeBar': False}, use_container_width=True)
            st.plotly_chart(self.plot_avg_comparison(ref_images, test_images), config={'displayModeBar': False}, use_container_width=True)

        


    @staticmethod
    def perform_ks_test(ref_sample, test_sample):
        """Perform the Kolmogorov-Smirnov test to detect significant change."""
        ks_stat, p_value = stats.ks_2samp(ref_sample, test_sample)
        return ks_stat, p_value
    

    @staticmethod
    def compute_color_relative_intensity(image, color='red'):
        """Compute the Red Relative Intensity (RRI) of an image."""
        r, g, b = cv2.split(image)
        total_intensity = (r + g + b).astype(np.float32) + 1e-6  # Avoid division by zero
        if color == 'red':
            r_relative = r.astype(np.float32) / total_intensity
        elif color == 'green':
            r_relative = g.astype(np.float32) / total_intensity
        elif color == 'blue':
            r_relative = b.astype(np.float32) / total_intensity
        else:
            raise ValueError("Color must be 'red', 'green', or 'blue'")

        return np.mean(r_relative)
    

    @staticmethod
    def plot_mrri_over_time(mrri_values, ref_start, ref_end, test_start, test_end):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=list(range(len(mrri_values))),y=mrri_values, name='MRRI', mode='lines+markers', marker=dict(size=4), line=dict(width=2)))
        fig.add_vrect(x0=ref_start, x1=ref_end, fillcolor="cornflowerblue", opacity=0.5, layer="below", line_width=0, name="Reference Images")
        fig.add_vrect(x0=test_start, x1=test_end, fillcolor="whitesmoke", opacity=0.5, layer="below", line_width=0, name="Test Images")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='lightgreen', opacity=0.5), name='Reference Images', showlegend=True))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='lightblue', opacity=0.5), name='Test Images',showlegend=True))

        fig.update_layout(
            height=250,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(t=0, b=0, l=0, r=0),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            xaxis=dict(range=[0, len(mrri_values)-1],autorange=False)
        )

        return fig
        
        
    @staticmethod
    def plot_avg_comparison(ref_images, test_images):
        # Compute average images
        ref_stack = np.stack(ref_images, axis=0)
        ref_avg = np.mean(ref_stack, axis=0).astype(np.uint8)
        test_stack = np.stack(test_images, axis=0)
        test_avg = np.mean(test_stack, axis=0).astype(np.uint8)
        
        # Create subplot with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            '<span style="color: cornflowerblue">Averaged RGB Reference Image</span>',
            '<span style="color: whitesmoke">Averaged RGB Test Image</span>'
        ), horizontal_spacing=0)
        
        # Add images to subplots
        fig.add_trace(go.Image(z=ref_avg), row=1, col=1)
        fig.add_trace(go.Image(z=test_avg), row=1, col=2)
        
        # Update layout
        fig.update_layout(
            margin=dict(t=40, b=10, l=0, r=0),
            showlegend=False
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
        
        return fig
    
    
    @staticmethod
    def apply_status_style(val):
        if val == 'No Drift Detected':
            return 'color: green'
        elif val == 'Drift Detected!':
            return 'color: red'