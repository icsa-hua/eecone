import pandas as pd
import streamlit as st
from archimedes_utils import *
from static.styles import apply_css
import os
import cv2


st.set_page_config(page_title="Drift Detection", page_icon="static/icsa_logo.png", layout="wide")
apply_css(sidebar_width=440)


def main():
    with st.sidebar:
        st.text("")
        icon, title = st.columns([0.3, 0.7])
        icon.markdown("<a href='https://icsa.hua.gr/'><img src='app/static/icsa_logo.png' style='width:95%;'></a>", unsafe_allow_html=True)
        title.markdown("<h1>Intelligent Computer Systems & Applications Archimedes</h1>", unsafe_allow_html=True)
        st.divider()

        images = load_images('data/thermal_ironbow/')

        with st.form('ref_test_ranges', border=False):
            st.session_state.ref_start, st.session_state.ref_end = st.slider("Reference Image Range", 0, len(images), (0, 321), 1, key="ref_slider")
            st.session_state.test_start, st.session_state.test_end = st.slider("Test Image Range", 0, len(images), (321, len(images)), 1, key="test_slider")
            st.number_input("KS Test Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="ks_threshold")
            st.number_input("Test Images Window Size", min_value=1, max_value=len(images), value=5, step=1, key="test_window_size")
            col1, col2 = st.columns(2)
            with col1:
                run_button = st.form_submit_button(label='Run')
            with col2:
                stream = st.toggle("Iterate test images", value=False, key="stream")

    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Image Drift Detection Report</h2>", unsafe_allow_html=True)
    if run_button:
        if stream:
            stream_app(1, images)
        else:
            run(images)


def run(images):
    ref_start, ref_end, test_start, test_end, ks_threshold = st.session_state.ref_start, st.session_state.ref_end, st.session_state.test_start, st.session_state.test_end, st.session_state.ks_threshold
    ref_images = images[ref_start:ref_end]
    if st.session_state.stream:
        test_end = test_start + st.session_state.test_window_size
        test_images = images[test_start:test_end]
        st.session_state.test_start = test_start + 1
    else:
        test_images = images[test_start:test_end]
    mrri_values = [compute_color_relative_intensity(img, 'red') for img in images]
    ref_mrri_values = [compute_color_relative_intensity(img, 'red') for img in ref_images]
    test_mrri_values = [compute_color_relative_intensity(img, 'red') for img in test_images]
    ks_stat, p_value = perform_ks_test(ref_mrri_values, test_mrri_values)
    is_drift_detected = p_value < ks_threshold
    print(f"KS Statistic: {ks_stat}, P-Value: {p_value}, Drift Detected: {is_drift_detected}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="KS Statistic", value=ks_stat)
    with col2:
        st.metric(label="P-Value", value=p_value)
    with col3:
        st.markdown(f"<h3 style='text-align: center; color: {'red' if is_drift_detected else 'green'};'>{'Fire Alert!' if is_drift_detected else 'All good :)'}</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.plotly_chart(plot_mrri_over_time(mrri_values, ref_start, ref_end, test_start, test_end), config={'displayModeBar': False}, use_container_width=True)
    with col2:
        st.plotly_chart(plot_avg_rgb(test_images), config={'displayModeBar': False}, use_container_width=True)


def stream_app(auto_refresh_rate, images):
    @st.fragment(run_every=auto_refresh_rate)
    def stream_app_inner(images):
        run(images)
    stream_app_inner(images)


if __name__ == "__main__":
    main()
