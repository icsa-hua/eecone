import streamlit as st

def apply_css(sidebar_width=450):
    st.markdown(f"""
        <style>
            .block-container {{
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                    margin-top: -0.5rem;  # -0.5rem without login / -4rem with login
            }}
            [data-testid="stSidebarHeader"] {{
                display: none
            }}
            [data-testid="stSidebarUserContent"] {{
                padding-top: 0.4rem;
                padding-right: 1.5rem;
                padding-left: 1.5rem;
                padding-bottom: 2rem;
            }}
            [data-testid="stSidebar"] {{
                width: {sidebar_width}px !important;
            }}
            [data-testid="stHeader"] {{
                background: transparent;
                pointer-events: none;
            }}
            [data-testid="stToolbar"] {{
                z-index: 1000;
                pointer-events: auto;
            }}
            [data-testid="StyledFullScreenButton"] {{
                right: 0rem;
                top: -0.5rem;
            }}
            hr {{
                margin: 0px;
            }}
            section.stSidebar > div:last-child > div {{
                display: none;
            }}
            .stColumn.st-emotion-cache-17ad2io .stElementToolbar {{
                display: none;
            }}
        </style>
        """, unsafe_allow_html=True)