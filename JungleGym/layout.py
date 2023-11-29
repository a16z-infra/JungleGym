"""
Layout Module for all pages
======================

This module is the general layout of the app for the left side menu and footer.

Author: Marco Mascorro (@mascobot) & Matt Bornstein
Created: November 2023
Version: 0.9 (Experimental)
Status: Development
Python version: 3.9.15
"""
import streamlit as st
import st_pages

def sidebar_footer():
    ##Add a sidebar with Resources:
    sidebar = st.sidebar
    sidebar.header('**Resources:**')
    sidebar.markdown("<a style='text-decoration:none;' href='https://github.com/a16z-infra/JungleGym'><font size=3>GitHub Repo</font></a>", unsafe_allow_html=True)
    sidebar.markdown("<a style='text-decoration:none;' href='https://docs.junglegym.ai/junglegym/junglegym-ai-overview'><font size=3>API Documentation</font></a>", unsafe_allow_html=True)
    sidebar.markdown("<a style='text-decoration:none;' href='https://arxiv.org/abs/2306.06070'><font size=3>Mind2Web Paper</font></a>", unsafe_allow_html=True)
    sidebar.markdown("<a style='text-decoration:none;' href='https://arxiv.org/abs/2307.13854'><font size=3>WebArena Paper</font></a>", unsafe_allow_html=True)
    sidebar.markdown("<a style='text-decoration:none;' href='https://arxiv.org/abs/2310.12823'><font size=3>AgentTuning Paper</font></a>", unsafe_allow_html=True)
    # sidebar.header('**Other Relevant Papers:**')
    # sidebar.markdown("<a style='text-decoration:none;' href='https://proceedings.mlr.press/v70/shi17a.html'><font size=3>World of Bits Paper</font></a>", unsafe_allow_html=True)
    # sidebar.markdown("<a style='text-decoration:none;' href='https://arxiv.org/abs/2308.03688'><font size=3>AgentBench Paper</font></a>", unsafe_allow_html=True)
    st.sidebar.markdown('**by a16z Infra**')

@st.cache_data()
def main_layout():
    # Set config for a cleaner menu, footer & background:
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                body {
                    background-color: #fff;
                    }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # show the menu
    st_pages.add_indentation()
    st_pages.show_pages_from_config()

    # expand page nav
    css='''
    div[data-testid="stSidebarNav"], div[data-testid="stSidebarNav"] > ul {
        min-height: 50vh
    }
    '''
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    # show the footer
    sidebar_footer()


def init_page():
    # Set page title page config:
    st.set_page_config(page_title="JungleGym", page_icon="🦉", layout="wide")

    main_layout()