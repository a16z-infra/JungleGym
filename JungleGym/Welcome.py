# -*- coding: utf-8 -*-
"""
Welcome Page
======================

This module displays the welcome message and a description of the project.

Author: Marco Mascorro (@mascobot) & Matt Bornstein
Created: November 2023
Version: 0.0.9 (Experimental)
Status: Development
Python version: 3.9.15
"""
#External libraries:
import streamlit as st
import os
import base64
import sentry_sdk
#Local libraries:
from layout import init_page

####Sentry####
try:
    SENTRY_DSN = os.environ.get('SENTRY_DSN', default='')
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )
except Exception as e:
    print(e)
###############
init_page()

#Create welcome page function:
def welcome():
    image_junglegym = os.path.join(os.getcwd(), "JungleGymLogo.png")
    with open(image_junglegym, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
    f'''
    <style>
    img {{
        margin-top: 0px;
        padding-top: 0px;
    }}
    </style>
    <div style="display: flex; justify-content: center; padding-bottom: 20px;"><img src="data:image/png;base64,{b64_string}" width="550"></div>
    ''',
    unsafe_allow_html=True,
)
    st.markdown(
        """
        <div style='text-align: center; font-size: 30px; font-weight: bold; padding-bottom: 20px;'>
            Open source playground for Building Autonomous Web Agents
        </div>
        """,
        unsafe_allow_html=True
    )

    video_html = """
    <div style="display: flex; justify-content: center; padding-bottom: 20px;">
        <iframe width="776" height="432" 
        src="https://player.vimeo.com/video/885186258?h=be3d2b842d&amp;badge=0&amp;autopause=0&amp;quality_selector=1&amp;player_id=0&amp;app_id=58479" 
        frameborder="0" 
        allow="autoplay; fullscreen; picture-in-picture" 
        title="JungleGym Demo"></iframe>
    </div>
    """

    st.markdown(video_html, unsafe_allow_html=True)

    st.markdown(
    """
    JungleGym is an open source playground for testing and developing autonomous web agents. This is not an Agent, but rather a tool to test and build agents with datasets.
    Here, you can download demonstration data, test your agents with ground-truth web trajectories (using the JungleGym APIs), and design your agents.
    These Datasets are available in the APIs with examples in the [JungleGym Repo](https://github.com/a16z-infra/JungleGym) and in this playground, using the links to the left.

    We're hosting 3 datasets:
    - **Mind2Web**: Ground truth for ~2k tasks across 137 websites, including full HTML page states and screenshots. Good for *broad* testing and development across a range of sites and tasks.
    - **WebArena**: 6 realistic, fully functional, sandboxed web sites. Good for *deep* testing of many tasks and paths on a single site.
    - **AgentInstruct**: ~1.8k agent trajectories designed for fine-tuning language models (i.e. Llama2, Mistral, etc) on agent tasks.

    We're also sharing one tool:
    - **TreeVoyager**: An LLM-based DOM parser that returns the best URL HTML/DOM element to interact with given a task (e.g. 'buy coffee').
    We found this to be a hard problem for properly parsing websites for web agents, so we're sharing our (partial) solution.
    """)
    
    st.write(
        """
            ### Why are we doing this?

            We wanted to make our minor open source contribution to the field of autonomous agents. This is a very early prototype version still in development.
            There's no standardized way to benchmark web agents, and datasets and tools are too sparse on the web. These agents operate in the wild web, but clear metrics, datasets, and insights into their effectiveness are lacking.

            It's early, and we hope this helps agent builders and developers test, benchmark, and explore datasets to continue the advancement in the field.
            
            Please share any feedback or suggestions and please contribute via the [Github repo](https://github.com/a16z-infra/JungleGym).

        """)

welcome()
