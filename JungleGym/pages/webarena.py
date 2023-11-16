# -*- coding: utf-8 -*-
"""
Analyze WebArena Dataset Module
======================

This module is designed to visualization and analyze the Mind2Web dataset. Functionality like filtering 
Domains and subdomains, see each task category and get annotation_id's for each task.

Author: Marco Mascorro (@mascobot) & Matt Bornstein
Created: November 2023
Version: 1.0.0
Status: Development
Python version: 3.9.15
"""
#External libraries:
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import os
#Local libraries:
from layout import init_page

# Get server API endpoint:
API_ENDPOINT = 'http://api.junglegym.ai'
MIND2WEB_API_KEY = os.environ.get('MIND2WEB_API_KEY', default='')
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {}'.format(MIND2WEB_API_KEY)
}

init_page()

plotly_config_set = {'displayModeBar': False, 'showlogo': False}

link_to_name = {
    'http://cms.junglegym.ai/admin': 'E-commerce Content Management System (CMS)',
    'http://forum.junglegym.ai': 'Social Forum',
    'http://git.junglegym.ai': 'Gitlab',
    'http://shop.junglegym.ai': 'Shopping',
    'http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000': 'Map',
}

st.title('🏟 WebArena')

st.subheader('Description')

st.markdown(
'''
WebArena is an environment for **deep** development and testing of web agents.
It consists of 8 complete web sites that web agents can interact with.
Each website is either a copy of an open-source app (e.g. [Gitlab](http://git.junglegym.ai/)) or meant to be a realistic simulation of a real app (e.g. [online store](http://shop.junglegym.ai)).
Tests run on WebArena data are unstructured (i.e. there are many different paths to accomplish one task) but repeatable (i.e. the sites never change).

You can read more about WebArena in the [paper](https://arxiv.org/abs/2307.13854) from Zhou, et. al. at Carnegie Mellon.
''')


# Get server API endpoint:
API_ENDPOINT = 'http://api.junglegym.ai'
API_KEY = os.environ.get('MIND2WEB_API_KEY', default='')
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {}'.format(API_KEY)
}

# Load light dataset and cache it for faster loading
@st.cache_data
def load_tasks():
    response = requests.get(API_ENDPOINT + '/get_webarena_tasks')
    if response.status_code == 200:
        try:
            response = json.loads(response.text)
            df = pd.DataFrame(response['data'])
            return df
        except json.JSONDecodeError as e:
            print(e)
            return None
    else:
        print("HTTP request failed with status code:", response.status_code)
        print("Response content: ", response.text)
        return pd.DataFrame()

df_tasks = load_tasks()

# stats_expander = st.expander("Dataset statistics")

# Replacing the links with their corresponding names
df_tasks['start_url_junglegym'] = df_tasks['start_url_junglegym'].replace(link_to_name)

# # Visualization
# col1, col2, col3 = st.columns([1,6,1])
# # Use the middle column for the chart
# with col2:
#     fig1 = px.pie(df_tasks, names='start_url_junglegym', title="Task ")
#     fig1.update_layout(
#         autosize=True, 
#         width=800, 
#         height=550,  # Adjust the size of the chart
#     )
#     st.plotly_chart(fig1, config=plotly_config_set)

st.subheader('How to Use It')

st.write('''
We mirror 6 WebArena environments as part of Junglegym.
You can point a web agent at these mirrors and do whatever you want - no rate limits, risk of spending real money, etc.

WebArena also includes a set of sample tasks for each web site, accessible via the [JungleGym API](https://docs.junglegym.ai/junglegym/junglegym-ai-overview).
These are just examples; you can run any task you'd like against the WebArena mirrors, including via [TreeVoyager](/TreeVoyager).
''')

st.subheader('Try it!')

st.write('''
Here are the web sites we mirror as part of Junglegym:
- [Online store](http://shop.junglegym.ai/)
- [Gitlab](http://git.junglegym.ai/) (You can create an account with a user and pwd)
- [Social Forum](http://forum.junglegym.ai/)
- [Wikipedia](http://wiki.junglegym.ai/)
- [E-Commerce CMS](http://cms.junglegym.ai/) Admin panel (cms.junglegym.ai/admin) u: admin, pwd: admin1234
- [Map](http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000/)
''')

st.write('''
You can browse the WebArena tasks here, to get a sense for what they look like:
''')

# #Download Full dataset: /get_full_webarena_dataset
# def download_full_dataset():
#     full_dataset_response = requests.get(API_ENDPOINT + "/get_full_webarena_dataset")
#     json_data = full_dataset_response.json()['data']
#     json_str = json.dumps(json_data)
#     json_bytes = json_str.encode('utf-8')
#     return json_bytes

# st.download_button(
#     label="Download Full Dataset (791KB)",
#     data=download_full_dataset(),
#     file_name="JungleGym_WebArena_FullDataset.json",
#     mime="application/json",
# )

nickname_to_name = {
    'E-commerce Content Management System (cms.junglegym.ai)': 'E-commerce Content Management System (CMS)',
    'Social Forum (forum.junglegym.ai)': 'Social Forum',
    'Gitlab (git.junglegym.ai)': 'Gitlab',
    'Shopping (shop.junglegym.ai)': 'Shopping',
    'Map': 'Map',
    }

domain_selection = st.selectbox('Choose domain(s)', options=['All'] + list(nickname_to_name.keys()))
if domain_selection != 'All':
    df_tasks = df_tasks[df_tasks['start_url_junglegym'] == nickname_to_name[domain_selection]]

height = 45 * len(df_tasks)#45 is roughly the height of each row in the dataframe
if height > 500:
    height = 500
df_tasks = df_tasks.rename(columns={'intent':'Task', 'start_url_junglegym': 'Domain'})#Rename columns for readability
df_tasks = df_tasks.reset_index(drop=True)
# st.dataframe(df_tasks, use_container_width=True, height=height)

# if domain_selection != 'All':
#     #Download filtered data as JSON:
#     def download_domain_dataset():    
#         domain = [key for key, val in link_to_name.items() if val == nickname_to_name[domain_selection]][0]
#         params = {"domain": domain} 
#         domain_dataset_response = requests.get(API_ENDPOINT + "/get_webarena_by_domain", params=params)
#         json_data = domain_dataset_response.json()
#         json_str = json.dumps(json_data)
#         json_bytes = json_str.encode('utf-8')
#         return json_bytes
#     st.download_button(
#         label="Download Dataset Selection",
#         data=download_domain_dataset(),
#         file_name="JungleGym_WebArena_selection.json",
#         mime="application/json",
#     )

task_list = df_tasks['Task'].tolist()
task = st.selectbox('Select task to see task data:', options =[''] + task_list)
if len(task) > 0:
    params = {"task": task} 
    task_response = requests.get(API_ENDPOINT + "/get_webarena_by_task", params=params)
    # Check if the request was successful
    if task_response.status_code == 200:
        data = task_response.json()['data'][0]
        # Extracting the specified fields
        extracted_data = {
            "task_id": data["task_id"],
            "require_login": data["require_login"],
            "start_url": data["start_url"],
            "intent": data["intent"],
            "eval": data["eval"]
        }
        ground_truth = data["eval"]['reference_answers']
        if ground_truth != None:
            if 'exact_match' in data["eval"]['reference_answers']:
                ground_truth = data["eval"]['reference_answers']['exact_match']
            elif 'must_include' in data["eval"]['reference_answers']:
                ground_truth = data["eval"]['reference_answers']['must_include']
        if isinstance(ground_truth, list):
            ground_truth = ", ".join(ground_truth)

        st.markdown(f"*Domain: " + link_to_name[data["start_url_junglegym"]] + "*")
        st.markdown(f"*Ground truth response (if applicable): " + str(ground_truth) + "*")
        st.markdown(f"*Start URL*: " + data["start_url"])
        if data["require_login"] == True:
            print (link_to_name[data["start_url_junglegym"]])
            if "http://cms.junglegym.ai/admin" == data["start_url_junglegym"]:
                st.markdown(f"*Login required*: Yes (username: admin, password: admin1234)")
            else:
                st.markdown(f"*Login required*: Yes")
        else:
            st.markdown(f"*Login required*: No")
        st.markdown(f"*Task JSON data*: ")
        # Convert JSON object to formatted string
        json_str = json.dumps(extracted_data, indent=4)
        # Display on Streamlit
        st.code(json_str, language='json')
        #Download filtered data as JSON:
        json_str = json.dumps(data)
        json_bytes = json_str.encode('utf-8')
        st.download_button(
        label="Download task data",
        data=json_bytes,
        file_name="JungleGym_WebArena_TaskData.json",
        mime="application/json",
        )
    else:
        print(f"Request failed with status code {task_response.status_code}")

