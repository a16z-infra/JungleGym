# -*- coding: utf-8 -*-
"""
Analyze Mind2Web Dataset Module
======================

This module is designed to visualization and analyze the Mind2Web dataset. Functionality like filtering 
Domains and subdomains, see each task category and get annotation_id's for each task.

Author: Marco Mascorro (@mascobot) & Matt Bornstein
Created: November 2023
Version: 0.9 (Experimental)
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
from PIL import Image
import base64
import io
#Local libraries:
from layout import init_page


MAX_IM_SIZE = 1000
# Get Mind2Web server API endpoint:
MIND2WEB_ENDPOINT = 'http://api.junglegym.ai'
MIND2WEB_API_KEY = os.environ.get('MIND2WEB_API_KEY', default='')
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {}'.format(MIND2WEB_API_KEY)
}

# if 'showing_subset_data' not in st.session_state:
#     st.session_state.showing_subset_data = False
if 'simulating' not in st.session_state:
    st.session_state.simulating = False

init_page()

plotly_config_set = {'displayModeBar': False, 'showlogo': False}

# Load light dataset and cache it for faster loading
@st.cache_data
def load_light_train_dataset():
    response = requests.get(MIND2WEB_ENDPOINT + '/load_light_train_dataset', headers=headers)
    if response.status_code == 200:
        try:
            response = json.loads(response.text)
            df = pd.DataFrame(response['data'])
            return df
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
    else:
        print("HTTP request failed with status code:", response.status_code)
        print("Response content: ", response.text)
        return pd.DataFrame()
    

st.title('🧠 Mind2Web')

st.subheader('Description')

st.write(
'''
Mind2Web is a dataset for **broad** development and testing of web agents.
It contains ~2k tasks across 137 real websites, each with associated actions and snapshots.
A task is represented by a string like "Find a mini van at Brooklyn City from April 5th to April 8th for a 22 year old renter."
Actions are represented by a target element (e.g. "[button] Vehicle Class") and an operation (e.g. "CLICK").

There's more information in the [website](https://osu-nlp-group.github.io/Mind2Web/) and [paper](https://arxiv.org/abs/2306.06070) from Deng, et. al. at Ohio State University.
''')


stats_expander = st.expander('Summary statistics')

df = load_light_train_dataset()
# Create two columns for the pie charts
col1, col2 = stats_expander.columns(2)

# Visualization for 'domain' column in the first column
fig1 = px.pie(df, names='domain', title="Domain")
fig1.update_layout(autosize=False, width=360, height=360)  # Adjust the size of the chart
col1.plotly_chart(fig1, config=plotly_config_set)

# Visualization for 'subdomain' column in the second column
fig2 = px.pie(df, names='subdomain', title="Subdomain")
fig2.update_layout(autosize=False, width=360, height=360)  # Adjust the size of the chart
col2.plotly_chart(fig2, config=plotly_config_set)

# Visualization for 'website' column - as horizontal bar chart
# st.subheader('Website breakdown')
df['website_counts'] = df.groupby('website')['website'].transform('count')
df = df.sort_values(by='website_counts', ascending=False)
fig3 = px.bar(df, y='website', hover_data=['website_counts'], labels={'website_counts': 'Count'}, title="Website breakdown")
fig3.update_traces(hovertemplate='Website: %{y}<br>Count: %{customdata[0]}')
fig3.update_layout(height=800)  # Set the vertical size of the chart
fig3.update_xaxes(title_text='Task count')  # Add your new label here
stats_expander.plotly_chart(fig3, config=plotly_config_set)


st.subheader('How to Use It')
         
st.markdown(
'''
The Mind2Web dataset is large (including full HTML page states and screenshots), so we're hosting it in the [JungleGym API](https://docs.junglegym.ai/junglegym/api-documentation/mind2web-api).
The most common way to use this is to:
1. Download a subset of Mind2Web annotation IDs
2. Execute your agent on the tasks and sites specified by those IDs
3. Use the API to fetch full ground truth for those tasks, and compare against agent output. Keep in mind, this is just one valid path to accomplish each task.

*Note: We are not including the Mind2Web test data in our API, since the authors don't want this exposed to the public internet in cleartext.*
''')


st.subheader('Try It!')

# subset_expander = st.expander("Download subset")

df_subset = load_light_train_dataset()

# Display a table with selected columns at the end of the app
# subset_expander.write(
# '''
# This page lets you batch download a subset of Mind2Web data for local training and evaluation.
# You can filter with any combination of Domain/ Subdomain/ Website, or list specific task IDs.
# '''
# )

st.write("First, filter the Mind2Web data (optional):")

# def show_data(show):
#     st.session_state.showing_subset_data = show

# Add selectbox widgets to allow the selection of specific items for each category
domain = st.selectbox('Choose domain', options=['All'] + df_subset['domain'].unique().tolist(), args=(False,))
if domain != 'All':
    df_subset = df_subset[df_subset['domain'] == domain]

subdomain = st.selectbox('Choose subdomain', options=['All'] + df_subset['subdomain'].unique().tolist(), args=(False,))
if subdomain != 'All':
    df_subset = df_subset[df_subset['subdomain'] == subdomain]

website = st.selectbox('Choose website', options=['All'] + df_subset['website'].unique().tolist(), args=(False,))
if website != 'All':
    df_subset = df_subset[df_subset['website'] == website]

options_ids = ['All'] + df_subset['annotation_id'].unique().tolist()
# annotation = st.multiselect('Or choose task ID(s)', options_ids, on_change=show_data, args=(False,))
# if 'All' not in annotation and annotation:
#     df_subset = df_subset[df_subset['annotation_id'].isin(annotation)]

df_subset = df_subset.rename(columns={'confirmed_task':'task', 'action_reprs': 'Trajectory (Demonstration data)'})#Rename columns for readability
df_subset = df_subset.reset_index(drop=True)
json_data = df_subset.to_json(orient='records')

st.write("Then, you can display and download the subset here:")

subset_expander = st.expander("Data display")
# subset_expander.button("Show subset", on_click=show_data, args=(True,))

# if st.session_state.showing_subset_data:
#     print ('Showing data')
    #Adjust height of dataframe:
subset_expander.write(f"{len(df_subset)} rows")
height = max(45 * len(df_subset), 100) #45 is roughly the height of each row in the dataframe
if height > 1000:
    height = 1000
subset_expander.dataframe(df_subset, use_container_width=True, height=height)

subset_expander.download_button(
    "Download subset",
    # disabled=(not st.session_state.showing_subset_data),
    data=json_data,
    file_name="JungleGym_Mind2Web_selection.json",
    mime="application/json"
    )



st.write("Or, run an ✨ agent trajectory simulation ✨ that steps through one task.")


@st.cache_data
def load_screenshots(annotation_id, headers):
    screenshots = []
    response = requests.get(MIND2WEB_ENDPOINT + '/get_raw_json_screenshots', params={"annotation_id" : annotation_id}, headers=headers)
    if response.status_code == 200:
        screenshots = response.json()['data']
    else:
        print('Error getting screenshots: ', response.status_code)
    return screenshots

@st.cache_data
def load_actions(annotation_id, headers):
    response = requests.get(MIND2WEB_ENDPOINT + '/get_list_of_actions', params={"annotation_id" : annotation_id}, headers=headers)
    actions_list = []
    action_reprs = []
    if response.status_code == 200:            
        actions_list = response.json()['actions']
        action_reprs = response.json()['action_reprs']
    else:
        print('Error getting actions: ', response.status_code)
    return actions_list, action_reprs

# st.write(
# '''
# This section steps through a single Mind2Web task, showing the actions taken and website snapshot at each step.
# This is real web data (i.e. actual web pages and controls).
# We call this an "agent simulation" because it shows the what a successful agent run would look like on each task.
# However, this represents only one path among potentially many paths to complete the given task.
# '''
# )

# # load metadata
# df_simulation = load_light_train_dataset()

# # select subset/ task

def reset_simulation():
    st.session_state.simulating = False
    st.session_state.action_slider = 0

# domain = st.selectbox('Choose domain', options=['All'] + df_simulation['domain'].unique().tolist(), on_change=reset_simulation)
# if domain != 'All':
#     df_simulation = df_simulation[df_simulation['domain'] == domain]

# subdomain = st.selectbox('Choose subdomain', options=['All'] + df_simulation['subdomain'].unique().tolist(), on_change=reset_simulation)
# if subdomain != 'All':
#     df_simulation = df_simulation[df_simulation['subdomain'] == subdomain]

# website = st.selectbox('Choose website', options=['All'] + df_simulation['website'].unique().tolist(), on_change=reset_simulation)
# if website != 'All':
#     df_simulation = df_simulation[df_simulation['website'] == website]

selected_columns_df = df_subset[['website', 'domain', 'subdomain', 'task']]
selected_columns_df = selected_columns_df.reset_index(drop=True)

task_list = selected_columns_df['task'].tolist()
task = st.selectbox('Choose one task', options =[''] + task_list, on_change=reset_simulation)

# simulate
def start_simulation_click():
    st.session_state.simulating = True

st.button('Start agent simulation', on_click=start_simulation_click)

if st.session_state.simulating:
    if len(task) > 0:

        # st.write('Running Simulation...')

        annotation_id = str(df_subset[df_subset['task'] == task]['annotation_id'].item())
        st.write("Website: " + df_subset[df_subset['task'] == task]['website'].item())
        st.write("Task ID: " + annotation_id)

        screenshots = load_screenshots(annotation_id, headers)
        actions_list, action_reprs = load_actions(annotation_id, headers)

        # Create slider
        action_slider_placeholder = st.empty()
        action = action_slider_placeholder.slider("Action number:", 0, len(screenshots)-1, value=st.session_state.action_slider)
        # action = action_slider_placeholder.slider("Action number:", 0, len(screenshots)-1, key="action_slider")

        placeholder = st.empty()
        with placeholder.container():
            try:
                html_code = actions_list[action]['raw_html']
                operation = actions_list[action]['operation']['op']
                value = actions_list[action]['operation']['value']
                backend_node_id = str(actions_list[action]['pos_candidates'][0]['backend_node_id'])
            except Exception as e:
                html_code = 'No HTML code available'
                print ('No HTML code available. Error: ', e)
            
            column1, column2 = st.columns(2)
            column1.markdown("#### Action list:")
            column2.markdown("#### Rendered page:")

            def action_click(i):
                st.session_state.action_slider = i

            for i, a in enumerate(action_reprs):
                button_type = "primary" if i == action else "secondary"
                column1.button(label=f"{i}\. {a}", on_click=action_click, args=(i,), type=button_type)

            for state in ['before', 'after']:
                image_exp = column2.expander(f"{state} {operation}", expanded=True)
                # image_exp = column2.expander(f"{state} {operation}", expanded=st.session_state[f"expanded_default_{state}"])
                b64_string = screenshots[action][state]["screenshot"]
                decoded_image = base64.b64decode(b64_string)
                image = Image.open(io.BytesIO(decoded_image))
                ratio = MAX_IM_SIZE / image.width
                new_height = int(image.height * ratio)
                image = image.resize((MAX_IM_SIZE, new_height))
                # column2.caption(f"{state} action:")
                image_exp.image(image, use_column_width=True)
            
            try:
                html_exp = column2.expander("raw html")
                html_exp.markdown(f'```html\n{html_code}\n```', unsafe_allow_html=True)
            except:
                column2.write('No HTML code available')
    else:
        st.write('Please select a task')
        st.stop()
