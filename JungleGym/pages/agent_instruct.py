# -*- coding: utf-8 -*-
"""
Analyze AgentTuning Dataset Module
======================

This module is designed to visualization and analyze the AgentTuning module. Functionality like filtering 
Domains and subdomains, see each task category and get annotation_id's for each task.

Author: Marco Mascorro (@mascobot) & Matt Bornstein
Created: November 2023
Version: 0.9 (Experimental)
Status: Development
Python version: 3.9.15
"""
# External libraries:
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import os
import re
# Local libraries:
from layout import init_page

# Get Mind2Web server API endpoint:
API_ENDPOINT = 'http://api.junglegym.ai'
API_KEY = os.environ.get('MIND2WEB_API_KEY', default='')
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {}'.format(API_KEY)
}

init_page()

plotly_config_set = {'displayModeBar': False, 'showlogo': False}

@st.cache_data
def load_agent_instruct():
    response = requests.get(API_ENDPOINT + '/load_agent_instruct', headers=headers)
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

st.title('👾 AgentInstruct')

st.subheader('Description')

st.markdown(
'''
AgentInstruct is a dataset for **training** (i.e. fine-tuning) LLMs for use as web agents.
It's designed for instruction tuning of open source LLMs like Llama 2 (most LLMs haven't been trained with Agent data).

It contains 1,866 trajectories across 6 categories of tasks. 
Where a dataset like Mind2Web provides ground truth for successful agent execution, AgentInstruct provides instructions (i.e. rationale) to help LLMs understand *how* to achieve the task successfully.

It was introduced in the [AgentTuning](https://arxiv.org/abs/2310.12823) paper (A Zeng et al., 2023), designed to improve LLMs’ generalized agent abilities.
''')


# data = {
#     'Task': ['ALFWorld (Shridhar et al., 2020)', 'WebShop (Yao et al., 2022)', 'Mind2Web (Deng et al., 2023)',
#             'Knowledge Graph (Liu et al., 2023)', 'Operating System (Liu et al., 2023)', 'Database (Liu et al., 2023)', 'Database (Liu et al., 2023)', 'AgentInstruct'],
#     'Inst. From': ['Train split', 'Train split', 'Train split', 'Train split', 'Self-Instruct', 'Self-Instruct', 'Task Deri.', '-'],
#     '# Inst': [954, 1485, 23378, 2501, 647, 1074, 5302, 35341],
#     '# Filt. Traj.': [336, 351, 122, 324, 195, 178, 360, 1866],
#     'Avg # Filt. Traj. Turns': [13.52, 3.68, 1.00, 6.04, 3.85, 2.13, 2.03, 5.24],
#     'Ratio': ['35.2%', '23.6%', '0.52%', '13.0%', '30.1%', '16.6%', '6.79%', '5.29%']
# }

# df_table = pd.DataFrame(data)

# # Styling
# st.markdown("""
# <style>
#     table {
#         font-size: 12px;
#     }
#     th {
#         font-size: 14px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Columns layout for table and pie chart
# col1, col2 = st.columns(2)




# with col1:
#     st.write("AgentInstruct Dataset Overview:")
#     st.table(df_table)

stats_expander = st.expander('Summary stats')

#Load AgentInstruct Data
df = load_agent_instruct()

def remove_loss_key(entry):
    if 'loss' in entry:
        del entry['loss']
    return entry

conversations = df['conversations']
for i, convo in enumerate(conversations):
    for j, entry in enumerate(convo):
        if 'loss' in entry:
            del entry['loss']
    conversations[i] = convo
df['conversations'] = conversations

df['category'] = df['id'].apply(lambda x: re.sub(r'_[0-9]+', '', x))

option_mapping = {
    "os": "Operating System",
    "webshop": "WebShop",
    "mind2web": "Mind2Web",
    "kg": "Knowledge Graph",
    "db": "Database",
    "alfworld": "ALFWorld"
}

readable_categories = [''] + ['All'] + [option_mapping.get(cat, cat) for cat in df['category'].unique()]


# Add the pie chart to the right of the table
category_counts = df['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']
category_counts['readable_category'] = category_counts['category'].map(option_mapping)

pie_chart = px.pie(category_counts, 
                values='count', 
                names='readable_category', 
                title='Task Distribution by Category',
                hole=0.3)

pie_chart.update_layout(showlegend=True)

# with col2:  # Displaying the pie chart in the second column (right side)
stats_expander.plotly_chart(pie_chart)

st.subheader('How to Use It')

st.write('''
The full dataset is available in JSONL format on [HuggingFace](https://huggingface.co/datasets/THUDM/AgentInstruct).
You can use the standard GPT-4, LLaMA2, Mistral, etc. fine-tuning frameworks and methods to run instruction tuning on this dataset.
''')

st.subheader('Try It!')

st.write('''
You can browse the AgentInstruct dataset by category here, and download category-level subsets.
''')

selected_readable_category = st.selectbox('Select Category:', readable_categories, index=0)
download_button_placeholder = st.empty()

if selected_readable_category in option_mapping.values():
    selected_original_category = [key for key, value in option_mapping.items() if value == selected_readable_category][0]
else:
    selected_original_category = selected_readable_category

if selected_original_category:
    if selected_original_category != 'All':
        filtered_df = df[df['category'] == selected_original_category]
    else:
        filtered_df = df.copy()

    st.write(f"Number of Rows: {len(filtered_df)}")

    filtered_df.reset_index(inplace=True, drop=True)
    filtered_df.drop(columns=['category', 'id'], inplace=True)

    for i, row in filtered_df.iterrows():
        preview = "No available preview"

        for entry in row['conversations']:
            if entry.get('from') == 'human':
                preview = entry.get('value', '')
                preview = (preview[:330] + '...') if len(preview) > 100 else preview
                break

        with st.expander(f"Conversation {i}: {preview}"):
            st.json(row['conversations'])

    def to_jsonl(df):
        jsonl_str = df.to_json(orient='records', lines=True)
        return jsonl_str

    if selected_original_category != 'All':
        download_df = df[df['category'] == selected_original_category]
    else:
        download_df = df.copy()

    len_data = str(len(download_df))
    download_button_placeholder.download_button(
        label="Download Selected Data of " + len_data + " rows",
        data=to_jsonl(download_df),
        file_name=f"{selected_readable_category}_data.jsonl",
        mime="text/jsonl"
    )
# else:
#     st.write("Please select a category to view and download data.")
