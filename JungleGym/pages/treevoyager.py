# -*- coding: utf-8 -*-
"""
TreeVoyager Module
======================

This module is a visualization tool for the TreeVoyager implementation by calling the API.

Author: Marco Mascorro (@mascobot) & Matt Bornstein
Created: November 2023
Version: 0.9 (Experimental)
Status: Development
Python version: 3.9.15
"""
#External libraries:
import streamlit as st
import pandas as pd
import requests
import os
import glob
import time
import json
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
import base64
#Internal libraries:
from layout import init_page
init_page()

BASE_DIR = os.path.dirname(os.getcwd())
API_ENDPOINT = 'http://api.junglegym.ai'
TREE_VOYAGER_SERVER_ENDPOINT = 'https://treevoyager.junglegym.ai'

st.title("🌳 TreeVoyager")

st.subheader('Description')

st.write('''
One of the hard problems with web agents is reliably parsing the HTML DOM.
There is no standard protocol yet that allows agents to interact with the web - each agent parses the DOM in a bespoke way.
Using normal prompting strategies, most LLMs hallucinate DOM entity details and don't consistently associate elements with correct actions.

We developed TreeVoyager to make progress on this problem. This is not an Agent, it's just a tool to mitigate the problem. To learn how TreeVoyager works, check out the [Github Repo](https://github.com/a16z-infra/JungleGym).
\nThis version uses GPT-4 Turbo 128K and it implements some principles from the [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (ToT) and [Minecraft's Voyager](https://voyager.minedojo.org/) papers.
Given a web page and a task as input, it can:
- Parse the DOM
- Generate curriculum (i.e. a plan of actions)
- Select correct HTML IDs
- Generate paths
- Create skills (memory) for the steps required in the agent trajectory.

This demo is only to show you step-by-step the suggested process for an agent's trajectory. It's not meant to be an Agent itself. In every step, you can edit the code, next URL and task.

*Note: TreeVoyager is not meant to solve CAPTCHA.*
''')

st.subheader('How to Use It')

st.write('''
You can run TreeVoyager using the [JungleGym API](https://docs.junglegym.ai/junglegym/api-documentation/treevoyager-api) or clone and customize the code from the Github repo. You can also try it out in the playground below.
''')

# image_treevoyager = os.path.join(os.getcwd(), "pages", "ImageTreeVoyager.png")
# # Convert the image to base64
# with open(image_treevoyager, "rb") as img_file:
#     b64_string = base64.b64encode(img_file.read()).decode()
# # Display the image
# st.markdown(
#     f'''
#     <style>
#     img {{
#         margin-top: 0px;
#         padding-top: 0px;
#     }}
#     </style>
#     <div style="display: flex; justify-content: center; padding-bottom: 20px;"><img src="data:image/png;base64,{b64_string}" width="650"></div>
#     ''',
#     unsafe_allow_html=True,
# )


st.subheader("Try it!")

st.write('''
Choose either a pre-loaded WebArena testing environment & task, or enter a custom URL & task.
After clicking the 'Start' button, TreeVoyager will attempt to generate a curriculum for the given web page and task, showing:
a screenshot of the page state; a table of "interactable" HTML elements; a suggested curriculum to execute the task; and sample Selenium code to execute the curriculum.
At each step, you can change the suggested action.
''')

st.markdown("<a name='top'></a>", unsafe_allow_html=True)

@st.cache_data
def load_tasks():#Loads WebArena tasks from API
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

def extract_domain(url) -> str:
    return urlparse(url).netloc

df = load_tasks()
df['domain'] = df['start_url_junglegym'].apply(extract_domain)
#df['task&domain'] = df['intent'] + " - (" + df['domain'] + ')'

def ensure_http_prefix(link) -> str:
    if not link.startswith(('http://', 'https://')):
        return 'http://' + link
    elif link.startswith('https://'):
        link = link.split('https://')[-1]
        return 'http://' + link
    return link

def get_screenshot(url: str, task: str) -> str:
    image_path = 'screenshots/' + task + '_screenshot.png'
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")#Wait until page is fully loaded
        page.screenshot(path=image_path)
        browser.close()
    return image_path

def run_task(task, url, curriculum, prev_code, step_name):
    params = {
        'task': task,
        'page': url,
        'curriculum': curriculum,
        'prev_code': prev_code,
        'step': step_name
    }
    response = requests.get(TREE_VOYAGER_SERVER_ENDPOINT + "/run_step", params=params, stream=True)
    if response.status_code != 200:
        st.error("Error running task: " + str(response.status_code))
        #st.stop()
    try:
        for line in response.iter_lines():
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data:"):
                data = json.loads(decoded_line[len("data:"):])
                if "actionable_elements" in data:
                    df_elements = pd.DataFrame(data['actionable_elements'])
                    df_elements.rename(columns={'id': 'HTML id'}, inplace=True)
                    yield {'actionable_elements': df_elements}
                if "curriculum" in data:
                    yield {'curriculum': data['curriculum']}
                if "step_list" in data:
                    yield {'step_list': data['step_list']}
                if 'step_name' in data:
                    yield {'step_name': data['step_name']}
                if 'step_code' in data:
                    code = data['step_code']
                    clean_code = code
                    if code.startswith("```") and code.endswith("```"):
                        clean_code = code[3:-3]
                    if clean_code.startswith("python"):
                        clean_code = clean_code[len("python"):].lstrip()
                    yield {'step_code': clean_code}
                # Display the streamed outputs of the API call in a box
                if 'step_tag_name' in data and 'step_field_name' in data and 'step_duration' in data and 'step_total_tokens' in data:
                    model_name = data['model']
                    if "gpt-4-1106-preview" in data['model']:
                        model_name = "gpt-4-1106-preview (GPT-4-Turbo 128K)"
                    info_data = {
                        'HTML id':data['step_html_id'],
                        'Step HTML Tag Name': data['step_tag_name'],
                        'Step HTML Field Name': data['step_field_name'],
                        'Step Duration (sec)': data['step_duration'],
                        'Total Step Tokens': data['step_total_tokens'],
                        'Model': model_name,
                    }
                    yield {'info_data': info_data}
    except requests.exceptions.ChunkedEncodingError:
        pass

def run():
    st.session_state['running'] = True

def reset_run():
    st.session_state['counter'] = 0
    st.session_state['url'] = ""
    st.session_state['task'] = ""
    st.session_state['actionable_elements'] = []
    st.session_state['image_path'] = ""
    st.session_state['curriculum'] = ""
    st.session_state['step_name'] = ""
    st.session_state['step_list'] = []
    st.session_state['step_code'] = None
    st.session_state['info_data'] = {}
    st.session_state['running'] = False

# WEBARENA
mapping = {'git.junglegym.ai': 'GitLab', 'forum.junglegym.ai': 'Social Forum', 'cms.junglegym.ai': 'E-Commerce CMS', 'shop.junglegym.ai': 'Online Store', 'ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000': 'Map'}
df['webdomain'] = df['domain'].map(mapping)
df['webdomain+link'] = df['webdomain'].astype(str) + ' (' + df['domain'].astype(str) + ')'
# Apply the exception for the special domain
special_domain = 'ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000'#This is the Map mirror domain
df.loc[df['domain'] == special_domain, 'webdomain+link'] = 'Map'
domains = df['webdomain+link'].unique().tolist()
domains.insert(0, "")

webarena_exp = st.expander("WebArena environments")
webarena_domain = webarena_exp.selectbox('Choose a WebArena environment:', domains, on_change=reset_run)
webarena_task = ""
if webarena_domain != "":
    df = df[df['webdomain+link'] == webarena_domain]
    webarena_tasks_list = df['intent'].unique().tolist()
    webarena_tasks_list.insert(0, "")
    webarena_task = webarena_exp.selectbox('Select a WebArena task:', webarena_tasks_list, on_change=reset_run)
else:
    webarena_exp.selectbox('Select a WebArena task:', [])

webarena_exp.markdown("<sub>Note: if you are using the http://cms.junglegym.ai/admin (E-commerce CMS Emulated Web Environment), the panel username is 'admin' and the password is 'admin1234'. If you are using git.junglegym.ai, you can create a new account with custom credentials.</sub>", unsafe_allow_html=True)


# CUSTOM URL
custom_url_exp = st.expander("Custom web page")
custom_url = custom_url_exp.text_input(f'Enter a custom URL (e.g. "yoursite.com", "shop.junglegym.ai", etc): ', key="external_url", on_change=reset_run)
custom_url = ensure_http_prefix(custom_url)
custom_task = custom_url_exp.text_input('Enter a custom task (e.g. buy coffee, rent the cheapest truck, etc):', key="task_input", on_change=reset_run)
custom_url_exp.markdown("<sub>Note: this will fail on pages requiring authentication.</sub>", unsafe_allow_html=True)

if webarena_task != "":
    task = webarena_task
    url = df[df['intent'] == webarena_task]['start_url_junglegym'].iloc[0]
else:
    task = custom_task
    url = custom_url

# RUN
run_button_placeholder = st.empty()
divider_placeholder = st.empty()

step_title_placeholder = st.empty()
image_title_placeholder = st.empty()
image_placeholder = st.empty()
html_interactable_elements_placeholder = st.empty()
table1_placeholder = st.empty()
curriculum_title_placeholder = st.empty()
curriculum_placeholder = st.empty()
step_name_placeholder = st.empty()
code_title_placeholder = st.empty()
code_placeholder = st.empty()
table2_placeholder = st.empty()
next_step_input_placeholder = st.empty()
next_step_button_placeholder = st.empty()

next_step = ""
if 'counter' not in st.session_state:
    st.session_state['counter'] = 0
if 'url' not in st.session_state:
    st.session_state['url'] = ""
if 'task' not in st.session_state:
    st.session_state['task'] = ""
if 'actionable_elements' not in st.session_state:
    st.session_state['actionable_elements'] = []
if 'image_path' not in st.session_state:
    st.session_state['image_path'] = ""
if 'curriculum' not in st.session_state:
    st.session_state['curriculum'] = ""
if 'step_name' not in st.session_state:
    st.session_state['step_name'] = ""
if 'step_list' not in st.session_state:
    st.session_state['step_list'] = []
if 'step_code' not in st.session_state:
    st.session_state['step_code'] = None
if 'info_data' not in st.session_state:
    st.session_state['info_data'] = {}
if 'running' not in st.session_state:
    st.session_state['running'] = False

run_button_placeholder.button("Start", on_click=run, disabled=st.session_state['running'])

if st.session_state['running']:
    st.session_state['task'] = task
    divider_placeholder.divider()
    step_title_placeholder.markdown(f"### Step {st.session_state['counter']+1}")
    image_title_placeholder.markdown("#### Page state (screenshot):")
    if url != st.session_state['url']:
        print ("New URL, taking screenshot", url)
        with (st.spinner('Getting page state...')):
            try:#Remove previous screenshot:
                os.remove(st.session_state['image_path'])
            except Exception as e: 
                print ("No image path found for a file.", e)
            image_path = get_screenshot(url, task)
            st.session_state['image_path'] = image_path
            st.session_state['url'] = url
            image_placeholder.image(image_path, caption="Page screenshot for this step", width=1000)
    else:
        print ("Same URL, using previous screenshot", url)
        image_placeholder.image(st.session_state['image_path'], caption="Page screenshot for this step", width=1000)
        html_interactable_elements_placeholder.markdown(f"#### Interactable HTML elements on page:")
        table1_placeholder.dataframe(st.session_state['actionable_elements'])
    if st.session_state['counter'] == 0:#if it is the first time running:
        with st.spinner('Generating response for first step...'):
            for result in run_task(task=task, url=url, curriculum=None, prev_code=None, step_name=None):
                if 'actionable_elements' in result:
                    html_interactable_elements_placeholder.markdown(f"### Interactable HTML elements on page:")
                    df_elements = result['actionable_elements']
                    st.session_state['actionable_elements'] = df_elements
                    table1_placeholder.dataframe(df_elements)
                if 'curriculum' in result:
                    st.session_state['curriculum'] = result['curriculum']
                    curriculum_title_placeholder.markdown(f"### Suggested Curriculum:")
                    curriculum_placeholder.write(f"{result['curriculum']}")
                if 'step_name' in result:
                    st.session_state['step_name'] = result['step_name']
                    step_name_placeholder.subheader('Step ' + result['step_name'])
                if 'step_list' in result:
                    if st.session_state['counter'] + 1 < len(result['step_list']):
                        st.session_state['step_list'] = result['step_list']
                        try:
                            next_step = result['step_list'][st.session_state['counter'] + 1]
                        except ValueError:
                            next_step = "Step not found in curriculum"
                            print("Step not found in curriculum")   
                    else:
                        #next_step = "Step not found in curriculum"
                        print("Step not found in curriculum")    
                if 'step_code' in result:
                    st.session_state['step_code'] = result['step_code']
                    code_title_placeholder.markdown('##### Step ' + st.session_state['step_name'])
                    code_placeholder.code(result['step_code'], language='python')
                if 'info_data' in result:
                    st.session_state['info_data'] = result['info_data']
                    table2_placeholder.table(pd.DataFrame(result['info_data'], index=[0]))
        st.session_state['counter'] += 1
    else:#If it is not the first time running:
        #with st.spinner('Generating response for step...'):# For some reason, the spinner break the for loop below early.
        print ('Running again!')
        curriculum = st.session_state['curriculum']
        prev_code = st.session_state['step_code']
        if st.session_state['counter'] < len(st.session_state['step_list']):
            step_name = st.session_state['step_list'][st.session_state['counter']]
        else:
            # handle the error or reset the counter
            print("Counter exceeds the length of step_list.")
        task_ = st.session_state['task']
        url_ = st.session_state['url']
        loading_message = st.text('Generating response for step {}...'.format(str(st.session_state['counter'] + 1)))
        for result in run_task(task=task_, url=url_, curriculum=curriculum, prev_code=prev_code, step_name=step_name):
            if 'actionable_elements' in result and url != st.session_state['url']:
                html_interactable_elements_placeholder.markdown(f"#### Interactable HTML elements for this page:")
                df_elements = result['actionable_elements']
                st.session_state['actionable_elements'] = df_elements
                table1_placeholder.dataframe(df_elements)
            if 'curriculum' in result:
                st.session_state['curriculum'] = result['curriculum']
                curriculum_title_placeholder.markdown(f"#### Suggested Curriculum:")
                curriculum_placeholder.write(f"{result['curriculum']}")
            if 'step_name' in result:
                st.session_state['step_name'] = result['step_name']
                step_name_placeholder.subheader('Step ' + result['step_name'])      
            if 'step_code' in result:
                st.session_state['step_code'] = result['step_code']
                code_title_placeholder.markdown('###### Sample code for this step (the code accounts for all previous steps):')
                code_placeholder.code(result['step_code'], language='python')
            if 'info_data' in result:
                st.session_state['info_data'] = result['info_data']
                table2_placeholder.table(pd.DataFrame(result['info_data'], index=[0]))
        loading_message.text('Done generating response for step {}.'.format(str(st.session_state['counter'] + 1)))
        st.session_state['counter'] += 1
        next_step = st.session_state['step_list'][st.session_state['counter']]
    next_step_input_placeholder.text_input('Next step (leave blank to keep original step from curriculum):', placeholder=next_step, key="next_step_name")
    next_step_button_placeholder.button(f"Run Step {st.session_state['counter']+1}")
    








