"""
TreeVoyager - A LLM-based DOM Parser that implements Tree of Thoughts (ToT), Voyager and a "SemioseVetter" for optimizing functional correctness in open-ended autonomous agents with LLMs.
======================
Description:
A protocol that implements some concepts from Tree of Thoughts (ToT) architecture for hierarchical information structuring,
the Voyager AI agent framework for curriculum generation, self-verification and continual skill learning
and a SemioseVetter for hallucination pruning and selection correctness. 

This initial release is for Web actions/agents (could be extended to other modalities and other AI agents in the future).
======================
Author: Marco Mascorro (@mascobot)
Created: November 2023
Version: 0.9.0 (Experimental)
Status: Development
Python version: 3.9.15
"""
#Import external libraries:
import requests
import json
import os
import torch
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup#For parsing the HTML
import openai
from selenium import webdriver #For taking web actions
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import faiss
import time
from datetime import datetime
import re
import pytz
from typing import Generator

from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)
RATE_LIMIT = "500/minute"

openai.api_key = os.environ.get('OPENAI_KEY', default='')
##Global variables
HTML_ID_NAME = 'id'#Most commonly used id name in HTML tags.
MODEL = 'gpt-4-1106-preview'#Default model. Options: gpt-3.5-turbo, gpt-4, gpt-4-1106-preview
SKILL_MEMORY = []

#driver = webdriver.Chrome()
URL_DICT = {'forum':
        {'link': 'http://forum.junglegym.ai/',
            'nickname': 'Reddit.com'},
        'shopping':
            {'link': 'http://shop.junglegym.ai/',
                'nickname': 'Amazon.com'},
        'gitlab':
            {'link': 'http://git.junglegym.ai/',
                'nickname': 'Gitlab.com'},
        'wiki': 
            {'link': 'http://wiki.junglegym.ai/',
            'nickname':'Wikipedia.org'}
        }

######################
#driver.get(page['link'])

def curriculum_generator(website, task, actionable_elements, model=MODEL):
    """
    Generate a curriculum for web interaction tasks.

    Parameters:
    website (str): The website where the task is to be performed.
    task (str): The task description.
    model (str): The GPT model to use for generating the curriculum. Default is 'gpt-4'.

    Returns:
    dict: A dictionary containing the generated curriculum, total tokens used, duration, and model used.
    """
    start_time = time.perf_counter()
    preprompt1 = f"You've successfully navigated to {website} using the Selenium webdriver (that means you to need to give instructions to navigate to the website anymore). You also have a list of dictionaries of the HTML elements.\n\nYou have being assigned the task: {task}.\n\nWhat are the Selenium steps you would execute to successfully accomplish the task? Only describe the specific steps in Python's Selenium to select the right HTML id or HTML field_name value and action. Indicate them step by step giving Selenium instructions to write the python code. Do no create names or ids.\n\nFor the initial HTML page, here is a parsed list of dictionaries with key and value of the HTML id, attributes, field_name, link and other fields for every single HTML element. Each dictionary contains additional context information of each element. In Selenium, when selecting an HTML element for an action, you can use the Selenium's methods 'By.ID' if the best element for the action has the 'id' key in the dictionary, 'By.NAME' if the best element for the action has the 'field_name' key in the dictionary and so on. You can also use other Selenium methods, but please use all information in the dictionary to make the best assessment to select the best HTML element and best method to make the task successful: \n\n {actionable_elements}"
    message_history=[
            {"role": "system", "content": "You are a helpful and smart software engineer using python's Selenium"},
            {"role": "user", "content": preprompt1}]
    response = openai.ChatCompletion.create(model=model, messages=message_history, temperature=0.0)
    response = response['choices'][0]['message']['content']
    #Self-verification implementation:
    message_history.append({"role": "assistant", "content": response})
    message_history.append({"role": "user", "content": 'How would you improve these steps?\n\nOnly list the steps you will take to accomplish the complete task successfully: {}\n\nDo not explain.'.format(task)})#Try one more time to remove any additional non necessary content
    response = openai.ChatCompletion.create(model=model, messages=message_history, temperature=0.0)
    # Final response and metadata
    total_tokens = response['usage']['total_tokens']
    response = response['choices'][0]['message']['content'].strip('.').strip()#remove periods from beginning and end if any.
    pattern = r"(\d+\.\s*.*?)(?:\n|$)"#Filter list elements
    tasks_list = re.findall(pattern, response)
    end_time = time.perf_counter()
    duration = round(end_time - start_time, 2)
    return {'curriculum':response, 'tasks_list':tasks_list,'total_tokens':total_tokens, 'duration':duration, 'model':model}

def get_actionable_elements(page_link):
    start_time = time.perf_counter()
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(page_link, headers=headers, timeout=15)

    if response.status_code != 200:
        print(f"Failed to get link data (requests): {response.status_code}")
        return None
    print("Successfully got link data (requests) for {}".format(page_link))

    soup = BeautifulSoup(response.content, 'html.parser')
    actionable_elements = ['a', 'button', 'input', 'textarea', 'select']
    event_attributes = ['onclick', 'onsubmit', 'onchange', 'onmouseover', 'onmouseout', 'onkeydown', 'onkeyup']
    elements_data = []
    div_summary = {}

    # Summarize div relationships
    for div in soup.find_all('div', {HTML_ID_NAME: True}):
        div_summary[div[HTML_ID_NAME]] = [child[HTML_ID_NAME] for child in div.find_all(True, {'id': True})]

    # Iterate over all elements
    for element in soup.find_all(True):
        tag_name = element.name
        attributes = element.attrs
        parent_html = element.find_parent()
        parent_class_name = parent_html.get('class', [None])[0] if parent_html else None
        parent_div_id = parent_html.get(HTML_ID_NAME) if parent_html and parent_html.name == 'div' else None
        id_ = None
        if element.has_attr(HTML_ID_NAME):
            id_ = element[HTML_ID_NAME]
            try:
                del attributes['id']#Already in id_
            except Exception as e:
                print("Error deleting 'id' from attributes")
        element_info = {
            'id': id_,
            'tag_name': tag_name,
            'attributes': attributes,
            'parent_class': parent_class_name,
            'parent_div_children': div_summary.get(parent_div_id, [])
        }

        # Additional details for actionable elements
        element_info['field_name'] = None
        element_info['link'] = None
        element_info['event_attributes'] = None
        if tag_name in actionable_elements:
            element_info['field_name'] = element.get('name', '').strip() or element.get('value', '').strip() or element.text.strip()
            element_info['link'] = element.get('href', '') if tag_name == 'a' else ''
            element_info['event_attributes'] = {attr: element.get(attr, '') for attr in event_attributes if element.has_attr(attr)}

        elements_data.append(element_info)
    #Remove empty elements:
    elements_data = [element for element in elements_data if not ((element['id'] == None or element['id'] == '') and (element['field_name'] == None or element['field_name'] == '') and (element['link'] == None or element['link'] == ''))]

    end_time = time.perf_counter()
    duration = round(end_time - start_time, 2)
    return {'actionable_elements': elements_data, 'duration': duration}

def get_pos_candidates_branch(website, task, curriculum, actionable_elements, step, model, CoT):
    """
    Generates a GPT-based selection for an HTML 'id' given a specific task and website context.
    
    Parameters:
    - website (str): The website context.
    - task (str): The task to be performed.
    - curriculum (str): The guide followed by the assistant.
    - actionable_elements (list): List of potential HTML elements.
    - step (str): Step name.
    - model (str): The GPT model being used.
    - CoT (bool): Whether to include a second round of checking (Confirmation of Task).
    
    Returns:
    dict: A dictionary containing the assistant's response, message history, total token count, duration, and the model used.
    """
    start_time = time.perf_counter()
    #actionable_elements = json.dumps(actionable_elements)
    enumerated_elements = "\n".join([f"{i+1}. {element}" for i, element in enumerate(actionable_elements)])
    preprompt = f"You want to accomplish this task: {task} in this website: {website} and you are following this guide:\n\n{curriculum}.\n\nFor the step: {step}, What is the best HTML 'id' or 'field_name', or 'tag_name' to select from the following dictionary list?\n\nDo not explain. Only respond with the exact 'id' or 'field_name' or 'tag_name' value from the following list of dictionaries, where each dictionary, contains additional information for each 'id', 'field_name' and 'tag_name': \n\n {enumerated_elements}"
    message_history = [
        {"role": "system", "content": f"You are an intelligent programmer using python's Selenium. You are helping a colleague select an HTML 'id' or HTML 'field_name' value saved in a list of dictionaries. You must not lie or make up 'id' or 'field_name' values. You only select an 'id' or 'field_name' value from the list of dictionaries."},
        {"role": "user", "content": preprompt}
    ]
    response = openai.ChatCompletion.create(model=model, messages=message_history, temperature=0.0)
    total_tokens = response['usage']['total_tokens']
    response = response['choices'][0]['message']['content'].strip('.').strip("'").strip('"')
    if CoT == False:
        end_time = time.perf_counter()
        duration = round(end_time - start_time, 2)
        return {'response': response, 'message_history': message_history, 'total_tokens': total_tokens, 'duration': duration, 'model': model}
    # CoT implementation:
    message_history.append({"role": "assistant", "content": response})
    message_history.append({"role": "user", "content": f"Is this 'id' or 'field_name' value in the list of dictionaries the best value for the given task? Check the list of dictionaries again. Here it is: \n{actionable_elements}\n. Only respond with the 'id' or 'field_name' value and do not explain."})
    response = openai.ChatCompletion.create(model=model, messages=message_history, temperature=0.0)
    total_tokens = response['usage']['total_tokens']
    response = response['choices'][0]['message']['content'].strip('.').strip("'").strip('"')
    end_time = time.perf_counter()
    duration = round(end_time - start_time, 2)
    return {'response': response, 'message_history': message_history, 'total_tokens': total_tokens, 'duration': duration, 'model': model}

def tree_thought_generation(website, task, curriculum, actionable_elements, CoT, branches, step, model=MODEL):
    """
    Generates thought processes using a tree-like structure with multiple branches.
    
    Parameters:
    - website (str): The website being analyzed.
    - task (str): The task to perform on the website.
    - curriculum (str): The curriculum being followed.
    - actionable_elements (List): List of elements that can be interacted with.
    - CoT (bool): Flag for Condition over Time implementation for each branch.
    - branches (int): Number of branches in the tree.
    - step (str): Step name.
    - model (str, optional): Name of the model to be used.
    
    Returns:
    dict: Contains various performance and result metrics like total tokens, duration, and individual branch statistics.
    """
    start_time = time.perf_counter()
    arguments = {
        'website': website,
        'task': task,
        'curriculum': curriculum,
        'actionable_elements': actionable_elements,
        'step': step,
        'model': model,
        'CoT': CoT#Pass True for CoT implementation for each branch. False will only return the first response with no CoT.
    }
    website_results_ext = []
    def run_tree():
        with ThreadPoolExecutor() as executor:#Run tree branches
            futures = [executor.submit(get_pos_candidates_branch, **arguments) for _ in range(branches)]#Run n tree branches
            for future in futures:
                result = future.result()
                website_results_ext.append(result)#append results of each branch
    run_tree()#Run first tree generation
    branch_stats = []
    for index, result in enumerate(website_results_ext):#Get individual branch stats for data logging
        index += 1
        branch_name = str('branch_' + str(index))
        branch_stats.append({branch_name: {'branch_result':result['response'], 'branch_duration': result['duration'], 'branch_tokens': result['total_tokens']}})
    total_tokens = 0
    for result in website_results_ext:#Get total tokens for all branches
        total_tokens += result['total_tokens']
    candiates = []
    for result in website_results_ext:
        candiates.append(result['response'])
    #Majority vote for id:
    count = Counter(candiates)
    if count:
        majority_id_vote = max(count, key=count.get)#majority vote for id
        if len(set(count.values())) == len(count.values()):#Check if all results are different? - If yes, re-run and append new results & do voting again
            for i in website_results_ext:
                if i['response'] == majority_id_vote:#Return the first result that has the majority vote with the chat history
                    end_time = time.perf_counter()
                    duration = round(end_time - start_time, 2)
                    return {'response':i, 'total_tokens':total_tokens, 'duration':duration, 'number_of_branches':len(website_results_ext), 'branch_stats':branch_stats, 'model':model}
        else:
            run_tree()#Re run tree generation to add more "n" branches
            total_tokens = 0
            for result in website_results_ext:
                total_tokens += result['total_tokens']
            candiates = []
            for result in website_results_ext:
                candiates.append(result[0])
            branch_stats = []
            for index, result in enumerate(website_results_ext):#Get individual branch stats for data logging
                index += 1
                branch_name = str('branch_' + str(index))
                branch_stats.append({branch_name: {'branch_result':result['response'], 'branch_duration': result['duration'], 'branch_tokens': result['total_tokens']}})
            count = Counter(candiates)
            majority_id_vote = max(count, key=count.get)#majority vote for id for 2 runs (from n*2 responses)
            for i in website_results_ext:#Return the first result that has the majority vote with the chat history
                if i['response'] == majority_id_vote:                       
                    end_time = time.perf_counter()
                    duration = round(end_time - start_time, 2)
                    return {'response':i, 'total_tokens':total_tokens, 'duration':duration, 'number_of_branches':len(website_results_ext), 'branch_stats':branch_stats, 'model':model}
    else:
        #('No results to process.')
        return {'response': None}
    
def semiose_vetter(actionable_elements, query, embeddings_model='thenlper/gte-base'):
    """
    Vets actionable elements against a query using semantic search with an embeddings model.
    
    Parameters: 
    - actionable_elements (List[Dict]): List of elements that can be acted upon.
    - query (str): Query string to search for in the actionable elements.
    - embeddings_model (str, optional): Model to use for generating embeddings, defaults to 'gte-base'.
    
    Returns:
    dict: Contains the ID of the most relevant actionable element, duration of the process, and the embeddings model used.
    """
    start_time = time.perf_counter()
    model = SentenceTransformer(embeddings_model)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    df = pd.DataFrame(actionable_elements)
    df['ID'] = df.index
    df['id'] = df['id'].astype(str)#This is the HTML id, not the pandas index
    df['field_name'] = df['field_name'].astype(str)
    # Create new columns with a concat of the columns "id" and "field_name". Omit concat if field_name column is less than 2 chars:
    df['element'] = df.apply(lambda row: f"{row['id']} - {row['field_name']}" if len(row['field_name']) > 2 else row['id'], axis=1)
    #Create embeddings:
    embeddings = model.encode(df.element.to_list(), show_progress_bar=True)
    embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, df.ID.values)
    vector = model.encode(list([query]))
    D, I = index.search(np.array(vector).astype("float32"), k=5)
    indexed_list = I.flatten().tolist()
    result_list = []
    for relevance_index, df_ID in enumerate(indexed_list):
        if df_ID >= 0:
            results_dict = {}
            results_dict['relevance_index'] = relevance_index
            results_dict['id'] = df.at[df_ID, 'id']
            result_list.append(results_dict)
    end_time = time.perf_counter()
    duration = round(end_time - start_time, 2)
    return {'id_result': result_list[0]['id'], 'duration':duration, 'embeddings_model':embeddings_model}

def code_generator(website, step_name, step_html_id, prev_code, model=MODEL):
    now_utc = datetime.now(pytz.timezone('UTC'))
    now_pacific = now_utc.astimezone(pytz.timezone('US/Pacific'))
    time = now_pacific.strftime('%H:%M, %d-%m-%Y')
    if prev_code != None:
        preprompt1 = f"Write the python code to execute in Selenium the following step: {step_name}, on the website {website}.\n\nThis is the codebase that was used to execute previous steps on this page, please incorporate it with your solution without missing any of the previous steps in the code: {prev_code}\n\nThe HTML id or HTML field_name to use with this code is: {step_html_id}.\n\nOnly respond with the code to perform the step. Do not explain. Do not create HTML ids or HTML field_names, only use the one provided. If the HTML element says 'id' then use Selenium's method 'By.ID', if the element says 'field_name' then use Selenium's 'By.NAME' and so on. Use your best assessment for every HTML element to decide what Selenium method to use. At the top of the code, after the '```python', write the comments: '#Suggested code generated by JungleGym to incorporate with your Agent. Generated at (hr:min, day-month-year) Pacific Time: {time}"
    else:
        if "cms.junglegym.ai" not in website:
            preprompt1 = f"Write the python code to execute in Selenium the following step: {step_name}, on the website {website}.\n\nThe HTML id or HTML field_name to use with this code is: {step_html_id}.\n\nOnly respond with the code to perform the task. Do not explain. Do not create HTML ids or HTML field_names, only use the one provided. If the HTML element says 'id' then use Selenium's method 'By.ID', if the element says 'field_name' then use Selenium's 'By.NAME' and so on. Use your best assessment for every HTML element to decide what Selenium method to use. At the top of the code, after the '```python', write the comments: '#Suggested code generated by JungleGym to incorporate with your Agent. Generated at (hr:min, day-month-year) Pacific Time: {time}"
        else:
            preprompt1 = f"Write the python code to execute in Selenium the following step: {step_name}, on the website {website}.\n\nIf you need to login to the website, the username is 'admin' and the password is 'admin1234'\n\nThe HTML id or HTML field_name to use with this code is: {step_html_id}.\n\nOnly respond with the code to perform the task. Do not explain. Do not create HTML id or HTML field_name values, only use the one provided. If the HTML element says 'id' then use Selenium's method 'By.ID', if the element says 'field_name' then use Selenium's 'By.NAME' and so on. Use your best assessment for every HTML element to decide what Selenium method to use. At the top of the code, after the '```python', write the comments: '#Suggested code generated by JungleGym to incorporate with your Agent. Generated at (hr:min, day-month-year) Pacific Time: {time}"
    message_history=[
            {"role": "system", "content": "You are a helpful and smart python developer using python's Selenium. You only respond with python code. You are only using the chrome driver."},
            {"role": "user", "content": preprompt1}]
    response = openai.ChatCompletion.create(model=model, messages=message_history, temperature=0.0)
    response = response['choices'][0]['message']['content']
    #Self-verification implementation:
    message_history.append({"role": "assistant", "content": response})
    message_history.append({"role": "user", "content": 'How would you improve the code if you are executing this task in Selenium {step_name}? Only respond with the executable code to perform the task. Do not explain.'})
    response = openai.ChatCompletion.create(model=model, messages=message_history, temperature=0.0)
    total_tokens = response['usage']['total_tokens']
    response = response['choices'][0]['message']['content']
    return {'response':response, 'message_history':message_history, 'total_tokens': total_tokens, 'model':model}

def save_skill_to_memory(curriculum, task, website, curriculum_tokens, step_name, step_html_id, step_tag_name, step_field_name, step_duration, step_total_tokens, number_of_branches, step_code, model, branch_stats):
    """
    Saves or updates a task and its associated curriculum and steps to the global SKILL_MEMORY list.

    Parameters:
    - curriculum (str): The curriculum associated with the task.
    - task (str): The task to be saved or updated.
    - website (str): The website where the task is performed.
    - curriculum_tokens (int): Token count for the curriculum.
    - step_name (str): Name of the step to be saved.
    - step_html_id (str): HTML ID for the step.
    - tag_name (str): HTML tag name for the step.
    - field_name (str): Field name associated with the step.
    - duration (float): Duration of the step in seconds.
    - step_total_tokens (int): Token count for the step.
    - step_branches (list): Branching details for the step.
    - model (str): Model associated with the step.
    - branch_stats (dict): Statistical details for the step branches.
    
    Output:
    Updates the global SKILL_MEMORY list and prints status messages.
    """
    #First element in memory:
    if len(SKILL_MEMORY) == 0:
        step_number = 0 #We will count the first (first or home page) step as 0
        task_element = {'task': task,
                        'website': website,
                        'curriculums':[{
                                'curriculum': curriculum,
                                'curriculum_tokens': curriculum_tokens,
                                'steps':[{
                                        'step_number': step_number,
                                        'step_name': step_name,
                                        'step_html_id': step_html_id,
                                        'step_tag_name': step_tag_name,
                                        'step_field_name': step_field_name,
                                        'step_duration': step_duration,
                                        'step_total_tokens': step_total_tokens,
                                        'number_of_branches': number_of_branches,
                                        'step_code': step_code,
                                        'model': model,
                                        'branch_stats': branch_stats,
                                    }]
                                }]
                        }
        SKILL_MEMORY.append(task_element)
    else:
        for t in SKILL_MEMORY:
            if t['task'] == task:#If the task is already in the memory
                for c in t['curriculums']:
                    if c['curriculum'] == curriculum:#If the curriculum is already in the memory:
                        step_number = len(c['steps']) + 1
                        c['steps'].append(
                            {
                            'step_number': step_number,
                            'step_name': step_name,
                            'step_html_id': step_html_id,
                            'step_tag_name': step_tag_name,
                            'step_field_name': step_field_name,
                            'step_duration': step_duration,
                            'step_total_tokens': step_total_tokens,
                            'number_of_branches': number_of_branches,
                            'step_code': step_code,
                            'model': model,
                            'branch_stats': branch_stats,
                            })
                        return
                    else:
                        pass
                #In case the curriculum is not in the memory for an existing task:
                step_number = 0#We will count the first (home page) step as 0
                t['curriculums'].append(
                        {'curriculum': curriculum,
                        'curriculum_tokens': curriculum_tokens,
                            'steps': [
                                {
                                'step_number': step_number,
                                'step_name': step_name,
                                'step_html_id': step_html_id,
                                'step_tag_name': step_tag_name,
                                'step_field_name': step_field_name,
                                'step_duration': step_duration,
                                'step_total_tokens': step_total_tokens,
                                'number_of_branches': number_of_branches,
                                'step_code': step_code,
                                'model': model,
                                'branch_stats': branch_stats,
                                }
                            ]
                        })
                return
            else:
                pass
        #If the task is not in the memory:
        step_number = 0 #We will count the first (first or home page) step as 0
        task_element = {'task': task,
                        'website': website,
                        'curriculums':[{
                                'curriculum': curriculum,
                                'curriculum_tokens': curriculum_tokens,
                                'steps':[{
                                        'step_number': step_number,
                                        'step_name': step_name,
                                        'step_html_id': step_html_id,
                                        'step_tag_name': step_tag_name,
                                        'step_field_name': step_field_name,
                                        'step_duration': step_duration,
                                        'step_total_tokens': step_total_tokens,
                                        'number_of_branches': number_of_branches,
                                        'step_code': step_code,
                                        'model': model,
                                        'branch_stats': branch_stats,
                                    }]
                                }]
                        }
        SKILL_MEMORY.append(task_element)

def skill_search(query, website, embeddings_model='thenlper/gte-base'):
    """
    Searches for a task relevant to a query string, using sentence embeddings and FAISS.

    Parameters:
    - query (str): The query string to search for.
    - website (str): The website associated with the search (not used in current implementation).
    - embeddings_model (str, optional): The SentenceTransformer model to use for embeddings. Defaults to 'thenlper/gte-base'.

    Output:
    Returns a dictionary containing:
    - 'task_result': The most relevant task to the query.
    - 'duration': Time taken for the search, rounded to 2 decimal places.
    - 'embeddings_model': The embeddings model used.

    Note: The function utilizes the global SKILL_MEMORY for data.
    """
    start_time = time.perf_counter()
    model = SentenceTransformer(embeddings_model)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    df = pd.json_normalize(SKILL_MEMORY, 'curriculums', ['task'])
    df['ID'] = df.index
    #Create embeddings:
    embeddings = model.encode(df.task.to_list(), show_progress_bar=True)
    embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, df.ID.values)
    vector = model.encode(list([query]))
    D, I = index.search(np.array(vector).astype("float32"), k=5)
    indexed_list = I.flatten().tolist()
    result_list = []
    for relevance_index, df_ID in enumerate(indexed_list):
        if df_ID >= 0:
            results_dict = {}
            results_dict['relevance_index'] = relevance_index
            results_dict['task'] = df.at[df_ID, 'task']
            result_list.append(results_dict)
    end_time = time.perf_counter()
    duration = round(end_time - start_time, 2)
    return {'task_result': result_list[0]['task'], 'duration':duration, 'embeddings_model':embeddings_model}

def chain_generator(website_name, task, curriculum, actionable_elements, step, prev_code):
    tot_response = tree_thought_generation(website=website_name, task=task, curriculum=curriculum, actionable_elements=actionable_elements, branches=3, step=step, CoT=False)
    exact_id = tot_response['response']['response']#Comment this line if you want to use the semiose_vetter (and comment the next line)
    if len(exact_id) > 20:#If it's too long, most likely it's not an id.
        return None
    #exact_id = semiose_vetter(actionable_elements=actionable_elements, query=tot_response['response'])['id_result']#Search for the exact ID with semantic search in case it hallucinated.
    total_step_tokens = tot_response['total_tokens']
    print ('HTML element for this step: ', exact_id)
    tag_name = ''
    field_name = ''
    for i in actionable_elements:#Look for the tag name and field name of the exact id to return it as "previous_step"
        if i['id'] == exact_id:
            tag_name = i['tag_name']
            field_name = i['field_name']
    for dictionary in actionable_elements:
        for key, value in dictionary.items():
            if value ==exact_id:
                exact_id = f"{key}: {value}"
    code = code_generator(website=website_name, step_name=step, step_html_id=exact_id, prev_code=prev_code)['response']
    step_metadata = {'step_name': step, 'step_html_id': exact_id, 'step_html_tag_name': tag_name, 'step_html_field_name': field_name, 'step_code': code, 'total_step_tokens': total_step_tokens, 'number_of_branches': tot_response['number_of_branches'], 'branch_stats': tot_response['branch_stats']}
    return {'step_metadata':step_metadata}

@app.get("/")
@limiter.limit(RATE_LIMIT)
async def root(request: Request):  # Added request: Request
    return {"message": "Hello from TreeVoyager's server. Check the full api documentation at: https://docs.junglegym.ai"}

@app.get("/run_step")
@limiter.limit(RATE_LIMIT)
async def run_task(request: Request, task: str = 'buy the cheapest coffee with best reviews', page: str = URL_DICT['shopping']['link'], curriculum=None, prev_code=None, step=None):
    def event_stream(task=task, page=page, curriculum=curriculum, prev_code=prev_code, step=step):
        if step == None:#First step (if query is just staring):
            print ("Running first time!")
            #Generate curriculum for new task:
            actionable_elements = get_actionable_elements(page)['actionable_elements']
            result = {
                    "actionable_elements": actionable_elements,
                    "model": MODEL,
                }
            yield f"data:{json.dumps(result)}\n\n"
            curriculum_response = curriculum_generator(page, task, actionable_elements=actionable_elements)
            step_list = curriculum_response['tasks_list']
            curriculum = curriculum_response['curriculum']
            result = {
                    "curriculum": curriculum,
                    "step_list": step_list,
                }
            yield f"data:{json.dumps(result)}\n\n"
            curriculum_tokens = curriculum_response['total_tokens']
            #Step 0: 
            start_time = time.perf_counter()
            prev_code = None
            chain_generator_result = chain_generator(website_name=page, task=task, curriculum=curriculum, actionable_elements=actionable_elements, step=step_list[0], prev_code=prev_code)
            if chain_generator_result == None: #Temp. TODO: modify this later, This is just a placeholder for testing 
                print ('No HTML id found for this step.')
                end_time = time.perf_counter()
                step_duration = round(end_time - start_time, 2)
                result = {
                    "curriculum": curriculum,
                    "step_name": step_name,
                    "step_list": step_list,
                    "step_code": 'No HTML id found for this step.',
                    "step_html_id": 'No HTML id found for this step.',
                    "step_tag_name": 'No HTML id found for this step.',
                    "step_field_name": 'No HTML id found for this step.',
                    "step_duration": step_duration,
                    "step_total_tokens": 0,
                    "number_of_branches": 0,
                    "branch_stats": [],
                }
                return result
            else:
                step_code = chain_generator_result['step_metadata']['step_code']
                clean_code = step_code
                prev_code = step_code
                if step_code.startswith("```") and step_code.endswith("```"):
                    clean_code = step_code[3:-3]
                if clean_code.startswith("python"):
                    clean_code = clean_code[len("python"):].lstrip()
                #Save to memory (broken down for readability):
                step_name = chain_generator_result['step_metadata']['step_name']
                step_html_id = chain_generator_result['step_metadata']['step_html_id']
                step_html_tag_name = chain_generator_result['step_metadata']['step_html_tag_name']
                step_html_field_name = chain_generator_result['step_metadata']['step_html_field_name']
                total_step_tokens = chain_generator_result['step_metadata']['total_step_tokens']
                number_of_branches = chain_generator_result['step_metadata']['number_of_branches']
                branch_stats = chain_generator_result['step_metadata']['branch_stats']
                end_time = time.perf_counter()
                step_duration = round(end_time - start_time, 2)
                #save_skill_to_memory(curriculum=curriculum, task=task, website=website_name, curriculum_tokens=curriculum_tokens, step_name=step_name, step_html_id=step_html_id, step_tag_name=step_html_tag_name, step_field_name=step_html_field_name, step_duration=step_duration, step_total_tokens=total_step_tokens, number_of_branches=number_of_branches, step_code=step_code, model=MODEL, branch_stats=branch_stats)
                result = {
                    "task": task,
                    "curriculum": curriculum,
                    "step_name": step_name,
                    "step_list": step_list,
                    "step_code": step_code,
                    "step_html_id": step_html_id,
                    "step_tag_name": step_html_tag_name,
                    "step_field_name": step_html_field_name,
                    "step_duration": step_duration,
                    "step_total_tokens": total_step_tokens,
                    "number_of_branches": number_of_branches,
                    "branch_stats": branch_stats,
                    "model": MODEL,
                }
                yield f"data:{json.dumps(result)}\n\n"
        
        else:
            print ("Running again!")
            actionable_elements = get_actionable_elements(page)['actionable_elements']
            start_time = time.perf_counter()
            chain_generator_result = chain_generator(website_name=page, task=task, curriculum=curriculum, step=step, prev_code=prev_code, actionable_elements=actionable_elements)
            if chain_generator_result == None: #Temp. TODO: modify this later, This is just for testing and don't breaking the code
                print ('No HTML id found for this step.')
                end_time = time.perf_counter()
                step_duration = round(end_time - start_time, 2)
                result = {
                    "curriculum": curriculum,
                    "step_name": step_name,
                    "step_list": step_list,
                    "step_code": 'No HTML id found for this step.',
                    "step_html_id": 'No HTML id found for this step.',
                    "step_tag_name": 'No HTML id found for this step.',
                    "step_field_name": 'No HTML id found for this step.',
                    "step_duration": step_duration,
                    "step_total_tokens": 0,
                    "number_of_branches": 0,
                    "branch_stats": [],
                }
                return result
            else:
                step_code = chain_generator_result['step_metadata']['step_code']
                clean_code = step_code
                prev_code = step_code
                if step_code.startswith("```") and step_code.endswith("```"):
                    clean_code = step_code[3:-3]
                if clean_code.startswith("python"):
                    clean_code = clean_code[len("python"):].lstrip()
                #Save to memory (broken down for readability):
                step_name = chain_generator_result['step_metadata']['step_name']
                step_html_id = chain_generator_result['step_metadata']['step_html_id']
                step_html_tag_name = chain_generator_result['step_metadata']['step_html_tag_name']
                step_html_field_name = chain_generator_result['step_metadata']['step_html_field_name']
                total_step_tokens = chain_generator_result['step_metadata']['total_step_tokens']
                number_of_branches = chain_generator_result['step_metadata']['number_of_branches']
                branch_stats = chain_generator_result['step_metadata']['branch_stats']
                end_time = time.perf_counter()
                step_duration = round(end_time - start_time, 2)
                result = {
                    "task": task,
                    "curriculum": curriculum,
                    "step_name": step_name,
                    "step_code": step_code,
                    "step_html_id": step_html_id,
                    "step_tag_name": step_html_tag_name,
                    "step_field_name": step_html_field_name,
                    "step_duration": step_duration,
                    "step_total_tokens": total_step_tokens,
                    "number_of_branches": number_of_branches,
                    "branch_stats": branch_stats,
                    "model": MODEL,
                }
                yield f"data:{json.dumps(result)}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
