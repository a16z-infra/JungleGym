# -*- coding: utf-8 -*-
"""
Mind2Web Server Engine
======================

This is the server that runs the Mind2Web engine that serves the partial and full dataset APIs.

Author: Marco Mascorro (@mascobot) & Matt Bornstein
Created: November 2023
Version: 0.0.9 (Experimental)
Status: Development
Python version: 3.9.15
"""
#External libraries:
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import pandas as pd
import json
import datetime
import subprocess
import sentry_sdk

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

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

#####################TOKEN API KEY################################
#FatsAPI keys auth: TODO: replace this with a database:
#API KEYS from database
API_KEYS = ['']
MIND2WEB_API_KEY = os.environ.get('MIND2WEB_API_KEY', default='')
API_KEYS.append(MIND2WEB_API_KEY)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # use token authentication

def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )
###################################################################

#Define app and allow all origins:
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

###Global variables:###
# Define dataset file names
light_dataset_file = 'df_light_combined_train_json_data.pkl'
full_dataset_file = 'df_combined_train_data.pkl'
webarena_dataset_file = 'webarena.json'
agent_instruct_file = 'AgentInstruct.jsonl'
#Get root directory
BASE_DIR = os.path.dirname(os.getcwd())#Gets the parent directory of the current working directory (app)
print ("BASE_DIR: ", BASE_DIR)
LIGHT_DATASET_PKL_PATH = os.path.join(BASE_DIR, 'src', light_dataset_file)#src because the root directory in Render is src
FULL_DATASET_PKL_PATH = os.path.join(BASE_DIR, 'src', full_dataset_file)
RAW_TASK_DATA_PATH = os.path.join(BASE_DIR, 'src', 'data', 'raw_dump', 'task')#Path to where the raw_dump task data is stored
WEBARENA_DATASET_PATH = os.path.join(BASE_DIR, 'src', webarena_dataset_file)
AGENT_INSTRUCT_FILE_PATH = os.path.join(BASE_DIR, 'src', agent_instruct_file)
RATE_LIMIT = "500/minute"
print (BASE_DIR)

# Utility function to sanitize a DataFrame
def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    def sanitize_value(val):
        if isinstance(val, float):
            if val != val:  # Check for NaN
                return "NaN"
            if val == float("inf"):
                return "Infinity"
            if val == float("-inf"):
                return "-Infinity"
        return val
    # Apply sanitize_value to every element in the DataFrame
    return df.applymap(sanitize_value)

try:
    DF_PARTIAL = pd.read_pickle(LIGHT_DATASET_PKL_PATH)
    DF_PARTIAL = sanitize_dataframe(DF_PARTIAL)
    print ("Light Dataset loaded successfully") 
except Exception as e:
    print("Error importing light DF from pkl", e)
    DF_PARTIAL = pd.DataFrame()
try:
    DF_FULL = pd.read_pickle(FULL_DATASET_PKL_PATH)
    DF_FULL = sanitize_dataframe(DF_FULL)
    print ("Full Dataset loaded successfully")
except Exception as e:
    print("Error importing df_combined_train from pkl", e)
    DF_FULL = pd.DataFrame()
try:
    WEBARENA_DF = pd.read_json(WEBARENA_DATASET_PATH)
    WEBARENA_DF = sanitize_dataframe(WEBARENA_DF)
    print("WebArena Dataset loaded successfully")
except Exception as e:
    print("Error importing webarena.json", e)
try:
    DF_AGENT_INSTRUCT = pd.read_json(AGENT_INSTRUCT_FILE_PATH, lines=True)
    print ("AgentInstruct DF loaded successfully") 
except Exception as e:
    print("Error importing AgentInstruct DF from jsonl", e)
    DF_AGENT_INSTRUCT = pd.DataFrame()

#Test api:
@app.get("/", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def root(request: Request):
    return {"message": "Hello World from the JungleGym dataset server! Check the full api documentation at: https://docs.junglegym.ai"}

#update_api_keys():
@app.get("/update_api_keys", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def update_api_keys(request: Request):
    try:
        update_api_keys()
        print("API keys updated")
        return {"data": "API keys updated"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error: {e}")

@app.get("/load_light_train_dataset", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def load_light_train_dataset(request: Request):
    try:
        return {"data": DF_PARTIAL.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for load_light_train_dataset. Error: {e}")

@app.get("/load_full_train_dataset", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def load_full_train_dataset(request: Request):
    try:
        return {"data": DF_FULL.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for load_full_train_dataset. Error: {e}")

@app.get("/get_list_of_actions", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_list_of_actions(request: Request, annotation_id: str):
    try:
        actions = DF_FULL[DF_FULL['annotation_id'] == annotation_id]['actions'].tolist()[0]
        action_reprs = DF_FULL[DF_FULL['annotation_id'] == annotation_id]['action_reprs'].tolist()[0]
        return {"actions": actions, "action_reprs": action_reprs}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for get_list_of_actions for annotation {annotation_id}. Error: {e}")

### RAW DUMP _TASK_ DATA ###
@app.get("/get_raw_json_screenshots", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_raw_json_snapshots(request: Request, annotation_id: str):
    try:
        path = os.path.join(RAW_TASK_DATA_PATH, annotation_id, 'processed', 'screenshot.json')
        with open(path, "r") as f:
            screenshots = json.load(f)
        return {"data": screenshots}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for get_raw_json_snapshots for annotation {annotation_id}. Error: {e}")

@app.get("/get_raw_dom_content", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_raw_dom_content(request: Request, annotation_id: str):
    try:
        path = os.path.join(RAW_TASK_DATA_PATH, annotation_id, 'processed', 'dom_content.json')
        with open(path, "r") as f:
            dom_content = json.load(f)
        return {"data": dom_content}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for get_raw_dom_content for annotation {annotation_id}. Error: {e}")

@app.get("/get_storage", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_raw_dom_content(request: Request, annotation_id: str):
    try:
        path = os.path.join(RAW_TASK_DATA_PATH, annotation_id, 'processed', 'storage.json')
        with open(path, "r") as f:
            storage = json.load(f)
        return {"data": storage}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for storage for annotation {annotation_id}. Error: {e}")

@app.get("/get_raw_trace_zip", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_raw_trace_zip(request: Request, annotation_id: str):
    try:
        path = os.path.join(RAW_TASK_DATA_PATH, annotation_id,'trace.zip')
        if os.path.exists(path):
            return FileResponse(path, media_type='application/zip')
        else:
            raise HTTPException(status_code=404, detail=f"No trace.zip found for annotation {annotation_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while trying to get trace.zip for annotation {annotation_id}. Error: {e}")
    
##### WEBARENA DATASET #####
@app.get("/get_full_webarena_dataset", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_webarena_dataset(request: Request):
    try:
        return {"data": WEBARENA_DF.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for get_webarena_dataset. Error: {e}")
    
@app.get("/get_webarena_tasks", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_webarena_task(request: Request):
    try:
        return {"data": WEBARENA_DF[['intent', 'start_url_junglegym']].to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for get_webarena_tasks. Error: {e}")
    
@app.get("/get_webarena_by_task", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_webarena_by_task(request: Request, task: str):
    try:
        return {"data": WEBARENA_DF[WEBARENA_DF['intent'] == task].to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for get_webarena_by_task. Error: {e}")
    
@app.get("/get_webarena_by_task_id", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_webarena_by_task_id(request: Request, task_id: str):
    try:
        return {"data": WEBARENA_DF[WEBARENA_DF['task_id'] == int(task_id)].to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for get_webarena_by_task_id. Error: {e}")

@app.get("/get_webarena_by_domain", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def get_webarena_by_domain(request: Request, domain: str):
    try:
        return {"data": WEBARENA_DF[WEBARENA_DF['start_url_junglegym'] == domain].to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for get_webarena_by_domain. Error: {e}")
    
@app.get("/load_agent_instruct", )#dependencies=[Depends(api_key_auth)]
@limiter.limit(RATE_LIMIT)
async def load_agent_instruct(request: Request):
    try:
        return {"data": DF_AGENT_INSTRUCT.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No data found for AgentInstruct dataset. Error: {e}")