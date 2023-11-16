import requests
import json
import pandas as pd

TREE_VOYAGER_SERVER_ENDPOINT = 'https://treevoyager.junglegym.ai'
"""
INSTRUCTIONS:
1. Run the function below with a task (like "buy coffee") and the URL of the website you want to run the task on. If you are running the task for the first time, you can leave the curriculum, prev_code and step_name as None (as it hasn't generated them yet), this will be return in the next step.
2. The function will return the HTML/DOM element your agent should interact with (it will parse it and select it for you!), and the suggested code for the step. It will also return a suggested curriculum (a plan), the next step (in Python Selenium's terms) and the code for the task.
3. To run the next step, you need to run the function again but now by passing the curriculum (plan), the new URL (in case you agent clicked on a button to the next page), the code and the step name (step name to execute from curriculum) from the previous step.
And so on... (this is a loop)
At any given step, you can modify the curriculum, the URL, the code and the step name and pass it to the function to get the next step.
"""

#This function is used to run a task on the TreeVoyager server. No need to modify it and just run the function below
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
        print("Error running task: " + str(response.status_code))
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

################### Example of how to use the function: ###################
#If you are running the task for the first time, you can leave the curriculum, prev_code and step_name as None (as it hasn't generated them yet)
#The curriculum (plan) and the step_name (next step) will be returned by the function and you can use them in the next turn.

for result in run_task(task="buy coffee", url='http://shop.junglegym.ai', curriculum=None, prev_code=None, step_name=None):
    if 'actionable_elements' in result:#This returns all parsed DOM Elements of the page.
        all_dom_elements = result['actionable_elements']
    if 'curriculum' in result:#This returns the suggested curriculum (plan) for the task.
        print("Curriculum: ", result['curriculum'])
    if 'step_name' in result:#This returns the suggested next step (in Python Selenium's terms) for the task.
        print("Step name: ", result['step_name'])
    if 'step_list' in result:#This returns the list of all suggested steps (in Python Selenium's terms) for the task.
        print("List of steps: ", result['step_list'])
    if 'step_code' in result:#This returns the suggested code for the step (only for this step) for the task.
        print("Code for this step: ", result['step_code'])
    if 'info_data' in result:
        data = result['info_data']
        html_element = data['HTML id']#This is the HTML id that your agent should interact with based on the task and current suggested step.
        tag_name = data['Step HTML Tag Name']#This is the HTML tag name that your agent should interact with based on the task and current suggested step.
        field_name = data['Step HTML Field Name']#This is the HTML field name that your agent should interact with based on the task and current suggested step.
        print("HTML id: ", html_element, " HTML tag name: ", tag_name, " HTML field name: ", field_name)