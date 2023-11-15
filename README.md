
![JungleGym Logo](https://github.com/a16z-infra/JungleGym/blob/main/JungleGymLogo.png)

---

✨ An Open Source Playground with Agent Datasets and APIs for building and testing your own Autonomous Web Agents 💫

---

https://github.com/a16z-infra/JungleGym/assets/5958899/db0d6613-e95e-48c7-98ee-c8cc79030382

---

## 🧠 Project Overview:

JungleGym is an open-source playground for testing and developing autonomous web agents. Here, you can download demonstration data and test your agents with ground-truth trajectories (using the JungleGym APIs). These Datasets are all available in the JungleGym API and in the [JungleGym](https://junglegym.ai) playground, including 6 realistic, fully functional, sandboxed websites (from WebArena) to test your Agents.

### ✅ How to use it and example code:

We're hosting 3 web agent datasets: Mind2Web, WebArena, and AgentInstruct.

1. Mind2Web: Ground truth for ~2k tasks across 137 websites, including full HTML page states and screenshots. Good for broad testing and development across a range of sites and tasks.
   Here is how you can use the Mind2Web Dataset API to test your Agent with the ground truth. You can find the website, task, and annotation ID in [JungleGym](https://junglegym.ai/Mind2Web) or in the API.

   Full Mind2Web [API endpoints docs](https://docs.junglegym.ai/junglegym/api-documentation/mind2web-api)
   
   Here is an example to get the ground truth actions of one task to compare your agent with ([one click run in Replit](https://replit.com/@mmascorro1/Example-of-how-to-use-the-Mind2Web-Dataset?v=1)):
   ```python
   import requests
   import json
   
   """
   Get the task annotation ID from the Mind2Web dataset page in junglegym.ai/Mind2Web (or from the API)
   Task details:
   Website = 'https://www.kohls.com'
   Task = "Add the cheapest Women's Sweaters to my shopping cart."
   """
   task_annotation_id = '4bc70fa1-e817-405f-b113-0919e8e94205'
   
   # Mind2Web API's endpoint to get ground truth for the list of actions given a task/annotation ID:
   url = f"http://api.junglegym.ai/get_list_of_actions?annotation_id={task_annotation_id}"
   response = requests.get(url)
   
   # Check if the request was successful:
   if response.status_code == 200:
      data = response.json()
   else:
      print(f"Failed to get data: {response.status_code}")
   
   print("Number of total steps to accomplish this task:", len(data['action_reprs']))
   
   print ("Ground truth action for first step:", data['action_reprs'][0])#-> This is the list of ground truth actions you should compare your agent with.
   
   print ("HTML Element data for this step:", data['actions'][0]['pos_candidates'])#-> These are the extended DOM elements of the first action.
   ```

3. WebArena: A task dataset and 6 realistic, fully functional, sandboxed websites. Good for deep testing of many tasks and paths on a single site.
   You can find a website, task, and task_id in [JungleGym](https://junglegym.ai/WebArena) or in the API.
   
   Full WebArena [API endpoints docs](https://docs.junglegym.ai/junglegym/api-documentation/webarena-api).
   
   Here is an example of how to get a WebArena task result. Unlike Mind2Web which shows every step in the DOM, WebArena only shows the final ground truth response [one click run in Replit](https://replit.com/@mmascorro1/WebArena-API-Task-Example?v=1):
   ```python
   import requests
   import json
   """
   Get the task from the WebArena dataset page in junglegym.ai/WebArena (or from the API)
   Task details:
   Website = 'http://shop.junglegym.ai' (WebArena's sandboxed emulated shopping website)
   Task = "What is the price range for products from ugreen?"
   """
   
   WebArena_task = 'What is the price range for products from ugreen?'
   # WebArena API's endpoint to get ground truth result given a task:
   url = f"http://api.junglegym.ai/get_webarena_by_task?task={WebArena_task}"
   # Send the GET request
   response = requests.get(url)
   # Check if the request was successful
   if response.status_code == 200:
     data = response.json()
   else:
     print(f"Failed to get data: {response.status_code}")
   
   print(
       data['data'][0]['eval']['reference_answers']['must_include']
   )  # -> This will give the final ground truth result for the task to compare with your Web agent's response. In this case for this task: ['6.99', '38.99']
   ```
   

5. AgentInstruct: ~1.8k agent trajectories designed for fine-tuning language models (i.e. llama2) on agent tasks. It was introduced with [AgentTuning](https://arxiv.org/abs/2310.12823).

6. TreeVoyager: An LLM-based (GPT-4 Turbo) DOM parser designed to implement some principles from the papers ['Tree of Thoughts'](https://arxiv.org/abs/2305.10601) (ToT) and ['Minecraft's Voyager'](https://arxiv.org/abs/2305.16291) to parse, generate curriculum, select HTML IDs, generate paths, and create skills (memory) for the steps required in the agent trajectory. Note: this is still in very early development, and we would be keen to hear your feedback or contributions.


---

#### 🌲 TreeVoyager in-depth:

- **Purpose**: TreeVoyager is a LLM-based (GPT-4 Turbo) DOM parser as described above. The goal of the TreeVoyager's Streamlit page is to show how it works (step-by-step) and to visualize its outputs and suggested code for every step.
- The source code of TreeVoyager is in this repo under the TreeVoyager folder.
- For instruction and documentation on how to use the TreeVoyager API, refer to the [docs](https://docs.junglegym.ai/junglegym/api-documentation/treevoyager-api).
- #### How does TreeVoyager work?
<div align="center">
  <img src="https://github.com/a16z-infra/JungleGym/blob/main/TreeVoyagerBlockSimple.png" width="40%">
</div>

- #### Inside TreeVoyager:
<div align="center">
  <img src="https://github.com/a16z-infra/JungleGym/blob/main/JungleGym/pages/ImageTreeVoyager.png" width="60%">
</div>

---
## 🔖 Version:

0.9.0 (Experimental) - November 2023
- This project is under development. Contributions are welcome!
  
---
## 👥 Authors:

- Marco Mascorro - [@mascobot](https://twitter.com/Mascobot)
- Matt Bornstein - [@BornsteinMatt](https://twitter.com/BornsteinMatt)

---
## 🔮 Future Work & Contributions:

- TreeVoyager is a very early release. We expect this to be an ongoing project that adds new features and improvements. 
- We think Vision will be a key component for web agents. Once Large Multimodal Models (LMMs) become more powerful and accessible, we think they will heavily contribute to the web agents field, whether it is a combination of HTML/DOM interaction with assisted vision or purely vision-based.
- This is a very early version of JungleGym and TreeVoyager. We would be keen to hear from you and from your contributions!
  
---
## 📔 Acknowledgements:

- Special thanks to the authors of Mind2Web, WebArena, Tree of Thoughts, Voyager and AgentTuning; and to a16z-infra, and the entire open-source community.
  
---
## Disclaimer:

- This is an experimental version of JungleGym, TreeVoyager, and their tools. Use at your own risk. While the app has been tested, the authors hold no liability for any kind of losses arising out of using this application.
- This tool is not designed for CAPTCHA bypass. Always consult a website's Terms of Service (ToS) before use.

---
## 🪪 License:
JungleGym and TreeVoyager are under the permissive Apache 2.0 license. Please refer to the License Agreement for the datasets and tools used with Mind2Web, WebArena, AgentInstruct, etc.
