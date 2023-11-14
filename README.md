
![JungleGym Logo](https://github.com/a16z-infra/JungleGym/blob/main/JungleGymLogo.png)

---

✨ An Open Source Playground with Agent Datasets and APIs for building and testing your own Autonomous Web Agents 💫

---

https://github.com/a16z-infra/JungleGym/assets/5958899/3b9a054a-442b-43d2-a67b-b25359e27713

---

## 🧠 Project Overview:

JungleGym is an open-source playground for testing and developing autonomous web agents. Here, you can download demonstration data, test your agents with ground-truth trajectories (using the JungleGym APIs), and design your web agents. These Datasets are all available in the JungleGym API and in this playground, including 6 fully functional emulated websites (from WebArena) to test your Agents.

This repo is broken down into three main components: 
1. JungleGym (the JungleGym folder) is a Streamlit app to visualize the web agent datasets, trajectories, steps, web snapshots and to download the agent datasets. You can use these Web Demonstration Datasets to train your LLMs or to use with your Agents. You can see it on the official [JungleGym website.](https://junglegym.ai)
2. TreeVoyager (the TreeVoyager folder) - a Python-based protocol designed to implement some principles from the papers ['Tree of Thoughts'](https://arxiv.org/abs/2305.10601) (ToT) and ['Minecraft's Voyager'](https://arxiv.org/abs/2305.16291) to parse, generate curriculum, select HTML IDs, generate paths, create skills (memory) and suggested code for the steps required in a web agent trajectory.
3. The APIs server for the web agent datasets Mind2Web, WebArena, AgentInstruct, and for Treevoyager.



### ✅ JungleGym Main Features:

1. Mind2Web (Generalist Agent for the Web) dataset integration.

2. WebArena Dataset integration with 6 different fully functional emulated web environments to test agents end-to-end: an online store, Git, a social forum, Wikipedia, an E-Commerce CMS, and a map.

3. AgentInstruct integration: AgentInstruct is a dataset that aims to improve LLMs’ generalized agent abilities. It was introduced with [AgentTuning](https://arxiv.org/abs/2310.12823): Enabling Generalized Agent Abilities for LLMs. The AgentInstruct dataset includes 1,866 trajectories from 6 agents' tasks.

4. TreeVoyager: a Python-based protocol designed to implement some principles from the papers 'Tree of Thoughts' (ToT) and 'Minecraft's Voyager' to parse, generate curriculum, select HTML IDs, generate paths, and create skills (memory) for the steps required in the agent trajectory. It currently uses GPT-4 Turbo. Note: this is still in very early development, and we would be keen to hear your feedback or contributions.

5. APIs for all Web Agent Datasets & TreeVoyager: JungleGym provides APIs for the Mind2Web dataset, WebArena dataset, AgentInstruct dataset, and TreeVoyager. You can use these APIs to create and download your own dataset subsets, and test agent trajectories using the TreeVoyager suggested code (with Python's Selenium).

For the full documentation, you can read the [JungleGym Docs here](https://docs.junglegym.ai/junglegym/junglegym-ai-overview).

---

#### 🌲 TreeVoyager in-depth:

- **Purpose**: TreeVoyager web agent Python-based protocol as described above. The goal of the TreeVoyager Streamlit page is to show how it works (step-by-step) and to visualize its outputs and suggested code for every step. It currently uses GPT-4 Turbo.
- The source code of TreeVoyager is in this repo under the TreeVoyager folder.
- For instruction and documentation on how to use the TreeVoyager API, refer to the [docs](https://docs.junglegym.ai/junglegym/api-documentation/treevoyager-api).
- #### TreeVoyager Functionality Diagram:
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
