
![JungleGym Logo](https://github.com/a16z-infra/JungleGym/blob/main/JungleGymLogo.png)

---

✨ An Open Source Playground with Agent Datasets and APIs for building and testing your own Autonomous Web Agents 💫

---

https://github.com/a16z-infra/JungleGym/assets/5958899/db0d6613-e95e-48c7-98ee-c8cc79030382

---

## 🧠 Project Overview:

JungleGym is an open-source playground for testing and developing autonomous web agents. Here, you can download demonstration data, test your agents with ground-truth trajectories (using the JungleGym APIs), and test your web agents. These Datasets are all available in the JungleGym API and in the [JungleGym](https://junglegym.ai) playground, including 6 fully functional emulated websites (from WebArena) to test your Agents.

### ✅ JungleGym Main Features:

We're hosting 3 web agent datasets: Mind2Web, WebArena and AgentInstruct.

1. Mind2Web: Ground truth for ~2k tasks across 137 websites, including full HTML page states and screenshots. Good for broad testing and development across a range of sites and tasks.

2. WebArena: A task dataset and 6 realistic, fully functional, sandboxed websites. Good for deep testing of many tasks and paths on a single site.

3. AgentInstruct: ~1.8k agent trajectories designed for fine-tuning language models (i.e. llama2) on agent tasks. It was introduced with [AgentTuning](https://arxiv.org/abs/2310.12823).

4. TreeVoyager: An LLM-based (GPT-4 Turbo) DOM parser designed to implement some principles from the papers ['Tree of Thoughts'](https://arxiv.org/abs/2305.10601) (ToT) and ['Minecraft's Voyager'](https://arxiv.org/abs/2305.16291) to parse, generate curriculum, select HTML IDs, generate paths, and create skills (memory) for the steps required in the agent trajectory. Note: this is still in very early development, and we would be keen to hear your feedback or contributions.

5. APIs for all Web Agent Datasets & TreeVoyager: JungleGym provides APIs for these three datasets and TreeVoyager. You can use these APIs to test ground-truths with your agent trajectories and you can use TreeVoyager for parsing the DOM with the suggested code (with Python's Selenium).

For the full documentation, you can read the [JungleGym Docs here](https://docs.junglegym.ai/junglegym/junglegym-ai-overview).

This Github repo structure is broken down into two main components: 
1. JungleGym (the JungleGym folder) is a Streamlit app to visualize the web agent datasets, trajectories, steps, and web snapshots and to download the agent datasets.
2. TreeVoyager (the TreeVoyager folder) - The LLM-based DOM parser described above in step 4.

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
