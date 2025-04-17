# LangChain Project

# 🌟 LangChain Project – Celebrity Explorer

This project utilizes [LangChain](https://github.com/langchain-ai/langchain) to build Streamlit apps powered by OpenAI's language models.

The latest version (`example1.py`) demonstrates a **multi-step reasoning chain** to:
1. Get details about a celebrity
2. Identify their birthdate
3. Retrieve major world events around that time

---

## 🧠 Features

- ✨ LangChain + OpenAI integration
- 📄 Prompt chaining with `SequentialChain`
- ⚡ Streamlit UI
- 🔐 API key security via `constants.py`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Langcahain
```


### 2. Set Up Conda Environment

Create a new Conda environment in the project directory with Python 3.9:
```bash
conda create -p langchainenv python=3.9 -y
```

Activate the environment:
```bash
conda activate ./langchainenv
```

To deactivate:
```bash
conda deactivate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4.	Add your OpenAI API key
Create a file named constants.py and add:
```bash
openai_key = "your-openai-key-here"
```


### 5.	Run the App
▶ Basic App
```bash
streamlit run main.py 
```
▶ Multi-Step Chain (Recommended)
```bash
streamlit run example1.py
```