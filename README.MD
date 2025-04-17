# LangChain Project

This project utilizes [LangChain](https://github.com/langchain-ai/langchain) to build applications powered by language models.

## Getting Started

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
```bash
streamlit run main.py
```