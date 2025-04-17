## Integrate our code OPENAI API
import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = openai_key

# Streamlit Framework
st.title('LangChain Demo with OpenAI API')
input_text = st.text_input("Search the topic you want")

# OpenAI LLM
llm = OpenAI(temperature=0.8)

if input_text:
    response = llm(input_text)
    st.write(response)