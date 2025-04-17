## Integrate our code OPENAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain 
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = openai_key

# Streamlit Framework
st.title('Celebrity search Results')
input_text = st.text_input(" your celebrity name you want to know about")

# Prompt Template1

first_input_prompt= PromptTemplate(
    input_variables=['name'],
    template="What is the name of the celebrity {name}"
)

#Memory
Person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
description_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')




# OpenAI LLMs
llm = OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt =first_input_prompt, verbose=True,output_key='person', memory=Person_memory)


# Prompt Template2
second_input_prompt= PromptTemplate(
    input_variables=['person'],
    template="when was the {person} born"
)
chain2= LLMChain (llm=llm,prompt =second_input_prompt, verbose=True,output_key='dob', memory=dob_memory)


# Prompt Template3
third_input_prompt= PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world"
)
chain3= LLMChain (llm=llm,prompt =third_input_prompt, verbose=True,output_key='description', memory=description_memory)




parent_chain=SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person','dob','description'], 
    verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander("Person Name"):
        st.write(Person_memory.buffer)
     
    with st.expander("Date of Birth"):
             st.write(dob_memory.buffer)
    
    with st.expander("Description"):
           st.write(description_memory.buffer)