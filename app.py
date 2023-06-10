import os

from apikey import apikey

import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

## App
st.title("CHAT GPT")
prompt = st.text_input("Plug your prompt here")

## Prompt template 
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a description about {topic}'
)

example_template = PromptTemplate(
    input_variables = ['title', 'wiki_research'],
    template = 'write me an example code about {title} while leveraging this wikipedia research: {wiki_research}'
)

## Memory
memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
example_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

## LLMS
llm = OpenAI(temperature=0)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose = True, output_key= 'title', memory=memory)
example_chain = LLMChain(llm=llm, prompt=example_template, verbose = True, output_key= 'example', memory=example_memory)

wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_search = wiki.run(prompt)
    example = example_chain.run(title=title, wiki_research=wiki_search)
    st.write(title)
    st.write(example)
    with st.expander('Memory history'):
        st.info(memory.buffer) 
    with st.expander('Example history'):
        st.info(example_memory.buffer)
    with st.expander('Wiki'):
        st.info(wiki_search)
