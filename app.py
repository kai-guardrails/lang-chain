import os

from apikey import apikey

import streamlit as st

from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = apikey

st.title("CHAT GPT")

prompt = st.text_input("Plug your prompt here")

## LLMS
llm = OpenAI(temperature=0)

if prompt:
    response = llm(prompt)
    st.write(response)
