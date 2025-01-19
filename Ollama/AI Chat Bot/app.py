import os
from dotenv import load_dotenv

from langchain_ollama.llms import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## Langchain Tracking
os.environ["LANCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")


## Prompt Template
prompt=ChatPromptTemplate.from_messages(

    [
        ("system", "You are a AI Expert Assistant. Please give best answers for the questions."),
        ("user", "Question:{input}")
    ]
)


## Streamlit Framework
st.title("AI Chat Bot with Llama 3.1")
input_text = st.text_input("Ask your question !")

## Ollama Llama 3.1 Model
llm = OllamaLLM(model="llama3.1")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"input":input_text}))
