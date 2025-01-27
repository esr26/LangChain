from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Prompt Template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate(
    [("system", system_template),
     ("user", '{text}')]
)
parser=StrOutputParser()

# Chain
chain = prompt_template|model|parser

# App
app=FastAPI(title="Language Translator",
            version="1.0",
            description="API server using Langchain runnable interfaces")

# Chain Routes
add_routes(
    app, 
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)