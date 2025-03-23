import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate 
from langchain.memory import ConversationBufferMemory
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

login(token=HF_TOKEN)


model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map = "auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Search the web for current information using DuckDuckGo."

    )
]

template = """
Answer the following questions as best you can. You have access to the following tools: {tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
(this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought: {agent_scratchpad}

"""

prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)


## Web App
st.title("Open Source Search Engine with LangChain")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "Bot", "content": "Hello, Ask me anythin."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Type your query here...")

if user_input:
    st.session_state.messages.append({"role":"User", "content":user_input})
    with st.chat_message("User"):
        st.markdown(user_input)

    with st.spinner("Searching..."):
        try:
            result = agent_executor.invoke({"input": user_input})
            response = result["output"]
        except Exception as e:
            response = f"Error : {str(e)}"
    
    st.session_state.messages.append({"role": "Bot", "content":response})
    with st.chat_message("Bot"):
        st.markdown(response)
