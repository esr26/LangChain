import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS  
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from datetime import datetime
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Please set it in your environment variables.")
    st.stop()


def reset_session():
    st.session_state.vector_store = None
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.messages = [{"role": "Bot", "content": "Hello! How can I help you today? (New session started)"}]
    
    # Delete FAISS index if exists
    if "faiss_index" in os.listdir():
        shutil.rmtree("faiss_index")

# Check if a new session is requested
if "new_session" in st.query_params and st.query_params["new_session"] == "true":
    reset_session()

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector store
if "vector_store" not in st.session_state:
    if "faiss_index" in os.listdir():
        try:
            st.session_state.vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Error loading FAISS index: {e}. Starting a new session.")
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# Initialize conversation memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize LLM model
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="qwen-2.5-32b" 
    )

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "Bot", "content": "Hello! How can I help you today?"}]

# Function to add messages to vector store
def add_message_to_store(text, sender):
    timestamp = datetime.now().isoformat()
    metadata = {"sender": sender, "timestamp": timestamp}
    try:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = FAISS.from_texts([text], embeddings, metadatas=[metadata])
        else:
            st.session_state.vector_store.add_texts([text], metadatas=[metadata])
    except Exception as e:
        st.error(f"Error adding message to vector store: {e}")

# Function to create conversational retrieval chain
def create_chain():
    if st.session_state.vector_store is None:
        st.error("No vector store found. Please start a new session.")
        st.stop()
    
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        return_source_documents=False,
        output_key="answer"
    )


st.title("Chat Bot with Message History")


if st.button("Start New Session"):
    reset_session()
    st.success("New session created!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "User", "content": user_input})
    add_message_to_store(user_input, "User")
    
    with st.chat_message("User"):
        st.markdown(user_input)
    
    with st.spinner("Thinking..."):
        try:
            chain = create_chain()
            result = chain.invoke({"question": user_input})
            response = result.strip() if isinstance(result, str) else result["answer"].strip()
        except Exception as e:
            response = f"Error processing request: {e}"
    
    st.session_state.messages.append({"role": "Bot", "content": response})
    add_message_to_store(response, "Bot")
    
    with st.chat_message("Bot"):
        st.markdown(response)


if st.button("Save Session"):
    try:
        st.session_state.vector_store.save_local("faiss_index")
        st.success("Session saved!")
    except Exception as e:
        st.error(f"Error saving session: {e}")
