# Chatbot with Message History Using LangChain

## Overview

This project is an interactive chatbot built with [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), and the [Groq API](https://groq.com/). It leverages a FAISS vector store to persist and retrieve message history, enabling context-aware conversations. The chatbot uses the Groq language model (`qwen-2.5-32b` or a fallback like `mixtral-8x7b-32768`) for response generation, with conversation memory managed by LangChain’s `ConversationBufferMemory`. Users can start new sessions, save conversations to disk, and interact via a clean web interface.

Key features:
- **Message History**: Stores and retrieves past messages using FAISS for semantic similarity search.
- **Session Management**: Supports starting new sessions and saving current ones.
- **Error Handling**: Robust try/except blocks for graceful error reporting.
- **Open-Source**: Built with open-source tools (except for the Groq API, which requires a free key).

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Groq API Key**: Obtain a free key from [Groq Console](https://console.groq.com/keys).
- **Anaconda (Optional)**: For environment management (used in the author’s setup).

## Installation

1. **Clone the Repository** (if hosted on GitHub):
   ```bash
   git clone https://github.com/yourusername/chatbot-with-message-history.git
   cd chatbot-with-message-history
2. **Set Up a Virtual Environment:**
   - Using venv
     ```bash
     python -m venv env
     source env/bin/activate  # macOS/Linux
     env\Scripts\activate     # Windows
     
   - Using Anaconda:
     ```bash
     conda create -n env python=3.9
     conda activate env

   - Install Dependencies:
     ```bash
     pip install streamlit langchain langchain-groq langchain-community faiss-cpu sentence-transformers python-dotenv

   - Configure Environment Variables:
      - Create a .env file in the project root:
        ```bash
        echo "GROQ_API_KEY=your-groq-api-key-here" > .env
      - Replace your-groq-api-key-here with your Groq API key.
##Usage
1. **Run the App:**
   From the project directory:
      ```bashh
      streamlit run chatbot.py --server.fileWatcherType none
  - The --server.fileWatcherType none flag avoids a known Streamlit/PyTorch compatibility issue.
  - Open http://localhost:8501 in your browser.

2. **Interact with the Chatbot:**
   - Chat: Type messages in the input box (e.g., "I like blue color") and press Enter.
   - New Session: Click "Start New Session" to reset memory and history.
   - Save Session: Click "Save Session" to persist the FAISS vector store to disk (faiss_index folder).
   - Automatic New Session: Start with a fresh session using http://localhost:8501/?new_session=true.

3. **Example:**
   - Input: "I like blue color"
   - Response: (Bot replies based on Groq model)
   - Save, reset, then ask "What color do I like?" → Won’t remember post-reset.
  

### Example Screenshots:

![image](https://github.com/user-attachments/assets/bfe66cec-b7a3-43ef-8d5c-d08d75a41b12)


     













     
