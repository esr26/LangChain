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
