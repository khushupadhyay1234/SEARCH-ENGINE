import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
)
from langchain_community.tools.ddg_search import DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# ---------------------- LOAD ENV ----------------------
load_dotenv()

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="LangChain - Chat with search", page_icon="🔎")

st.title("🔎 LangChain - Chat with search")
st.markdown(
    "This chatbot can search the web, Arxiv, and Wikipedia using LangChain agents."
)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

# ---------------------- TOOLS ----------------------
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
)

# Use DuckDuckGoSearchResults for better reliability (fixes "No good DuckDuckGo Search Result was found")
search = DuckDuckGoSearchResults(num_results=5)

tools = [search, arxiv, wiki]

# ---------------------- SESSION STATE ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you?",
        }
    ]

# ---------------------- DISPLAY CHAT ----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------- MAIN LOGIC ----------------------
if api_key:

    # 🔥 Initialize LLM with DIRECT API KEY (FIX)
    llm = ChatGroq(
        groq_api_key=api_key,   # ✅ FIXED HERE
        model="llama-3.1-8b-instant",
        streaming=True,
        temperature=0
    )

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # Chat input
    if prompt := st.chat_input("Ask something..."):

        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Assistant response
        with st.chat_message("assistant"):

            # Thinking UI
            st_callback = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=True,
                collapse_completed_thoughts=True
            )

            try:
                response = agent.run(
                    prompt,
                    callbacks=[st_callback]
                )
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

            st.write(response)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

else:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")