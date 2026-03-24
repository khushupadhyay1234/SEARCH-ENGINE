import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchResults
)

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.callbacks import StreamlitCallbackHandler

# ---------------------- LOAD ENV ----------------------
load_dotenv()

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="LangChain - Chat with search", page_icon="🔎")

st.title("🔎 LangChain - Chat with search")
st.markdown("Chatbot with Web + Arxiv + Wikipedia search")

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

search = DuckDuckGoSearchResults(num_results=5)

tools = [search, arxiv, wiki]

# ---------------------- SESSION ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I can search the web. Ask me anything!"}
    ]

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------- MAIN ----------------------
if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        streaming=True,
        temperature=0
    )

    # 🔥 NEW AGENT SYSTEM
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    # Chat input
    if prompt_input := st.chat_input("Ask something..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt_input}
        )

        with st.chat_message("user"):
            st.write(prompt_input)

        with st.chat_message("assistant"):

            st_callback = StreamlitCallbackHandler(st.container())

            try:
                response = agent_executor.invoke(
                    {"input": prompt_input},
                    {"callbacks": [st_callback]}
                )["output"]
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

            st.write(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

else:
    st.warning("⚠️ Enter Groq API key to continue")
