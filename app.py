import streamlit as st
from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

# Tools
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchResults
)

# ---------------------- LOAD ENV ----------------------
load_dotenv()

# ---------------------- UI ----------------------
st.set_page_config(page_title="Smart Search Chatbot", page_icon="🔎")
st.title("🔎 Smart Search Chatbot")
st.markdown("Search Web + Arxiv + Wikipedia without agent errors 🚀")

# Sidebar
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# ---------------------- TOOLS ----------------------
search = DuckDuckGoSearchResults(num_results=3)

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)

# ---------------------- SESSION ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything 🔍"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------- MAIN ----------------------
if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    if prompt := st.chat_input("Ask something..."):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):

            try:
                # 🔥 Run tools manually
                web_result = search.run(prompt)
                wiki_result = wiki.run(prompt)
                arxiv_result = arxiv.run(prompt)

                # Combine results
                combined = f"""
Web Search:
{web_result}

Wikipedia:
{wiki_result}

Arxiv:
{arxiv_result}

Answer the user clearly using above info:
{prompt}
"""

                response = llm.invoke(combined).content

            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

            st.write(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

else:
    st.warning("Enter API key to continue")
