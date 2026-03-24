import streamlit as st
from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

# Tools
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun
)

from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

# Messages
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------- LOAD ENV ----------------------
load_dotenv()

# ---------------------- UI ----------------------
st.set_page_config(page_title="Smart AI Agent", page_icon="🤖")

st.title("🤖 Smart AI Agent (No Errors Version)")
st.markdown("Web + Wikipedia + Arxiv with Memory 🚀")

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# ---------------------- TOOLS ----------------------
search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

tools = {
    "search": search_tool,
    "wiki": wiki_tool,
    "arxiv": arxiv_tool
}

# ---------------------- SESSION MEMORY ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hi! I can search the web, Wikipedia, and Arxiv. Ask me anything!")
    ]

# Display chat
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.write(msg.content)

# ---------------------- MAIN ----------------------
if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    if user_input := st.chat_input("Ask something..."):

        st.session_state.messages.append(HumanMessage(content=user_input))

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):

            st.info("🔎 Thinking...")

            try:
                # STEP 1: Decide tool
                tool_prompt = f"""
Decide best tool for this query:

Options:
- search (for current/general info)
- wiki (for definitions/concepts)
- arxiv (for research papers)

Query: {user_input}

Answer ONLY one word: search / wiki / arxiv
"""
                tool_choice = llm.invoke(tool_prompt).content.lower().strip()

                # STEP 2: Run tool safely
                try:
                    if "wiki" in tool_choice:
                        tool_output = tools["wiki"].run(user_input)
                    elif "arxiv" in tool_choice:
                        tool_output = tools["arxiv"].run(user_input)
                    else:
                        tool_output = tools["search"].run(user_input)
                except Exception:
                    tool_output = "⚠️ Tool failed. Answering from general knowledge."

                # STEP 3: Build memory context
                history_text = "\n".join([m.content for m in st.session_state.messages])

                final_prompt = f"""
Conversation History:
{history_text}

Tool Result:
{tool_output}

User Question:
{user_input}

Give a helpful, clear answer:
"""

                # STEP 4: Generate response safely
                try:
                    response = llm.invoke(final_prompt).content
                except Exception:
                    response = "⚠️ AI failed to respond. Please try again."

            except Exception as e:
                response = f"⚠️ Unexpected Error: {str(e)}"

            st.write(response)

        st.session_state.messages.append(AIMessage(content=response))

else:
    st.warning("⚠️ Enter Groq API key to continue")
