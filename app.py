"""
app.py — Streamlit Chatbot App (Part 3 integration)
Run with: streamlit run app.py
"""

import os
import streamlit as st
from agents import Head_Agent

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Textbook Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("📚 ML Textbook Chatbot")
st.caption("Ask me anything about Machine Learning! Powered by a multi-agent RAG system.")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: API Keys & Config
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    openai_key = st.text_input(
        "OpenAI API Key", type="password",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    pinecone_key = st.text_input(
        "Pinecone API Key", type="password",
        value=os.getenv("PINECONE_API_KEY", "")
    )
    pinecone_index = st.text_input(
        "Pinecone Index Name",
        value=os.getenv("PINECONE_INDEX", "ml-textbook")
    )
    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        if "head_agent" in st.session_state:
            st.session_state.head_agent.reset_history()
        st.rerun()

    st.divider()
    st.markdown("**Agent Pipeline:**")
    st.markdown(
        "1. 🚫 Obnoxious Check\n"
        "2. 👋 Greeting Handler\n"
        "3. ✏️ Context Rewriter\n"
        "4. 🔍 Topic Relevance\n"
        "5. 📄 Pinecone Retrieval\n"
        "6. ✅ Doc Relevance Check\n"
        "7. 💬 Answer Generation"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Head Agent (cached per session)
# ─────────────────────────────────────────────────────────────────────────────
def get_agent(openai_key: str, pinecone_key: str, index_name: str) -> Head_Agent:
    """Create or retrieve the cached Head_Agent."""
    if "head_agent" not in st.session_state or st.session_state.get("agent_config") != (openai_key, pinecone_key, index_name):
        with st.spinner("Initializing agents..."):
            agent = Head_Agent(openai_key, pinecone_key, index_name)
        st.session_state.head_agent = agent
        st.session_state.agent_config = (openai_key, pinecone_key, index_name)
    return st.session_state.head_agent

# ─────────────────────────────────────────────────────────────────────────────
# Chat History
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("agent"):
            st.caption(f"🔧 Agent path: `{msg['agent']}`")

# ─────────────────────────────────────────────────────────────────────────────
# Handle new input
# ─────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a Machine Learning question..."):
    if not openai_key or not pinecone_key:
        st.error("Please enter your OpenAI and Pinecone API keys in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from Head Agent
    try:
        agent = get_agent(openai_key, pinecone_key, pinecone_index)
        with st.spinner("Thinking..."):
            response, agent_path = agent.chat(prompt)
    except Exception as e:
        response = f"⚠️ Error: {e}"
        agent_path = "Error"

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(f"🔧 Agent path: `{agent_path}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "agent": agent_path,
    })
