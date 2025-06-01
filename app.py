import streamlit as st
from agents.literature_agent import LiteratureAgent
from agents.tool_agent import ToolUsingAgent
from PyPDF2 import PdfReader
from utils.helper import save_uploaded_pdf,save_uploaded_text,rebuild_index

# --- Streamlit UI ---

st.set_page_config(page_title="AutoMed Agent — Hugging Face Edition", layout="centered")

st.title("🤖 Auto-Med-Agent — Hugging Face Edition")

st.markdown(
    """
    Upload medical articles (.txt or .pdf), rebuild the search index, then ask questions.
    The AI agent uses a mix of retrieval and tools to answer your queries.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload .txt or .pdf article", type=["txt", "pdf"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        save_uploaded_pdf(uploaded_file)
    else:
        save_uploaded_text(uploaded_file)
    st.success(" File uploaded! Please rebuild the index to include this document.")

# Index rebuild button
if st.button("🔄 Rebuild Index"):
    with st.spinner("Rebuilding index... This may take a moment."):
        success = rebuild_index()
    if success:
        st.success("Index rebuilt successfully!")
    else:
        st.error("Failed to rebuild index. Please check logs.")

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# User query input
query = st.text_input("Enter your medical question:")

# Search button and query processing
if st.button("🔍 Search") and query.strip():
    agent = ToolUsingAgent(memory=st.session_state.chat_memory)
    answer, summary = agent.run(query.strip())
    
    # Append to history and memory
    st.session_state.history.append((query.strip(), answer, summary))
    st.session_state.chat_memory.append((query.strip(), answer, summary))

# Display conversation history as chat bubbles
for q, a, _ in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {q}")
    with st.chat_message("assistant"):
        st.markdown(f"**Agent:** {a}")

# Optional: Show summary below the last answer
if st.session_state.history and st.session_state.history[-1][2]:
    last_summary = st.session_state.history[-1][2]
    st.markdown(f"**Summary:** {last_summary}")