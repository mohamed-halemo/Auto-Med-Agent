import streamlit as st
from agents.literature_agent import LiteratureAgent
from PyPDF2 import PdfReader

def save_uploaded_text(file):
    content = file.read().decode("utf-8")
    with open(f"data/pubmed_papers/{file.name}", "w") as f:
        f.write(content)

def save_uploaded_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open(f"data/pubmed_papers/{file.name}.txt", "w") as f:
        f.write(text)

st.title("🔬 AutoMed Agent — Hugging Face Edition")

uploaded_file = st.file_uploader("Upload .txt or .pdf article", type=["txt", "pdf"])
if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        save_uploaded_pdf(uploaded_file)
    else:
        save_uploaded_text(uploaded_file)
    st.success("✅ File uploaded! Please rebuild index.")

if st.button("Rebuild Index"):
    import os
    os.system("python retriever/build_faiss.py")
    st.success("Index rebuilt!")

query = st.text_input("Enter your medical question:")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Search"):
    if query:
        agent = LiteratureAgent()
        answer, summary = agent.run(query)
        st.session_state.history.append((query, answer, summary))

for q, a, s in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    st.markdown(f"_Summary:_ {s}")
    st.markdown("---")
