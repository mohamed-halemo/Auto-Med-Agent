import streamlit as st
from agents.literature_agent import LiteratureAgent

st.title("🧠 AutoMed Agent — Literature QA (Hugging Face Edition)")

query = st.text_input("Enter a medical question:")

if st.button("Search"):
    if query:
        agent = LiteratureAgent()
        response = agent.run(query)
        st.write("### Answer:")
        st.write(response)
    else:
        st.warning("Please enter a query.")
