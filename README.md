# AutoMed Agent 

> **An AI-powered medical assistant using Generative AI + RAG + Agentic Workflows**

AutoMed Agent is a smart, multi-agent system built to help with:

* Medical literature research using RAG (Retrieval-Augmented Generation)
* Patient diagnosis from clinical notes with ICD-10 mapping
* Simple orchestration to route queries to the correct expert agent

---


## 🔄 Architecture Overview

```mermaid
graph TD
    UI[User Interface (Streamlit)] Query Orchestrator
    Orchestrator |Research| LiteratureAgent
    Orchestrator |Diagnosis| DiagnosisAgent
    LiteratureAgent |Query PubMed FAISS| RAGModule
    DiagnosisAgent |Analyze Text| GPTModule
    DiagnosisAgent |Lookup| ICD10DB
```

* **Orchestrator**: Determines query type and routes to the right agent
* **LiteratureAgent**: Uses FAISS + RAG to answer research questions
* **DiagnosisAgent**: Suggests diagnosis + ICD-10 code from patient notes

---

## ⚙️ Tech Stack

| Component       | Tech                                 |
| --------------- | ------------------------------------ |
| UI              | Streamlit                            |
| LLM             | OpenAI / BioGPT                      |
| Agent Framework | LangChain                            |
| RAG             | FAISS + LangChain Retriever          |
| Data            | Sample PubMed PDFs or CSVs           |
| Deployment      | Streamlit Cloud / HuggingFace Spaces |

---

## 🚀 Features (MVP)

* ✉️ **Chat interface** to ask medical questions
* 📃 **Upload patient notes** to get diagnosis
* 🔎 **RAG-based literature search** with summaries
* ⚖️ **ICD-10 code extraction** from diagnosis

---

## ⚡ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/mohamed-halemo/Auto-Med-Agent
cd Auto-Med-Agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create `.env` file:

```env
OPENAI_API_KEY=your_openai_key
```

---

## 🔄 Project Structure

```bash
Auto-Med-Agent/
├── agents/
│   ├── orchestrator.py
│   ├── literature_agent.py
│   └── diagnosis_agent.py
├── data/
│   ├── pubmed_papers/
│   └── icd10.csv
├── retriever/
│   ├── build_faiss.py
│   └── query_faiss.py
├── app.py  # Streamlit UI
├── .env.example
├── requirements.txt
└── README.md
```

---

## ✅ How It Works

### 1. User sends query from Streamlit UI

* The Orchestrator detects whether it's a research query or diagnosis

### 2. For research:

* The Literature Agent runs a RAG pipeline
* FAISS retrieves chunks
* GPT generates answer

### 3. For diagnosis:

* GPT processes input
* Extracts symptoms + maps to ICD-10 from CSV

### 4. Results shown in the UI

---

## 🚧 Example Prompts

* “Latest treatments for diabetic neuropathy?” → Literature Agent
* Upload a patient note → Diagnosis Agent

---

## 🌟 Stretch Goals

* Add synthetic case generator
* Allow CSV upload of notes for batch diagnosis
* Save chat history (LangChain memory)
* Fine-tune a BioGPT model

---

## 🚪 License

MIT

---

## 🚀 Author

Built with ❤️ by \[Mohamed Hafez] | AI Research Engineer | 2025
