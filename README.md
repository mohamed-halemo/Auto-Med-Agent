# 🧠 AutoMed AI Agent – Generative Medical Research Assistant

> A generative AI agent that reads medical research papers, answers domain-specific questions, summarizes papers, performs PubMed searches, handles calculations, and learns from previous conversations.

---

## 📌 Project Highlights

- 🧾 **Literature-Aware RAG Agent** using PubMed articles (PDF or Abstracts)
- 🧠 **Conversational Memory** to retain context
- 🔧 **Tools Integration**: PubMed search, calculator
- 📚 **Automatic Question Generation** from research papers
- 📏 **Evaluation** with ROUGE + BLEU
- 💡 **Powered by**: HuggingFace Transformers, Sentence Transformers, Streamlit, FAISS

---

## 🏗️ System Architecture

graph TD

    A[User Input] --> B{Query Type?}
    B -->|Medical Q/A| C[RAG Agent (PDF / Abstracts)]
    B -->|Math| D[Calculator Tool]
    B -->|PubMed| E[PubMed Search Tool]
    C --> F[Answer + Summary]
    D --> F
    E --> F
    F --> G[Conversation Memory]
    G --> A
---


| Layer         | Tools Used                                   |
| ------------- | -------------------------------------------- |
| Frontend      | Streamlit                                    |
| Backend Logic | Python + LangChain-like agent design         |
| NLP Models    | Sentence Transformers (MiniLM), HF Pipelines |
| Tools         | Custom Python tools (PubMed, Math)           |
| Memory        | Session-based Memory                         |
| Evaluation    | ROUGE, BLEU                                  |

---

git clone https://github.com/mohamed-halemo/Auto-Med-Agent
cd automed-ai-agent

# Install requirements
pip install -r requirements.txt

# Run the app
```bash
streamlit run app.py

automed-ai-agent/
│
├── app.py                         # Streamlit frontend
├── agents/
│   ├── literature_agent.py       # Handles RAG with summarization
│   └── tool_agent.py             # Routes queries to tools or literature
│
├── tools/
│   └── toolkit.py                # Calculator, PubMed search
│
├── evaluation/
│   ├── generate_questions.py     # Uses QA pipeline to generate test set
│   └── evaluate_answers.py       # BLEU + ROUGE evaluations
│
├── data/
│   └── generated_qas.json        # Auto-generated Q/A for evaluation
│
├── utils/
│   └── helper.py                 # PDF & text loading utilities
│
├── requirements.txt
└── README.md
```
# How It Works
> Upload a PDF or enter a PubMed topic.

> The agent creates document embeddings and builds a FAISS retriever.

> You can ask:

> Domain questions → retrieved + summarized

> calculate 7+3 → calculator tool

> pubmed cancer → returns top IDs

> Memory tracks past Q&A to allow follow-ups.

> Metrics + Evaluation
> Generated 20 Q/A pairs from documents using Hugging Face's transformers pipeline.

> Evaluated answers using nltk BLEU and rouge_score metrics.

> Used this to benchmark the effectiveness of the current RAG setup.

