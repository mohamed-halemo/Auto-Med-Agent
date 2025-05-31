import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class LiteratureAgent:
    def __init__(self, index_path="data/faiss_index"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        with open(os.path.join(index_path, "texts.pkl"), "rb") as f:
            self.texts = pickle.load(f)
        self.index = faiss.read_index(os.path.join(index_path, "index.faiss"))

    def run(self, query: str) -> str:
        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, k=3)
        contexts = [self.texts[i] for i in I[0]]

        best_answer = ""
        best_score = 0
        summary = ""

        for context in contexts:
            result = self.qa_pipeline(question=query, context=context)
            if result["score"] > best_score:
                best_answer = result["answer"]
                summary = self.summarizer(context[:1000])[0]["summary_text"]
                best_score = result["score"]

        return best_answer if best_answer else "No confident answer found.", summary
