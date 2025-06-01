# Import libraries
import faiss                         # Facebook AI Similarity Search – used for efficient vector search
import pickle                        # Used for loading stored Python objects (like text lists)
import os                            # For file path manipulation
from sentence_transformers import SentenceTransformer  # For embedding queries and texts
from transformers import pipeline    # For QA and summarization pipelines

class LiteratureAgent:
    def __init__(self, index_path="data/faiss_index"):
        """
        Initializes the LiteratureAgent by loading the models and precomputed index.
        """
        # Load a sentence transformer for embedding queries
        self.model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

        # Load a question answering pipeline using a fine-tuned RoBERTa model
        self.qa_pipeline = pipeline("question-answering", model="ktrapeznikov/biobert_v1.1_pubmed_squad_v2")

        # Load a summarization pipeline using a BART model
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Load precomputed texts (contexts) from a pickle file
        with open(os.path.join(index_path, "texts.pkl"), "rb") as f:
            self.texts = pickle.load(f)

        # Load the FAISS index which allows fast similarity search over embeddings
        self.index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    def run(self, query: str) -> tuple[str, str]:
        """
        Runs a semantic search + QA + guided summarization pipeline for the input query.

        Args:
            query (str): The question asked by the user.

        Returns:
            tuple[str, str]: The best answer found, and a guided summary of the context.
        """

        # Step 1: Embed the input query
        query_embedding = self.model.encode([query])

        # Step 2: Search the FAISS index for top 3 most similar documents
        distances, indices = self.index.search(query_embedding, k=3)

        # Step 3: Retrieve the top contexts (texts) using the indices
        top_contexts = [self.texts[i] for i in indices[0]]

        best_answer = ""
        best_score = 0
        summary = ""

        # Step 4: Loop over top contexts to get the best answer using the QA model
        for context in top_contexts:
            result = self.qa_pipeline(question=query, context=context)

            if result["score"] > best_score:
                best_answer = result["answer"]
                best_score = result["score"]

                # Step 5: Guide the summarizer to focus on answering the query
                prompt = f"Answer the question: '{query}' based on the following passage:\n\n{context[:1000]}"
                summary = self.summarizer(prompt)[0]["summary_text"]

        return (
            best_answer if best_answer else "No confident answer found.",
            summary
        )
