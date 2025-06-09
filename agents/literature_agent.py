# Import libraries
import faiss                         # Facebook AI Similarity Search – used for efficient vector search
import pickle                        # Used for loading stored Python objects (like text lists)
import os                            # For file path manipulation
import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline    # For QA and summarization pipelines
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data."""
    required_packages = ['punkt', 'stopwords']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"Downloaded {package}")

# Ensure NLTK data is downloaded
download_nltk_data()

class LiteratureAgent:
    def __init__(self, index_path="data/faiss_index"):
        """
        Initializes the LiteratureAgent with improved models and preprocessing.
        """
        # Load a more powerful sentence transformer for embedding
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
        # Load a cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Load a question answering pipeline using a fine-tuned model
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
        
        # Load a summarization pipeline using a BART model
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )
        
        # Load precomputed texts and index
        try:
            with open(os.path.join(index_path, "texts.pkl"), "rb") as f:
                self.texts = pickle.load(f)
            self.index = faiss.read_index(os.path.join(index_path, "index.faiss"))
        except Exception as e:
            print(f"Error loading index: {e}")
            self.texts = []
            self.index = None

    def preprocess_query(self, query: str) -> str:
        """
        Preprocesses the query to improve retrieval quality.
        """
        try:
            # Convert to lowercase
            query = query.lower()
            
            # Remove special characters and extra whitespace
            query = re.sub(r'[^\w\s]', ' ', query)
            query = ' '.join(query.split())
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(query)
            filtered_tokens = [word for word in tokens if word not in stop_words]
            
            return ' '.join(filtered_tokens)
        except Exception as e:
            print(f"Error in query preprocessing: {e}")
            return query  # Return original query if preprocessing fails

    def retrieve_documents(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieves and re-ranks documents using both bi-encoder and cross-encoder.
        """
        if not self.index or not self.texts:
            return []

        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Extract key terms from query for topic matching
        key_terms = set(word.lower() for word in processed_query.split())
        
        # Get initial embeddings and search
        query_embedding = self.model.encode([processed_query])
        distances, indices = self.index.search(query_embedding, k=k*2)  # Retrieve more candidates for re-ranking
        
        # Get candidate documents and filter by topic relevance
        candidates = []
        for i, d in zip(indices[0], distances[0]):
            text = self.texts[i].lower()
            # Check if the text contains any of the key terms
            if any(term in text for term in key_terms):
                candidates.append((self.texts[i], 1.0 / (1.0 + d)))
        
        if not candidates:
            return []

        # Re-rank using cross-encoder
        pairs = [(processed_query, doc) for doc, _ in candidates]
        scores = self.cross_encoder.predict(pairs)
        
        # Combine scores and sort
        reranked = [(doc, score) for (doc, _), score in zip(candidates, scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:k]

    def run(self, query: str) -> tuple[str, str]:
        """
        Runs an improved semantic search + QA + guided summarization pipeline.
        """
        # Retrieve and re-rank documents
        retrieved_docs = self.retrieve_documents(query, k=8)
        
        if not retrieved_docs:
            return "No relevant documents found.", ""

        best_answer = ""
        best_score = 0
        summary = ""
        all_contexts = []
        all_answers = []

        # Extract key terms for topic verification
        key_terms = set(word.lower() for word in query.split())

        # Process each retrieved document
        for context, score in retrieved_docs:
            # Verify topic relevance
            context_lower = context.lower()
            if not any(term in context_lower for term in key_terms):
                continue

            all_contexts.append(context)
            
            # Run QA on the context with adjusted parameters
            result = self.qa_pipeline(
                question=query,
                context=context,
                max_answer_len=150,
                handle_impossible_answer=True,
                topk=3
            )

            # Handle both single answer and multiple answers
            if isinstance(result, dict):
                answers = [result]
            else:
                answers = result

            for answer in answers:
                # Adjust confidence threshold and scoring
                confidence = answer["score"]
                if confidence > 0.1:
                    # Verify answer relevance
                    answer_lower = answer["answer"].lower()
                    if any(term in answer_lower for term in key_terms):
                        all_answers.append((answer["answer"], confidence * score))

        # Select best answer from all candidates
        if all_answers:
            # Sort by combined score
            all_answers.sort(key=lambda x: x[1], reverse=True)
            best_answer = all_answers[0][0]
            
            # If we have multiple good answers, combine them
            if len(all_answers) > 1 and all_answers[1][1] > 0.3:
                additional_info = [ans[0] for ans in all_answers[1:3] if ans[1] > 0.3]
                if additional_info:
                    best_answer += " Additionally, " + " ".join(additional_info)

        # Generate a comprehensive summary using all retrieved contexts
        if all_contexts:
            # Filter contexts by topic relevance
            relevant_contexts = [ctx for ctx in all_contexts if any(term in ctx.lower() for term in key_terms)]
            if not relevant_contexts:
                relevant_contexts = all_contexts

            # Use more contexts for summary
            combined_context = " ".join(relevant_contexts[:5])
            prompt = f"""Based on the following medical information about {query}, provide a detailed summary:

{combined_context[:3000]}

Focus specifically on information related to the question."""

            try:
                summary = self.summarizer(
                    prompt,
                    max_length=200,
                    min_length=50,
                    do_sample=False
                )[0]["summary_text"]

                # Verify summary relevance
                if not any(term in summary.lower() for term in key_terms):
                    summary = "No relevant summary could be generated."
            except Exception as e:
                print(f"Error in summarization: {e}")
                summary = ""

        # If no confident answer was found but we have a summary, use the first sentence of the summary
        if not best_answer and summary and summary != "No relevant summary could be generated.":
            best_answer = summary.split('.')[0] + '.'

        return (
            best_answer if best_answer else "No confident answer found.",
            summary
        )
