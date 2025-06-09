# Import necessary libraries
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_faiss_index(data_path="data/pubmed_papers", index_path="data/faiss_index"):
    """
    Builds an improved FAISS index from documents in the specified folder.
    Uses better chunking strategy and embedding model.
    """
    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    documents = []
    
    # Load documents from both .txt and .pdf files
    for fname in os.listdir(data_path):
        file_path = os.path.join(data_path, fname)
        try:
            if fname.endswith(".txt"):
                loader = TextLoader(file_path)
                docs = loader.load()
            elif fname.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            else:
                continue
                
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} docs from {fname}")
        except Exception as e:
            logger.error(f"Error loading {fname}: {e}")
            continue

    if not documents:
        raise ValueError("No documents found to index. Check your input files.")

    # Improved text splitter with better chunking strategy
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for better precision
        chunk_overlap=100,  # More overlap for better context
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    # Split and clean documents
    raw_chunks = splitter.split_documents(documents)
    cleaned_chunks = []
    
    for chunk in raw_chunks:
        text = chunk.page_content.strip()
        # Improved filtering criteria
        if (len(text) > 100 and  # Minimum length
            len(text) < 2000 and  # Maximum length
            not text.lower().startswith("news quiz") and
            "cnn" not in text.lower() and
            not text.isspace() and  # Not just whitespace
            len(text.split()) > 10):  # Minimum word count
            cleaned_chunks.append(text)

    logger.info(f"🔍 {len(cleaned_chunks)} clean chunks after filtering")

    if len(cleaned_chunks) == 0:
        raise ValueError("No valid chunks found to index. Check your input files.")

    # Embed chunks with progress tracking
    logger.info("Generating embeddings...")
    embeddings = model.encode(
        cleaned_chunks,
        show_progress_bar=True,
        batch_size=32,
        convert_to_tensor=False
    )

    # Normalize embeddings for better similarity search
    faiss.normalize_L2(embeddings)

    # Build FAISS index with better configuration
    dim = len(embeddings[0])
    
    # Use IVF index for better performance with large datasets
    nlist = min(100, len(embeddings) // 10)  # Number of clusters
    quantizer = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train the index if we have enough vectors
    if len(embeddings) > nlist:
        index.train(embeddings)
    
    # Add vectors to the index
    index.add(embeddings)
    
    # Save index and texts
    os.makedirs(index_path, exist_ok=True)
    
    with open(os.path.join(index_path, "texts.pkl"), "wb") as f:
        pickle.dump(cleaned_chunks, f)
    
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))
    
    logger.info(f"\nFAISS index saved to {index_path}")
    logger.info(f"Total embedded chunks: {len(cleaned_chunks)}")
    logger.info(f"Index type: IVF with {nlist} clusters")

# Run
if __name__ == "__main__":
    os.makedirs("data/faiss_index", exist_ok=True)
    build_faiss_index()
