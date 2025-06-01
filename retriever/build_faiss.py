# Import necessary libraries
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_faiss_index(data_path="data/pubmed_papers", index_path="data/faiss_index"):
    """
    Builds a FAISS index from .txt documents in the specified folder.
    Only keeps chunks that are useful (not empty, short, or junk).
    """
    
    # Load embedding model
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    documents = []

    for fname in os.listdir(data_path):
        if fname.endswith(".txt"):
            file_path = os.path.join(data_path, fname)
            loader = TextLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f" Loaded {len(docs)} docs from {fname}")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    # Split and clean
    raw_chunks = splitter.split_documents(documents)
    cleaned_chunks = []
    for chunk in raw_chunks:
        text = chunk.page_content.strip()
        # Skip empty or low-info chunks
        if len(text) > 210 and not text.lower().startswith("news quiz") and "cnn" not in text.lower():
            cleaned_chunks.append(text)

    print(f"🔍 {len(cleaned_chunks)} clean chunks after filtering")

    if len(cleaned_chunks) == 0:
        raise ValueError(" No valid chunks found to index. Check your input .txt files.")

    # Embed
    embeddings = model.encode(cleaned_chunks, convert_to_tensor=False)

    # Build FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    os.makedirs(index_path, exist_ok=True)
    with open(os.path.join(index_path, "texts.pkl"), "wb") as f:
        pickle.dump(cleaned_chunks, f)

    faiss.write_index(index, os.path.join(index_path, "index.faiss"))

    print(f"\n FAISS index saved to {index_path}")
    print(f" Total embedded chunks: {len(cleaned_chunks)}")

# Run
if __name__ == "__main__":
    os.makedirs("data/faiss_index", exist_ok=True)
    build_faiss_index()
