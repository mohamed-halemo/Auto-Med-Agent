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

    Args:
        data_path (str): Folder containing .txt files with documents.
        index_path (str): Output folder where index and texts will be saved.
    """
    
    # Load a pre-trained SentenceTransformer for embedding text chunks
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    documents = []

    # Loop through each file in the data folder
    for fname in os.listdir(data_path):
        if fname.endswith(".txt"):
            # Load the .txt file using LangChain's TextLoader
            loader = TextLoader(os.path.join(data_path, fname))
            
            # Load returns a list of documents (usually one per file)
            documents.extend(loader.load())

    # Use RecursiveCharacterTextSplitter to break each document into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Each chunk will be up to 1000 characters
        chunk_overlap=100       # Overlap 100 characters between chunks for context continuity
    )

    # Split all loaded documents into chunks
    docs = splitter.split_documents(documents)

    # Extract plain text content from each document chunk
    texts = [doc.page_content for doc in docs]

    # Convert text chunks into dense vector embeddings
    embeddings = model.encode(texts, convert_to_tensor=False)

    # Get the embedding vector dimension (required for FAISS)
    dim = len(embeddings[0])

    # Create a FAISS index for L2 (Euclidean) distance
    index = faiss.IndexFlatL2(dim)

    # Add all embeddings to the FAISS index
    index.add(embeddings)

    # Save the text chunks (to retrieve the actual content during search)
    with open(os.path.join(index_path, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    # Save the FAISS index to disk
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))

    print(f"FAISS index and texts saved to {index_path}")

# Run the script when executed directly
if __name__ == "__main__":
    # Create the output folder if it doesn't exist
    os.makedirs("data/faiss_index", exist_ok=True)

    # Build the FAISS index from text documents
    build_faiss_index()
