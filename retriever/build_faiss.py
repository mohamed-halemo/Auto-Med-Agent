import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_faiss_index(data_path="data/pubmed_papers", index_path="data/faiss_index"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = []
    
    for fname in os.listdir(data_path):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(data_path, fname))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]

    embeddings = model.encode(texts, convert_to_tensor=False)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open(os.path.join(index_path, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))

    print(f"FAISS index and texts saved to {index_path}")

if __name__ == "__main__":
    os.makedirs("data/faiss_index", exist_ok=True)
    build_faiss_index()
