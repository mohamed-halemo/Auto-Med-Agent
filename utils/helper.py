# helpers.py

from PyPDF2 import PdfReader

def save_uploaded_text(file, save_dir="data/pubmed_papers"):
    content = file.read().decode("utf-8")
    filepath = f"{save_dir}/{file.name}"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath

def save_uploaded_pdf(file, save_dir="data/pubmed_papers"):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    filepath = f"{save_dir}/{file.name}.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath

import os

def rebuild_index():
    result = os.system("python retriever/build_faiss.py")
    return result == 0
