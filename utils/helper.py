from PyPDF2 import PdfReader

# Function to save uploaded plain text files to a specific directory
def save_uploaded_text(file, save_dir="data/pubmed_papers"):
    # Read and decode the uploaded file content (assuming UTF-8 text format)
    content = file.read().decode("utf-8")
    # Define the file path where the file will be saved
    filepath = f"{save_dir}/{file.name}"
    # Write the content to the target file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    # Return the path of the saved file
    return filepath

# Function to extract and save text content from uploaded PDF files
def save_uploaded_pdf(file, save_dir="data/pubmed_papers"):
    # Create a PDF reader object from the uploaded file
    reader = PdfReader(file)
    text = ""
    # Extract text from each page and append to a string
    for page in reader.pages:
        text += page.extract_text() + "\n"
    # Define the output file path (save as .txt file)
    filepath = f"{save_dir}/{file.name}.txt"
    # Save the extracted text to a new file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    # Return the path of the saved text file
    return filepath

import os

# Function to rebuild the FAISS index by calling the build script
def rebuild_index():
    # Execute the index building script via system shell command
    result = os.system("python retriever/build_faiss.py")
    # Return True if the command succeeded (exit code 0), False otherwise
    return result == 0
