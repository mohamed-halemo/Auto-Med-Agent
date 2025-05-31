from transformers import pipeline  # For loading pretrained NLP pipelines
import os
import json  # For saving QA pairs to JSON
from glob import glob  # For reading all .txt files in a folder
import re  # For regex-based cleaning

# Load the question generation pipeline using a T5-based model from HuggingFace
# This model expects input in the form: "generate question: <hl> answer <hl> context..."
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def highlight_answer(text, answer):
    """
    Highlights the answer in the text using <hl> tags which the model understands.
    Replaces the first instance of the answer in the full text.
    """
    return text.replace(answer, f"<hl> {answer} <hl>")

def clean_answer(answer):
    """
    Cleans the answer string:
    - Removes common metadata like 'Title:' or 'Abstract:'
    - Strips unnecessary whitespace
    """
    cleaned = re.sub(r"\b(Title|Abstract):\s*", "", answer, flags=re.IGNORECASE)
    return cleaned.strip()

def generate_qas(text):
    """
    Generates question-answer pairs from the input text:
    - Splits text into sentences
    - Highlights each sentence to treat it as an answer
    - Feeds to the model to generate a question
    - Extracts clean answer from <hl> tags
    - Returns up to 5 QA pairs per document
    """
    inputs = []
    sentences = text.split(". ")  # Naive sentence split using periods

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # Skip very short sentences
            continue
        # Highlight the sentence in the full text
        highlighted = highlight_answer(text, sentence)
        inputs.append(highlighted)

    qas = []
    for inp in inputs[:5]:  # Limit to 5 questions per document for speed
        # Generate question from highlighted text
        output = qg_pipeline(f"generate question: {inp}", max_length=64, truncation=True)[0]['generated_text']
        
        try:
            # Extract the answer from the highlighted input
            raw_answer = inp.split("<hl>")[1].strip()
        except IndexError:
            continue  # If <hl> tags not found, skip this entry

        # Clean the answer string
        cleaned_answer = clean_answer(raw_answer)

        # Add QA pair to the list
        qas.append({
            "question": output,
            "answer": cleaned_answer
        })

    return qas

def run_on_folder(folder="data/pubmed_papers/", output="data/generated_qas.json"):
    """
    Processes all .txt files in a given folder:
    - Reads each text file
    - Generates QA pairs using generate_qas()
    - Saves all QA pairs to a single JSON file
    """
    all_qas = []

    # Iterate over all text files in the specified folder
    for path in glob(f"{folder}/*.txt"):
        with open(path, "r") as f:
            text = f.read()
        
        # Generate QA pairs from the file
        qas = generate_qas(text)
        all_qas.extend(qas)

    # Save all generated QA pairs to a JSON file
    with open(output, "w") as f:
        json.dump(all_qas, f, indent=2)

    print(f" Saved {len(all_qas)} QA pairs to {output}")

# Run the script when it's executed directly
if __name__ == "__main__":
    run_on_folder()
