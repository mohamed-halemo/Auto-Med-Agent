from transformers import pipeline
import os
import json
from glob import glob
from transformers import AutoTokenizer

qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def highlight_answer(text, answer):
    return text.replace(answer, f"<hl> {answer} <hl>")

def generate_qas(text):
    inputs = []
    sentences = text.split(". ")
    for sentence in sentences:
        if len(sentence) < 20:
            continue
        inputs.append(highlight_answer(text, sentence.strip()))
    
    qas = []
    for inp in inputs[:5]:  # Limit for speed
        output = qg_pipeline(f"generate question: {inp}", max_length=64, truncation=True)[0]['generated_text']
        answer = inp.split("<hl>")[1].strip()
        qas.append({"question": output, "answer": answer})
    return qas

def run_on_folder(folder="data/pubmed_papers/", output="data/generated_qas.json"):
    all_qas = []
    for path in glob(f"{folder}/*.txt"):
        with open(path, "r") as f:
            text = f.read()
        qas = generate_qas(text)
        all_qas.extend(qas)
    
    with open(output, "w") as f:
        json.dump(all_qas, f, indent=2)
    print(f"✅ Saved {len(all_qas)} QA pairs to {output}")

if __name__ == "__main__":
    run_on_folder()
