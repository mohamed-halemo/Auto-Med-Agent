import json
import evaluate

qa_data = json.load(open("data/generated_qas.json"))
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def evaluate_model(agent):
    generated_answers = []
    references = []
    for qa in qa_data[:10]:  # Run a small eval batch
        question = qa["question"]
        reference = qa["answer"]
        answer, _ = agent.run(question)
        generated_answers.append(answer)
        references.append(reference)

    rouge_score = rouge.compute(predictions=generated_answers, references=references)
    bleu_score = bleu.compute(predictions=generated_answers, references=[[ref] for ref in references])
    
    print("📊 Evaluation Results:")
    print("ROUGE:", rouge_score)
    print("BLEU:", bleu_score)

if __name__ == "__main__":
    from agents.literature_agent import LiteratureAgent
    evaluate_model(LiteratureAgent())
