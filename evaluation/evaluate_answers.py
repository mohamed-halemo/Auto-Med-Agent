import json
import evaluate  # Hugging Face's `evaluate` library for metrics like ROUGE and BLEU

# Load generated QA pairs from file (used as evaluation reference)
qa_data = json.load(open("data/generated_qas.json"))

# Load evaluation metrics
rouge = evaluate.load("rouge")  # ROUGE measures word/phrase overlap (e.g., ROUGE-1, ROUGE-L)
bleu = evaluate.load("bleu")    # BLEU measures n-gram overlap (e.g., BLEU-1 to BLEU-4)

def evaluate_model(agent):
    """
    Evaluates a QA agent using ROUGE and BLEU.
    Compares the agent's generated answers to the reference answers from `qa_data`.
    """
    generated_answers = []  # Stores answers predicted by the agent
    references = []         # Stores ground truth (reference) answers

    # Limit evaluation to the first 10 QA pairs for speed
    for qa in qa_data[:10]:
        question = qa["question"]
        reference = qa["answer"]

        # Run the agent on the question to get its answer
        answer, _ = agent.run(question)  # Second return value is summary, which is ignored

        generated_answers.append(answer)
        references.append(reference)

    # Compute ROUGE score: overlap of words/sequences between prediction and reference
    rouge_score = rouge.compute(predictions=generated_answers, references=references)

    # Compute BLEU score: needs references to be wrapped in a list of lists
    bleu_score = bleu.compute(predictions=generated_answers, references=[[ref] for ref in references])

    # Print results
    print("📊 Evaluation Results:")
    print("ROUGE:", rouge_score)
    print("BLEU:", bleu_score)

# Entry point: evaluate LiteratureAgent if script is run directly
if __name__ == "__main__":
    from agents.literature_agent import LiteratureAgent
    evaluate_model(LiteratureAgent())
