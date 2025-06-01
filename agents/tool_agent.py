from tools.toolkit import clinical_trial_search, pubmed_search
from agents.literature_agent import LiteratureAgent

class ToolUsingAgent:
    def __init__(self, memory=None):
        self.memory = memory or []
        self.rag_agent = LiteratureAgent()

    def run(self, query: str) -> tuple[str, str]:
        # Tool 1: Clinical trial search
        if "clinical trial" in query.lower():
            condition = query.lower().replace("clinical trial", "").strip()
            return clinical_trial_search(condition), ""

        # Tool 2: PubMed search
        elif "pubmed" in query.lower():
            topic = query.lower().replace("pubmed", "").strip()
            return pubmed_search(topic), ""

        else:
            # RAG fallback
            last_memory = self.memory[-3:]
            full_context = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in last_memory])
            query_with_context = f"{full_context}\nQ: {query}" if full_context else query
            return self.rag_agent.run(query)
