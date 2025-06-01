from tools.toolkit import calculator, pubmed_search
from agents.literature_agent import LiteratureAgent

class ToolUsingAgent:
    def __init__(self, memory=None):
        self.memory = memory or []
        self.rag_agent = LiteratureAgent()

    def run(self, query: str) -> tuple[str, str]:
        # Tool: calculator
        if "calculate" in query.lower():
            expr = query.lower().replace("calculate", "").strip()
            return calculator(expr), ""

        # Tool: PubMed search
        elif "pubmed" in query.lower():
            topic = query.lower().replace("pubmed", "").strip()
            return pubmed_search(topic), ""

        else:
            # Use up to last 3 QA pairs from memory
            last_memory = self.memory[-3:]
            full_context = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in last_memory])
            
            # Combine with current query
            query_with_context = f"{full_context}\nQ: {query}" if full_context else query
            
            # Pass contextualized query to the RAG agent
            return self.rag_agent.run(query_with_context)
