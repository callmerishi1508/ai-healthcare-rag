from agents.retriever_agent import RetrieverAgent
from agents.reasoning_agent import ReasoningAgent
from agents.critic_agent import CriticAgent

class RAGPipeline:
    def __init__(self):
        self.retriever = RetrieverAgent()
        self.reasoner = ReasoningAgent()
        self.critic = CriticAgent()
        self.history = []

    def _build_context_query(self, query):
        if not self.history:
            return query
        recent = " ".join(item["query"] for item in self.history[-3:])
        return f"{query} (follow-up on: {recent})"

    def run(self, query, use_history=True, key_facts=None):
        try:
            expanded_query = self._build_context_query(query) if use_history else query
            complex_query = self.retriever.is_complex(expanded_query)
            sub_queries = self.retriever.decompose_query(expanded_query) if complex_query else [expanded_query]

            all_context = []
            for sq in sub_queries:
                retrieved = self.retriever.retrieve(sq, k=5, key_facts=key_facts)
                all_context.extend(retrieved)

            seen = set()
            unique_context = []
            for c in all_context:
                key = (c["meta"]["doc_id"], c["text"])
                if key not in seen:
                    seen.add(key)
                    unique_context.append(c)

            answer, trace = self.reasoner.generate_answer(expanded_query, unique_context)
            critique = self.critic.validate(answer, unique_context)

            record = {
                "query": query,
                "expanded_query": expanded_query,
                "sub_queries": sub_queries,
                "context": unique_context,
                "answer": answer,
                "trace": trace,
                "critique": critique,
            }
            self.history.append(record)

            return record
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            return {
                "query": query,
                "expanded_query": query,
                "sub_queries": ["Query processing failed"],
                "context": [],
                "answer": error_msg,
                "trace": "",
                "critique": "Unable to validate due to pipeline error",
            }
