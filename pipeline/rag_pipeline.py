from agents.retriever_agent import RetrieverAgent
from agents.reasoning_agent import ReasoningAgent
from agents.critic_agent import CriticAgent

class RAGPipeline:
    def __init__(self):
        self.retriever = RetrieverAgent()
        self.reasoner = ReasoningAgent()
        self.critic = CriticAgent()

    def run(self, query, use_history=True):
        # Step 1: Retrieve
        context, sub_queries = self.retriever.retrieve(query)

        # 🔥 REMOVE DUPLICATES
        unique = {}
        for c in context:
            doc_id = c["meta"]["doc_id"]
            if doc_id not in unique:
                unique[doc_id] = c
        context = list(unique.values())

        # Step 2: Reason
        answer, reasoning = self.reasoner.generate(query, context)

        # Step 3: Critic
        review = self.critic.review(answer, context)

        return {
            "answer": answer,
            "trace": reasoning,
            "context": context,
            "sub_queries": sub_queries,
            "review": review
        }