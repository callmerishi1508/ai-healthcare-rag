class ReasoningAgent:
    def generate_answer(self, query, context):
        if context:
            return f"Based on the retrieved information: {context[0]['text'][:500]}..."
        else:
            return "No relevant information found in the knowledge base."