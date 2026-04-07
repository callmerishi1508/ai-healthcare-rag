class RAGPipeline:
    def run(self, query):
        try:
            sub_queries = self.retriever.decompose_query(query)

            all_context = []
            for sq in sub_queries:
                retrieved = self.retriever.retrieve(sq, k=3)
                all_context.extend(retrieved)

            # Remove duplicates
            seen = set()
            unique_context = []
            for c in all_context:
                key = (c["meta"]["doc_id"], c["text"])
                if key not in seen:
                    seen.add(key)
                    unique_context.append(c)

            answer = self.reasoner.generate_answer(query, unique_context)
            critique = self.critic.validate(answer, unique_context)

            return {
                "sub_queries": sub_queries,
                "context": unique_context,
                "answer": answer,
                "critique": critique
            }
        except Exception as e:
            # Handle OpenAI rate limits or other errors
            error_msg = f"Error processing query: {str(e)}"
            return {
                "sub_queries": ["Query processing failed"],
                "context": [],
                "answer": error_msg,
                "critique": "Unable to validate due to API error",
            }