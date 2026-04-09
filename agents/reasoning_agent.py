import re

class ReasoningAgent:
    def _split_sentences(self, text):
        return [sentence.strip() for sentence in re.split(r'(?<=[.?!])\s+', text) if sentence.strip()]

    def _extract_relevant_sentences(self, query, context):
        # Return the full text if it matches query terms
        query_terms = set(re.findall(r"\w+", query.lower()))
        text = context["text"]
        lowered = text.lower()
        if any(term in lowered for term in query_terms):
            return [text]
        return []

    def _synthesize_answer(self, query, relevant_sentences, doc_map):
        if not relevant_sentences:
            return "No relevant information found."
        
        text = relevant_sentences[0]
        doc = doc_map.get(text, "Unknown")
        
        # For percentage questions, extract key facts
        if "percentage" in query.lower() or "%" in query:
            nums = re.findall(r'\b\d+(?:\.\d+)?%', text)
            if nums:
                # Find sentences with numbers
                sentences = self._split_sentences(text)
                for sent in sentences:
                    if '%' in sent:
                        return f"{sent} [Citation: {doc}]"
                return f"{nums[0]}. [Citation: {doc}]"
        
        # Otherwise, return the first sentence or summary
        sentences = self._split_sentences(text)
        if sentences:
            return f"{sentences[0]} [Citation: {doc}]"
        return f"{text[:200]}... [Citation: {doc}]"

    def generate_answer(self, query, context):
        if not context:
            return "No relevant information found in the knowledge base.", "No reasoning trace available."

        all_relevant = []
        doc_map = {}
        for ctx in context:
            doc_id = ctx["meta"]["doc_id"]
            relevant = self._extract_relevant_sentences(query, ctx)
            if relevant:
                all_relevant.extend(relevant)
                for sent in relevant:
                    if sent not in doc_map:
                        doc_map[sent] = doc_id

        if not all_relevant:
            return "No relevant information found in the knowledge base.", "No reasoning trace available."

        # Synthesize concise answer
        answer = self._synthesize_answer(query, all_relevant, doc_map)
        
        # Trace
        steps = []
        for i, (sent, doc) in enumerate(doc_map.items(), 1):
            steps.append(f"Step {i}: Reviewed evidence from {doc}.")
        
        trace = "\n".join(steps)
        return answer, trace
