import re

class ReasoningAgent:
    def _split_sentences(self, text):
        return [sentence.strip() for sentence in re.split(r'(?<=[.?!])\s+', text) if sentence.strip()]

    def _extract_relevant_sentences(self, query, context):
        query_terms = set(re.findall(r"\w+", query.lower()))
        sentences = self._split_sentences(context["text"])
        relevant = []

        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in query_terms):
                relevant.append(sentence)

        if not relevant and sentences:
            return [sentences[0]]
        return relevant[:3]

    def generate_answer(self, query, context):
        if not context:
            return "No relevant information found in the knowledge base.", "No reasoning trace available."

        steps = []
        claims = []
        seen_docs = set()

        for idx, ctx in enumerate(context, start=1):
            doc_id = ctx["meta"]["doc_id"]
            if doc_id in seen_docs:
                continue

            title = ctx["meta"].get("title", "Unknown title")
            relevant_sentences = self._extract_relevant_sentences(query, ctx)

            if not relevant_sentences:
                continue

            sentence = relevant_sentences[0]
            steps.append(f"Step {len(seen_docs)+1}: Reviewed evidence from {doc_id} ({title}).")
            claims.append(f"{sentence} [Citation: {doc_id}]")
            seen_docs.add(doc_id)

            if len(seen_docs) >= 5:
                break

        if not claims:
            return "No relevant information found in the knowledge base.", "No reasoning trace available."

        answer = " ".join(claims)
        trace = "\n".join(steps)
        return answer, trace
