import re

class CriticAgent:
    def validate(self, answer, context):
        cited_docs = set(re.findall(r"DOC-\d+", answer))
        expected_docs = {ctx["meta"]["doc_id"] for ctx in context}
        missing = expected_docs - cited_docs

        if not answer:
            return "FAIL: No answer generated."

        if missing:
            missing_docs = ", ".join(sorted(missing))
            return f"WARNING: Missing citations for retrieved sources: {missing_docs}."

        if cited_docs:
            return "PASS: Answer contains citations for all retrieved sources."

        return "FAIL: Answer does not contain any document citations."
