class CriticAgent:
    def review(self, answer, context):
        if "DOC-" not in answer:
            return "⚠️ Missing citations"
        return "✅ PASS: citations present"