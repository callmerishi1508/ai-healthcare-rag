class CriticAgent:
    def validate(self, answer, context):
        if context and "information" in answer.lower():
            return "PASS: Answer appears to be based on retrieved context."
        else:
            return "FAIL: Answer may not be sufficiently supported by context."