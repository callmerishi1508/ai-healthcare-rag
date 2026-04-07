from openai import OpenAI

client = OpenAI()

class CriticAgent:
    def validate(self, answer, context):
        prompt = f"""
        Check if this answer is fully supported by context.

        Answer:
        {answer}

        Context:
        {context}

        Output:
        - Unsupported claims
        - Missing citations
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content