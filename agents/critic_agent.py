import os
from openai import OpenAI


class CriticAgent:

    def validate(self, answer, context):
        context_text = "\n".join([c["text"] for c in context])

        prompt = f"""
Check the answer against context.

Answer:
{answer}

Context:
{context_text}

Return:
- Unsupported claims
- Missing citations
- Final verdict (PASS/FAIL)
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content