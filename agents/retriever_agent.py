import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI


class RetrieverAgent:
    def __init__(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(os.path.join(root_dir, 'index.faiss'))

        with open(os.path.join(root_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.chunks = data['chunks']
        self.metadata = data['meta']
        self.client = None

    def _openai(self):
        if self.client is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise RuntimeError('OPENAI_API_KEY must be set to decompose queries.')
            self.client = OpenAI(api_key=api_key)
        return self.client

    def decompose_query(self, query):
        prompt = f"""
Break this question into 2-4 focused sub-queries for retrieval.

Rules:
- Cover all aspects of the question
- Keep each sub-query short
- Avoid redundancy

Question:
{query}

Output:
- sub query 1
- sub query 2
"""
        response = self._openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return [q.strip("- " ).strip() for q in response.choices[0].message.content.split("\n") if q.strip()]

    def retrieve(self, query, k=3):
        query_vec = np.array(self.model.encode([query]), dtype='float32')
        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'text': self.chunks[idx],
                'meta': self.metadata[idx],
                'score': float(dist),
            })

        return results