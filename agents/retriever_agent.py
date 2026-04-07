import faiss
import json
import numpy as np
from openai import OpenAI

client = OpenAI()

class RetrieverAgent:
    def __init__(self):
        self.index = faiss.read_index("index.faiss")
        with open("metadata.json") as f:
            data = json.load(f)
            self.chunks = data["chunks"]
            self.meta = data["meta"]

    def embed_query(self, query):
        emb = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        return np.array(emb.data[0].embedding)

    def retrieve(self, query, k=5):
        q_emb = self.embed_query(query).reshape(1, -1)
        D, I = self.index.search(q_emb, k)
        
        results = []
        for idx in I[0]:
            results.append({
                "text": self.chunks[idx],
                "meta": self.meta[idx]
            })
        
        return results

    def decompose_query(self, query):
        prompt = f"""
        Break this into sub-queries for multi-step reasoning:
        {query}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.split("\n")