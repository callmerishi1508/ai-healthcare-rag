import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RetrieverAgent:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        index_path = root / "index.faiss"
        metadata_path = root / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            from indexing.build_index import build_index
            build_index()

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["meta"]
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def decompose_query(self, query):
        return [query]

    def retrieve(self, query, k=3):
        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({
                "text": self.chunks[idx],
                "meta": self.metadata[idx],
                "score": float(score)
            })
        return results