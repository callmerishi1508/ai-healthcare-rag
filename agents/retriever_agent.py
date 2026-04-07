import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
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
        self.tokenized_chunks = data.get("tokenized_chunks") or [self._tokenize(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _tokenize(self, text):
        tokens = re.findall(r"\w+", text.lower())
        return [token for token in tokens if len(token) > 1]

    def is_complex(self, query):
        lower = query.lower()
        complex_signals = ["compare", "difference", "vs", "versus", "and", "or", "evidence", "impact", "effect", "benefits", "risks", "how many", "what was", "what is", "why", "when"]
        return len(query.split()) > 20 or any(signal in lower for signal in complex_signals)

    def decompose_query(self, query):
        parts = re.split(r"\band\b|\bor\b|\bversus\b|\bcompare\b|,|;|\bplus\b", query, flags=re.I)
        parts = [part.strip() for part in parts if part.strip() and len(part.split()) > 3]
        if len(parts) > 1:
            subqueries = []
            for part in parts:
                if part.lower() not in [q.lower() for q in subqueries]:
                    subqueries.append(part)
            if query.strip() not in subqueries:
                subqueries.append(query.strip())
            return subqueries
        return [query.strip()]

    def retrieve(self, query, k=5):
        query_tokens = self._tokenize(query)
        dense_embedding = self.model.encode([query]).astype(np.float32)
        dense_scores, dense_indices = self.index.search(dense_embedding, k)

        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:k * 3]

        dense_map = {int(idx): float(score) for idx, score in zip(dense_indices[0], dense_scores[0]) if idx >= 0}
        bm25_map = {int(idx): float(bm25_scores[idx]) for idx in bm25_top}

        dense_max = max(dense_map.values()) if dense_map else 1.0
        bm25_max = max(bm25_map.values()) if bm25_map else 1.0

        combined = {}
        for idx, score in dense_map.items():
            combined[idx] = combined.get(idx, 0.0) + 0.65 * (score / dense_max)
        for idx, score in bm25_map.items():
            combined[idx] = combined.get(idx, 0.0) + 0.35 * (score / bm25_max)

        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:k]]

        results = []
        for idx in top_indices:
            if idx < 0 or idx >= len(self.chunks):
                continue
            score = combined.get(idx, 0.0)
            results.append({
                "text": self.chunks[idx],
                "meta": self.metadata[idx],
                "score": float(score)
            })

        return results
