import faiss
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class RetrieverAgent:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.index = faiss.read_index("index.faiss")

        with open("metadata.json") as f:
            self.metadata = json.load(f)

        with open("knowledge_base_ai_healthcare.json") as f:
            data = json.load(f)
            self.documents = data["documents"]

        # BM25 setup
        corpus = [doc["text"].split() for doc in self.documents]
        self.bm25 = BM25Okapi(corpus)

    def decompose(self, query):
        return [
            query,
            f"healthcare AI statistics {query}",
            f"comparison data {query}"
        ]

    def retrieve(self, query, top_k=8):
        sub_queries = self.decompose(query)

        all_results = []

        for q in sub_queries:
            emb = self.model.encode([q])
            faiss.normalize_L2(emb)

            scores, indices = self.index.search(emb, top_k)

            # BM25
            tokenized = q.split()
            bm25_scores = self.bm25.get_scores(tokenized)

            for i, idx in enumerate(indices[0]):
                dense_score = scores[0][i]
                sparse_score = bm25_scores[idx]

                final_score = 0.7 * dense_score + 0.3 * sparse_score

                doc = self.documents[idx]

                all_results.append({
                    "text": doc["text"],
                    "meta": doc,
                    "score": final_score
                })

        # sort
        all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

        return all_results[:8], sub_queries