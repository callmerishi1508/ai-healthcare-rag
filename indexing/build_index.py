import json
import os
import re
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_path = os.path.join(base_dir, "knowledge_base_ai_healthcare.json")

    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data["documents"]


def tokenize_text(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def detokenize(tokens):
    text = " ".join(tokens)
    text = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
    text = text.replace(" ;", ";").replace(" :", ":")
    return text


def chunk_docs(docs):
    chunk_size = 600
    chunk_overlap = 100

    chunks, metadata = [], []

    for doc in docs:
        tokens = tokenize_text(doc["text"])
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = detokenize(chunk_tokens)
            chunk_index += 1

            chunks.append(chunk_text)
            metadata.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source_type": doc.get("source_type"),
                "source": doc.get("source"),
                "url": doc.get("url"),
                "date": doc.get("date"),
                "chunk_id": f"{doc['doc_id']}-{chunk_index}",
                "token_count": len(chunk_tokens),
            })

            if end == len(tokens):
                break
            start = end - chunk_overlap

    return chunks, metadata


def tokenize(text):
    tokens = re.findall(r"\w+", text.lower())
    return [token for token in tokens if len(token) > 1]


def embed_chunks(chunks):
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


def build_index():
    docs = load_data()
    chunks, metadata = chunk_docs(docs)
    embeddings = embed_chunks(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    faiss.write_index(index, "index.faiss")

    bundle = {
        "chunks": chunks,
        "meta": metadata,
        "tokenized_chunks": [tokenize(chunk) for chunk in chunks],
    }

    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f)


if __name__ == "__main__":
    build_index()
