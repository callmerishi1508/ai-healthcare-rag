import json
import os
import re
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_path = os.path.join(base_dir, "knowledge_base_ai_healthcare.json")

    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data["documents"]


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks, metadata = [], []

    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for chunk_index, chunk in enumerate(splits):
            chunks.append(chunk)
            metadata.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source_type": doc.get("source_type"),
                "source": doc.get("source"),
                "url": doc.get("url"),
                "date": doc.get("date"),
                "chunk_id": f"{doc['doc_id']}-{chunk_index + 1}",
            })

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
