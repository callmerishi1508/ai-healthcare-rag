import json
import faiss
import numpy as np
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    # Get project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct correct path
    kb_path = os.path.join(base_dir, "knowledge_base_ai_healthcare.json")
    
    print("Loading from:", kb_path)  # debug line
    
    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["documents"]

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    
    chunks, metadata = [], []
    
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for chunk in splits:
            chunks.append(chunk)
            metadata.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"]
            })
    
    return chunks, metadata

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

    with open("metadata.json", "w") as f:
        json.dump({"chunks": chunks, "meta": metadata}, f)

if __name__ == "__main__":
    build_index()