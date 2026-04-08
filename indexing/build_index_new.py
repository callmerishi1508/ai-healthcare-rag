import pdfplumber
import re
import requests
from bs4 import BeautifulSoup
import tiktoken
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from pathlib import Path

CORPUS_PDF = "healthcare_ai_corpus_v2.pdf"
OUTPUT_JSON = "knowledge_base_ai_healthcare.json"
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"

def extract_articles_from_pdf(pdf_path):
    articles = []
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    
    # Find all article blocks
    # Pattern: number TYPE\nTitle\nSource · Date · Qtags\nURL
    pattern = r'(\d{2}) (\w+)\n(.*?)\n(.*?)\n(https?://[^\s]+)'
    matches = re.findall(pattern, text, re.MULTILINE)
    
    for match in matches:
        number = int(match[0])
        article_type = match[1]
        title = match[2].strip()
        source_date = match[3].strip()
        url = match[4].strip()
        
        # Parse source_date
        source = ""
        date = ""
        q_tags = []
        if '·' in source_date:
            parts = source_date.split('·')
            source = parts[0].strip()
            for part in parts[1:]:
                part = part.strip()
                if re.match(r'\d{4}', part) or any(month in part for month in ['Fall', 'Dec', 'Oct', 'Aug', 'Nov', 'Jul', 'Apr', 'Mar', 'Jan']):
                    date = part
                elif part.startswith('Q'):
                    q_tags.extend(re.findall(r'Q\d+', part))
        
        articles.append({
            'number': number,
            'type': article_type,
            'title': title,
            'source': source,
            'date': date,
            'q_tags': q_tags,
            'url': url
        })
    
    return articles

def fetch_article(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' in content_type:
            # Handle PDF
            import io
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            return text
        else:
            # Handle HTML
            soup = BeautifulSoup(response.content, 'lxml')
            # Remove scripts, styles, etc.
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def clean_text(text):
    # Remove excessive whitespace, headers, footers, etc.
    # This is basic; can be improved
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks

def main():
    articles = extract_articles_from_pdf(CORPUS_PDF)
    print(f"Found {len(articles)} articles")
    for art in articles:
        print(f"{art['number']}: {art['title']} - {art['url']}")
    
    documents = []
    for art in articles:
        print(f"Fetching article {art['number']}: {art['title']}")
        text = fetch_article(art['url'])
        if text:
            text = clean_text(text)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                doc = {
                    "doc_id": f"DOC-{art['number']:03d}",
                    "title": art['title'],
                    "source": art['source'],
                    "url": art['url'],
                    "date": art['date'],
                    "type": art['type'],
                    "q_tags": art['q_tags'],
                    "text": chunk,
                    "chunk_id": f"DOC-{art['number']:03d}-{i+1}"
                }
                documents.append(doc)
        else:
            print(f"Failed to fetch {art['url']}")
    
    # Save to JSON
    data = {
        "metadata": {
            "dataset_name": "AI in Healthcare Knowledge Base",
            "version": "2.0",
            "created": "2026-04-09",
            "domain": "AI in Healthcare",
            "total_documents": len(documents),
            "source_types": {"fetched": len(documents)}
        },
        "documents": documents
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    if documents:
        # Now, build index as before
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [doc['text'] for doc in documents]
        embeddings = model.encode(texts)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        faiss.write_index(index, INDEX_FILE)

        # Metadata
        metadata = []
        for doc in documents:
            metadata.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source": doc["source"],
                "url": doc["url"],
                "date": doc["date"],
                "chunk_id": doc["chunk_id"]
            })
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print("Indexing complete")
    else:
        print("No documents to index")

if __name__ == "__main__":
    main()