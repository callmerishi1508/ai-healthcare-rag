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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================== CONFIG ==================
CORPUS_PDF = "healthcare_ai_corpus_v2.pdf"
OUTPUT_JSON = "knowledge_base_ai_healthcare.json"
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"

# ================== SESSION ==================
def create_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

SESSION = create_session()

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'text/html'
}

# ================== PDF PARSER ==================
def extract_articles_from_pdf(pdf_path):
    articles = []
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'

    pattern = r'(\d{2}) (\w+)\n(.*?)\n(.*?)\n(https?://[^\s]+)'
    matches = re.findall(pattern, text, re.MULTILINE)

    for match in matches:
        number = int(match[0])
        article_type = match[1]
        title = match[2].strip()
        source_date = match[3].strip()
        url = match[4].strip()

        source, date, q_tags = "", "", []

        if '·' in source_date:
            parts = source_date.split('·')
            source = parts[0].strip()
            for part in parts[1:]:
                part = part.strip()
                if re.match(r'\d{4}', part) or any(m in part for m in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Fall']):
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

# ================== HTML CLEAN ==================
def extract_text_from_html(content):
    soup = BeautifulSoup(content, 'lxml')

    for tag in soup(['script','style','header','footer','nav','aside','form','noscript']):
        tag.decompose()

    selectors = ['article', 'main', '.content', '.post', '.article-body']

    for sel in selectors:
        section = soup.select_one(sel)
        if section:
            text = section.get_text(separator=' ')
            break
    else:
        text = soup.get_text(separator=' ')

    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ================== JINA FALLBACK ==================
def fetch_with_jina(url):
    try:
        jina_url = 'https://r.jina.ai/http://' + url.replace('https://', '')
        response = SESSION.get(jina_url, timeout=20)
        response.raise_for_status()

        text = response.text
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)

        return text
    except:
        return None

# ================== FETCH ==================
def fetch_article(url):
    try:
        response = SESSION.get(url, headers=HEADERS, timeout=15)

        if response.status_code >= 400:
            return fetch_with_jina(url)

        content_type = response.headers.get('content-type','')

        if 'pdf' in content_type:
            import io
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = ''
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t
                return text

        return extract_text_from_html(response.content)

    except:
        return fetch_with_jina(url)

# ================== CLEAN ==================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'cookie|privacy policy|terms of use', '', text, flags=re.I)
    return text.strip()

# ================== CHUNK ==================
def chunk_text(text, chunk_size=800, overlap=150):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk_words = words[start:start+chunk_size]
        chunk = " ".join(chunk_words)

        if len(chunk) > 200:
            chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks

# ================== MAIN ==================
def main():
    articles = extract_articles_from_pdf(CORPUS_PDF)
    print(f"Found {len(articles)} articles")

    documents = []
    seen_chunks = set()

    for art in articles:
        print(f"\nFetching {art['number']}...")

        text = fetch_article(art['url'])

        if not text or len(text) < 2000:
            print("❌ Skipped (low content)")
            continue

        text = clean_text(text)

        # Debug save
        with open(f"raw_DOC_{art['number']:03d}.txt", "w", encoding="utf-8") as f:
            f.write(text[:5000])

        if "AI" not in text and "healthcare" not in text:
            print("⚠️ Irrelevant skipped")
            continue

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            if chunk in seen_chunks:
                continue
            seen_chunks.add(chunk)

            documents.append({
                "doc_id": f"DOC-{art['number']:03d}",
                "title": art['title'],
                "source": art['source'],
                "url": art['url'],
                "date": art['date'],
                "type": art['type'],
                "q_tags": art['q_tags'],
                "text": chunk,
                "chunk_id": f"DOC-{art['number']:03d}-{i+1}"
            })

    print(f"\nTotal chunks: {len(documents)}")

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"documents": documents}, f, indent=2)

    # ================== EMBEDDINGS ==================
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts)

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    # Metadata
    metadata = [{
        "doc_id": d["doc_id"],
        "title": d["title"],
        "url": d["url"],
        "chunk_id": d["chunk_id"]
    } for d in documents]

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ INDEX BUILT SUCCESSFULLY")

# ================== RUN ==================
if __name__ == "__main__":
    main()