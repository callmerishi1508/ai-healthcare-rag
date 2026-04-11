import requests
import urllib.parse
import json
from bs4 import BeautifulSoup

def clean_html(html):
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return text


def fetch_with_fallback(url):
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'text/html'
    }

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200 and len(r.text) > 1000:
            return clean_html(r.text)
    except:
        pass

    # fallback → jina
    try:
        jina_url = 'https://r.jina.ai/http/' + urllib.parse.quote(url, safe=':/')
        r = requests.get(jina_url, timeout=15)
        if r.status_code == 200:
            return r.text
    except:
        pass

    return None


urls = [
    ("DOC-001", "AI Healthcare State", "https://menlovc.com/perspective/2025-the-state-of-ai-in-healthcare/"),
    ("DOC-002", "WEF AI Healthcare", "https://www.weforum.org/stories/2025/08/ai-transforming-global-health/"),
    ("DOC-003", "NCBI Clinical AI", "https://www.ncbi.nlm.nih.gov/books/NBK613808/"),
    ("DOC-004", "Scottsdale Survey", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12202002/"),
    ("DOC-005", "ScienceDirect Study", "https://www.sciencedirect.com/science/article/abs/pii/S0031699725075118"),
    ("DOC-006", "ACS Drug Discovery", "https://pubs.acs.org/doi/10.1021/acsomega.5c00549"),
]


data = []

for doc_id, title, url in urls:
    print(f"Fetching {doc_id}...")

    text = fetch_with_fallback(url)

    if text is None or len(text) < 1000:
        print(f"❌ Failed: {doc_id}")
        continue

    data.append({
        "doc_id": doc_id,
        "title": title,
        "url": url,
        "text": text
    })

    print(f"✅ Done: {doc_id} | Length: {len(text)}")


# SAVE FILE
with open("cleaned_kb.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("🔥 Saved cleaned_kb.json")