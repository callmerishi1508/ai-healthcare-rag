def fetch_with_jina(url):
    jina_url = 'https://r.jina.ai/http://' + url.replace('https://', '')
    try:
        response = SESSION.get(jina_url, timeout=20)
        response.raise_for_status()

        text = response.text

        # 🔥 REMOVE markdown artifacts
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)

        return text
    except Exception as e:
        print(f'Jina fallback failed for {url}: {e}')
        return None