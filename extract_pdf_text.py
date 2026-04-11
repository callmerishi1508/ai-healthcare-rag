def extract_text_from_html(content):
    soup = BeautifulSoup(content, 'lxml')

    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'noscript']):
        tag.decompose()

    # Try strong selectors first
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