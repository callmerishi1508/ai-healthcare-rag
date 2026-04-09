import requests
import urllib.parse

urls=[
 'https://menlovc.com/perspective/2025-the-state-of-ai-in-healthcare/',
 'https://www.weforum.org/stories/2025/08/ai-transforming-global-health/',
 'https://www.ncbi.nlm.nih.gov/books/NBK613808/',
 'https://www.sciencedirect.com/science/article/abs/pii/S0031699725075118',
 'https://pubs.acs.org/doi/10.1021/acsomega.5c00549',
 'https://www.thehastingscenter.org/briefingbook/ai-in-healthcare/'
]
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
for url in urls:
    print('URL', url)
    try:
        r = requests.get(url, headers=headers, timeout=15)
        print('status', r.status_code, 'content-type', r.headers.get('content-type'))
        print('len', len(r.text))
    except Exception as e:
        print('ERROR', e)
    try:
        jina = 'https://r.jina.ai/http/' + urllib.parse.quote(url, safe=':/')
        r = requests.get(jina, timeout=15)
        print('jina status', r.status_code, 'len', len(r.text))
    except Exception as e:
        print('jina error', e)
    print('---')
