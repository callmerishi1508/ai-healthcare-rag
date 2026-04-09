import requests
urls=['https://www.weforum.org/stories/2025/08/ai-transforming-global-health/']
for url in urls:
    u='https://r.jina.ai/http://'+url.replace('https://','')
    print('trying',u)
    try:
        r=requests.get(u,timeout=15)
        print(r.status_code, len(r.text))
        print(r.text[:200])
    except Exception as e:
        print('error',e)
