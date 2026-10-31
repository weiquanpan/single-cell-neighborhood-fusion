from pathlib import Path
import re,json,concurrent.futures,requests
OUT=Path(__file__).resolve().parent
old=(OUT/'manuscript_en/references.tex').read_text()
dois=re.findall(r'https://doi.org/([^}\s]+)',old)+['10.1126/science.aam8940','10.1126/science.aar4237','10.1038/s41467-017-02001-5','10.1038/nn.4462','10.1242/dev.151142','10.1007/BF01908075','10.1002/nav.3800020109','10.1186/s13059-019-1738-8']
def get(d):
    try:
        r=requests.get('https://api.crossref.org/works/'+d,timeout=30);r.raise_for_status();j=r.json()['message'];return {'doi':d,'status':'verified_metadata','title':j.get('title'),'authors':j.get('author'),'published':j.get('published'),'journal':j.get('container-title'),'volume':j.get('volume'),'page':j.get('page'),'article':j.get('article-number'),'url':j.get('URL')}
    except Exception as e:return {'doi':d,'status':'unresolved','error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:out=list(pool.map(get,sorted(set(dois))))
(OUT/'audit/citation_metadata.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for r in out:print(r['doi'],r['status'],r.get('title'),r.get('published'))
