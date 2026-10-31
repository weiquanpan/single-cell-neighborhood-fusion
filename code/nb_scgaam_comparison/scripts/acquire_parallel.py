"""Bounded range downloads preserving the prior prefix and every retry event."""
from common import *
import concurrent.futures, threading, time, urllib.request, shutil, traceback
statepath=R/'logs/download_state.json';state=json.loads(statepath.read_text());state.update(pid=os.getpid(),status='DOWNLOADING',mode='four_range_workers',resumed_utc=utc())
save(statepath,state);mutex=threading.Lock()
entries={e['key']:e for e in json.loads((R/'provenance/zenodo_15272789.json').read_text())['files']}
def event(**row):
    with mutex:
        with (R/'logs/download_events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(time=utc(),**row))+'\n')
def download_chunk(name,start,end,path):
    for attempt in range(1,5):
        offset=path.stat().st_size if path.exists() else 0
        if offset==end-start+1:return
        try:
            lo=start+offset;url=f'https://zenodo.org/records/15272789/files/{name}?download=1'
            req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{end}','User-Agent':'NB-scGAAM-research/1.0'})
            with urllib.request.urlopen(req,timeout=90) as response:
                assert response.status==206
                assert response.headers['Content-Range']==f'bytes {lo}-{end}/{entries[name]["size"]}',response.headers['Content-Range']
                with path.open('ab') as f:
                    while True:
                        data=response.read(1024**2)
                        if not data:break
                        f.write(data)
            assert path.stat().st_size==end-start+1
            event(event='chunk_complete',file=name,start=start,end=end,attempt=attempt);return
        except Exception as exc:
            event(event='download_attempt_failed',file=name,start=start,attempt=attempt,error=repr(exc))
            if attempt==4:raise
            time.sleep(5*attempt)
try:
 for name in ['Tosches.h5','Bach.h5']:
    if state['files'].get(name,{}).get('status')=='VERIFIED':continue
    e=entries[name];prefix=R/'data/raw'/(name+'.partial');prefix_size=prefix.stat().st_size if prefix.exists() else 0
    chunksdir=R/'downloads/chunks'/name;chunksdir.mkdir(parents=True,exist_ok=True)
    tasks=[];chunksize=64*1024**2
    for lo in range(prefix_size,e['size'],chunksize):
        hi=min(e['size']-1,lo+chunksize-1);tasks.append((lo,hi,chunksdir/f'{lo}_{hi}.part'))
    state.update(current=name,expected_bytes=e['size'],prefix_bytes=prefix_size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(download_chunk,name,lo,hi,path) for lo,hi,path in tasks]
        while not all(f.done() for f in futures):
            state.update(downloaded_bytes=prefix_size+sum(p.stat().st_size for _,_,p in tasks if p.exists()),updated_utc=utc());save(statepath,state);time.sleep(10)
        for f in futures:f.result()
    assembled=R/'data/raw'/(name+'.assembled');md=hashlib.md5();h=hashlib.sha256()
    with assembled.open('wb') as dest:
        parts=([prefix] if prefix_size else [])+[p for _,_,p in tasks]
        for part in parts:
            with part.open('rb') as src:
                for block in iter(lambda:src.read(8*1024**2),b''):dest.write(block);md.update(block);h.update(block)
    assert assembled.stat().st_size==e['size'] and 'md5:'+md.hexdigest()==e['checksum']
    os.replace(assembled,R/'data/raw'/name)
    state['files'][name]=dict(status='VERIFIED',size=e['size'],md5=md.hexdigest(),sha256=h.hexdigest(),url=f'https://zenodo.org/records/15272789/files/{name}?download=1',verified_utc=utc());save(statepath,state)
    print(name,'VERIFIED',flush=True)
 state.update(status='COMPLETED',end_utc=utc());save(statepath,state);save(R/'provenance/data_download_manifest.json',state)
except BaseException as error:
 state.update(status='FAILED',error=repr(error),traceback=traceback.format_exc(),end_utc=utc());save(statepath,state);raise
