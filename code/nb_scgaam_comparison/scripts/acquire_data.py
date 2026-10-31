"""Download only the four preregistered public files; verify archive checksums."""
import datetime, hashlib, json, os, time, urllib.request
from pathlib import Path
R=Path(__file__).resolve().parents[1]
FILES=['Quake_10x_Limb_Muscle.h5','worm_neuron_cell.h5','Tosches.h5','Bach.h5']
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf8');os.replace(tmp,p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def hashes(p):
    md=hashlib.md5();sha=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):md.update(b);sha.update(b)
    return md.hexdigest(),sha.hexdigest()
entries={e['key']:e for e in json.loads((R/'provenance/zenodo_15272789.json').read_text())['files']}
state={'status':'DOWNLOADING','pid':os.getpid(),'start_utc':stamp(),'files':{}}
statepath=R/'logs/download_state.json'
for name in FILES:
    e=entries[name];dest=R/'data/raw'/name;part=dest.with_suffix('.h5.partial');url=e['links']['self']
    state['current']=name;save(statepath,state)
    if not dest.exists():
        done=part.stat().st_size if part.exists() else 0
        req=urllib.request.Request(url,headers={'User-Agent':'NB-scGAAM-research/1.0',**({'Range':f'bytes={done}-'} if done else {})})
        with urllib.request.urlopen(req,timeout=90) as response:
            if done and response.status!=206:raise RuntimeError('Server did not honor range; retained partial')
            with part.open('ab' if done else 'xb') as f:
                last=0
                while True:
                    b=response.read(4*1024**2)
                    if not b:break
                    f.write(b);done+=len(b)
                    if time.monotonic()-last>10:
                        state.update(downloaded_bytes=done,expected_bytes=e['size'],updated_utc=stamp());save(statepath,state);last=time.monotonic()
        assert done==e['size'],(name,done,e['size'])
        md,sha=hashes(part);assert 'md5:'+md==e['checksum'],name
        os.replace(part,dest)
    else:
        md,sha=hashes(dest);assert dest.stat().st_size==e['size'] and 'md5:'+md==e['checksum']
    state['files'][name]={'status':'VERIFIED','size':dest.stat().st_size,'md5':md,'sha256':sha,'url':url,'verified_utc':stamp()}
    save(statepath,state);print(name,'VERIFIED',flush=True)
state.update(status='COMPLETED',end_utc=stamp());save(statepath,state);save(R/'provenance/data_download_manifest.json',state)
