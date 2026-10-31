"""Download and verify repository blobs without opening expression or labels."""
from common import *
import urllib.request
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'cross-dataset-research'}),timeout=120) as r:return r.read()
def main():
    repo='xuebaliang/scziDesk'
    commit=json.loads(get(f'https://api.github.com/repos/{repo}/commits/master'))['sha']
    tree=json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'))
    files={f['path']:f for f in tree['tree']};manifest={'repository':repo,'commit':commit,'labels_read':False,'expression_read':False,'files':{}}
    for name in CFG['heldout_names']:
        rel=f'dataset/{name}/data.h5';item=files[rel];dest=EXP/'heldout_sealed'/f'{name}.h5';dest.parent.mkdir(exist_ok=True)
        url=f'https://raw.githubusercontent.com/{repo}/{commit}/{rel}'
        data=dest.read_bytes() if dest.exists() else get(url)
        assert len(data)==item['size']
        git=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest();assert git==item['sha']
        if not dest.exists():dest.write_bytes(data)
        manifest['files'][name]={'url':url,'git_blob_sha1':git,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'status':'SEALED_VERIFIED_BYTES_ONLY'}
        save(EXP/'heldout_manifest.json',manifest);print(name,len(data),flush=True)
if __name__=='__main__':main()
