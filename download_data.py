from pathlib import Path
import subprocess,hashlib,json,zipfile,shutil
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
sources={
 'ames_housing.txt':'https://jse.amstat.org/v19n3/decock/AmesHousing.txt',
 'dry_bean.zip':'https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip',
 'annealing.zip':'https://archive.ics.uci.edu/static/public/3/annealing.zip',
 'ai_human_2026.zip':'https://www.kaggle.com/api/v1/datasets/download/alitaqishah/ai-vs-human-text-classification-dataset-2026',
}
for name in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']:
    sources[name]='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'+name
def download(item):
    name,url=item;p=DATA/name
    if not p.exists():subprocess.run([shutil.which('curl.exe') or shutil.which('curl') or 'curl','-f','-L','--retry','2','--max-time','180','-sS',url,'-o',str(p)],check=True)
    print('DOWNLOADED',name,p.stat().st_size,flush=True)
    return name,{'url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
manifest={}
with ThreadPoolExecutor(max_workers=4) as pool:
    for name,entry in pool.map(download,sources.items()):manifest[name]=entry
(DATA/'sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for p in DATA.glob('*.zip'):
    with zipfile.ZipFile(p) as z:
        print('ZIP',p.name,[(n.filename,n.file_size) for n in z.infolist()],flush=True)
        for info in z.infolist():
            if info.filename.endswith(('.xlsx','.csv','.data','.test','.names')):
                assert info.file_size<300_000_000
                dest=DATA/Path(info.filename).name
                dest.write_bytes(z.read(info))
