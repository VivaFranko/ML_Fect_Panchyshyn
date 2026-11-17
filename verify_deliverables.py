from pathlib import Path
import sys,json,hashlib,zipfile,base64,subprocess
import numpy as np
import pandas as pd
import nbformat
import pymupdf as fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'project'))
from course_text import FIGURES

manifest=json.loads((ROOT/'data/sources.json').read_text(encoding='utf-8'))
for name,v in manifest.items():assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==v['sha256']
profile=json.loads((ROOT/'profile.json').read_text(encoding='utf-8'))
assert profile['confirmed'] and profile['number']==20
checks={'source_hashes':len(manifest),'completed_labs':{},'student':profile['student'],'group':profile['group'],'individual_number':20,'individual_seed':62}
workspace=next(p for p in ROOT.parents if (p/'_system/studykit').is_dir())
root_python=workspace/'.venv/Scripts/python.exe'
for n in [1,2,3,4,5]:
    out=ROOT/f'lab_{n:02d}';r=json.loads((out/'results/results.json').read_text(encoding='utf-8'))
    nb=nbformat.read(next(out.glob('Практична*.ipynb')),4);code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert all(o.output_type!='error' for c in code for o in c.outputs)
    image_hashes={hashlib.sha256(base64.b64decode(o.data['image/png'])).hexdigest() for c in code for o in c.outputs if o.output_type=='display_data' and 'image/png' in o.get('data',{})}
    doc=out/f'Звіт_ПР_{n}_Панчишин.docx';pdf=doc.with_suffix('.pdf')
    from docx import Document
    word=Document(doc);paragraph_text='\n'.join(p.text for p in word.paragraphs)
    assert profile['student'] in paragraph_text and profile['group'] in paragraph_text and profile['teacher'] in paragraph_text
    assert 'ЧЕРНЕТКА' not in paragraph_text
    if n==4:assert r['number']==20 and r['seed']==62 and all(len(v['folds'])==5 for v in r['nested'].values())
    for args in [['-m','studykit','check',str(doc),'--pdf',str(pdf)],['-m','studykit.documents.figure_refs',str(doc)]]:
        subprocess.run([str(root_python)]+args,check=True)
    with zipfile.ZipFile(doc) as z:word_images={hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist() if name.startswith('word/media/')}
    for name,_,_ in FIGURES[n]:
        sha=hashlib.sha256((out/'results'/f'{name}.png').read_bytes()).hexdigest()
        assert sha in word_images,(n,name,'DOCX')
        assert sha in image_hashes,(n,name,'Notebook')
    data=np.load(out/'results/split_indices.npz')
    for i,a in enumerate(data.files):
        if a.startswith('group_'):continue
        assert len(np.unique(data[a]))==len(data[a])
        for b in data.files[i+1:]:
            if b.startswith('group_'):continue
            assert len(np.intersect1d(data[a],data[b]))==0,(n,a,b)
    document=fitz.open(pdf);lengths=[len(p.get_text().strip()) for p in document];assert min(lengths)>30
    checks['completed_labs'][n]={'executed_cells':len(code),'figures_verified':len(FIGURES[n]),'pages':len(document),'no_empty_pages':True,'split_disjoint':True}
    for start in range(0,len(document),8):
        sheet=Image.new('RGB',(1400,2040),'#bcc4ca');draw=ImageDraw.Draw(sheet)
        for i in range(start,min(start+8,len(document))):
            pix=document[i].get_pixmap(matrix=fitz.Matrix(.65,.65),alpha=False);im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((650,465));x=(i-start)%2*700+(700-im.width)//2;y=(i-start)//2*510+25
            sheet.paste(im,(x,y));draw.text((x,y-20),f'Practice {n}, page {i+1}',fill='black')
        sheet.save(ROOT/'analysis'/f'report_{n}_{start}.jpg')
    print('REPORT_VERIFIED',n,'PAGES',len(document),'FIGURES',len(FIGURES[n]),flush=True)
raw=pd.read_csv(ROOT/'lab_01/results/test_predictions.csv');r=json.loads((ROOT/'lab_01/results/results.json').read_text(encoding='utf-8'));r2=1-np.sum((raw.actual-raw.prediction)**2)/np.sum((raw.actual-raw.actual.mean())**2);assert np.isclose(r2,r['models'][r['winner']]['test']['r2'])
ann=np.load(ROOT/'lab_05/results/test_predictions.npz');r=json.loads((ROOT/'lab_05/results/results.json').read_text(encoding='utf-8'));assert np.isclose(np.mean(ann['probabilities'].argmax(1)==ann['y']),r['test']['accuracy'])
checks['independent_r2']=float(r2);checks['independent_ann_accuracy']=float(np.mean(ann['probabilities'].argmax(1)==ann['y']))
(ROOT/'analysis/deliverable_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('VERIFICATION_PASSED',json.dumps(checks,ensure_ascii=False),flush=True)
