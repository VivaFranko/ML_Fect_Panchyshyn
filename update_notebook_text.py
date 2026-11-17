from pathlib import Path
import sys,json
import nbformat
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'project'))
from course_text import THEORY,FIGURES
from result_tables import tables
for n in [1,2,3,4,5]:
    path=next((ROOT/f'lab_{n:02d}').glob('Практична*.ipynb'));book=nbformat.read(path,4)
    for cell in book.cells:
        if cell.cell_type=='markdown' and cell.source.startswith('### Таблиця '):
            import re
            number=int(re.search(r'\d+',cell.source)[0])
            data=json.loads((ROOT/f'lab_{n:02d}/results/results.json').read_text(encoding='utf-8'))
            title,headers,rows,comment=tables(n,data)[number-1]
            cell.source=f'### Таблиця {number}. {title}\n\nУ таблиці {number} наведено '+title[0].lower()+title[1:]+'. '+comment
        if cell.cell_type=='markdown' and cell.source.startswith('## 1. Теоретичні'):
            cell.source='## 1. Теоретичні пояснення та відповіді на питання\n\n'+'\n\n'.join('### '+title+'\n'+text for title,text in THEORY[n])
        if n==5 and cell.cell_type=='markdown' and 'експеримент ' in cell.source:
            results=json.loads((ROOT/'lab_05/results/results.json').read_text(encoding='utf-8'));names=list(results['experiments'])
            import re
            cell.source=re.sub(r'експеримент (\d+)',lambda m:names[int(m[1])-1],cell.source)
    nbformat.write(book,path)
