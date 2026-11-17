from pathlib import Path
import nbformat,sys
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent
for number in [int(n) for n in sys.argv[1:]] or [1,2,3,5]:
    path=next((ROOT/f'lab_{number:02d}').glob('Практична*.ipynb'))
    print('EXECUTING',number,flush=True)
    book=nbformat.read(path,as_version=4)
    try:
        NotebookClient(book,timeout=7200,kernel_name='ml-panch2026',resources={'metadata':{'path':str(path.parent)}}).execute()
    except Exception:
        nbformat.write(book,path);raise
    nbformat.write(book,path)
    assert all(out.output_type!='error' for cell in book.cells if cell.cell_type=='code' for out in cell.get('outputs',[]))
    print('NOTEBOOK_PASSED',number,sum(c.cell_type=='code' for c in book.cells),flush=True)
