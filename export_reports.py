from pathlib import Path
import sys
import pythoncom,win32com.client
ROOT=Path(__file__).resolve().parent
pythoncom.CoInitialize();word=win32com.client.DispatchEx('Word.Application');word.Visible=False;word.DisplayAlerts=0;word.AutomationSecurity=3
try:
    for path in sorted(ROOT.glob('lab_*/*.docx')):
        if sys.argv[1:] and int(path.parent.name.split('_')[1]) not in {int(n) for n in sys.argv[1:]}:
            continue
        d=word.Documents.Open(str(path),ReadOnly=False,AddToRecentFiles=False)
        try:
            d.Fields.Update()
            for toc in d.TablesOfContents:toc.Update();toc.Range.ParagraphFormat.Alignment=3
            d.Repaginate()
            for toc in d.TablesOfContents:toc.UpdatePageNumbers()
            d.Save();d.ExportAsFixedFormat(str(path.with_suffix('.pdf')),17,OpenAfterExport=False)
            print('PDF',path.name,'PAGES',d.ComputeStatistics(2),flush=True)
        finally:d.Close(False)
finally:word.Quit();pythoncom.CoUninitialize()
