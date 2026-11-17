from pathlib import Path
import json
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
errors=[];report=[]
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8766')
    page.locator('#diagnostic-questions [data-q="0"] [data-a="1"]').click()
    page.locator('#diagnostic-questions [data-q="1"] [data-a="0"]').click()
    page.locator('#diagnostic-questions [data-q="2"] [data-a="1"]').click()
    assert '3/3' in page.locator('#diagnostic-result').inner_text()
    for i in range(6):
        page.goto(f'http://127.0.0.1:8766/#lesson-{i}')
        page.wait_for_selector('#lab')
        assert page.locator('#questions [data-q]').count()==4
        if i==0:
            page.locator('#fit-all').click();assert '34' in page.locator('#mean').inner_text()
            page.locator('#fit-train').click();assert '12' in page.locator('#mean').inner_text()
        if i==1:
            page.locator('#gd-many').click();assert 'Крок 20' in page.locator('#gd-result').inner_text()
            assert '0,00' in page.locator('#gd-result').inner_text()
        if i==2:
            assert '4' in page.locator('#fed-result').inner_text()
            page.locator('#unweighted').click();assert '3' in page.locator('#fed-result').inner_text()
            page.locator('#scenario').select_option('Dirichlet 1');assert page.locator('#clients-table td').count()>20
        if i==3:
            page.locator('#idf').fill('9');page.locator('#idf').dispatch_event('input');assert '0,1' in page.locator('#doc-vector').inner_text()
        if i==4:
            assert '99%' in page.locator('#rare-result').inner_text()
            page.locator('#rare').fill('10');page.locator('#rare').dispatch_event('input');assert '100%' in page.locator('#rare-result').inner_text()
        if i==5:
            assert '0,6' in page.locator('#clip-result').inner_text()
            page.locator('#nn-mode').select_option('Dropout 0.5');assert 'Validation accuracy' in page.locator('#nn-result').inner_text()
        for j in range(4):
            answer=page.evaluate(f'COURSE.lessons[{i}].quiz[{j}].answer')
            page.locator(f'#questions [data-q="{j}"] [data-a="{answer}"]').click()
        page.locator('details').last.locator('summary').click()
        page.locator('#notes').fill(f'Нотатка уроку {i}')
        assert page.locator('body').evaluate('(el)=>el.scrollWidth <= innerWidth')
        page.screenshot(path=str(HERE/f'assets/lesson_{i}_desktop.png'))
        report.append({'lesson':i,'passed':True})
    assert '24 / 24' in page.locator('#score').inner_text()
    page.reload();assert '24 / 24' in page.locator('#score').inner_text()
    with page.expect_download() as dl:
        page.locator('#export').click()
    exported=dl.value
    payload=json.loads(Path(exported.path()).read_text(encoding='utf-8'))
    assert len(payload['answers'])==24
    page.on('dialog',lambda d:d.accept())
    page.locator('#reset').click()
    assert '0 / 24' in page.locator('#score').inner_text()
    page.locator('#import-file').set_input_files(exported.path())
    page.wait_for_timeout(150)
    assert '24 / 24' in page.locator('#score').inner_text()
    page.goto('http://127.0.0.1:8766/#exam')
    assert page.locator('#exam-questions [data-q]').count()==12
    page.locator('#exam-questions [data-q="0"] [data-a="0"]').click()
    page.locator('#retry').click();assert page.locator('#exam-questions [data-q]:visible').count()==12
    page.locator('#exam-questions [data-q="0"] [data-a="1"]').click()
    page.locator('#retry').click();assert page.locator('#exam-questions [data-q]:visible').count()==11
    for i in range(6):
        page.set_viewport_size({'width':390,'height':844})
        page.goto(f'http://127.0.0.1:8766/#lesson-{i}')
        assert page.locator('body').evaluate('(el)=>el.scrollWidth <= innerWidth'),f'mobile overflow lesson{i}'
    page.screenshot(path=str(HERE/'assets/mobile.png'),full_page=True)
    page.goto((HERE/'index.html').as_uri()+'#lesson-2')
    assert '4' in page.locator('#fed-result').inner_text()
    page.goto((HERE/'index.html').as_uri()+'#video')
    page.wait_for_function('document.querySelector("video").readyState>=1')
    assert page.locator('[data-time]').count()==18
    assert page.evaluate('document.querySelector("video").textTracks[0].cues.length')==76
    assert abs(page.evaluate('document.querySelector("video").duration')-422.844)<.2
    page.locator('[data-time]').nth(4).click()
    assert page.evaluate('document.querySelector("video").currentTime')>90
    browser.close()
assert not errors,errors
(HERE/'checks.json').write_text(json.dumps({'lessons':report,'js_errors':errors,'desktop':'1440x1000','mobile':'390x844','checks':['diagnostic','all 6 interactives','24 graded answers','persist after reload','export/reset/import progress','exam mistake filtering','mobile overflow','offline file:// lesson','offline MP4 metadata, 18 chapters, 76 subtitle cues and seeking']},ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: six lessons, 24 answers, simulations, persistence, exam, mobile, offline')
