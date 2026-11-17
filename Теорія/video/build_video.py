"""Author deterministic HyperFrames chapters and synthesize free Ukrainian narration."""
from pathlib import Path
import asyncio, json, html, shutil, subprocess, urllib.request, wave, math, struct
import edge_tts

HERE=Path(__file__).resolve().parent
AS=HERE/'assets'; AS.mkdir(exist_ok=True)
SCENES=[
('Вступ','Ти вже маєш код.\nЯк його пояснити?','Одна логіка для п’яти робіт','intro','Уяви, що на захисті просять не запустити ноутбук, а пояснити власне рішення. Почни з чотирьох речей: які дані, що прогнозує модель, як вона навчається і як ти перевірив результат. Ця логіка об’єднує всі п’ять практичних робіт.'),
('Основа','Навчання ≠\nперевірка','Train → Validation → Test','split','Навчальна вибірка потрібна для ваг і перетворень. Валідація допомагає обрати параметри. Тест — фінальний незалежний екзамен. Якщо обирати найкращу модель за тестом, він перестає бути незалежним. Тому спочатку фіксуємо рішення і тільки потім дивимось підсумкову оцінку.'),
('Основа','Витік буває\nбез цільової змінної','fit лише на Train','leak','Навчальний приклад: у тренувальній частині числа десять, дванадцять і чотирнадцять, а в тесті — сто. Середнє тренувальної частини дорівнює дванадцяти. Середнє всіх даних — тридцять чотири. Навіть без цільової змінної тест уже змінив параметр перетворення. Це витік інформації.'),
('01 / Регресія','Прогноз — це\nзважена сума','ŷ = Xw + b','regression','У першій роботі прогнозується ціна будинку. Лінійна модель додає внески його характеристик. На спрощеному графіку нахил визначає вага, а зсув — вільний член. Це навчальна ілюстрація, не дані Еймс. Реальна робота використовує числові ознаки, категорії, заповнення пропусків і масштабування.'),
('01 / Регресія','Помилка підказує\nнапрям кроку','w ← w − α∇J','gradient','Градієнт показує напрям зростання втрати. Ваги оновлюємо у протилежний бік. Великий крок може проскочити мінімум і спричинити розбіжність. У повному градієнтному спуску один крок використовує всю вибірку, у стохастичному — один приклад. Тому однакова кількість епох не означає однакову кількість оновлень.'),
('01 / Регресія','Обрано SGD\nза Validation','Test R² 0.8630 · RMSE 30 555 USD','ames','У твоєму експерименті стохастичний градієнтний спуск обрано за валідацією. На тесті коефіцієнт детермінації становить приблизно нуль цілих вісімдесят шість сотих, а середньоквадратична помилка — близько тридцяти тисяч п’ятисот п’ятдесяти п’яти доларів. Кращий тестовий результат іншого алгоритму не є приводом змінювати вибір заднім числом.'),
('02 / FedAvg','Сім класів —\nсім ймовірностей','Softmax: сума ймовірностей = 1','softmax','У другій роботі класифікуються сім видів квасолі. Модель обчислює сім логітів, а софтмакс перетворює їх на ймовірності із сумою один. Перед експонентою віднімаємо найбільший логіт. Ймовірності не змінюються, зате зменшується ризик числового переповнення.'),
('02 / FedAvg','Клієнти навчають.\nСервер агрегує.','Навчальний приклад: глобальна вага = 4','fed','Федеративне усереднення враховує кількість прикладів клієнта. Нехай клієнти мають сто, триста і шістсот прикладів, а їхні ваги дорівнюють одному, трьом і п’яти. Зважене середнє дорівнює чотирьом, а звичайне — трьом. Це навчальна арифметика; у реальній роботі агрегуються матриці ваг і зміщення.'),
('02 / FedAvg','Більше локальних епох —\nбільше обчислень','60 раундів ≠ однаковий бюджет','beans','У реальному експерименті є п’ять клієнтів і шістдесят раундів. Неоднорідні розподіли класів створюються через розподіл Діріхле. Більше локальних епох означає більше роботи між обмінами й може посилювати дрейф клієнтів. Федеративна схема сама собою не гарантує приватність: тут немає криптографічного захисту оновлень.'),
('03 / Тексти','Слова → вектори\n→ документ','FastText + Mean / TF-IDF','text','Третя робота переводить слова у вектори за допомогою фасттексту. Просте усереднення надає словам однакову вагу. Зважене усереднення враховує частоту слова та його інформативність у навчальному корпусі. На схемі лише два виміри для наочності; у твоєму коді вектор має сто вимірів.'),
('03 / Тексти','100% на Test —\nще не універсальний детектор','866 унікальних текстів · шаблонний корпус','limits','Усі шість основних моделей дали стовідсоткову точність на тестовій частині. Але після видалення повторів залишилося лише вісімсот шістдесят шість текстів, а джерело має шаблонне походження. Додаткова перевірка на інших темах корисна, проте не замінює перевірку на незалежних реальних авторах і нових моделях.'),
('04 / Дерева','Підбір усередині.\nОцінка зовні.','Inner CV обирає · Outer CV перевіряє','nested','Четверта робота використовує вкладену кросвалідацію. Зовнішній цикл відкладає частину навчальних даних. На решті внутрішня кросвалідація підбирає параметри через Оптуну. Потім обраний пайплайн оцінюється на відкладеній зовнішній частині. Тест у цих циклах не бере участі, а перетворення навчаються заново всередині кожного розбиття.'),
('04 / Дерева','Accuracy приховує\nрідкісний клас','RF: accuracy 0.9944 · macro-F1 0.9326','rare','На тесті випадковий ліс має майже дев’яносто дев’ять із половиною відсотків правильних відповідей. Однак макро еф один нижчий. Причина — однакова вага класів у цій метриці. Для найрідкіснішого класу є лише два тестові приклади, і модель правильно знаходить один. Одна помилка тут сильно змінює оцінку класу.'),
('05 / Нейромережа','784 → 64 → 48\n→ 32 → 10','Forward: матриці + ReLU + Softmax','network','П’ята робота реалізує мережу власноруч у нампай. Зображення двадцять вісім на двадцять вісім стає вектором із семисот вісімдесяти чотирьох чисел. Три приховані шари з релу будують нові представлення. Фінальний софтмакс дає ймовірності десяти класів одягу. Це прямий прохід.'),
('05 / Нейромережа','Похідні йдуть назад.\nВаги змінюються.','Backpropagation = правило ланцюга','backprop','Зворотне поширення використовує правило ланцюга: обчислює вплив кожної ваги на втрату, починаючи з виходу. Градієнтний спуск потім оновлює ваги. Перевірка скінченними різницями підтверджує правильність похідних на вибраних параметрах. Вона не доводить, що модель добре узагальнює: для цього потрібна незалежна оцінка.'),
('05 / Нейромережа','Регуляризація —\nне кнопка «краще»','13 режимів · спільні умови порівняння','regularization','Усі тринадцять режимів починаються з однакових ваг і порівнюються за спільним протоколом. Базова модель перемогла за валідаційною кросентропією. Сильний дропаут погіршив якість. Обмеження норми градієнта не тотожне обмеженню ваг, а тимчасовий нуль від дропауту не означає мертвий нейрон.'),
('Закріплення','Передбач. Зміни.\nПоясни результат.','Не заучуй оцінку — відтвори логіку','practice','Тепер відкрий інтерактивний курс. Зроби один крок градієнтного спуску, порахуй федеративне середнє та порівняй метрики для рідкісного класу. Перед натисканням спочатку передбач результат. Якщо помилишся, прочитай пояснення і повтори експеримент з іншими числами.'),
('Закріплення','Дані → модель\n→ чесна перевірка','Одна зрозуміла система','outro','На захисті рухайся від задачі до доказів. Назви джерело даних і ціль, покажи ключову формулу, поясни спосіб вибору моделі та межі висновків. Пройди фінальний тест сьогодні, повтори завтра й через тиждень. Так код перетворюється на знання, які ти можеш застосувати самостійно.')
]

async def voices():
    sem=asyncio.Semaphore(3)
    async def one(i,s):
        path=AS/f'voice_{i:02}.mp3'
        if path.exists() and path.stat().st_size>2000:return
        async with sem:
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(s[4],'uk-UA-OstapNeural',rate='-3%').save(str(path));break
                except Exception:
                    if attempt==2:raise
                    await asyncio.sleep(2)
        print('VOICE',i,flush=True)
    await asyncio.gather(*(one(i,s) for i,s in enumerate(SCENES)))

def duration(p):
    return float(subprocess.check_output(['ffprobe','-v','quiet','-show_entries','format=duration','-of','csv=p=0',str(p)],text=True).strip())

def svg(content):return f'<svg viewBox="0 0 1000 650" width="1000" height="650" aria-label="Навчальна схема" style="overflow:visible">{content}</svg>'
def text(x,y,s,size=40,color='#c8e7d4',cls='item'):
    return f'<text class="{cls}" x="{x}" y="{y}" font-size="{max(size,36)}" fill="{color}" font-family="MLSans">{html.escape(s)}</text>'
def diagram(kind):
    if kind in ('intro','outro','practice'):
        return svg(''.join(f'<path class="line" d="M {150+k*230} {180 if k%2==0 else 370} L {380+k*230} {370 if k%2==0 else 180}" stroke="#428c78" stroke-width="4" fill="none"/>' for k in range(3)) + ''.join(f'<circle class="node" cx="{150+k*230}" cy="{180 if k%2==0 else 370}" r="48" fill="{["#b5ebc9","#eea76e","#b5ebc9","#eea76e"][k]}"/>'+text(70+k*230,500,t,34) for k,t in enumerate(['Дані','Модель','Вибір','Test'])))
    if kind in ('split','nested'):
        labels=['TRAIN','VALIDATION','TEST'] if kind=='split' else ['OUTER TRAIN','INNER CV','OUTER VALID']
        out=''
        for j,t in enumerate(labels):
            y=100+j*160;out+=f'<rect class="node" x="90" y="{y}" width="780" height="105" rx="8" fill="{["#b5ebc9","#eea76e","#789dbe"][j]}"/>'+text(125,y+65,t,44,'#102f2d')
        return svg(out+text(95,635,'Test залишається поза підбором',32))
    if kind=='leak':
        return svg(text(90,130,'Train: 10, 12, 14',52)+text(90,230,'Test: 100',52,'#eea76e')+f'<path class="line" d="M100 285 H850" stroke="#5e9b87" stroke-width="4"/>'+text(90,390,'12',110)+text(450,390,'34',110,'#eea76e')+text(90,470,'fit(Train)',35)+text(450,470,'fit(усі дані)',35)+text(90,590,'Test змінив середнє → витік',38,'#eea76e'))
    if kind=='gradient':
        return svg('<path d="M90 530 H920 M90 530 V55" stroke="#57967f" stroke-width="3" fill="none"/><path class="line" d="M140 110 Q500 950 860 110" stroke="#b5ebc9" stroke-width="6" fill="none"/>'+''.join(f'<circle class="node" cx="{x}" cy="{y}" r="12" fill="#eea76e"/>' for x,y in [(180,200),(270,355),(360,455),(450,505)])+text(100,80,'J(w)',40)+text(890,585,'w',40)+text(100,635,'Навчальна схема · кроки до мінімуму',34))
    if kind=='regression':
        pts=[(140,470),(280,415),(410,345),(560,260),(710,210),(850,135)]
        return svg('<path d="M90 530 H920 M90 530 V55" stroke="#57967f" stroke-width="3" fill="none"/>'+''.join(f'<circle class="node" cx="{x}" cy="{y}" r="10" fill="#b5ebc9"/>' for x,y in pts)+'<path class="line" d="M110 500 L880 120" fill="none" stroke="#eea76e" stroke-width="7"/>'+text(110,625,'Навчальний приклад · не дані Ames',31))
    if kind=='ames':
        return svg(text(100,150,'SGD',70)+text(100,310,'0.8630',130,'#b5ebc9')+text(100,375,'Test R²',38)+text(100,510,'30 555 USD',80,'#eea76e')+text(100,575,'Test RMSE · зафіксована модель',33))
    if kind=='softmax':
        heights=[.05,.1,.12,.2,.08,.4,.05]
        return svg(''.join(f'<rect class="bar" x="{95+j*115}" y="{500-v*800}" width="65" height="{v*800}" rx="6" fill="{("#eea76e" if j==5 else "#b5ebc9")}"/>'+text(95+j*115,550,str(j+1),30) for j,v in enumerate(heights))+text(95,70,'Ілюстративні ймовірності',35)+text(95,630,'Σ p = 1',45))
    if kind=='fed':
        out=''
        for j,(n,w) in enumerate([(100,1),(300,3),(600,5)]):
            x=120+j*320;out+=f'<circle class="node" cx="{x}" cy="140" r="65" fill="#b5ebc9"/>'+text(x-15,155,str(w),48,'#102f2d')+text(x-65,255,f'n={n}',33)+f'<path class="line" d="M{x} 290 L460 410" stroke="#528e7a" stroke-width="4"/>'
        return svg(out+f'<circle class="node" cx="460" cy="485" r="75" fill="#eea76e"/>'+text(435,510,'4',70,'#102f2d')+text(250,630,'0.1×1 + 0.3×3 + 0.6×5',35))
    if kind=='beans':
        return svg(''.join(f'<rect class="bar" x="{110+j*170}" y="{140+(j%2)*40}" width="100" height="{310-(j%2)*40}" rx="10" fill="{["#b5ebc9","#eea76e","#789dbe","#b5ebc9","#eea76e"][j]}"/>'+text(115+j*170,520,str(j+1),38) for j in range(5))+text(100,80,'5 клієнтів',50)+text(100,630,'Схема неоднорідності · не частоти класів',30))
    if kind=='text':
        return svg(text(100,100,'«текст»',55)+text(550,100,'«ознака»',55)+f'<path class="line" d="M220 150 L380 290 M650 150 L480 290" stroke="#58937f" stroke-width="5"/>'+text(250,360,'[0.5, 0.5]',80,'#eea76e')+f'<path class="line" d="M100 560 L860 560 M100 560 L100 400" stroke="#508572" stroke-width="3"/>'+text(130,630,'2D-ілюстрація · модель: 100D',35))
    if kind=='limits':
        return svg(text(110,200,'1.0',160,'#b5ebc9')+text(110,285,'на заданому Test',44)+f'<path class="line" d="M100 340 H890" stroke="#58937f" stroke-width="4"/>'+text(110,430,'Нові жанри?',50,'#eea76e')+text(110,505,'Нові автори?',50,'#eea76e')+text(110,580,'Незалежний корпус?',50,'#eea76e'))
    if kind=='rare':
        return svg(text(110,110,'Клас 1 · Test: 2 приклади',42)+f'<circle class="node" cx="270" cy="300" r="90" fill="#b5ebc9"/><circle class="node" cx="660" cy="300" r="90" fill="#eea76e"/>'+text(235,320,'✓',70,'#102f2d')+text(635,320,'×',70,'#102f2d')+text(110,500,'Recall = 1 / 2 = 0.5',65)+text(110,605,'Одна помилка має велику вагу',34))
    if kind in ('network','backprop'):
        out=''
        ns=[4,5,4,3,2]
        for j in range(4):
            for k in range(ns[j]):
                for l in range(ns[j+1]):out+=f'<path class="line" d="M{110+j*190} {160+k*85} L{300+j*190} {160+l*85}" stroke="#447c69" stroke-width="2" opacity=".6"/>'
        for j,n in enumerate(ns):
            for k in range(n):out+=f'<circle class="node" cx="{110+j*190}" cy="{160+k*85}" r="20" fill="{("#eea76e" if kind=="backprop" else "#b5ebc9")}"/>'
        return svg(out+text(90,615,'Схема шарів · кількість вузлів скорочена',30))
    if kind=='regularization':
        names=['Baseline','L2 0.01','Dropout 0.5']
        metrics=json.loads((HERE.parent.parent/'lab_05/results/results.json').read_text(encoding='utf-8'))['experiments']
        values=[metrics[n]['validation_accuracy'] for n in names]
        return svg(text(90,80,'Validation accuracy',42)+''.join(text(90,170+j*150,n,36)+f'<rect class="bar" x="410" y="{130+j*150}" width="{v*480}" height="65" rx="5" fill="{("#b5ebc9" if j==0 else "#eea76e")}"/>'+text(835,175+j*150,f'{v:.3f}',34) for j,(n,v) in enumerate(zip(names,values)))+text(90,625,'Дані: lab_05/results/results.json',30))
    raise ValueError(kind)

def build():
    # Freeze openly licensed fonts with Cyrillic and their redistribution license.
    fonts=AS/'fonts';fonts.mkdir(exist_ok=True)
    for name in ['Regular','Bold']:
        target=fonts/f'NotoSans-{name}.ttf'
        if not target.exists():urllib.request.urlretrieve(f'https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-{name}.ttf',target)
    if not (fonts/'OFL.txt').exists():urllib.request.urlretrieve('https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/OFL.txt',fonts/'OFL.txt')
    for legacy in fonts.glob('Arial*.ttf'):legacy.unlink()
    if not (AS/'gsap.min.js').exists():urllib.request.urlretrieve('https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js',AS/'gsap.min.js')
    chapters=[];start=0
    for i,s in enumerate(SCENES):
        d=duration(AS/f'voice_{i:02}.mp3')+.65
        chapters.append(dict(id=f's{i:02}',title=s[0]+' · '+s[1].replace('\n',' '),start=round(start,3),duration=round(d,3),text=s[4]));start+=d
    total=round(start,3)
    comps=HERE/'compositions';comps.mkdir(exist_ok=True)
    hosts=[];audios=[]
    for i,(s,ch) in enumerate(zip(SCENES,chapters)):
        sid=ch['id'];d=ch['duration'];headline='<br>'.join(html.escape(x) for x in s[1].split('\n'))
        scene=f'''<template><style>
@font-face{{font-family:MLSans;src:url('assets/fonts/NotoSans-Regular.ttf')}}@font-face{{font-family:MLSans;src:url('assets/fonts/NotoSans-Bold.ttf');font-weight:700}}
#root{{position:absolute;inset:0;background:#102f2d;color:#eff7ed;font-family:MLSans,sans-serif;overflow:hidden}}
.chapter{{position:absolute;left:194px;top:120px;font-size:32px;color:#a6d1bd;letter-spacing:3px}}
.headline{{position:absolute;left:194px;top:235px;width:700px;font-size:68px;line-height:1.13;font-weight:700;letter-spacing:-2px;margin:0}}
.diagram{{position:absolute;left:940px;top:210px;width:810px;height:527px}}.diagram svg{{width:100%;height:100%}}
.bottom{{position:absolute;left:194px;top:835px;right:194px;font-size:38px;color:#c8e7d4;border-top:2px solid #467b67;padding-top:28px}}
.count{{position:absolute;right:194px;top:120px;font-size:30px;color:#a6d1bd}}
.meter{{position:absolute;left:194px;bottom:110px;width:1532px;height:4px;background:#285848}}.meter span{{display:block;width:100%;height:100%;background:#eea76e;transform-origin:left}}
</style><div id="root" data-composition-id="{sid}" data-width="1920" data-height="1080" data-hold="Narrated explanation with time to inspect the diagram">
<div id="{sid}-chapter" class="chapter">{html.escape(s[0])}</div><div class="count">{i+1:02} / {len(SCENES)}</div><h1 id="{sid}-headline" class="headline">{headline}</h1><div id="{sid}-diagram" class="diagram">{diagram(s[3])}</div><div id="{sid}-bottom" class="bottom">{html.escape(s[2])}</div><div class="meter"><span id="{sid}-meter"></span></div></div>
<script>
const tl=gsap.timeline({{paused:true}});
tl.fromTo('#{sid}-headline',{{opacity:0.3,y:22}},{{opacity:1,y:0,duration:.7,ease:'power2.out'}},0);
if(document.querySelector('#{sid}-diagram .node'))tl.fromTo('#{sid}-diagram .node',{{scale:0.7,opacity:0,transformOrigin:'center'}},{{scale:1,opacity:1,duration:.6,stagger:.12,ease:'power2.out'}},.5);
if(document.querySelector('#{sid}-diagram .line'))tl.fromTo('#{sid}-diagram .line',{{strokeDasharray:1600,strokeDashoffset:1600}},{{strokeDashoffset:0,duration:2.5,stagger:.03,ease:'power1.inOut'}},1);
if(document.querySelector('#{sid}-diagram .item'))tl.fromTo('#{sid}-diagram .item',{{opacity:0,y:12}},{{opacity:1,y:0,duration:.6,stagger:.45,ease:'power2.out'}},1.4);
if(document.querySelector('#{sid}-diagram .bar'))tl.fromTo('#{sid}-diagram .bar',{{scaleX:0,transformOrigin:'left center'}},{{scaleX:1,duration:1.6,stagger:.2,ease:'power2.out'}},1);
tl.fromTo('#{sid}-bottom',{{opacity:.25,y:10}},{{opacity:1,y:0,duration:.7}},Math.min(7,{d}/3));
tl.fromTo('#{sid}-meter',{{scaleX:0}},{{scaleX:1,duration:{d},ease:'none'}},0);
window.__timelines=window.__timelines||{{}};window.__timelines['{sid}']=tl;
</script></template>'''
        (comps/f'{sid}.html').write_text(scene,encoding='utf-8')
        hosts.append(f'<div id="{sid}" class="clip" data-composition-id="{sid}" data-composition-src="compositions/{sid}.html" data-start="{ch["start"]}" data-duration="{d}" data-width="1920" data-height="1080" data-track-index="1" data-track-kind="graphics"></div>')
        audios.append(f'<audio id="voice-{i}" src="assets/voice_{i:02}.mp3" data-start="{ch["start"]}" data-duration="{duration(AS/f"voice_{i:02}.mp3"):.3f}" data-track-index="2" data-volume="1"></audio>')
    # Quiet, deterministic original ambient bed; no third-party licensing or provider costs.
    sr=24000
    with wave.open(str(AS/'bed.wav'),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
        for sec in range(math.ceil(total)):
            block=bytearray()
            for k in range(sr):
                t=sec+k/sr;fade=min(1,t/3,max(0,(total-t)/4));val=sum(math.sin(2*math.pi*hz*t) for hz in (130.8128,196,261.6256))/3
                block+=struct.pack('<h',int(170*fade*val*(.7+.3*math.cos(2*math.pi*t/8))))
            f.writeframes(block)
    root='''<!doctype html><html lang="uk"><head><meta charset="utf-8"><title>ML · Пояснення п’яти робіт</title><script src="assets/gsap.min.js"></script><style>html,body{margin:0;width:100%;height:100%;background:#102f2d}#root{position:absolute;inset:0}.clip{position:absolute;inset:0;width:100%;height:100%}</style></head><body>'''
    root+=f'<div id="root" data-composition-id="main" data-start="0" data-duration="{total}" data-width="1920" data-height="1080" data-fps="30">'+''.join(hosts)+''.join(audios)+f'<audio id="music-bed" src="assets/bed.wav" data-start="0" data-duration="{total}" data-volume="1" data-fade-in="2" data-fade-out="3" data-track-index="3"></audio></div><script>window.__timelines=window.__timelines||{{}};const tl=gsap.timeline({{paused:true}});tl.to({{}},{{duration:{total}}});window.__timelines.main=tl;</script></body></html>'
    (HERE/'index.html').write_text(root,encoding='utf-8')
    (HERE/'chapters.json').write_text(json.dumps(chapters,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'SCRIPT.md').write_text('# Відеопояснення\n\n'+'\n\n'.join(f'## {c["start"]:.1f} с · {c["title"]}\n\n{c["text"]}' for c in chapters),encoding='utf-8')
    def stamp(t):
        ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}'
    cues=['WEBVTT\n']; cue_data=[]
    for c in chapters:
        sentences=[s.strip()+'.' for s in c['text'].split('.') if s.strip()]
        weight=sum(len(s) for s in sentences);pos=c['start']
        for sent in sentences:
            d=(c['duration']-.65)*len(sent)/weight;cues.append(f'{stamp(pos)} --> {stamp(pos+d)}\n{sent}\n');cue_data.append({'start':round(pos,3),'end':round(pos+d,3),'text':sent});pos+=d
    (HERE/'subtitles.vtt').write_text('\n'.join(cues),encoding='utf-8')
    (HERE.parent/'video-data.js').write_text('window.VIDEO_DATA = '+json.dumps({'duration':total,'chapters':chapters,'cues':cue_data},ensure_ascii=False)+';',encoding='utf-8')
    (HERE/'SPEC.md').write_text(f'''# ML explainer\n1920×1080, 30 fps, {total:.3f} s. Ukrainian synthesized narration, uk-UA-OstapNeural, Edge TTS, rate −3%.\n\nAudience: owner of the five submitted labs; beginner to intermediate.\nStory: one shared experimental protocol, then one conceptual mechanism and one boundary per lab. Alternatives considered: long code walkthrough; exam-only flashcards. The chosen overview links to deeper interactive lessons.\n\nDesign: dark green #102f2d, mint #b5ebc9, orange #eea76e. Noto Sans OFL files frozen locally, 68 px headline, 38 px conclusion, 36+ px diagram labels. Safe x=194..1726, main text y=120..930.\nBANNED: stock footage, invented screenshots, tiny code, three-card bullet slides, fake numerical results, universal-detector claims.\n\nSource of numerical results: ../content.js and ../../lab_0*/results/results.json. Educational diagrams marked as illustrative. Regularization bars adapted from installed animated-bar-chart component (growth transforms). Original quiet sine-wave bed created by build_video.py, no licensed music.\nCaptions are sentence-length approximate cues proportionally aligned to narration; they are not word-level forced alignment.\n''',encoding='utf-8')
    (HERE/'STORYBOARD.md').write_text('# Storyboard\n\n'+'\n'.join(f'{i+1:02}. {c["start"]:.2f}–{c["start"]+c["duration"]:.2f}s · {c["title"]} · diagram={SCENES[i][3]}' for i,c in enumerate(chapters)),encoding='utf-8')
    (HERE/'assets/ledger.json').write_text(json.dumps(dict(voice={'provider':'Edge TTS','voice':'uk-UA-OstapNeural','cost':'free','files':'voice_00..17.mp3'},music={'provider':'original procedural synthesis','file':'bed.wav'},fonts={'source':'notofonts/noto-fonts','license':'OFL-1.1','license_file':'fonts/OFL.txt'},gsap={'url':'https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js'},diagrams={'type':'authored SVG','source':'course formulas and verified results'}),ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPOSITION_READY',total,flush=True)

if __name__=='__main__':
    asyncio.run(voices());build()
