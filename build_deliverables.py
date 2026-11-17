"""Збирає ноутбуки та Word-звіти з результатів власних експериментів."""

from pathlib import Path
import ast
import sys, json, argparse, subprocess

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "project"))
from course_text import TITLES, GOALS, THEORY, FIGURES, SOURCES
from result_tables import tables, conclusions
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH as A
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import nbformat


def clean(x):
    return str(x).replace("—", "-").replace("–", "-")


class Report:
    def __init__(self, n, profile, draft):
        self.n = n
        self.d = Document()
        self.fig = 0
        self.tab = 0
        sec = self.d.sections[0]
        sec.page_width = Cm(21)
        sec.page_height = Cm(29.7)
        sec.left_margin = Cm(3)
        sec.right_margin = Cm(1.5)
        sec.top_margin = sec.bottom_margin = Cm(2)
        sec.different_first_page_header_footer = True
        for name in ["Normal", "Heading 1", "Heading 2"]:
            s = self.d.styles[name]
            s.font.name = "Times New Roman"
            s.font.size = Pt(14)
            s.font.color.rgb = RGBColor(0, 0, 0)
            s.paragraph_format.line_spacing = 1.5
            s.paragraph_format.space_after = Pt(0)
            s.paragraph_format.first_line_indent = Cm(1.25 if name == "Normal" else 0)
            fonts = s._element.get_or_add_rPr().get_or_add_rFonts()
            for key in ["ascii", "hAnsi", "eastAsia", "cs"]:
                fonts.set(qn("w:" + key), "Times New Roman")
            for key in ["asciiTheme", "hAnsiTheme", "eastAsiaTheme", "csTheme"]:
                fonts.attrib.pop(qn("w:" + key), None)
        for name in ["Heading 1", "Heading 2"]:
            s = self.d.styles[name]
            s.font.bold = True
            s.paragraph_format.keep_with_next = True
            s.paragraph_format.space_before = Pt(10)
            s.paragraph_format.space_after = Pt(8)
        self.d.styles["Heading 1"].paragraph_format.page_break_before = True
        for text in [
            "Міністерство освіти і науки України",
            "Львівський національний університет імені Івана Франка",
        ]:
            self.p(text, A.CENTER, indent=False)
        for _ in range(3):
            self.blank()
        self.p(
            f"ЗВІТ ПРО ВИКОНАННЯ ПРАКТИЧНОЇ РОБОТИ №{n}",
            A.CENTER,
            bold=True,
            size=16,
            indent=False,
        )
        self.p(profile.get("discipline", "Машинне навчання"), A.CENTER, indent=False)
        self.p(TITLES[n], A.CENTER, bold=True, indent=False)
        if draft:
            self.p(
                "ЧЕРНЕТКА: титульні дані очікують підтвердження",
                A.CENTER,
                size=11,
                indent=False,
            )
        for _ in range(3):
            self.blank()
        for text in [
            "Виконав:",
            profile.get("student", "Панчишин ____________________"),
            "Група: " + profile.get("group", "____________________"),
            "Перевірив(ла): " + profile.get("teacher", "____________________"),
        ]:
            self.p(text, A.RIGHT, indent=False)
        for _ in range(4):
            self.blank()
        self.p("Львів - " + str(profile.get("year", "2026")), A.CENTER, indent=False)
        p = self.p("ЗМІСТ", A.CENTER, bold=True, indent=False)
        p.paragraph_format.page_break_before = True
        p = self.d.add_paragraph()
        p.alignment = A.JUSTIFY
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), 'TOC \\o "1-2" \\h \\z \\u')
        p._p.append(field)
        p = sec.footer.paragraphs[0]
        p.alignment = A.CENTER
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        p._p.append(field)
        self.d.core_properties.author = profile.get("student", "Панчишин")
        self.d.core_properties.title = TITLES[n]

    def p(self, text, alignment=A.JUSTIFY, bold=False, size=14, indent=True):
        p = self.d.add_paragraph()
        p.alignment = alignment
        p.paragraph_format.first_line_indent = Cm(1.25 if indent else 0)
        r = p.add_run(clean(text))
        r.bold = bold
        r.font.size = Pt(size)
        return p

    def blank(self):
        p = self.d.add_paragraph()
        p.paragraph_format.line_spacing = 1
        p.add_run().font.size = Pt(6)

    def h(self, text, level=1):
        p = self.d.add_paragraph(clean(text), "Heading " + str(level))
        p.alignment = A.CENTER if level == 1 else A.JUSTIFY

    def table(self, title, headers, rows, comment):
        self.tab += 1
        num = self.tab
        self.p(f"У таблиці {num} наведено " + title[0].lower() + title[1:] + ".")
        p = self.p("Таблиця " + str(num), A.RIGHT, size=12, indent=False)
        p.paragraph_format.keep_with_next = True
        p = self.p(title, A.CENTER, bold=True, size=12, indent=False)
        p.paragraph_format.keep_with_next = True
        table = self.d.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        first = 6.3 if len(headers) <= 5 else 4.0
        widths = [first] + [(16.5 - first) / (len(headers) - 1)] * (len(headers) - 1)
        for col, w in zip(table.columns, widths):
            col.width = Cm(w)
        if not rows:
            rows = [["Немає"] + ["-"] * (len(headers) - 1)]
        for cells, values in [(table.rows[0].cells, headers)] + [
            (table.add_row().cells, row) for row in rows
        ]:
            for cell, value, width in zip(cells, values, widths):
                cell.width = Cm(width)
                cell.text = clean(value)
        table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for i, row in enumerate(table.rows):
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p in cell.paragraphs:
                    p.alignment = A.LEFT if i > 0 else A.CENTER
                    p.paragraph_format.first_line_indent = Cm(0)
                    p.paragraph_format.line_spacing = 1
                    p.paragraph_format.space_after = Pt(3)
                    p.paragraph_format.space_before = Pt(3)
                    if len(table.rows) <= 16 and i < len(table.rows) - 1:
                        p.paragraph_format.keep_with_next = True
                    for r in p.runs:
                        r.font.name = "Times New Roman"
                        r.font.size = Pt(12)
                        r.bold = i == 0
        self.blank()
        self.p(comment)

    def figure(self, path, caption, comment):
        self.fig += 1
        num = self.fig
        self.p(
            f"На рисунку {num} зображено "
            + caption[0].lower()
            + caption[1:]
            + ". "
            + comment
        )
        p = self.d.add_paragraph()
        p.alignment = A.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.line_spacing = 1
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(16.1))
        p = self.p(f"Рис. {num}. {caption}", A.CENTER, size=12, indent=False)
        for run in p.runs:
            run.italic = True
        self.blank()

    def equation(self, text):
        p = self.d.add_paragraph()
        p.alignment = A.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        mathpara = OxmlElement("m:oMathPara")
        mathobj = OxmlElement("m:oMath")
        run = OxmlElement("m:r")
        node = OxmlElement("m:t")
        node.text = text
        run.append(node)
        mathobj.append(run)
        mathpara.append(mathobj)
        p._p.append(mathpara)


def make_report(n, data, profile, draft):
    rows = tables(n, data)
    figures = [
        item
        for item in FIGURES[n]
        if (ROOT / f"lab_{n:02d}/results" / f"{item[0]}.png").exists()
    ]
    r = Report(n, profile, draft)
    r.h("ВСТУП")
    r.p("Мета роботи - " + GOALS[n][0].lower() + GOALS[n][1:])
    r.p(
        "Для виконання використав методичні вказівки з навчального репозиторію викладача та первинні файли рекомендованого датасету. Усі наведені числові оцінки отримані запуском власного коду; готові навчальні результати з лекційних прикладів не підставлялися."
    )
    r.p(
        "Дані поділив на незалежні вибірки до навчання перетворень. Відбір ознак, параметри масштабування та налаштування моделей обмежив навчальними й валідаційними даними відповідно до умов роботи."
    )
    r.p(
        "Реалізацію зберіг у project/ та виконаному Jupyter Notebook. Разом із кодом збережено індекси вибірок, метрики, графіки й параметри моделей для перевірки та відтворення результатів."
    )
    r.p(
        f"Звіт містить вступ, опис реалізації, таблиці результатів, їхню візуалізацію, висновки та джерела. У роботі {len(rows)} таблиць і {len(figures)} рисунків."
    )
    r.h("1. Дані та реалізація")
    if n == 1:
        r.p(
            "Датасет Ames Housing містить 2930 спостережень. SalePrice - ціна в доларах США; числові та категоріальні ознаки описують нерухомість. Для навчання вибрав десять числових ознак за абсолютною Train-кореляцією з ціною та чотири категоріальні: Neighborhood, Bldg Type, House Style, Kitchen Qual."
        )
    elif n == 2:
        r.p(
            f"Dry Bean містив {data['raw_shape'][0]} записів; до поділу видалив {data['duplicates_removed']} точних дублікатів. Використав усі 16 геометричних характеристик зерен. Точні дублікати не можуть одночасно потрапити в Train і Test."
        )
    elif n == 3:
        r.p(
            "Для класифікації використав тільки text_content. spaCy виконує лематизацію, видалення стоп-слів і небуквених токенів та переведення до нижнього регістру. Назви тем і способи генерації використано лише для аудиту й додаткового групового поділу."
        )
    elif n == 4:
        r.p(
            f"Використав UCI Annealing: 798 рядків anneal.data і 100 рядків anneal.test об’єднав у набір із 898 об’єктів та 38 ознак. Це відповідає дозволеній повній версії. Індивідуальний номер N={data['number']}, усі джерела випадковості ініціалізовано SEED={data['seed']}."
        )
    elif n == 5:
        r.p(
            "Використав незмінні офіційні train/test файли Fashion-MNIST. Із 60000 навчальних зображень стратифіковано відібрав 15000: 12000 для Train і 3000 для Validation. Методичка дозволяє підмножину для пришвидшення. Офіційний Test із 10000 зображень залишив окремим. Класи: 0 - футболка/топ, 1 - штани, 2 - светр, 3 - сукня, 4 - пальто, 5 - сандалі, 6 - сорочка, 7 - кросівки, 8 - сумка, 9 - черевики."
        )
    for title, text in THEORY[n]:
        r.h(title, 2)
        r.p(text)
    r.h("Основні розрахункові формули", 2)
    equations = {
        1: [
            "ŷ = Xw + b; J = ||ŷ - y||² / (2m)",
            "∇wJ = Xᵀ(ŷ - y)/m; ∂J/∂b = mean(ŷ - y)",
            "R² = 1 - SSE/SST",
        ],
        2: [
            "Pᵢc = exp(Zᵢc) / Σⱼ exp(Zᵢⱼ)",
            "∇W L = Xᵀ(P - Y)/m + λW",
            "Wglobal = Σₖ (nₖ / Σⱼ nⱼ) Wₖ",
        ],
        3: ["vmean = Σᵢ vᵢ / m", "vTFIDF = Σₜ tfidf(t,d) vₜ / Σₜ tfidf(t,d)"],
        4: [
            "t = mean(d) / √[(1/n + ntest/ntrain) Var(d)]",
            "f(x) = E[f(X)] + Σⱼ SHAPⱼ(x)",
        ],
        5: [
            "Zˡ = Aˡ⁻¹ Wˡ + bˡ; Aˡ = ReLU(Zˡ)",
            "δout = (P - Y)/m; ∇Wˡ = (Aˡ⁻¹)ᵀδˡ",
            "L = CE + λ₁Σ|W| + (λ₂/2)ΣW²",
            "gclip = g · min(1, c/||g||₂)",
        ],
    }
    for equation in equations[n]:
        r.equation(equation)
    r.h("2. Результати експериментів")
    for title, headers, values, comment in rows:
        r.table(title, headers, values, comment)
    r.h("3. Візуальний аналіз")
    for stem, caption, comment in figures:
        if n == 5 and stem.startswith("weights_"):
            index = int(stem.split("_")[1])
            method = list(data["experiments"])[index]
            caption = "Розподіли ваг прихованих шарів: " + method
        r.figure(ROOT / f"lab_{n:02d}/results" / f"{stem}.png", caption, comment)
    r.h("ВИСНОВКИ")
    for text in conclusions(n, data):
        r.p(text)
    r.h("СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ")
    refs = (
        [
            "Sinkevych O. Machine Learning: навчальні матеріали та практичні завдання. https://github.com/olehsinkevych/machine-learning"
        ]
        + SOURCES[n]
        + [
            "scikit-learn. Документація перетворень і метрик. https://scikit-learn.org/stable/"
        ]
    )
    for i, text in enumerate(refs, 1):
        r.p(str(i) + ". " + text.replace("/", "/\u200b"))
    assert (r.tab, r.fig) == (len(rows), len(figures))
    out = ROOT / f"lab_{n:02d}"
    path = out / (("Чернетка_" if draft else "") + f"Звіт_ПР_{n}_Панчишин.docx")
    r.d.save(path)
    print("DOCX", path.relative_to(ROOT), "TABLES", r.tab, "FIGURES", r.fig, flush=True)


def make_notebook(n, profile):
    book = nbformat.v4.new_notebook()
    cells = []

    def md(s):
        cells.append(nbformat.v4.new_markdown_cell(clean(s)))

    def code(s):
        cells.append(nbformat.v4.new_code_cell(clean(s)))

    md(
        f"# Практична робота №{n}\n## {TITLES[n]}\n\nСтудент: {profile.get('student', 'Панчишин (повне ПІБ очікує підтвердження)')}\n\n**Мета:** {GOALS[n]}\n\nДжерело завдання: https://github.com/olehsinkevych/machine-learning\n\n**Відтворення:** середовище Python 3.12, залежності в requirements.txt; перед запуском усієї роботи завантажте дані командою `python download_data.py`. Усі результати нижче отримуються виконанням коду, а не копіюванням готових таблиць."
    )
    code(
        "from pathlib import Path\nimport sys, os\nfor name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:\n    os.environ.setdefault(name, '2')\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'project' / 'common.py').is_file())\nsys.path.insert(0, str(ROOT / 'project'))\nimport common\nfrom IPython.display import display, Image\nimport pandas as pd\nimport json"
    )
    md(
        "## 1. Теоретичні пояснення та відповіді на питання\n\n"
        + "\n\n".join("### " + title + "\n" + text for title, text in THEORY[n])
    )
    md(
        "## 2. Повна реалізація\n\nДалі міститься власна реалізація обчислень цієї роботи. Код також винесено в окремий файл `project/lab%02d.py`."
        % n
    )
    source = (
        (ROOT / "project" / f"lab{n:02d}.py")
        .read_text(encoding="utf-8")
        .rsplit("\nif __name__", 1)[0]
    )
    source_lines = source.splitlines(keepends=True)
    starts = sorted({0} | {
        node.lineno - 1 for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    })
    for start, end in zip(starts, starts[1:] + [len(source_lines)]):
        chunk = ''.join(source_lines[start:end]).strip()
        if chunk:
            code(chunk)
    if n == 4:
        md(
            "### Індивідуальний параметр\nN є номером студента у списку групи. Його не можна замінювати довільним seed."
        )
        code(
            "NUMBER = "
            + repr(profile.get("number"))
            + '\nif NUMBER is None:\n    raise ValueError("Потрібен підтверджений номер студента N у списку групи")\nrun(NUMBER)'
        )
    else:
        code("run()")
    code(
        f"result_path = ROOT / 'lab_{n:02d}' / 'results' / 'results.json'\nR = json.loads(result_path.read_text(encoding='utf-8'))\nfrom result_tables import tables, conclusions"
    )
    data_path = ROOT / f"lab_{n:02d}/results/results.json"
    if data_path.exists():
        r = json.loads(data_path.read_text(encoding="utf-8"))
        items = tables(n, r)
        md("## 3. Результати та їх інтерпретація")
        for i, (title, headers, rows, comment) in enumerate(items):
            md(
                f"### Таблиця {i + 1}. {title}\n\nУ таблиці {i + 1} наведено "
                + title[0].lower()
                + title[1:]
                + ". "
                + comment
            )
            code(
                f"entry = tables({n}, R)[{i}]\ndisplay(pd.DataFrame(entry[2], columns=entry[1]))"
            )
    else:
        md(
            "## 3. Результати\nТаблиці й рисунки додаються після виконання індивідуального запуску."
        )
        code(
            f"for caption, headers, rows, comment in tables({n}, R):\n    print(caption)\n    display(pd.DataFrame(rows, columns=headers))\n    print(comment)"
        )
    md("## 4. Графіки")
    for i, (stem, caption, comment) in enumerate(FIGURES[n], 1):
        if n == 5 and stem.startswith("weights_"):
            index = int(stem.split("_")[1])
            caption = "Розподіли ваг прихованих шарів: " + list(r["experiments"])[index]
        md(
            f"### Рисунок {i}. {caption}\n\nНа рисунку {i} зображено "
            + caption[0].lower()
            + caption[1:]
            + ". "
            + comment
        )
        code(
            f"display(Image(filename=str(ROOT / 'lab_{n:02d}' / 'results' / '{stem}.png')))"
        )
    md("## Висновки")
    code(f"for text in conclusions({n}, R):\n    print(text)")
    md("## Джерела\n\n" + "\n\n".join(SOURCES[n]))
    book.cells = cells
    book.metadata = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    path = ROOT / f"lab_{n:02d}" / f"Практична_{n}_Панчишин.ipynb"
    path.parent.mkdir(exist_ok=True)
    nbformat.write(book, path)
    print("NOTEBOOK", path.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    path = ROOT / "profile.json"
    profile = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not args.draft:
        assert (
            profile.get("student") and profile.get("group") and profile.get("confirmed")
        ), "Титульні дані ще не підтверджені"
    for n in range(1, 6):
        if not args.reports_only:
            make_notebook(n, profile)
        result = ROOT / f"lab_{n:02d}/results/results.json"
        if result.exists():
            make_report(
                n, json.loads(result.read_text(encoding="utf-8")), profile, args.draft
            )
    freeze = subprocess.check_output(
        ["uv", "pip", "freeze", "--python", sys.executable], text=True
    )
    (ROOT / "requirements.txt").write_text(freeze, encoding="utf-8")
