"""Build the RealFact final presentation on the user's template (its master, layouts and section order)."""
import sys
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu
from lxml import etree

TEMPLATE, OUT, FIG = sys.argv[1], sys.argv[2], sys.argv[3]

NAVY = RGBColor(0x1B, 0x2A, 0x4A)
BLUE = RGBColor(0x2F, 0x5B, 0xD3)
TINT = RGBColor(0xEC, 0xF1, 0xFC)
GRAY = RGBColor(0x55, 0x5D, 0x6B)
LIGHT = RGBColor(0xF4, 0xF5, 0xF7)
LINE = RGBColor(0xD5, 0xD9, 0xE0)
GREEN = RGBColor(0x1F, 0x8A, 0x57)
RED = RGBColor(0xC4, 0x3C, 0x3C)
AMBER = RGBColor(0xB7, 0x79, 0x1F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation(TEMPLATE)
L = {l.name: l for l in prs.slide_layouts}
n_original = len(prs.slides)


# ------------------------------------------------------------------ helpers
def title_slide(text, layout="TITLE_ONLY"):
    s = prs.slides.add_slide(L[layout])
    s.shapes.title.text = text
    for p in s.shapes.title.text_frame.paragraphs:
        for r in p.runs:
            r.font.color.rgb = NAVY
            r.font.bold = True
            r.font.size = Pt(24)
    return s


def box(s, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None, radius=0.08):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.25)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    st = sh._element.find(qn("p:style"))
    if st is not None:
        sh._element.remove(st)
    sh.text_frame.text = ""
    if name:
        sh.name = name
    return sh


def text(s, x, y, w, h, paras, size=13, color=GRAY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=4, font=None):
    """paras: str, or list of str / list of runs [(text, {opts})]."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = o.get("color", color)
            if font or o.get("font"):
                f.name = o.get("font", font)
    return tb


def bullets(s, x, y, w, h, items, size=13, color=GRAY, spacing=6, bullet_color=BLUE):
    tb = text(s, x, y, w, h, [""], size=size, color=color)
    tf = tb.text_frame
    tf.paragraphs[0]._p.getparent().remove(tf.paragraphs[0]._p)
    for it in items:
        p = tf.add_paragraph()
        p.space_after = Pt(spacing)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.2))))
        pPr.set("indent", str(-Emu(Inches(0.2))))
        buClr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(buClr, qn("a:srgbClr")).set("val", str(bullet_color))
        etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
        etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        runs = it if isinstance(it, list) else [(it, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", False)
            r.font.color.rgb = o.get("color", color)
    return tb


def arrow(s, x1, y1, x2, y2, color=GRAY, width=1.5):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    st = c._element.find(qn("p:style"))
    if st is not None:
        c._element.remove(st)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def badge(s, x, y, d, label, fill=BLUE, size=13):
    c = box(s, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = WHITE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return c


def stat(s, x, y, w, value, label, color=BLUE, vsize=30, lsize=11):
    text(s, x, y, w, 0.55, [[(value, {"size": vsize, "bold": True, "color": color})]])
    text(s, x, y + 0.55, w, 0.6, label, size=lsize, color=GRAY)


def node(s, x, y, w, h, title, sub=None, novel=False, size=11):
    b = box(s, x, y, w, h, fill=TINT if novel else LIGHT, line=BLUE if novel else LINE, radius=0.12)
    paras = [[(title, {"bold": True, "color": NAVY if not novel else BLUE, "size": size})]]
    if sub:
        paras.append([(sub, {"size": size - 2, "color": GRAY})])
    text(s, x + 0.06, y + 0.04, w - 0.12, h - 0.08, paras, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=1)
    return b


def style_chart(ch, legend=True):
    ch.font.size = Pt(10)
    ch.font.name = "Arial"
    ch.font.color.rgb = GRAY
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(10)
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(10)
    ca = ch.category_axis
    ca.format.line.color.rgb = LINE
    ca.tick_labels.font.size = Pt(10)


def table(s, x, y, w, h, rows, col_w, header_fill=NAVY, size=11, highlight_row=None, colors=None):
    gt = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    t = gt.table
    tblPr = t._tbl.tblPr
    for k in ("bandRow", "firstRow"):
        tblPr.set(k, "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid"
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = header_fill
            elif highlight_row is not None and i == highlight_row:
                c.fill.fore_color.rgb = TINT
            else:
                c.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = ""
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = "Arial"
            r.font.bold = i == 0 or (highlight_row is not None and i == highlight_row and j == 0)
            r.font.color.rgb = WHITE if i == 0 else NAVY
            if colors and (i, j) in colors:
                r.font.color.rgb = colors[(i, j)]
                r.font.bold = True
    return gt


# ================================================================== 1. Title
s = prs.slides.add_slide(L["TITLE"])
s.shapes.title.text = "RealFact"
tp = s.shapes.title.text_frame.paragraphs[0]
tp.runs[0].font.size = Pt(54)
tp.runs[0].font.bold = True
tp.runs[0].font.color.rgb = NAVY
p2 = s.shapes.title.text_frame.add_paragraph()
p2.alignment = PP_ALIGN.CENTER
r = p2.add_run()
r.text = "OpenMP-Accelerated, Independence-Aware Real-Time News Verification"
r.font.size = Pt(20)
r.font.color.rgb = BLUE
sub = s.placeholders[1]
sub.text_frame.text = ""
names = "Adithya Shyam D (241IT005)   ·   Krishna (241IT039)   ·   P Suman (241IT049)   ·   Saurabh (241IT072)"
sp = sub.text_frame.paragraphs[0]
sp.alignment = PP_ALIGN.CENTER
rr = sp.add_run(); rr.text = names; rr.font.size = Pt(13); rr.font.color.rgb = NAVY; rr.font.bold = True
sp2 = sub.text_frame.add_paragraph(); sp2.alignment = PP_ALIGN.CENTER
rr = sp2.add_run(); rr.text = "IT 301 Parallel Computing  ·  Department of Information Technology, NIT Karnataka, Surathkal"
rr.font.size = Pt(12); rr.font.color.rgb = GRAY
# small verdict chips as a visual signature
chips = [("Supported", GREEN), ("Contradicted", RED), ("Conflicting Evidence", AMBER), ("Unverified", GRAY)]
cx = 1.55
for lab, col in chips:
    w = 0.32 + 0.085 * len(lab)
    b = box(s, cx, 4.45, w, 0.36, fill=WHITE, line=col, radius=0.5)
    text(s, cx, 4.45, w, 0.36, [[(lab, {"bold": True, "color": col, "size": 11})]], align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    cx += w + 0.22
s.notes_slide.notes_text_frame.text = (
    "Good morning. Our project is RealFact: a system that checks whether a news claim is supported by live news, "
    "and does the heavy similarity work in parallel with OpenMP. The four coloured labels are the four answers it can "
    "give: Supported, Contradicted, Conflicting Evidence, or Unverified.")

# ================================================================== 2. Problem statement
s = title_slide("Problem Statement")
text(s, 0.45, 1.2, 9.1, 0.6, [[("Checking a claim means comparing it against thousands of live articles in real time, "
                                "and many of those “confirmations” are copies of a single wire story.",
                                {"size": 15, "color": NAVY})]])
cards = [
    ("1", "What is broken today", ["Style classifiers judge the tone, not the truth",
                                   "Evidence systems count every copy as a new source",
                                   "Speed and scalability are never measured"]),
    ("2", "Stakeholders", ["Readers checking what they see online",
                                       "Journalists and fact-checking desks",
                                       "Platforms and content moderators"]),
    ("3", "How they benefit", ["A cited verdict in about 0.1 s",
                               "Each independent source counted once",
                               "Honest “Unverified” and “Conflicting” answers"]),
]
for i, (num, head, items) in enumerate(cards):
    x = 0.45 + i * 3.08
    box(s, x, 1.9, 2.88, 2.4, fill=LIGHT)
    badge(s, x + 0.18, 2.05, 0.42, num)
    text(s, x + 0.72, 2.05, 2.05, 0.42, [[(head, {"bold": True, "color": NAVY, "size": 14})]], anchor=MSO_ANCHOR.MIDDLE)
    bullets(s, x + 0.2, 2.62, 2.55, 1.6, items, size=12, spacing=5)
stats = [("13.1%", "of live articles have a near-copy"), ("23", "copies of one story across 10 outlets"),
         ("6.6M", "article pairs to compare")]
for i, (v, lab) in enumerate(stats):
    stat(s, 0.55 + i * 3.08, 4.4, 2.7, v, lab, vsize=22, lsize=11)
s.notes_slide.notes_text_frame.text = (
    "The problem has two parts. First, verifying a claim against live news is a large similarity-search problem: every "
    "claim is compared with every article. Second, outlets republish the same wire report, so ten 'confirmations' can be "
    "one story. In our own live corpus, 13 percent of articles have a near-copy, and one story appeared 23 times across "
    "10 outlets. The stakeholders are readers, journalists and fact-checkers, and platforms. They get a cited answer in "
    "about a tenth of a second, independent sources are counted once, and the system says 'I don't know' when it should.")

# ================================================================== 3. Methodology block diagram
s = title_slide("Methodology Block Diagram")
text(s, 0.45, 1.13, 1.2, 0.3, [[("BUILD THE INDEX", {"bold": True, "color": GRAY, "size": 10})]])
node(s, 0.45, 1.45, 1.55, 0.62, "Live RSS", "42 feeds, every 20 min")
node(s, 0.45, 2.17, 1.55, 0.62, "Web search", "Google News + Wikipedia")
node(s, 2.35, 1.8, 1.55, 0.62, "Clean + dedupe", "English, exact-copy hash")
node(s, 4.25, 1.8, 1.55, 0.62, "Embed (BGE)", "384-d vectors")
node(s, 6.15, 1.8, 1.55, 0.62, "Embedding matrix", "N × 384, in memory")
node(s, 8.05, 1.8, 1.5, 0.62, "Kernel 2", "syndication map", novel=True)
arrow(s, 2.0, 1.76, 2.35, 2.05); arrow(s, 2.0, 2.48, 2.35, 2.17)
arrow(s, 3.9, 2.11, 4.25, 2.11); arrow(s, 5.8, 2.11, 6.15, 2.11); arrow(s, 7.7, 2.11, 8.05, 2.11)
text(s, 0.45, 2.98, 1.6, 0.3, [[("VERIFY A CLAIM", {"bold": True, "color": GRAY, "size": 10})]])
row = [("Claim", "embed the text", False), ("Kernel 1", "OpenMP scan, top-M", True),
       ("Kernel 3", "cluster duplicates", True), ("NLI model", "support / contradict", False),
       ("Weighted vote", "w = 1 + λ·log₂n", True), ("Verdict", "4-way, with sources", False)]
xs = [0.45 + i * 1.55 for i in range(6)]
for (t, sub_, nov), x in zip(row, xs):
    node(s, x, 3.3, 1.32, 0.72, t, sub_, novel=nov, size=11)
for x in xs[:-1]:
    arrow(s, x + 1.32, 3.66, x + 1.55, 3.66)
for (a, b, c, d, arr) in ((6.92, 2.42, 6.92, 2.95, False), (6.92, 2.95, 2.66, 2.95, False), (2.66, 2.95, 2.66, 3.3, True)):
    if arr:
        arrow(s, a, b, c, d, color=BLUE, width=1.25)
    else:
        ln_ = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(a), Inches(b), Inches(c), Inches(d))
        st_ = ln_._element.find(qn("p:style"))
        if st_ is not None:
            ln_._element.remove(st_)
        ln_.line.color.rgb = BLUE
        ln_.line.width = Pt(1.25)
text(s, 3.4, 2.66, 3.4, 0.26, [[("every claim scans the whole matrix", {"italic": True, "color": BLUE, "size": 10})]],
     align=PP_ALIGN.CENTER)
box(s, 0.45, 4.25, 0.22, 0.22, fill=TINT, line=BLUE, radius=0.2)
text(s, 0.78, 4.23, 8.5, 0.3, "Blue = this project’s contribution (three C++/OpenMP kernels + weighted voting)", size=11)
text(s, 0.45, 4.62, 9.1, 0.5, [[("Key idea: ", {"bold": True, "color": NAVY}),
                                 ("retrieve wide (10,000 candidates), group copies of the same story, then let each "
                                  "story vote with a weight that grows only logarithmically with its copies.", {})]],
     size=12)
s.notes_slide.notes_text_frame.text = (
    "This is the whole system. The top row builds the evidence index: articles from 42 RSS feeds, and optionally a live "
    "web search for the claim, are cleaned, de-duplicated, embedded into 384-number vectors and kept in one big matrix. "
    "The bottom row verifies a claim. Kernel 1 scans the entire matrix in parallel to find the 10,000 most similar "
    "articles. Kernel 3 groups copies of the same story. An NLI model reads one article per story and says support, "
    "contradict or neutral. Then each story votes, with weight one plus half log-two of its number of copies, and we "
    "return one of four verdicts with sources. Kernel 2 runs over the whole corpus and finds all syndicated copies.")

# ================================================================== 4. Differences from existing solutions
s = title_slide("How RealFact Differs from Existing Solutions")
rows = [["", "Existing (FEVER-style pipelines)", "RealFact"],
        ["Evidence", "Static Wikipedia dump", "Live news + web search"],
        ["Duplicate copies", "Each copy is one vote", "One story, weighted vote"],
        ["Verdicts", "True / false (3-way)", "4-way, every source cited"],
        ["Conflicting evidence", "Folded into “not enough info”", "Its own verdict"],
        ["Speed reported?", "Accuracy only", "Speedup, efficiency, load balance"]]
table(s, 0.45, 1.25, 5.6, 3.3, rows, [1.55, 2.0, 2.05], size=11,
      colors={(i, 2): BLUE for i in range(1, 6)})
cd = CategoryChartData()
cd.categories = ["1", "2", "4", "8", "16", "32"]
cd.add_series("Naive count", (1, 2, 4, 8, 16, 32))
cd.add_series("RealFact (λ = 0.5)", (1, 1.5, 2, 2.5, 3, 3.5))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(6.3), Inches(1.2), Inches(3.3), Inches(3.0), cd)
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Votes for n copies of a story"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
for ser, col in zip(ch.series, (RED, BLUE)):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(2.25)
    ser.marker.style = XL_MARKER_STYLE.CIRCLE
    ser.marker.size = 5
    ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = col
    ser.smooth = False
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "copies n"
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
text(s, 6.3, 4.3, 3.3, 0.6, "16 copies of one report count as 3 votes, not 16", size=11, color=NAVY, align=PP_ALIGN.CENTER)
text(s, 0.45, 4.75, 9.1, 0.3, [[("Reference: ", {"bold": True, "color": NAVY}),
                                ("Guo, Schlichtkrull & Vlachos, “A Survey on Automated Fact-Checking,” TACL 2022", {})]],
     size=10)
s.notes_slide.notes_text_frame.text = (
    "Compared with the standard FEVER-style pipeline from our reference survey, RealFact differs in five ways. It uses "
    "live news instead of a frozen Wikipedia dump. It groups copies of one story instead of counting each copy as a "
    "vote. It has four verdicts and cites every source. Conflicting evidence is a verdict of its own, which the survey "
    "lists as an open problem. And we measure speed and scalability, which the surveyed systems do not report. The "
    "chart shows the weighting: naive counting grows linearly with copies, ours grows logarithmically, so 16 copies "
    "count as 3 votes.")

# ================================================================== 5. Parallel design
s = title_slide("Parallel Design: Three OpenMP Kernels")
kern = [
    ("1", "Exact similarity scan", "Compares the claim with every article and keeps the top 10,000",
     ["parallel for, schedule(runtime)", "Private top-M heap per thread, padded to 64 B (no false sharing)",
      "16 independent sums per dot product (ILP)"], "Memory-bound"),
    ("2", "All-pairs syndication", "Compares every article with every other one and groups copies",
     ["Triangular loop: early rows carry more work", "dot4 register blocking, lock-free edge buffers",
      "dynamic vs static scheduling decides balance"], "Compute-bound"),
    ("3", "Duplicate clustering", "Groups the retrieved articles into independent stories",
     ["The algorithm is sequential by definition", "Block speculation: parallel similarity, short serial pass",
      "Output identical to the sequential algorithm"], "Sequential → parallel"),
]
for i, (num, head, what, items, tag) in enumerate(kern):
    x = 0.45 + i * 3.08
    box(s, x, 1.25, 2.88, 3.2, fill=LIGHT)
    badge(s, x + 0.18, 1.4, 0.42, num)
    text(s, x + 0.72, 1.4, 2.05, 0.42, [[(head, {"bold": True, "color": NAVY, "size": 14})]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.2, 1.95, 2.5, 0.6, what, size=11, color=GRAY)
    bullets(s, x + 0.2, 2.6, 2.55, 1.35, items, size=11, spacing=4)
    tg = box(s, x + 0.2, 3.98, 1.75, 0.3, fill=TINT, radius=0.5)
    text(s, x + 0.2, 3.98, 1.75, 0.3, [[(tag, {"bold": True, "color": BLUE, "size": 10})]], align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.45, 4.6, 9.1, 0.5, fill=TINT, radius=0.3)
text(s, 0.65, 4.6, 8.7, 0.5, [[("Correctness: ", {"bold": True, "color": NAVY}),
                               ("each kernel is compiled twice (with and without OpenMP); every parallel run must return "
                                "results bit-identical to the serial build. 14 automated checks.", {})]],
     size=11, anchor=MSO_ANCHOR.MIDDLE)
s.notes_slide.notes_text_frame.text = (
    "The parallel computing part has three kernels, and each shows a different lesson from the course. Kernel 1 is the "
    "scan: a parallel for over all articles, with a private heap per thread padded to a cache line so threads don't "
    "fight over memory. It is limited by memory bandwidth. Kernel 2 compares every pair of articles. The work forms a "
    "triangle, so the first rows have far more work than the last; that makes the OpenMP schedule matter. It is limited "
    "by computation. Kernel 3 parallelises an algorithm that is sequential by nature, using block speculation, and still "
    "gives exactly the same answer. Every kernel is checked against a serial build, bit for bit.")

# ================================================================== 6. Results 1: parallel performance
s = title_slide("Results 1: Parallel Speedup")
cd = CategoryChartData()
cd.categories = ["1", "2", "4", "8", "11"]
series = [("static", (1.01, 1.33, 2.22, 4.06, 4.98), GRAY), ("static,1", (1.00, 1.88, 3.64, 5.08, 5.56), AMBER),
          ("guided", (1.00, 1.97, 3.82, 5.71, 6.69), GREEN), ("dynamic", (1.00, 1.90, 3.70, 5.86, 6.92), BLUE)]
for name, vals, _ in series:
    cd.add_series(name, vals)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.35), Inches(1.15), Inches(5.3), Inches(3.55), cd)
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Kernel 2 speedup vs threads (live corpus, 6.6M pairs)"
rr = ch.chart_title.text_frame.paragraphs[0].runs[0]; rr.font.size = Pt(11); rr.font.bold = True; rr.font.color.rgb = NAVY
for ser, (_, _, col) in zip(ch.series, series):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(2.25)
    ser.marker.style = XL_MARKER_STYLE.CIRCLE
    ser.marker.size = 5
    ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = col
    ser.smooth = False
last = ch.series[3].points[4]
last.data_label.has_text_frame = True
last.data_label.text_frame.text = "6.92×"
last.data_label.position = XL_LABEL_POSITION.ABOVE
last.data_label.text_frame.paragraphs[0].runs[0].font.bold = True
last.data_label.text_frame.paragraphs[0].runs[0].font.color.rgb = BLUE
ch.value_axis.maximum_scale = 8
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "threads"
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
stats = [("6.92×", "all-pairs kernel, dynamic, 11 threads (static only 4.98×)"),
         ("4.61×", "similarity scan at 500K articles, 16 threads"),
         ("2.6×", "faster serial scan after the ILP rewrite (96 GB/s ceiling)"),
         ("5.8×", "parallel clustering, identical output")]
for i, (v, lab) in enumerate(stats):
    y = 1.2 + i * 0.86
    text(s, 5.95, y, 1.25, 0.6, [[(v, {"size": 24, "bold": True, "color": BLUE})]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.2, y, 2.4, 0.6, lab, size=11, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.45, 4.78, 9.1, 0.4, [[("Why static loses: ", {"bold": True, "color": NAVY}),
                                ("each thread gets one block of rows, and early rows have the most pairs, so the "
                                 "busiest thread does 1.9× the average work.", {})]], size=11)
s.notes_slide.notes_text_frame.text = (
    "Here are the parallel results on our Apple M3 Pro with 11 cores. The chart is kernel 2 on the live corpus. Dynamic "
    "and guided scheduling reach almost 7 times speedup; plain static stops at 5, because with contiguous blocks the "
    "first thread gets the rows with the most pairs and does 1.9 times the average work. On the right: the similarity "
    "scan reached 4.6 times at half a million articles; rewriting the dot product for instruction-level parallelism "
    "made the serial version 2.6 times faster and moved the kernel to the memory-bandwidth ceiling of about 96 GB/s; "
    "and the parallel clustering reached 5.8 times with exactly the same output. The ceiling near 7 times is the "
    "hardware: 6 of the 11 cores are slower efficiency cores.")

# ================================================================== 7. Results 2: verification quality
s = title_slide("Results 2: Verification Accuracy")
cd = CategoryChartData()
cd.categories = ["Small NLI model", "FEVER-trained NLI", "+ sentence evidence,\ntuned on held-out claims"]
cd.add_series("Accuracy (%)", (58.7, 74.3, 81.0))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.35), Inches(1.15), Inches(5.3), Inches(3.45), cd)
ch = gf.chart
style_chart(ch, legend=False)
ch.has_title = True
ch.chart_title.text_frame.text = "Accuracy on 179 FEVER test claims (25,324 Wikipedia pages)"
rr = ch.chart_title.text_frame.paragraphs[0].runs[0]; rr.font.size = Pt(11); rr.font.bold = True; rr.font.color.rgb = NAVY
ch.value_axis.maximum_scale = 100
ch.value_axis.minimum_scale = 0
plot = ch.plots[0]
plot.gap_width = 70
plot.has_data_labels = True
plot.data_labels.number_format = '0.0"%"'
plot.data_labels.number_format_is_linked = False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size = Pt(12)
plot.data_labels.font.bold = True
plot.data_labels.font.color.rgb = NAVY
for i, col in enumerate((LINE, RGBColor(0x9D, 0xB2, 0xEA), BLUE)):
    pt = ch.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
stats = [("81.0%", "accuracy (macro F1 0.807)"), ("92.4%", "correct page ranked first"),
         ("114 ms", "per claim, end to end")]
for i, (v, lab) in enumerate(stats):
    stat(s, 6.0, 1.15 + i * 0.85, 3.5, v, lab, vsize=26)
box(s, 5.95, 3.75, 3.6, 0.85, fill=LIGHT)
text(s, 6.1, 3.8, 3.35, 0.75, [[("Fair tuning: ", {"bold": True, "color": NAVY}),
                                 ("settings chosen on 180 other claims; the 179 test claims were scored once.", {})]],
     size=11, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.45, 4.78, 9.1, 0.4, [[("LIAR (2007–2016 claims vs today’s news): ", {"bold": True, "color": NAVY}),
                                ("99% correctly answered “Unverified”. That test measures corpus coverage, not reasoning.", {})]],
     size=11)
s.notes_slide.notes_text_frame.text = (
    "To measure accuracy we need claims whose evidence is in the corpus, so we built a FEVER demo corpus: 179 test "
    "claims against 25,000 Wikipedia pages. With a small NLI model we got 59 percent; with a model trained on FEVER, "
    "74; after checking evidence sentence by sentence and tuning, 81 percent, with the correct page ranked first 92 "
    "percent of the time. Importantly, all tuning was done on 180 different claims, and the test claims were scored "
    "only once, so the number is not inflated. On LIAR, whose claims are from 2007 to 2016, the system correctly says "
    "'Unverified' for almost everything, because today's news simply doesn't cover them.")

# ================================================================== 8. Results 3: comparison + metrics
s = title_slide("Results 3: Comparison and Metric Adequacy")
text(s, 0.45, 1.15, 5.0, 0.35, [[("Same evidence, one story copied ×10", {"bold": True, "color": NAVY, "size": 13})]])
rows = [["Aggregation method", "Verdict", "Votes"],
        ["RealFact (log-weighted)", "Conflicting", "6.2"],
        ["One vote per story", "Conflicting", "4.0"],
        ["Naive counting", "Contradicted", "15.0"],
        ["Plain top-K (FEVER-style)", "Contradicted", "10.0"]]
table(s, 0.45, 1.55, 5.0, 2.2, rows, [2.6, 1.5, 0.9], size=11, highlight_row=1,
      colors={(1, 1): AMBER, (2, 1): AMBER, (3, 1): RED, (4, 1): RED})
text(s, 0.45, 3.85, 5.0, 0.95, "The claim had genuinely mixed sources. Copying one contradicting article 10 times flips "
                               "naive and FEVER-style counting to “Contradicted”; RealFact still reports the "
                               "real disagreement.", size=11)
text(s, 5.85, 1.15, 3.7, 0.35, [[("Are our metrics good enough?", {"bold": True, "color": NAVY, "size": 13})]])
checks = [("✓", GREEN, "Speed", "speedup, efficiency, Karp–Flatt, load balance, Wilcoxon test"),
          ("✓", GREEN, "Correctness", "bit-identical to serial in every configuration"),
          ("✓", GREEN, "Accuracy", "accuracy, macro F1, Recall@K on a held-out split"),
          ("!", AMBER, "Limit", "no labelled live-news dataset; FEVER set is curated")]
for i, (mk, col, head, desc) in enumerate(checks):
    y = 1.6 + i * 0.8
    badge(s, 5.85, y + 0.04, 0.36, mk, fill=col, size=12)
    text(s, 6.35, y, 3.2, 0.75, [[(head, {"bold": True, "color": NAVY, "size": 12})], [(desc, {"size": 11})]],
         spacing=1)
s.notes_slide.notes_text_frame.text = (
    "This slide compares our approach with existing aggregation methods on exactly the same evidence. The claim had "
    "mixed evidence: one supporting and one contradicting source. When we copy the contradicting article ten times, "
    "naive counting and the plain top-K approach used by FEVER-style systems both flip to 'Contradicted'. RealFact "
    "counts the eleven copies as 2.7 votes and correctly still reports conflicting evidence. On whether our metrics "
    "are adequate: for speed we use the standard parallel metrics plus a significance test; for correctness every run "
    "is compared with the serial build; for accuracy we report accuracy, F1 and recall on a held-out split. The honest "
    "limit is that no labelled dataset exists for today's news.")

# ================================================================== 9. Achievements
s = title_slide("What We Achieved")
pic = s.shapes.add_picture(f"{FIG}/ui_verdict.png", Inches(0.45), Inches(1.25), width=Inches(5.6))
pic.line.color.rgb = LINE
pic.line.width = Pt(0.75)
text(s, 0.45, 1.25 + 5.6 * 956 / 2400 + 0.08, 5.6, 0.3, "The running web app: verdict, timed pipeline, OpenMP vs serial check",
     size=10, color=GRAY, align=PP_ALIGN.CENTER)
for i, (v, lab) in enumerate([("42", "news feeds, refreshed every 20 min"), ("~0.1 s", "per verified claim"),
                               ("14", "automated correctness checks")]):
    stat(s, 0.55 + i * 1.9, 3.85, 1.75, v, lab, vsize=24, lsize=10)
items = [[("Working system: ", {"bold": True, "color": NAVY}), ("web app with verify, benchmark, corpus, evaluation and history views", {})],
         [("Live evidence: ", {"bold": True, "color": NAVY}), ("42 RSS feeds every 20 min, plus on-demand web search", {})],
         [("Practical use: ", {"bold": True, "color": NAVY}), ("newsroom fact-check desk, classroom media literacy, platform moderation", {})],
         [("Deployment: ", {"bold": True, "color": NAVY}), ("runs locally on a laptop; open source on GitHub", {})],
         [("Deliverables: ", {"bold": True, "color": NAVY}), ("IEEE-format report, 14 automated tests, reproducible benchmarks", {})]]
bullets(s, 6.35, 1.25, 3.25, 3.6, items, size=11, spacing=7)
s.notes_slide.notes_text_frame.text = (
    "What we achieved: a complete working system, not just a benchmark. The screenshot is the real app: it shows the "
    "verdict, the time spent in each stage, and confirms the OpenMP result matches the serial build. It ingests 42 news "
    "feeds every 20 minutes and can search the web for a specific claim. Practical uses are a newsroom fact-check desk, "
    "media-literacy teaching, and platform moderation. It runs locally on a laptop and the code is open source on "
    "GitHub, with an IEEE-format report and automated tests.")

# ================================================================== 10. Conclusion
s = title_slide("Conclusion")
take = [("1", "Count stories, not copies",
         "Grouping syndicated copies and weighting them logarithmically keeps one wire story from outvoting independent sources."),
        ("2", "Three parallel regimes, measured",
         "Memory-bound scan (bandwidth ceiling), compute-bound all-pairs (6.92×, schedule matters), sequential clustering made parallel."),
        ("3", "Fast, correct and honest",
         "About 0.1 s per claim, bit-identical to serial, 81.0% on held-out FEVER claims, and “Unverified” when evidence is missing.")]
for i, (num, head, desc) in enumerate(take):
    y = 1.25 + i * 0.95
    badge(s, 0.45, y + 0.05, 0.45, num)
    text(s, 1.1, y, 4.9, 0.85, [[(head, {"bold": True, "color": NAVY, "size": 14})], [(desc, {"size": 11})]], spacing=2)
box(s, 6.35, 1.25, 3.2, 2.8, fill=LIGHT)
text(s, 6.55, 1.38, 2.9, 0.35, [[("Future work", {"bold": True, "color": NAVY, "size": 14})]])
bullets(s, 6.55, 1.82, 2.85, 2.2, ["Index the full FEVER dump and FakeNewsNet",
                                   "Run on a pinned multi-socket server",
                                   "float16 / int8 vectors to lift the bandwidth ceiling",
                                   "Generated explanations from cited sources"], size=11, spacing=6)
text(s, 0.45, 4.3, 9.1, 0.6, [[("Research question answered: ", {"bold": True, "color": BLUE}),
                                ("multi-source evidence retrieval runs efficiently on a multicore CPU with OpenMP, "
                                 "without counting duplicates as independent evidence.", {"color": NAVY})]], size=12)
s.notes_slide.notes_text_frame.text = (
    "To conclude, three takeaways. One: counting stories rather than copies matters; it changes verdicts exactly when "
    "syndication is present. Two: our three kernels show three regimes from the course, a memory-bound scan, a "
    "compute-bound all-pairs kernel where scheduling decides the speedup, and a sequential algorithm made parallel. "
    "Three: the system is fast, provably correct against the serial build, and honest about what it doesn't know. "
    "Future work is a bigger evidence corpus, a server-class machine, compressed vectors, and generated explanations. "
    "So our research question is answered: yes, this runs efficiently on a multicore CPU without counting duplicates.")

# ================================================================== 11. Thank you
s = prs.slides.add_slide(L["TITLE"])
s.shapes.title.text = "Thank You"
tr = s.shapes.title.text_frame.paragraphs[0].runs[0]
tr.font.size = Pt(48); tr.font.bold = True; tr.font.color.rgb = NAVY
sub = s.placeholders[1]
sub.text_frame.text = ""
p = sub.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Questions?"; r.font.size = Pt(20); r.font.color.rgb = BLUE
p = sub.text_frame.add_paragraph(); p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "github.com/Adi-Shyam/RealFact"; r.font.size = Pt(13); r.font.color.rgb = GRAY
s.notes_slide.notes_text_frame.text = "Thank you. We are happy to take questions, or to run a live demo of the system."

# ------------------------------------------------------------------ drop the template's placeholder slides
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst)[:n_original]:
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)
prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
