from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SRC = Path(r"D:\Cross-modal conversion\output\BridgeRefine_Chinese_manuscript_raw.docx")
OUT = Path(r"D:\Cross-modal conversion\output\BridgeRefine_Chinese_manuscript.docx")
doc = Document(SRC)

def set_font(run, east="宋体", latin="Times New Roman", size=None, bold=None):
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rf = rpr.rFonts
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
    rf.set(qn("w:ascii"), latin); rf.set(qn("w:hAnsi"), latin); rf.set(qn("w:eastAsia"), east)
    if size: run.font.size = Pt(size)
    if bold is not None: run.bold = bold

sec = doc.sections[0]
sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.2)
sec.left_margin = Cm(2.35); sec.right_margin = Cm(2.35)
sec.header_distance = Cm(1.0); sec.footer_distance = Cm(1.0)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"; normal.font.size = Pt(10.5)
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
normal.paragraph_format.first_line_indent = Cm(0.74)
normal.paragraph_format.line_spacing = 1.25
normal.paragraph_format.space_after = Pt(5)

for name, size, before, after in [("Title",18,0,12),("Heading 1",15,14,7),("Heading 2",12.5,10,5),("Heading 3",11,8,4)]:
    if name not in doc.styles: continue
    s=doc.styles[name]; s.font.name="Times New Roman"; s.font.size=Pt(size); s.font.bold=True; s.font.color.rgb=RGBColor(0,0,0)
    s._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    s.paragraph_format.space_before=Pt(before); s.paragraph_format.space_after=Pt(after); s.paragraph_format.keep_with_next=True
    if name=="Title": s.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER

for p in doc.paragraphs:
    for run in p.runs: set_font(run)
    if p.style.name == "Title":
        p.paragraph_format.first_line_indent = None
        for run in p.runs: set_font(run,east="黑体",size=18,bold=True)
    if p.style.name.startswith("Heading"):
        p.paragraph_format.first_line_indent = None
        for run in p.runs: set_font(run,east="黑体",size=p.style.font.size.pt if p.style.font.size else 11,bold=True)
    if p.style.name == "Caption":
        p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.first_line_indent=None
        p.paragraph_format.space_before=Pt(3); p.paragraph_format.space_after=Pt(8)
        for run in p.runs: set_font(run,east="宋体",size=9)

# Quiet running header and page number.
header=sec.header.paragraphs[0]; header.text="BridgeRefine：基于扩散桥与监督精炼的脑 CT 到 MRI 转换"
header.alignment=WD_ALIGN_PARAGRAPH.LEFT
for r in header.runs: set_font(r,east="宋体",size=8.5)
footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=footer.add_run(); fld=OxmlElement("w:fldSimple"); fld.set(qn("w:instr"),"PAGE"); r._r.addnext(fld)

for table in doc.tables:
    table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=True
    for ri,row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr=cell._tc.get_or_add_tcPr()
            mar=tcPr.first_child_found_in("w:tcMar")
            if mar is None: mar=OxmlElement("w:tcMar"); tcPr.append(mar)
            for side in ("top","bottom","start","end"):
                node=mar.find(qn(f"w:{side}"))
                if node is None: node=OxmlElement(f"w:{side}"); mar.append(node)
                node.set(qn("w:w"),"90" if side in ("top","bottom") else "110"); node.set(qn("w:type"),"dxa")
            if ri==0:
                shd=tcPr.find(qn("w:shd"))
                if shd is None: shd=OxmlElement("w:shd"); tcPr.append(shd)
                shd.set(qn("w:fill"),"E8EEF5")
            for p in cell.paragraphs:
                p.paragraph_format.first_line_indent=None; p.paragraph_format.space_after=Pt(2); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs: set_font(run,east="宋体",size=8.5,bold=(ri==0))

# Keep images centered and within usable page width.
for p in doc.paragraphs:
    if p._p.xpath('.//a:blip'):
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=None
        p.paragraph_format.space_before=Pt(5); p.paragraph_format.space_after=Pt(3)

props=doc.core_properties
props.title="BridgeRefine：基于扩散桥与监督精炼的脑 CT 到 MRI 转换"
props.author=""; props.last_modified_by=""; props.comments="事实锁定中文稿"
doc.save(OUT)
print(OUT)
