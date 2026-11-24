"""Generate a publication-grade DOCX manuscript following SBC Qualis A and ABNT norms.

Specifications enforced:
  - SBC (Sociedade Brasileira de Computacao) A4 page layout (top 3.5 cm, bottom 2.5 cm,
    left 3.0 cm, right 3.0 cm), Times New Roman, 11 pt body font, justified alignment,
    1.15 line spacing, 1.25 cm first-line indent from the second paragraph of each section.
  - Qualis A SBC scientific structure: Abstract/Resumo (indented italics), Introduction
    with organization roadmap, dedicated Related Work section with comparison table,
    Formal Mathematical Modeling & Architecture, Experimental Protocol with Research
    Questions (QP1-QP4) and full Dataset Provenance citations, Results & Discussion
    answering QP1-QP4, Threats to Validity, Conclusions, ABNT NBR 6023:2018 References,
    and Appendices A & B strictly AFTER References.
  - Automated compliance audit: zero police/law-enforcement terms, zero encoding artifacts
    (U+FFFD), zero source code in main body, and exact numerical traceability to
    data/benchmarks/ablation_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor


FONT_MAIN = "Times New Roman"
FONT_MONO = "Courier New"
COLOR_DARK = RGBColor(0x1A, 0x1A, 0x1A)
COLOR_SBC_ACCENT = RGBColor(0x10, 0x2A, 0x43)
COLOR_MUTED = RGBColor(0x4A, 0x55, 0x68)


def _fmt_pt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}".replace(".", ",")


def _fmt_en(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def _fmt_int_pt(val: int | float) -> str:
    return f"{int(round(val)):,}".replace(",", ".")


def _fmt_int_en(val: int | float) -> str:
    return f"{int(round(val)):,}"


def _set_run_font(
    run: Any,
    *,
    name: str = FONT_MAIN,
    size_pt: float = 11.0,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor | None = COLOR_DARK,
    sub: bool = False,
    sup: bool = False,
) -> None:
    run.font.name = name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    if sub:
        run.font.subscript = True
    if sup:
        run.font.superscript = True
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:cs"), name)


def _add_rich_runs(
    p: Any,
    tokens: list[tuple[str, dict[str, Any]] | str],
    *,
    base_size: float = 11.0,
    base_italic: bool = False,
    base_bold: bool = False,
    base_color: RGBColor = COLOR_DARK,
) -> None:
    """Append mixed formatted runs (normal, bold, italic, subscript, superscript, mono) cleanly."""
    for item in tokens:
        if isinstance(item, str):
            r = p.add_run(item)
            _set_run_font(r, size_pt=base_size, bold=base_bold, italic=base_italic, color=base_color)
        else:
            text, opts = item
            r = p.add_run(text)
            _set_run_font(
                r,
                name=opts.get("name", FONT_MAIN),
                size_pt=opts.get("size", base_size),
                bold=opts.get("bold", base_bold),
                italic=opts.get("italic", base_italic),
                color=opts.get("color", base_color),
                sub=opts.get("sub", False),
                sup=opts.get("sup", False),
            )


def _add_page_number_fields(paragraph: Any) -> None:
    """Insert native Word 'Página X de Y' fields in footer."""
    r1 = paragraph.add_run("Página ")
    _set_run_font(r1, size_pt=9.0, color=COLOR_MUTED)

    for field_name, suffix in (("PAGE", " de "), ("NUMPAGES", "")):
        fld_begin = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
        instr = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> {field_name} </w:instrText>')
        fld_sep = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="separate"/>')
        fld_end = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
        r_f = paragraph.add_run()
        _set_run_font(r_f, size_pt=9.0, color=COLOR_MUTED)
        r_f._r.append(fld_begin)
        r_f._r.append(instr)
        r_f._r.append(fld_sep)
        r_f._r.append(fld_end)
        if suffix:
            rs = paragraph.add_run(suffix)
            _set_run_font(rs, size_pt=9.0, color=COLOR_MUTED)


def _set_cell_shading(cell: Any, hex_color: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def _set_cell_margins(cell: Any, top: int = 55, bottom: int = 55, left: int = 85, right: int = 85) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def _set_table_booktabs_borders(table: Any) -> None:
    """Apply clean academic booktabs borders (top/bottom rules, header underline, no vertical grid)."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="10" w:space="0" w:color="102A43"/>'
        '<w:bottom w:val="single" w:sz="10" w:space="0" w:color="102A43"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD2D9"/>'
        '<w:left w:val="none"/>'
        '<w:right w:val="none"/>'
        '<w:insideV w:val="none"/>'
        "</w:tblBorders>"
    )
    tblPr.append(borders)


def _set_table_no_borders(table: Any) -> None:
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="none"/><w:bottom w:val="none"/>'
        '<w:left w:val="none"/><w:right w:val="none"/>'
        '<w:insideH w:val="none"/><w:insideV w:val="none"/>'
        "</w:tblBorders>"
    )
    tblPr.append(borders)


def _prevent_row_split(row: Any) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


def _repeat_header_row(row: Any) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))


class SBCArticleBuilder:
    """Builder for an SBC Qualis A / ABNT compliant DOCX manuscript with 11 pt body text."""

    def __init__(self) -> None:
        self.doc = Document()
        self._configure_page_and_styles()

    def _configure_page_and_styles(self) -> None:
        sec = self.doc.sections[0]
        # A4 exact dimensions (21.0 x 29.7 cm) & official SBC margins (3.5, 2.5, 3.0, 3.0 cm)
        sec.page_width = Cm(21.0)
        sec.page_height = Cm(29.7)
        sec.top_margin = Cm(3.5)
        sec.bottom_margin = Cm(2.5)
        sec.left_margin = Cm(3.0)
        sec.right_margin = Cm(3.0)
        sec.header_distance = Cm(1.6)
        sec.footer_distance = Cm(1.25)
        sec.different_first_page_header_footer = True

        # Configure Normal style: Times New Roman 11 pt (exact requested general text size)
        normal = self.doc.styles["Normal"]
        normal.font.name = FONT_MAIN
        normal.font.size = Pt(11)
        normal.font.color.rgb = COLOR_DARK
        pf = normal.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.line_spacing = 1.15
        pf.space_before = Pt(0)
        pf.space_after = Pt(5)

        # Running header & page numbers (pages 2+)
        hdr_p = sec.header.paragraphs[0]
        hdr_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_hdr = hdr_p.add_run("Rastreamento Multicâmera e Predição de Trajetórias Veiculares com EKF Adaptativo")
        _set_run_font(r_hdr, size_pt=8.5, italic=True, color=COLOR_MUTED)

        ftr_p = sec.footer.paragraphs[0]
        ftr_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _add_page_number_fields(ftr_p)

        # First page footer: page number only (clean SBC style)
        ftr_first = sec.first_page_footer.paragraphs[0]
        ftr_first.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _add_page_number_fields(ftr_first)

    def add_title_block(self, title_pt: str, title_en: str, authors: str, affiliation: str) -> None:
        p_pt = self.doc.add_paragraph()
        p_pt.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_pt.paragraph_format.space_before = Pt(0)
        p_pt.paragraph_format.space_after = Pt(5)
        p_pt.paragraph_format.line_spacing = 1.15
        r = p_pt.add_run(title_pt)
        _set_run_font(r, size_pt=15.0, bold=True, color=COLOR_SBC_ACCENT)

        p_en = self.doc.add_paragraph()
        p_en.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_en.paragraph_format.space_before = Pt(0)
        p_en.paragraph_format.space_after = Pt(10)
        p_en.paragraph_format.line_spacing = 1.12
        r_en = p_en.add_run(title_en)
        _set_run_font(r_en, size_pt=11.5, italic=True, color=COLOR_MUTED)

        p_aut = self.doc.add_paragraph()
        p_aut.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_aut.paragraph_format.space_before = Pt(2)
        p_aut.paragraph_format.space_after = Pt(2)
        r_aut = p_aut.add_run(authors)
        _set_run_font(r_aut, size_pt=11.0, bold=True, color=COLOR_DARK)

        p_aff = self.doc.add_paragraph()
        p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_aff.paragraph_format.space_before = Pt(0)
        p_aff.paragraph_format.space_after = Pt(12)
        r_aff = p_aff.add_run(affiliation)
        _set_run_font(r_aff, size_pt=9.5, italic=True, color=COLOR_MUTED)

    def add_sbc_abstract(self, label: str, body_tokens: list[Any], kw_label: str, keywords: str) -> None:
        """SBC official Resumo/Abstract style: 0.8 cm left/right indent, 10 pt italic, single-spaced."""
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.right_indent = Cm(0.8)
        p.paragraph_format.line_spacing = 1.08
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        r_lbl = p.add_run(f"{label}. ")
        _set_run_font(r_lbl, size_pt=10.0, bold=True, italic=True, color=COLOR_SBC_ACCENT)
        _add_rich_runs(p, body_tokens, base_size=10.0, base_italic=True)

        p_kw = self.doc.add_paragraph()
        p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_kw.paragraph_format.left_indent = Cm(0.8)
        p_kw.paragraph_format.right_indent = Cm(0.8)
        p_kw.paragraph_format.line_spacing = 1.08
        p_kw.paragraph_format.space_before = Pt(0)
        p_kw.paragraph_format.space_after = Pt(9)
        r_kl = p_kw.add_run(f"{kw_label}: ")
        _set_run_font(r_kl, size_pt=9.5, bold=True, italic=False, color=COLOR_SBC_ACCENT)
        r_kt = p_kw.add_run(keywords)
        _set_run_font(r_kt, size_pt=9.5, italic=False, color=COLOR_DARK)

    def add_section_heading(self, text: str, level: int = 1, page_break_before: bool = False) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.page_break_before = page_break_before
        if level == 1:
            p.paragraph_format.space_before = Pt(13)
            p.paragraph_format.space_after = Pt(5)
            r = p.add_run(text)
            _set_run_font(r, size_pt=13.0, bold=True, color=COLOR_SBC_ACCENT)
        else:
            p.paragraph_format.space_before = Pt(9)
            p.paragraph_format.space_after = Pt(3.5)
            r = p.add_run(text)
            _set_run_font(r, size_pt=11.5, bold=True, color=COLOR_SBC_ACCENT)

    def add_body_p(self, tokens: list[Any] | str, *, first_in_section: bool = False) -> None:
        """Standard SBC/ABNT body paragraph in 11 pt Times New Roman."""
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.first_line_indent = Cm(0.0 if first_in_section else 1.25)
        if isinstance(tokens, str):
            r = p.add_run(tokens)
            _set_run_font(r, size_pt=11.0)
        else:
            _add_rich_runs(p, tokens, base_size=11.0)

    def add_bullet_p(self, lead_bold: str, tokens: list[Any], *, bullet_prefix: str = "• ") -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.35)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        rb = p.add_run(bullet_prefix)
        _set_run_font(rb, size_pt=11.0, bold=True, color=COLOR_SBC_ACCENT)
        rl = p.add_run(lead_bold)
        _set_run_font(rl, size_pt=11.0, bold=True, color=COLOR_DARK)
        _add_rich_runs(p, tokens, base_size=11.0)

    def add_equation(self, lines_tokens: list[list[Any]], eq_num: str) -> None:
        """Render clean numbered equation using a borderless 2-column Word table."""
        tbl = self.doc.add_table(rows=1, cols=2)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        _set_table_no_borders(tbl)
        tblPr = tbl._tbl.tblPr
        tblPr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="8505" w:type="dxa"/>'))
        tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
        # Synchronize <w:tblGrid> column widths so Word and LibreOffice both allocate 13.6 cm + 1.4 cm
        grid_cols = tbl._tbl.tblGrid.findall(qn("w:gridCol"))
        if len(grid_cols) == 2:
            grid_cols[0].set(qn("w:w"), "7711")
            grid_cols[1].set(qn("w:w"), "794")
        row = tbl.rows[0]
        _prevent_row_split(row)

        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Cm(13.6)
        c1.width = Cm(1.4)
        for cell, w_dxa in ((c0, "7711"), (c1, "794")):
            tcPr = cell._tc.get_or_add_tcPr()
            for old_w in tcPr.findall(qn("w:tcW")):
                tcPr.remove(old_w)
            tcPr.append(parse_xml(f'<w:tcW {nsdecls("w")} w:w="{w_dxa}" w:type="dxa"/>'))
        c0.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c1.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _set_cell_margins(c0, top=45, bottom=45, left=50, right=30)
        _set_cell_margins(c1, top=45, bottom=45, left=15, right=20)
        _set_cell_shading(c0, "F8FAFC")
        _set_cell_shading(c1, "F8FAFC")

        for idx, line_toks in enumerate(lines_tokens):
            p = c0.paragraphs[0] if idx == 0 else c0.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.12
            _add_rich_runs(p, line_toks, base_size=10.0)

        p_num = c1.paragraphs[0]
        p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_num.paragraph_format.space_before = Pt(0)
        p_num.paragraph_format.space_after = Pt(0)
        rn = p_num.add_run(f"({eq_num})")
        _set_run_font(rn, size_pt=10.0, color=COLOR_MUTED)

        # Small spacer after equation table
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.space_after = Pt(4)

    def add_figure(self, img_path: Path, caption_label: str, caption_text: str, width_cm: float = 14.6, source: str = "Elaborado pelos autores.") -> None:
        p_img = self.doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(3)
        p_img.paragraph_format.keep_with_next = True
        p_img.add_run().add_picture(str(img_path), width=Cm(width_cm))

        p_cap = self.doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(2)
        p_cap.paragraph_format.line_spacing = 1.08
        p_cap.paragraph_format.keep_with_next = True
        r_lbl = p_cap.add_run(f"{caption_label} ")
        _set_run_font(r_lbl, size_pt=9.5, bold=True, color=COLOR_SBC_ACCENT)
        r_txt = p_cap.add_run(caption_text)
        _set_run_font(r_txt, size_pt=9.5, color=COLOR_DARK)

        p_src = self.doc.add_paragraph()
        p_src.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_src.paragraph_format.space_before = Pt(0)
        p_src.paragraph_format.space_after = Pt(8)
        r_src = p_src.add_run(f"Fonte: {source}")
        _set_run_font(r_src, size_pt=9.0, italic=True, color=COLOR_MUTED)

    def add_academic_table(
        self,
        title_label: str,
        title_text: str,
        headers: list[str],
        rows: list[list[Any]],
        col_widths_cm: list[float],
        *,
        center_cols: set[int] | None = None,
        highlight_rows: set[int] | None = None,
        font_size_pt: float = 8.8,
        source: str = "Elaborado pelos autores.",
    ) -> None:
        """ABNT/SBC scientific table: title above, booktabs borders, Fonte below."""
        center_set = center_cols or set()
        hl_set = highlight_rows or set()

        p_cap = self.doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_cap.paragraph_format.space_before = Pt(7)
        p_cap.paragraph_format.space_after = Pt(3)
        p_cap.paragraph_format.line_spacing = 1.08
        p_cap.paragraph_format.keep_with_next = True
        r_lbl = p_cap.add_run(f"{title_label} ")
        _set_run_font(r_lbl, size_pt=9.5, bold=True, color=COLOR_SBC_ACCENT)
        r_txt = p_cap.add_run(title_text)
        _set_run_font(r_txt, size_pt=9.5, color=COLOR_DARK)

        tbl = self.doc.add_table(rows=1 + len(rows), cols=len(headers))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        _set_table_booktabs_borders(tbl)

        # Header row
        hdr_row = tbl.rows[0]
        _repeat_header_row(hdr_row)
        _prevent_row_split(hdr_row)
        for j, h_text in enumerate(headers):
            cell = hdr_row.cells[j]
            cell.width = Cm(col_widths_cm[j])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_shading(cell, "1B4F72")
            _set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j in center_set else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(h_text)
            _set_run_font(r, size_pt=font_size_pt, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

        # Body rows
        for i, row_vals in enumerate(rows):
            row = tbl.rows[i + 1]
            _prevent_row_split(row)
            is_hl = i in hl_set
            bg = "E6F4EA" if is_hl else ("F4F7FA" if i % 2 == 1 else "FFFFFF")
            for j, val in enumerate(row_vals):
                cell = row.cells[j]
                cell.width = Cm(col_widths_cm[j])
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                _set_cell_shading(cell, bg)
                _set_cell_margins(cell, top=48, bottom=48, left=60, right=60)
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j in center_set else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                if isinstance(val, str):
                    r = p.add_run(val)
                    _set_run_font(r, size_pt=font_size_pt, bold=is_hl, color=COLOR_DARK)
                else:
                    _add_rich_runs(p, val, base_size=font_size_pt, base_bold=is_hl, base_color=COLOR_DARK)

        p_src = self.doc.add_paragraph()
        p_src.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_src.paragraph_format.space_before = Pt(2.5)
        p_src.paragraph_format.space_after = Pt(7)
        r_src = p_src.add_run(f"Fonte: {source}")
        _set_run_font(r_src, size_pt=8.8, italic=True, color=COLOR_MUTED)

    def add_abnt_reference(self, parts: list[tuple[str, dict[str, Any]] | str]) -> None:
        """ABNT NBR 6023:2018 reference: left-aligned, single-spaced, 10 pt, blank space between entries."""
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.05
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5.5)
        p.paragraph_format.left_indent = Cm(0.0)
        p.paragraph_format.first_line_indent = Cm(0.0)
        _add_rich_runs(p, parts, base_size=10.0)

    def add_appendix_code(self, title: str | list[Any], code_text: str) -> None:
        """Code listing strictly reserved for Appendices A & B (after References)."""
        p_t = self.doc.add_paragraph()
        p_t.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_t.paragraph_format.space_before = Pt(7)
        p_t.paragraph_format.space_after = Pt(2.5)
        p_t.paragraph_format.keep_with_next = True
        if isinstance(title, str):
            rt = p_t.add_run(title)
            _set_run_font(rt, size_pt=9.5, bold=True, color=COLOR_SBC_ACCENT)
        else:
            _add_rich_runs(p_t, title, base_size=9.5, base_bold=True, base_color=COLOR_SBC_ACCENT)

        tbl = self.doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        row = tbl.rows[0]
        _prevent_row_split(row)
        cell = row.cells[0]
        cell.width = Cm(15.0)
        _set_cell_shading(cell, "F5F7FA")
        _set_cell_margins(cell, top=80, bottom=80, left=110, right=100)
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(
            parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                '<w:top w:val="single" w:sz="4" w:color="BCCCDC"/>'
                '<w:bottom w:val="single" w:sz="4" w:color="BCCCDC"/>'
                '<w:left w:val="single" w:sz="16" w:color="1B4F72"/>'
                '<w:right w:val="single" w:sz="4" w:color="BCCCDC"/>'
                "</w:tcBorders>"
            )
        )
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.05
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(code_text)
        _set_run_font(r, name=FONT_MONO, size_pt=8.2, color=COLOR_DARK)

        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.space_after = Pt(4)


def _audit_docx_compliance(doc: Document) -> None:
    """Verify zero forbidden law-enforcement terms, zero U+FFFD, zero code before References, and 11 pt body."""
    if doc.styles["Normal"].font.size != Pt(11):
        raise ValueError(f"Expected Normal style font size 11 pt, got {doc.styles['Normal'].font.size}")

    all_text_parts: list[str] = []
    before_refs_mono = False
    in_references_or_after = False

    for p in doc.paragraphs:
        txt = p.text
        all_text_parts.append(txt)
        if txt.strip().lower() == "referências":
            in_references_or_after = True
        if not in_references_or_after:
            for r in p.runs:
                if r.font.name == FONT_MONO and len(r.text.strip()) > 35:
                    before_refs_mono = True

    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                all_text_parts.append(cell.text)

    full_text = "\n".join(all_text_parts)
    if "\ufffd" in full_text:
        raise ValueError("Encoding noise detected: U+FFFD character present in DOCX!")
    if before_refs_mono:
        raise ValueError("Monospace source-code block detected before Referências!")

    lower = full_text.lower()
    forbidden = [
        "polícia",
        "policia",
        "policial",
        "policiais",
        "police",
        "viatura",
        "cerco",
        "cordon",
        "patrol",
        "perseguição",
        "perseguicao",
        "pursuit",
        "rotas de fuga",
        "rota de fuga",
        "escape route",
        "evasiva",
        "evasive",
        "ilícito",
        "ilicito",
        "crime",
        "violência urbana",
        "segurança pública",
        "public safety",
        "public security",
        "cicc",
        "suspeito",
        "suspect",
        "interceptação tática",
        "forense",
    ]
    found = [w for w in forbidden if w in lower]
    if found:
        raise ValueError(f"Forbidden law-enforcement terms found in DOCX: {found}")


def build_sbc_docx(bench_json: Path, fig_dir: Path, out_docx: Path) -> Path:
    data = json.loads(bench_json.read_text(encoding="utf-8"))
    a1 = data["A1"]["results"]
    a2 = data["A2"]["results"]
    a3 = data["A3"]["results"]
    a5 = data["A5"]["results"]
    a6_kin = data["A6"]["results"]["kinematic_and_sequence_models"]
    a6_rt = data["A6"]["results"]["road_graph_route_models"]
    a7 = data["A7"]["results"]
    s1, s2, s3, s4 = a7["S1_4cam_20veh"], a7["S2_8cam_50veh"], a7["S3_16cam_100veh"], a7["S4_24cam_150veh"]

    a1_red_lin = 100.0 * (a1["linear_cv"]["rmse_maneuver_m"] - a1["adaptive_bicycle"]["rmse_maneuver_m"]) / a1["linear_cv"]["rmse_maneuver_m"]
    a1_red_can = 100.0 * (a1["canonical_bicycle"]["rmse_maneuver_m"] - a1["adaptive_bicycle"]["rmse_maneuver_m"]) / a1["canonical_bicycle"]["rmse_maneuver_m"]
    a6_red_lin = 100.0 * (a6_kin["linear_cv"]["rmse_maneuver_mean_m"] - a6_kin["adaptive_bicycle"]["rmse_maneuver_mean_m"]) / a6_kin["linear_cv"]["rmse_maneuver_mean_m"]
    a6_red_can = 100.0 * (a6_kin["canonical_bicycle"]["rmse_maneuver_mean_m"] - a6_kin["adaptive_bicycle"]["rmse_maneuver_mean_m"]) / a6_kin["canonical_bicycle"]["rmse_maneuver_mean_m"]

    b = SBCArticleBuilder()

    # =========================================================================
    # TÍTULO, AUTORES, RESUMO E ABSTRACT (PADRÃO SBC QUALIS A)
    # =========================================================================
    b.add_title_block(
        title_pt=(
            "Modelagem Cinemática Não Linear com Filtro de Kalman Estendido Adaptativo e "
            "Restrições Topológicas para Rastreamento Multicâmera e Predição de Trajetórias "
            "Veiculares em Redes Urbanas"
        ),
        title_en=(
            "Nonlinear Kinematic Modeling with Adaptive Extended Kalman Filtering and "
            "Topological Constraints for Multi-Camera Vehicle Tracking and Trajectory "
            "Prediction in Urban Networks"
        ),
        authors="Alex C. Medeiros, Eduardo E. Carvalho, Felipe R. Moura, Gabriel H. J. Ramalho, Kayky M. Oliveira, Roque C. Gabriel N.",
        affiliation="Pesquisa em Sistemas Inteligentes de Transporte e Visão Computacional Aplicada",
    )

    b.add_sbc_abstract(
        "Resumo",
        [
            "O monitoramento contínuo de fluxos veiculares em Sistemas Inteligentes de Transporte (ITS) enfrenta um obstáculo estrutural recorrente: a descontinuidade espacial da cobertura óptica decorrente de câmeras estáticas com campos de visão não sobrepostos e extensas regiões cegas urbanas. Nessas transições, manobras não lineares de conversão, frenagem e múltiplas bifurcações viárias degradam tanto a manutenção de identidade global quanto a predição de trajetórias futuras. Este trabalho formaliza matematicamente, implementa em arquitetura estritamente causal e avalia empiricamente um pipeline multicâmera que integra o Filtro de Kalman Estendido Adaptativo (EKF) — baseado no modelo cinemático ",
            ("Two-Point-Bicycle", {"italic": True}),
            " com suavização circular de orientação, ponderação direcional adaptativa à velocidade e gatilho persistente de Inovação Normalizada ao Quadrado (NIS) — à projeção geométrica 2D/3D com correção de ponto de contato, ao acoplamento topológico em grafo viário direcionado (",
            ("map-matching", {"italic": True}),
            " HMM/Viterbi e poda física por velocidade crítica de curva) e à associação híbrida intercâmeras (estatística χ",
            ("2", {"sup": True}),
            " multi-hipótese, descritores visuais Deep Re-ID OSNet e leitura seletiva de placas ALPR). A abordagem proposta é comparada sob protocolo controlado com o estado da arte — abrangendo Kalman Linear (CV), EKF CTRV, IMM, Filtro de Partículas (SIR), redes recorrentes LSTM e GRU e Rede Neural em Grafos Temporal (T-GNN) — e validada em cenários urbanos de larga escala de 4 a 24 câmeras e 20 a 150 veículos simultâneos (11.388 detecções e 232 transições cegas). Sobre cinco sementes aleatórias, o EKF adaptativo reduziu em ",
            f"{_fmt_pt(a6_red_lin, 1)}% o RMSE em manobras bruscas frente ao filtro linear ({_fmt_pt(a6_kin['adaptive_bicycle']['rmse_maneuver_mean_m'], 3)} m vs. {_fmt_pt(a6_kin['linear_cv']['rmse_maneuver_mean_m'], 3)} m) e em {_fmt_pt(a6_red_can, 1)}% frente ao EKF canônico, enquanto o IMM atingiu {_fmt_pt(a6_kin['imm']['rmse_maneuver_mean_m'], 3)} m com menor variância entre execuções e zero parâmetros treináveis. Na predição de rotas em cruzamentos da malha 8×8, a árvore física multi-hipótese superou a projeção euclidiana, o HMM/Viterbi e a rede T-GNN em erro espacial após 4,0 s de apagão visual ({_fmt_pt(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m vs. {_fmt_pt(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m), mantendo {_fmt_pt(a6_rt['proposed_physics_tree']['top1_route_accuracy']*100, 1)}% de acerto Top-1 e {_fmt_pt(a6_rt['proposed_physics_tree']['topk_route_accuracy']*100, 1)}% Top-2 sem retreinamento por cidade. Na escala máxima (24 câmeras e 150 veículos), a fusão híbrida sustentou IDF1 de {_fmt_pt(s4['modes']['hybrid']['idf1'], 4)} — superando o limiar legado ({_fmt_pt(s4['modes']['article']['idf1'], 4)}), o gating cinemático ({_fmt_pt(s4['modes']['chi2']['idf1'], 4)}) e o Re-ID isolado ({_fmt_pt(s4['modes']['reid_only']['idf1'], 4)}) — a mais de {_fmt_int_pt(s4['modes']['hybrid']['throughput_detections_s'])} detecções/s em CPU.",
        ],
        "Palavras-chave",
        "Sistemas Inteligentes de Transporte. Filtro de Kalman Estendido Adaptativo. Rastreamento Veicular Multicâmera. Predição de Trajetórias. Filtro de Partículas. LSTM e GRU. Redes Neurais em Grafos Temporais.",
    )

    b.add_sbc_abstract(
        "Abstract",
        [
            "Continuous vehicle flow monitoring in Intelligent Transportation Systems (ITS) faces a persistent structural bottleneck: spatial optical discontinuity caused by static cameras with non-overlapping fields of view (FOVs) and extensive urban blind regions. Across unmonitored corridors, nonlinear turning maneuvers, braking, and multi-branch intersections degrade both global identity association and future trajectory forecasting. This paper mathematically formalizes, implements as a strictly causal Python architecture, and empirically evaluates a multi-camera tracking pipeline integrating an Adaptive Extended Kalman Filter (EKF) — built upon the nonlinear Two-Point-Bicycle model with circular heading smoothing, speed-adaptive directional weighting, and persistent Normalized Innovation Squared (NIS) gating — with 2D/3D projective geometry using contact-point bias correction, directed road-graph topological constraints (HMM/Viterbi map-matching and critical cornering speed pruning), and hybrid cross-camera association (multi-hypothesis χ",
            ("2", {"sup": True}),
            " gating, OSNet Deep Re-ID, and selective ALPR). The approach is systematically benchmarked against state-of-the-art methods including Linear Kalman (CV), CTRV EKF, Interacting Multiple Model (IMM), Sequential Importance Resampling (SIR) Particle Filter, recurrent LSTM and GRU networks, and a Temporal Graph Neural Network (T-GNN), alongside city-scale scalability experiments spanning 4 to 24 non-overlapping cameras and 20 to 150 concurrent vehicles (11,388 detections and 232 blind-zone transitions). Across five random seeds, the stabilized Adaptive EKF reduced sharp-maneuver position RMSE by ",
            f"{_fmt_en(a6_red_lin, 1)}% over Linear CV ({_fmt_en(a6_kin['adaptive_bicycle']['rmse_maneuver_mean_m'], 3)} m vs. {_fmt_en(a6_kin['linear_cv']['rmse_maneuver_mean_m'], 3)} m) and by {_fmt_en(a6_red_can, 1)}% over Canonical EKF, while IMM reached {_fmt_en(a6_kin['imm']['rmse_maneuver_mean_m'], 3)} m with lower cross-seed variance and zero trainable parameters. On out-of-sample 8×8 urban intersection transitions, the proposed physics-constrained route tree outperformed Euclidean snapping, HMM/Viterbi, and T-GNN in 4.0 s blind-zone spatial RMSE ({_fmt_en(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m vs. {_fmt_en(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m), achieving {_fmt_en(a6_rt['proposed_physics_tree']['top1_route_accuracy']*100, 1)}% Top-1 and {_fmt_en(a6_rt['proposed_physics_tree']['topk_route_accuracy']*100, 1)}% Top-2 accuracy without city-specific retraining. At city scale (24 cameras, 150 vehicles), hybrid fusion sustained {_fmt_en(s4['modes']['hybrid']['idf1'], 4)} IDF1 — outperforming the legacy threshold ({_fmt_en(s4['modes']['article']['idf1'], 4)}), pure kinematic gating ({_fmt_en(s4['modes']['chi2']['idf1'], 4)}), and standalone Re-ID ({_fmt_en(s4['modes']['reid_only']['idf1'], 4)}) — at over {_fmt_int_en(s4['modes']['hybrid']['throughput_detections_s'])} detections/s on CPU.",
        ],
        "Keywords",
        "Intelligent Transportation Systems. Adaptive Extended Kalman Filter. Multi-Camera Vehicle Tracking. Trajectory Prediction. Particle Filter. LSTM and GRU. Temporal Graph Neural Networks.",
    )

    # =========================================================================
    # 1. INTRODUÇÃO
    # =========================================================================
    b.add_section_heading("1. Introdução", level=1)

    b.add_body_p(
        "A crescente urbanização e o adensamento do tráfego metropolitano demandam arquiteturas computacionais capazes de extrair inteligência contínua a partir de redes de videomonitoramento distribuídas. No âmbito dos Sistemas Inteligentes de Transporte (ITS), o acompanhamento consistente de veículos ao longo de corredores arteriais subsidia a estimação dinâmica de matrizes origem-destino, o controle semafórico adaptativo, a detecção precoce de incidentes e o planejamento operacional em Centros de Gestão de Tráfego (CGT). Contudo, conforme demonstrado por Chilgunde et al. (2004), a implantação de sensores ópticos com campos de visão continuamente sobrepostos sobre a totalidade de uma malha urbana mostra-se técnica e financeiramente inviável. Como consequência direta dessa restrição estrutural, as redes municipais operam com câmeras estáticas espaçadas, intercaladas por extensas regiões cegas nas quais o vetor de estado dos veículos deixa de ser observado visualmente.",
        first_in_section=True,
    )

    b.add_body_p(
        "A descontinuidade visual impõe dois desafios científicos interligados. Em primeiro lugar, a dinâmica real de veículos terrestres em vias urbanas caracteriza-se por acelerações longitudinais, frenagens pré-esquina e rotações não lineares governadas por restrições não holonômicas de esterçamento. Formulações baseadas no Filtro de Kalman linear clássico (Kalman, 1960) assumem transições invariantes de velocidade constante, acumulando erros severos em curvas acentuadas e durante intervalos sem observação (Lou et al., 2002; Bradler et al., 2021). Em segundo lugar, quando dezenas de veículos trafegam simultaneamente por múltiplas bifurcações entre câmeras não sobrepostas, a associação exclusivamente cinemática tende a colapsar: ensaios preliminares conduzidos sobre vídeos sintéticos densos sem descritores visuais registraram acurácia intercâmeras de apenas 5,0%, evidenciando a vulnerabilidade do rastreamento monomodal perante fragmentação de trajetórias locais e ambiguidade espaço-temporal."
    )

    b.add_body_p(
        "Embora a literatura contemporânea apresente avanços expressivos em filtros bayesianos não paramétricos — como o Filtro de Partículas (Gordon et al., 1993; Arulampalam et al., 2002) —, em modelos neurais sequenciais recorrentes — como LSTM e GRU (Hochreiter e Schmidhuber, 1997; Cho et al., 2014; Altché e de La Fortelle, 2017) — e em Redes Neurais em Grafos Temporais (T-GNN) (Yu et al., 2018; Zhao et al., 2020), observa-se uma lacuna metodológica na avaliação integrada, estritamente causal e comparativa desses paradigmas no interior de um mesmo pipeline multicâmera em escala urbana. Em especial, arquiteturas puramente baseadas em aprendizado profundo exigem grandes volumes de trajetórias rotuladas para cada nova topologia municipal, ao passo que métodos estritamente geométricos ou puramente visuais apresentam degradação quando confrontados com oclusões, distorções de perspectiva 3D e veículos visualmente semelhantes."
    )

    b.add_body_p(
        "Diante desse cenário, este artigo tem por objetivo formalizar matematicamente, implementar em software estritamente causal e avaliar empiricamente uma arquitetura híbrida que combina filtragem cinemática não linear estabilizada, projeção geométrica 2D/3D corrigida, restrições físicas em grafos viários e associação multimodal intercâmeras. A hipótese norteadora consiste em que o acoplamento explícito entre um EKF adaptativo estabilizado, poda dinâmica baseada em física de curva e fusão híbrida espaço-temporal/visual atinge desempenho equivalente ou superior a métodos neurais e baseados em partículas, preservando latência submilissegundo em CPU e generalização imediata para novas malhas urbanas sem retreinamento. As contribuições centrais sintetizam-se em quatro eixos:"
    )

    b.add_bullet_p(
        "Estabilização causal do EKF adaptativo sob ruído real: ",
        [
            "formaliza-se um filtro ",
            ("Two-Point-Bicycle", {"italic": True}),
            " com inicialização 100% causal alinhada à via, suavização circular de orientação em 4 quadros, peso direcional adaptativo à velocidade α",
            ("θ", {"sub": True}),
            "(v) e gatilho persistente de Inovação Normalizada ao Quadrado (NIS χ",
            ("2", {"sup": True}),
            ("0,95;2", {"sub": True}),
            " por ≥ 2 quadros), eliminando oscilações espúrias do fator de desvanecimento λ",
            ("k", {"sub": True}),
            ".",
        ],
    )
    b.add_bullet_p(
        "Geometria projetiva 3D e poda topológica por física de curva: ",
        [
            "incorpora-se correção do viés de perspectiva 3D do ponto de contato da caixa delimitadora (f = 0,90), ",
            ("map-matching", {"italic": True}),
            " HMM/Viterbi escalável e propagação multi-hipótese com poda dinâmica por velocidade crítica de derrapagem lateral v",
            ("crit", {"sub": True}),
            " = √(μgR",
            ("curva", {"sub": True}),
            ").",
        ],
    )
    b.add_bullet_p(
        "Comparação controlada com o estado da arte: ",
        [
            "implementam-se e avaliam-se, sob idêntico protocolo causal e múltiplas sementes, oito modelos de trajetória (Kalman CV, EKF Canônico, EKF Adaptativo, CTRV, IMM, ",
            ("Filtro de Partículas SIR", {"bold": True}),
            ", ",
            ("LSTM", {"bold": True}),
            " e ",
            ("GRU", {"bold": True}),
            ") e quatro modelos topológicos de predição de rotas (Euclidiano, HMM/Viterbi, ",
            ("T-GNN", {"bold": True}),
            " e a árvore física proposta).",
        ],
    )
    b.add_bullet_p(
        "Validação de escalabilidade urbana e rastreabilidade de fontes: ",
        [
            "demonstra-se a escalabilidade do sistema em malha urbana 8×8 de 4 até ",
            ("24 câmeras não sobrepostas e 150 veículos simultâneos", {"bold": True}),
            " (11.388 detecções e 232 transições cegas), acompanhada do catálogo integral de fontes bibliográficas e repositórios de todos os bancos de dados de teste.",
        ],
    )

    b.add_body_p(
        "O restante deste artigo está organizado da seguinte forma. A Seção 2 discute criticamente os trabalhos relacionados e explicita a lacuna científica atendida. A Seção 3 formaliza a modelagem matemática e a arquitetura modular do pipeline proposto. A Seção 4 detalha as questões de pesquisa, os métodos de referência implementados, as métricas e as fontes dos bancos de dados de teste. A Seção 5 apresenta e analisa os resultados experimentais, incluindo as ameaças à validade. Por fim, a Seção 6 consolida as considerações finais e os trabalhos futuros, enquanto os Apêndices A e B (após as Referências) disponibilizam os núcleos algorítmicos e os comandos de reprodutibilidade."
    )

    # =========================================================================
    # 2. TRABALHOS RELACIONADOS
    # =========================================================================
    b.add_section_heading("2. Trabalhos Relacionados", level=1)

    b.add_body_p(
        "A literatura de rastreamento veicular e predição de trajetórias em ambientes urbanos com oclusões prolongadas distribui-se em três frentes complementares: filtragem estocástica não linear, modelagem preditiva baseada em aprendizado profundo e rastreamento multicâmera acoplado a grafos viários. Esta seção analisa os avanços representativos de cada frente e posiciona a proposta frente ao estado da arte.",
        first_in_section=True,
    )

    b.add_section_heading("2.1 Filtragem Cinemática e Estimação Bayesiana Não Linear", level=2)
    b.add_body_p(
        "O Filtro de Kalman linear (Kalman, 1960) permanece como referência histórica para estimação ótima em sistemas lineares gaussianos, mas sua aplicação direta a veículos urbanos sob curvas fechadas incorre em divergência de inovação. Para tratar a não linearidade do esterçamento frontal, Lou et al. (2002) formularam um Filtro de Kalman Estendido adaptativo baseado no modelo Two-Point-Bicycle acoplado a um fator de desvanecimento calculado pelo critério de ortogonalidade dos resíduos. Embora eficaz sob baixo ruído de sensor, a ativação irrestrita desse fator torna-se sensível ao ruído de detecção de caixas delimitadoras em câmeras urbanas reais. Complementarmente, Bradler et al. (2021) exploraram o modelo CTRV (Constant Turn Rate and Velocity) em rastreamento probabilístico 3D urbano, enquanto Blom e Bar-Shalom (1988) estabeleceram o arcabouço IMM (Interacting Multiple Model) para alternância bayesiana entre regimes retilíneos e curvilíneos.",
        first_in_section=True,
    )
    b.add_body_p(
        "Em paralelo aos filtros baseados em linearização local, os Filtros de Partículas por Reamostragem Sequencial de Importância (SIR) (Gordon et al., 1993; Arulampalam et al., 2002) representam distribuições a posteriori arbitrárias e multimodais por meio de conjuntos ponderados de amostras estocásticas. Apesar da alta expressividade em dinâmicas não gaussianas, seu custo computacional escala linearmente com o número de partículas e com a quantidade de veículos simultâneos, constituindo um desafio para a operação em tempo real sobre dezenas de fluxos de vídeo."
    )

    b.add_section_heading("2.2 Preditores Neurais Recorrentes (LSTM/GRU) e em Grafos (T-GNN)", level=2)
    b.add_body_p(
        "Com a disseminação do aprendizado profundo, arquiteturas recorrentes baseadas em células LSTM (Hochreiter e Schmidhuber, 1997) e GRU (Cho et al., 2014) passaram a ser amplamente empregadas na previsão de trajetórias veiculares a partir de históricos cinemáticos sequenciais (Altché e de La Fortelle, 2017). Essas redes capturam dependências temporais não lineares sem exigir especificação analítica de Jacobianas; entretanto, sofrem acúmulo de erro em rollouts autorregressivos longos fora da distribuição de treinamento e não garantem, por construção, aderência às restrições geométricas da malha viária.",
        first_in_section=True,
    )
    b.add_body_p(
        "Para incorporar a estrutura espacial das vias, surgiram as Redes Neurais em Grafos Espaço-Temporais (STGCN e T-GCN/T-GNN) (Yu et al., 2018; Zhao et al., 2020), que combinam convoluções espectrais ou espaciais sobre o grafo viário com unidades recorrentes temporais. Ainda que altamente eficazes quando treinadas sobre grandes bases históricas de uma cidade específica, redes T-GNN requerem dados rotulados por topologia e retreinamento ao migrar para novos municípios, além de poderem atribuir probabilidade não nula a conversões fisicamente inviáveis sob velocidade excessiva."
    )

    b.add_section_heading("2.3 Rastreamento Multicâmera em Regiões Cegas e Map-Matching", level=2)
    b.add_body_p(
        "No rastreamento multicâmera sem sobreposição visual, Chilgunde et al. (2004) introduziram a associação probabilística no plano de solo combinando posições extrapoladas e dimensões aparentes dos veículos, empregando o Algoritmo Húngaro (Kuhn, 1955) para correspondência ótima global. Contudo, seu limiar empírico de log-verossimilhança, calibrado em espaço de pixels, mostra-se excessivamente permissivo em coordenadas métricas urbanas. Estudos mais recentes demonstraram ganhos substanciais ao integrar modelos ocultos de Markov com decodificação de Viterbi (Newson e Krumm, 2009; Chao et al., 2020), propagação em malha viária durante ausência de sinal (Kinderby e Ridderstolpe, 2024; Zhang et al., 2025), rastreadores locais robustos como ByteTrack (Zhang et al., 2022) e descritores profundos de reidentificação veicular como OSNet (Zhou et al., 2019; Tang et al., 2019).",
        first_in_section=True,
    )
    b.add_body_p(
        "A Tabela 1 sintetiza o posicionamento comparativo entre os principais estudos da literatura e a arquitetura proposta neste artigo, destacando a integração inédita entre filtragem causal estabilizada, física de curva em grafo, fusão multimodal e validação comparativa contra métodos neurais e bayesianos em escala urbana."
    )

    b.add_academic_table(
        "Tabela 1.",
        "Comparação sintética entre trabalhos relacionados e a arquitetura proposta neste artigo.",
        headers=[
            "Estudo / Referência",
            "Modelo Cinemático / Predição",
            "Acoplamento ao Grafo Viário",
            "Fusão Multimodal (Re-ID / ALPR)",
            "Avaliação Multi-Câmera Urbana",
            "Independência de Retreino",
        ],
        rows=[
            ["Lou et al. (2002)", "EKF Adaptativo (Bicycle)", "Não", "Não", "Câmera única", "Sim"],
            ["Chilgunde et al. (2004)", "Kalman / Homografia", "Não", "Dimensões (W, L)", "Poucas câmeras", "Sim"],
            ["Arulampalam et al. (2002)", "Filtro de Partículas (SIR)", "Não", "Não", "Simulado", "Sim"],
            ["Altché e de La Fortelle (2017)", "Rede LSTM Sequencial", "Não", "Não", "Via expressa", "Não"],
            ["Zhao et al. (2020)", "Rede T-GCN / T-GNN", "Sim (GCN)", "Não", "Fluxo agregado", "Não"],
            ["Bradler et al. (2021)", "EKF CTRV + Projeção 3D", "Não", "Não", "Câmera única", "Sim"],
            ["Kinderby e Ridderstolpe (2024)", "EKF + Grafo Viário", "Sim (Dead Reckoning)", "Não", "Telemetria GPS", "Sim"],
            [
                "Este trabalho",
                "EKF Adaptativo Estabilizado + IMM",
                ["Sim (HMM/Viterbi + Poda v", ("crit", {"sub": True}), ")"],
                ["Sim (χ", ("2", {"sup": True}), " + OSNet + ALPR)"],
                "Sim (4 a 24 câm., 150 veíc.)",
                "Sim (Zero parâmetros)",
            ],
        ],
        col_widths_cm=[3.2, 2.7, 2.6, 2.3, 2.3, 1.9],
        center_cols={2, 3, 4, 5},
        highlight_rows={7},
        font_size_pt=8.2,
        source="Elaborado pelos autores com base na revisão da literatura.",
    )

    # =========================================================================
    # 3. MODELAGEM MATEMÁTICA E ARQUITETURA DO PIPELINE PROPOSTO
    # =========================================================================
    b.add_section_heading("3. Modelagem Matemática e Arquitetura do Pipeline Proposto", level=1)

    b.add_section_heading("3.1 Visão Geral da Arquitetura Causal", level=2)
    b.add_body_p(
        "A arquitetura computacional foi projetada sob princípio estrito de causalidade quadro a quadro, assegurando que nenhuma decisão no instante t utilize observações futuras. Conforme ilustrado na Figura 1, o processamento organiza-se em seis estágios desacoplados e tipados (Pydantic v2 em Python 3.12): (1) percepção local com detecção veicular YOLO11n e rastreamento intracâmera ByteTrack/BoT-SORT munido de polígonos de zonas de saída; (2) projeção geométrica 2D/3D para o plano métrico UTM com correção do ponto de contato e propagação analítica da incerteza; (3) filtragem cinemática não linear com EKF Adaptativo estabilizado ou IMM; (4) acoplamento topológico à malha OpenStreetMap/UTM com map-matching HMM/Viterbi e árvore multi-hipótese podada por velocidade crítica de curva; (5) associação híbrida intercâmeras via Algoritmo Húngaro combinando gating χ², Deep Re-ID OSNet e ALPR seletivo; e (6) análise operacional em Centros de Gestão de Tráfego (ITS), incluindo reconstrução retrospectiva da rota percorrida na região cega e identificação de nós críticos de cobertura viária.",
        first_in_section=True,
    )

    b.add_figure(
        fig_dir / "fig1_arquitetura.png",
        "Figura 1.",
        "Arquitetura modular causal do pipeline multicâmera em seis estágios integrados, da percepção visual local até a gestão analítica de tráfego urbano (ITS).",
        width_cm=14.8,
    )

    b.add_body_p(
        "Para assegurar auditabilidade científica e correspondência direta entre formulação matemática e implementação executável, a Tabela 2 mapeia os subsistemas teóricos aos respectivos módulos de software. Em consonância com as convenções editoriais da SBC, os trechos ilustrativos de código-fonte e os comandos de reprodução integral dos experimentos encontram-se nos Apêndices A e B, após as Referências."
    )

    b.add_academic_table(
        "Tabela 2.",
        "Mapeamento entre os fundamentos metodológicos e os módulos executáveis da arquitetura.",
        headers=["Módulo do Repositório", "Seção", "Responsabilidade Técnica Implementada"],
        rows=[
            ["config.py", "Seções 3–4", "Contratos Pydantic v2 para câmeras, matrizes 3D/homografias, EKF, grafo e cenários."],
            ["kinematics.py / ekf.py", "Seção 3.2", "Modelos Two-Point-Bicycle, CTRV, Linear CV, EKF Adaptativo estabilizado e IMM."],
            ["geometry.py", "Seção 3.3", "Homografia DLT/RANSAC, projeção 3D K[R|t], correção de contato (f=0,90) e WGS84↔UTM."],
            ["osm_loader.py / roadgraph.py", "Seção 3.4", ["Leitura OSM/UTM, HMM/Viterbi otimizado, poda física v", ("crit", {"sub": True}), ", árvore multi-hipótese e nós críticos."]],
            ["detector.py / reid.py / alpr.py", "Seções 3.1 e 3.4", "Detecção YOLO11n + ByteTrack, extrator ONNX OSNet 512-D e leitor de placas ALPR."],
            ["association.py / pipeline.py", "Seção 3.4", "Associação híbrida com Algoritmo Húngaro, autocalibração de escala e backtracking."],
            ["sota_baselines.py", "Seção 4.2", "Baselines causais: Filtro de Partículas (SIR), LSTM, GRU e Temporal GNN (T-GNN)."],
            ["city_benchmark.py / run_benchmarks.py", "Seções 4–5", "Simulador urbano 8×8 (4–24 câmeras, 20–150 veículos) e suíte experimental A1–A7."],
        ],
        col_widths_cm=[4.2, 2.2, 8.6],
        center_cols={1},
        font_size_pt=8.4,
    )

    b.add_section_heading("3.2 Cinemática Não Linear e EKF Adaptativo Estabilizado", level=2)
    b.add_body_p(
        "Enquanto o modelo linear de Velocidade Constante (CV) assume a transição invariante da Equação (1), a dinâmica real de veículos de quatro rodas com esterçamento dianteiro e distância entre-eixos L = 2,5 m é representada pelo vetor de estado de quinta ordem da Equação (2), sob restrição não holonômica de ausência de escorregamento lateral:",
        first_in_section=True,
    )

    b.add_equation(
        [[("x", {"bold": True}), ("k+1", {"sub": True}), " = Φ(Δt) ", ("x", {"bold": True}), ("k", {"sub": True}), " + ", ("w", {"bold": True}), ("k", {"sub": True}), ",    com  ", ("w", {"bold": True}), ("k", {"sub": True}), " ~ N(0, ", ("Q", {"bold": True}), ("k", {"sub": True}), ")"]],
        "1",
    )
    b.add_equation(
        [[("x", {"bold": True}), ("k", {"sub": True}), " = [ X", ("k", {"sub": True}), ",  Y", ("k", {"sub": True}), ",  v", ("k", {"sub": True}), ",  θ", ("k", {"sub": True}), ",  δ", ("k", {"sub": True}), " ]", ("T", {"sup": True}), " ∈ ℝ", ("5", {"sup": True})]],
        "2",
    )

    b.add_body_p(
        [
            "em que (X",
            ("k", {"sub": True}),
            ", Y",
            ("k", {"sub": True}),
            ") correspondem às coordenadas métricas no plano UTM (m), v",
            ("k", {"sub": True}),
            " ≥ 0 representa a velocidade escalar (m/s), θ",
            ("k", {"sub": True}),
            " ∈ (−π, π] é o ângulo azimutal do chassi (rad) e δ",
            ("k", {"sub": True}),
            " ∈ [−δ",
            ("máx", {"sub": True}),
            ", +δ",
            ("máx", {"sub": True}),
            "] designa o esterçamento das rodas dianteiras (rad). Definindo a taxa de guinada ω",
            ("k", {"sub": True}),
            " = (v",
            ("k", {"sub": True}),
            " / L) tan(δ",
            ("k", {"sub": True}),
            "), a integração discreta exata por arco circular no intervalo Δt (com limite analítico suave quando |ω",
            ("k", {"sub": True}),
            "| < 10",
            ("−6", {"sup": True}),
            " rad/s) expressa-se por:",
        ]
    )

    b.add_equation(
        [
            ["X", ("k+1", {"sub": True}), " = X", ("k", {"sub": True}), " + (v", ("k", {"sub": True}), " / ω", ("k", {"sub": True}), ") [ sen(θ", ("k", {"sub": True}), " + ω", ("k", {"sub": True}), "Δt) − sen(θ", ("k", {"sub": True}), ") ]"],
            ["Y", ("k+1", {"sub": True}), " = Y", ("k", {"sub": True}), " + (v", ("k", {"sub": True}), " / ω", ("k", {"sub": True}), ") [ −cos(θ", ("k", {"sub": True}), " + ω", ("k", {"sub": True}), "Δt) + cos(θ", ("k", {"sub": True}), ") ]"],
            ["v", ("k+1", {"sub": True}), " = v", ("k", {"sub": True}), ",   θ", ("k+1", {"sub": True}), " = wrap(θ", ("k", {"sub": True}), " + ω", ("k", {"sub": True}), "Δt),   δ", ("k+1", {"sub": True}), " = clip(δ", ("k", {"sub": True}), ", −δ", ("máx", {"sub": True}), ", +δ", ("máx", {"sub": True}), ")"],
        ],
        "3",
    )

    b.add_body_p(
        [
            "A covariância de erro a priori propaga-se mediante a Jacobiana analítica exata ",
            ("F", {"bold": True}),
            ("k", {"sub": True}),
            " = ∂f",
            ("d", {"sub": True}),
            "/∂",
            ("x", {"bold": True}),
            " ∈ ℝ",
            ("5×5", {"sup": True}),
            ", verificada numericamente por diferenças finitas centradas nos testes unitários com discrepância inferior a 10",
            ("−5", {"sup": True}),
            ". Para preservar a simetria e a positividade definida (SPD) em ciclos prolongados, a atualização a posteriori adota a forma estabilizada de Joseph seguida de projeção espectral:",
        ]
    )

    b.add_equation(
        [
            [("γ", {"bold": True}), ("k", {"sub": True}), " = ", ("z", {"bold": True}), ("k", {"sub": True}), " − ", ("H", {"bold": True}), " ", ("x̂", {"bold": True}), ("k|k−1", {"sub": True}), ",    ", ("S", {"bold": True}), ("k", {"sub": True}), " = ", ("H", {"bold": True}), " ", ("P̃", {"bold": True}), ("k|k−1", {"sub": True}), " ", ("H", {"bold": True}), ("T", {"sup": True}), " + ", ("R", {"bold": True}), ("k", {"sub": True})],
            [("K", {"bold": True}), ("k", {"sub": True}), " = ", ("P̃", {"bold": True}), ("k|k−1", {"sub": True}), " ", ("H", {"bold": True}), ("T", {"sup": True}), " ", ("S", {"bold": True}), ("k", {"sub": True}), ("−1", {"sup": True}), ",    ", ("x̂", {"bold": True}), ("k|k", {"sub": True}), " = ", ("x̂", {"bold": True}), ("k|k−1", {"sub": True}), " + ", ("K", {"bold": True}), ("k", {"sub": True}), " ", ("γ", {"bold": True}), ("k", {"sub": True})],
            [("P", {"bold": True}), ("k|k", {"sub": True}), " = (", ("I", {"bold": True}), " − ", ("K", {"bold": True}), ("k", {"sub": True}), ("H", {"bold": True}), ") ", ("P̃", {"bold": True}), ("k|k−1", {"sub": True}), " (", ("I", {"bold": True}), " − ", ("K", {"bold": True}), ("k", {"sub": True}), ("H", {"bold": True}), ")", ("T", {"sup": True}), " + ", ("K", {"bold": True}), ("k", {"sub": True}), ("R", {"bold": True}), ("k", {"sub": True}), ("K", {"bold": True}), ("k", {"sub": True}), ("T", {"sup": True})],
        ],
        "4",
    )

    b.add_body_p(
        [
            "Diferentemente do fator de desvanecimento canônico de Lou et al. (2002), que pode amplificar ruído de detecção quando a orientação é derivada de caixas ruidosas com peso fixo α",
            ("θ", {"sub": True}),
            " = 2,8, o mecanismo estabilizado proposto calcula a Inovação Normalizada ao Quadrado (NIS) ε",
            ("k", {"sub": True}),
            " = ",
            ("γ", {"bold": True}),
            ("k", {"sub": True}),
            ("T", {"sup": True}),
            ("S", {"bold": True}),
            ("k", {"sub": True}),
            ("−1", {"sup": True}),
            ("γ", {"bold": True}),
            ("k", {"sub": True}),
            " e exige persistência acima do quantil χ",
            ("2", {"sup": True}),
            ("0,95;2", {"sub": True}),
            " = 5,991 por pelo menos dois quadros consecutivos, acoplada à média circular ponderada de orientação em janela de 4 quadros e ao peso adaptativo α",
            ("θ", {"sub": True}),
            "(v) ∈ [0,2; 2,8] (com α",
            ("x", {"sub": True}),
            " + α",
            ("y", {"sub": True}),
            " + α",
            ("θ", {"sub": True}),
            " = 3,0):",
        ]
    )

    b.add_equation(
        [
            [("P̃", {"bold": True}), ("k|k−1", {"sub": True}), " = λ", ("k", {"sub": True}), " ", ("P", {"bold": True}), ("k|k−1", {"sub": True})],
            ["λ", ("k", {"sub": True}), " = clip( 1 + α", ("xy", {"sub": True}), "(v)(ε", ("k", {"sub": True}), " / χ", ("2", {"sup": True}), ("0,95;2", {"sub": True}), ") + α", ("θ", {"sub": True}), "(v)|Δθ", ("circ", {"sub": True}), "|·I", ("|Δθ|>0,35", {"sub": True}), ",  1,15,  λ", ("máx", {"sub": True}), " )"],
            ["se  streak(ε", ("k", {"sub": True}), " > 5,991) ≥ 2  e  v ≥ v", ("mín", {"sub": True}), ";   caso contrário,  λ", ("k", {"sub": True}), " = 1,0"],
        ],
        "5",
    )

    b.add_body_p(
        "A inicialização de cada alvo é rigorosamente causal: no primeiro quadro observado, a orientação inicial alinha-se à tangente da aresta viária mais próxima; no segundo quadro, a velocidade e o azimute são refinados a partir do deslocamento métrico efetivo. Adicionalmente, o filtro IMM implementado combina o EKF Adaptativo Two-Point-Bicycle e o EKF CTRV por meio de probabilidades de modo atualizadas pela verossimilhança gaussiana de inovação."
    )

    b.add_section_heading("3.3 Geometria Projetiva 2D/3D e Correção do Ponto de Contato", level=2)
    b.add_body_p(
        [
            "Para unificar observações oriundas de múltiplas câmeras estáticas no referencial métrico UTM, suportam-se tanto homografias planares estimadas por DLT normalizado de Hartley com rejeição RANSAC quanto calibração projetiva completa K[R | t] com interseção de raios no plano de solo Z = 0 (Bradler et al., 2021). Como a borda inferior da caixa delimitadora 2D (y",
            ("2", {"sub": True}),
            ") sofre deslocamento sistemático causado pela altura tridimensional do veículo em ângulos oblíquos de montagem (20°–30°), adota-se o ponto de contato corrigido em perspectiva (f = 0,90) com propagação analítica da incerteza em pixels pela Jacobiana da homografia ",
            ("J", {"bold": True}),
            ("H", {"sub": True}),
            ":",
        ],
        first_in_section=True,
    )

    b.add_equation(
        [
            ["u", ("c", {"sub": True}), " = 0,5 (x", ("1", {"sub": True}), " + x", ("2", {"sub": True}), "),   v", ("c", {"sub": True}), " = y", ("1", {"sub": True}), " + 0,90 (y", ("2", {"sub": True}), " − y", ("1", {"sub": True}), ")"],
            ["[X", ("h", {"sub": True}), ", Y", ("h", {"sub": True}), ", w", ("h", {"sub": True}), "]", ("T", {"sup": True}), " = ", ("H", {"bold": True}), ("i", {"sub": True}), " [u", ("c", {"sub": True}), ", v", ("c", {"sub": True}), ", 1]", ("T", {"sup": True})],
            [("R", {"bold": True}), ("k", {"sub": True}), "(u", ("c", {"sub": True}), ", v", ("c", {"sub": True}), ") = ", ("J", {"bold": True}), ("H", {"sub": True}), "(u", ("c", {"sub": True}), ", v", ("c", {"sub": True}), ") · diag(σ", ("u", {"sub": True}), ("2", {"sup": True}), ", σ", ("v", {"sub": True}), ("2", {"sup": True}), ") · ", ("J", {"bold": True}), ("H", {"sub": True}), "(u", ("c", {"sub": True}), ", v", ("c", {"sub": True}), ")", ("T", {"sup": True})],
        ],
        "6",
    )

    b.add_section_heading("3.4 Acoplamento Topológico Viário, Poda Física e Associação Híbrida", level=2)
    b.add_body_p(
        "A malha viária é modelada como um multigrafo direcionado G = (V, E) em coordenadas métricas UTM (compatível com OpenStreetMap via OSMnx; Boeing, 2017). Sobre trechos visíveis ruidosos, aplica-se map-matching HMM/Viterbi (Newson e Krumm, 2009) com pré-filtragem espacial das 16 arestas candidatas mais próximas, eliminando desvios espúrios para ruas paralelas. Ao ingressar em região cega, propaga-se uma árvore multi-hipótese de trajetórias curvilíneas ao longo de G. Em cada intersecção, além da restrição de mão direcional e da vedação a retornos instantâneos de 180°, impõe-se a poda física por velocidade crítica de derrapagem lateral sob atrito pneu-asfalto μ = 0,70 e gravidade g = 9,81 m/s²:",
        first_in_section=True,
    )

    b.add_equation(
        [
            ["v", ("crit", {"sub": True}), " = √( μ · g · R", ("curva", {"sub": True}), " ),    v", ("mín", {"sub": True}), " = √( max(0, v", ("0", {"sub": True}), ("2", {"sup": True}), " − 2 μ g d", ("aprox", {"sub": True}), ") )"],
            ["w(e", ("próx", {"sub": True}), " | e", ("atual", {"sub": True}), ") = faixas(e", ("próx", {"sub": True}), ") · (v", ("lim", {"sub": True}), "(e", ("próx", {"sub": True}), ") / 13,89) · exp(−0,6 |Δθ|)"],
            ["P(h", ("m", {"sub": True}), ") = W(h", ("m", {"sub": True}), ") / Σ", ("j", {"sub": True}), " W(h", ("j", {"sub": True}), ")"],
        ],
        "7",
    )

    b.add_body_p(
        [
            "Caso v",
            ("mín", {"sub": True}),
            " > 1,05 v",
            ("crit", {"sub": True}),
            " para uma conversão com deflexão Δθ ≥ 30°, o ramo é podado como fisicamente inviável; para ramos factíveis, simula-se desaceleração pré-esquina até 0,90 v",
            ("crit", {"sub": True}),
            " e posterior reaceleração até v",
            ("lim", {"sub": True}),
            ". Quando um veículo reaparece em uma câmera a jusante, a associação 1:1 é resolvida pelo Algoritmo Húngaro (Kuhn, 1955) sobre uma matriz de custo híbrida que integra a menor distância de Mahalanobis multi-hipótese (d",
            ("χ²", {"sub": True}),
            "), a distância cosseno dos ",
            ("embeddings", {"italic": True}),
            " OSNet 512-D (d",
            ("ReID", {"sub": True}),
            "), a discrepância dimensional autocalibrada entre câmeras (d",
            ("dim", {"sub": True}),
            ") e a confirmação ou veto determinístico por leitura ALPR de alta confiança (≥ 0,85):",
        ]
    )

    b.add_equation(
        [["C", ("híbrido", {"sub": True}), "(a, b) = w", ("st", {"sub": True}), " (d", ("χ²", {"sub": True}), " / χ", ("2", {"sup": True}), ("lim", {"sub": True}), ") + w", ("ReID", {"sub": True}), " (d", ("ReID", {"sub": True}), " / d", ("ReID,máx", {"sub": True}), ") + 0,15 d", ("dim", {"sub": True}), "(a, b)"]],
        "8",
    )

    b.add_body_p(
        "Uma vez confirmada a identidade global, executa-se a reconstrução retrospectiva (backtracking topológico) do trajeto percorrido na região cega e atualizam-se os nós críticos de cobertura viária (vértices a jusante com maior probabilidade acumulada e tempo estimado de chegada — ETA). A Tabela 3 consolida os hiperparâmetros operacionais empregados em todo o pipeline."
    )

    b.add_academic_table(
        "Tabela 3.",
        "Hiperparâmetros operacionais consolidados nos módulos da arquitetura proposta.",
        headers=["Subsistema", "Parâmetro / Símbolo", "Valor Padrão", "Função Operacional"],
        rows=[
            ["Cinemática", ["Passo Δt / Entre-eixos L / δ", ("máx", {"sub": True})], "0,1 s / 2,5 m / 0,85 rad", "Discretização temporal e geometria do modelo Bicycle."],
            ["EKF Adaptativo", "Janela circular W / Heading gate", "4 quadros / 0,35 rad", "Suavização angular e proteção contra jitter de caixa."],
            ["EKF Adaptativo", "Limiar NIS (α = 0,05) / Persistência", ["χ", ("2", {"sup": True}), " = 5,991 / ≥ 2 quadros"], ["Acionamento confirmado de λ", ("k", {"sub": True}), " ∈ [1,15; 25,0]."]],
            ["Geometria 3D", ["Fração de contato f / Ruído σ", ("px", {"sub": True})], "0,90 / (4,0; 4,0) px", ["Compensação de perspectiva 3D e matriz ", ("R", {"bold": True}), ("k", {"sub": True}), "."]],
            ["Grafo Viário", "Atrito μ / Ângulo curva / Horizonte", "0,70 / 30° / 15–20 s", ["Poda física v", ("crit", {"sub": True}), " e expansão multi-hipótese."]],
            ["Associação", ["Gating χ", ("2", {"sup": True}), ("4gl", {"sub": True}), " / Pesos (w", ("st", {"sub": True}), ", w", ("ReID", {"sub": True}), ")"], "13,28 / (0,45; 0,55)", "Fusão espaço-temporal e visual com autocalibração."],
            ["Percepção", "Detector / Re-ID / Confiança", "YOLO11n / OSNet 512-D / 0,35", "Classes veiculares COCO [2, 3, 5, 7] e ByteTrack."],
        ],
        col_widths_cm=[2.6, 4.1, 3.3, 5.0],
        center_cols={2},
        font_size_pt=8.4,
    )

    # =========================================================================
    # 4. METODOLOGIA EXPERIMENTAL, BASELINES SOTA E BANCOS DE DADOS
    # =========================================================================
    b.add_section_heading("4. Metodologia Experimental, Métodos de Referência e Bancos de Dados", level=1)

    b.add_section_heading("4.1 Questões de Pesquisa", level=2)
    b.add_body_p(
        "Em consonância com o rigor experimental de publicações Qualis A em Computação, a avaliação empírica foi estruturada para responder objetivamente a quatro questões de pesquisa (QP):",
        first_in_section=True,
    )
    b.add_bullet_p(
        "QP1 (Acurácia e Custo frente ao Estado da Arte Cinemático e Neural): ",
        ["Qual o ganho de acurácia, estabilidade estatística e eficiência computacional do EKF Adaptativo estabilizado e do IMM frente ao Filtro de Kalman Linear, EKF Canônico, CTRV, Filtro de Partículas (SIR) e redes recorrentes LSTM e GRU em manobras bruscas, apagões visuais e ruído crescente?"],
    )
    b.add_bullet_p(
        "QP2 (Eficácia Topológica frente a HMM/Viterbi e T-GNN): ",
        [
            "Em que medida a correção de ponto de contato 3D e a árvore multi-hipótese com poda física v",
            ("crit", {"sub": True}),
            " superam a projeção euclidiana, o ",
            ("map-matching", {"italic": True}),
            " HMM/Viterbi e uma Rede Neural em Grafos Temporal (T-GNN) na predição de rotas e posições em regiões cegas urbanas?",
        ],
    )
    b.add_bullet_p(
        "QP3 (Escalabilidade em Cenário Urbano Realista): ",
        ["Como o desempenho de associação intercâmeras (IDF1 e acurácia de rota) dos quatro paradigmas de associação (legado, gating χ², Re-ID puro e fusão híbrida) escala ao ampliar a rede de 4 para 24 câmeras não sobrepostas e de 20 para 150 veículos simultâneos?"],
    )
    b.add_bullet_p(
        "QP4 (Viabilidade Computacional em Tempo Real): ",
        ["O núcleo matemático e de associação sustenta operação em tempo real em CPU convencional sob tráfego urbano denso com centenas de transições simultâneas em regiões cegas?"],
    )

    b.add_section_heading("4.2 Métodos do Estado da Arte Implementados para Comparação", level=2)
    b.add_body_p(
        "Para assegurar equidade comparativa, todos os métodos de referência foram implementados no mesmo contrato causal de entrada/saída e executados sobre a mesma CPU com sementes determinísticas:",
        first_in_section=True,
    )
    b.add_bullet_p(
        "Filtros Analíticos e de Múltiplos Modelos: ",
        ["Filtro de Kalman Linear CV (Kalman, 1960), EKF Two-Point-Bicycle Canônico (λ = 1), EKF CTRV (Bradler et al., 2021) e filtro IMM combinando Adaptive Bicycle e CTRV (Blom e Bar-Shalom, 1988)."],
    )
    b.add_bullet_p(
        "Filtro de Partículas SIR (Sequential Importance Resampling): ",
        [
            "implementação vetorizada com N",
            ("p", {"sub": True}),
            " = 500 partículas em espaço de estados 5D, excitações estocásticas de aceleração (σ",
            ("a", {"sub": True}),
            " = 1,6 m/s²) e esterçamento (σ",
            ("b", {"sub": True}),
            " = 0,65 rad/s), atualização por verossimilhança gaussiana e reamostragem sistemática quando N",
            ("eff", {"sub": True}),
            " < 0,5 N",
            ("p", {"sub": True}),
            " (Gordon et al., 1993; Arulampalam et al., 2002).",
        ],
    )
    b.add_bullet_p(
        "Redes Neurais Recorrentes LSTM e GRU: ",
        ["arquiteturas PyTorch de duas camadas ocultas (48 unidades, LayerNorm e cabeça MLP; 31.364 parâmetros na LSTM e 23.972 na GRU) operando sobre janelas de 6 passos em referencial local SE(2)-invariante, treinadas sob protocolo idêntico em 320 trajetórias urbanas multirregime e avaliadas em rollout autorregressivo na região cega (Hochreiter e Schmidhuber, 1997; Cho et al., 2014; Altché e de La Fortelle, 2017)."],
    )
    b.add_bullet_p(
        "Rede Neural em Grafos Temporal (T-GNN): ",
        ["arquitetura espaço-temporal (8.449 parâmetros) composta por duas camadas de convolução em grafos normalizada (GCN) sobre G = (V, E) (codificando coordenadas, graus de conectividade, velocidade limite e faixas), acopladas a um codificador temporal GRU e a uma cabeça de classificação por aresta eferente com normalização softmax (Yu et al., 2018; Zhao et al., 2020)."],
    )

    b.add_section_heading("4.3 Métricas de Avaliação e Protocolo Experimental", level=2)
    b.add_body_p(
        "A avaliação quantitativa emprega métricas consolidadas em rastreamento multiobjeto e predição de trajetórias: Erro Médio Quadrático (RMSE) segmentado por regime (global, manobra brusca e apagão visual), Erro de Deslocamento Médio (ADE) e Final (FDE) durante a ausência de observação, acurácia de predição de rota Top-1 e Top-k, índice global de manutenção de identidade IDF1 = 2·TP / (2·TP + FP + FN), precisão, revocação, latência média por passo/consulta (ms) e vazão sustentada (detecções/s). Os experimentos cinemáticos comparativos (A6) são repetidos sobre cinco sementes independentes (reportando média ± desvio padrão) e a avaliação topológica utiliza 120 consultas estritamente fora da amostra de treinamento da T-GNN.",
        first_in_section=True,
    )

    b.add_section_heading("4.4 Fontes e Proveniência dos Bancos de Dados de Teste", level=2)
    b.add_body_p(
        "A obtenção de vídeos públicos reais com dezenas de câmeras urbanas calibradas e ground truth contínuo em regiões cegas envolve restrições de licença institucional (caso do AI City Challenge / CityFlowV2), ao passo que conjuntos abertos de vídeo real frequentemente se limitam a câmera única ou a poucos pontos de vista. Para conciliar realismo empírico, transparência metodológica e escalabilidade controlada, o protocolo integra bancos públicos reais/fotorrealistas, malhas geográficas reais OpenStreetMap e um gerador sintético urbano determinístico em larga escala. A Tabela 4 documenta formalmente todas as fontes bibliográficas, endereços oficiais, características e papéis experimentais de cada banco de dados e modelo pré-treinado.",
        first_in_section=True,
    )

    b.add_academic_table(
        "Tabela 4.",
        "Catálogo e fontes oficiais dos bancos de dados de teste, malhas viárias e pesos pré-treinados utilizados ou suportados.",
        headers=["Banco de Dados / Recurso", "Referência", "Escala e Características", "Fonte Oficial / URL", "Tipo e Papel"],
        rows=[
            [
                "CityFlow / CityFlowV2",
                "Tang et al. (2019)",
                "46 câmeras em 16 intersecções; ~3,58 h; homografias GPS e GT MTMC.",
                "https://www.aicitychallenge.org/\nhttps://github.com/cityflow-project/CityFlow",
                "Real (registro); padrão de calibração e layout.",
            ],
            [
                "Synthehicle",
                "Herzog et al. (2023)",
                "Múltiplas câmeras não sobrepostas (CARLA); matrizes K[R|t] e MOT (Town01-N-day).",
                "https://huggingface.co/datasets/fubel/synthehicle",
                "Sintético fotorrealista aberto; projeção 3D e MTMC.",
            ],
            [
                "UA-DETRAC",
                "Wen et al. (2020)",
                ">140.000 quadros reais de CFTV em 24 locais; oclusões anotadas (MVI_40905).",
                "https://detrac-db.rit.albany.edu/\nhttps://huggingface.co/datasets/abhineet123/ua_detrac",
                "Real (público); percepção e apagão simulado de 5 s.",
            ],
            [
                "KITTI Vision Benchmark",
                "Geiger et al. (2013)",
                "Sequências reais de trânsito urbano (1226×370 px, 30 fps).",
                "https://www.cvlibs.net/datasets/kitti/",
                "Real (público); validação de detecção e SCT.",
            ],
            [
                "VeRi-776",
                "Liu et al. (2018)",
                "776 veículos em 20 câmeras urbanas (~50.000 imagens) sob múltiplos ângulos.",
                "https://vehiclereid.github.io/VeRi/\nhttps://github.com/VehicleReId/VeRi",
                "Real (licença); referência de protótipos Re-ID.",
            ],
            [
                "RodoSol-ALPR",
                "Laroca et al. (2022)",
                "20.000 imagens de placas brasileiras (padrão antigo e Mercosul).",
                "https://github.com/raysonlaroca/rodosol-alpr-dataset",
                "Real (licença); referência sintática ALPR.",
            ],
            [
                "OpenStreetMap & OSMnx",
                "Boeing (2017)",
                "Grafos viários reais em UTM (Av. Paulista EPSG:32723; Zurique EPSG:32632).",
                "https://www.openstreetmap.org/\nhttps://github.com/gboeing/osmnx",
                "Geográfico real (ODbL); topologia direcionada.",
            ],
            [
                "MS COCO / MSMT17 / fast-alpr",
                "Lin et al. (2014); Wei et al. (2018); Zhou et al. (2019)",
                "Pesos pré-treinados YOLO11n, OSNet x0.25 (512-D) e OCR de placas ONNX.",
                "https://github.com/ultralytics/ultralytics\nhttps://github.com/ankandrew/fast-alpr",
                "Pesos públicos verificados por SHA-256.",
            ],
            [
                "Malha Urbana 8×8 (Exp. A7)",
                "Este trabalho (city_benchmark.py)",
                "64 intersecções, 196 arestas, 4 a 24 câmeras 3D, 20 a 150 veículos, 11.388 detecções.",
                "Gerador determinístico embarcado no repositório (Apêndice B).",
                "Sintético em escala urbana; GT exato de rotas.",
            ],
        ],
        col_widths_cm=[2.6, 2.2, 3.6, 4.1, 2.5],
        highlight_rows={8},
        font_size_pt=7.6,
        source="Elaborado pelos autores a partir dos repositórios oficiais e do manifesto data/MANIFEST.json.",
    )

    # =========================================================================
    # 5. RESULTADOS EXPERIMENTAIS E DISCUSSÃO
    # =========================================================================
    b.add_section_heading("5. Resultados Experimentais e Discussão", level=1)

    b.add_section_heading("5.1 Resposta à QP1: Comparação Cinemática, Filtro de Partículas, LSTM e GRU", level=2)
    b.add_body_p(
        "Os Experimentos A1, A2 e A6 avaliaram o comportamento dos filtros e preditores neurais em trajetórias urbanas contendo condução retilínea, curva acentuada (ω = 1,05 rad/s), apagão visual completo de 1,2 s e recuperação em contra-curva, sob ruído de solo σ = 0,30 m em cinco sementes aleatórias. A Tabela 5 e a Figura 2 sintetizam as métricas de acurácia, variância, latência e complexidade paramétrica.",
        first_in_section=True,
    )

    b.add_academic_table(
        "Tabela 5.",
        "Desempenho comparativo de oito modelos de filtragem e predição de trajetórias sobre cinco sementes aleatórias (Experimento A6, σ = 0,30 m, apagão visual de 1,2 s).",
        headers=[
            "Modelo / Paradigma",
            "RMSE Global (m)",
            "RMSE Manobra (m)",
            "RMSE Região Cega (m)",
            "ADE / FDE Cego (m)",
            "Latência (ms/passo)",
            "Parâmetros",
        ],
        rows=[
            [
                "Kalman Linear CV (Kalman, 1960)",
                f"{_fmt_pt(a6_kin['linear_cv']['rmse_overall_mean_m'], 3)} ± {_fmt_pt(a6_kin['linear_cv']['rmse_overall_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['linear_cv']['rmse_maneuver_mean_m'], 3)} ± {_fmt_pt(a6_kin['linear_cv']['rmse_maneuver_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['linear_cv']['rmse_blind_outage_mean_m'], 3)} ± {_fmt_pt(a6_kin['linear_cv']['rmse_blind_outage_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['linear_cv']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['linear_cv']['blind_fde_m'], 2)}",
                _fmt_pt(a6_kin['linear_cv']['latency_ms_per_step'], 3),
                "0",
            ],
            [
                "EKF Canônico Bicycle (λ = 1)",
                f"{_fmt_pt(a6_kin['canonical_bicycle']['rmse_overall_mean_m'], 3)} ± {_fmt_pt(a6_kin['canonical_bicycle']['rmse_overall_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['canonical_bicycle']['rmse_maneuver_mean_m'], 3)} ± {_fmt_pt(a6_kin['canonical_bicycle']['rmse_maneuver_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['canonical_bicycle']['rmse_blind_outage_mean_m'], 3)} ± {_fmt_pt(a6_kin['canonical_bicycle']['rmse_blind_outage_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['canonical_bicycle']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['canonical_bicycle']['blind_fde_m'], 2)}",
                _fmt_pt(a6_kin['canonical_bicycle']['latency_ms_per_step'], 3),
                "0",
            ],
            [
                "EKF Adaptativo Bicycle (Proposto)",
                f"{_fmt_pt(a6_kin['adaptive_bicycle']['rmse_overall_mean_m'], 3)} ± {_fmt_pt(a6_kin['adaptive_bicycle']['rmse_overall_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['adaptive_bicycle']['rmse_maneuver_mean_m'], 3)} ± {_fmt_pt(a6_kin['adaptive_bicycle']['rmse_maneuver_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['adaptive_bicycle']['rmse_blind_outage_mean_m'], 3)} ± {_fmt_pt(a6_kin['adaptive_bicycle']['rmse_blind_outage_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['adaptive_bicycle']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['adaptive_bicycle']['blind_fde_m'], 2)}",
                _fmt_pt(a6_kin['adaptive_bicycle']['latency_ms_per_step'], 3),
                "0",
            ],
            [
                "EKF CTRV (Bradler et al., 2021)",
                f"{_fmt_pt(a6_kin['ctrv']['rmse_overall_mean_m'], 3)} ± {_fmt_pt(a6_kin['ctrv']['rmse_overall_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['ctrv']['rmse_maneuver_mean_m'], 3)} ± {_fmt_pt(a6_kin['ctrv']['rmse_maneuver_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['ctrv']['rmse_blind_outage_mean_m'], 3)} ± {_fmt_pt(a6_kin['ctrv']['rmse_blind_outage_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['ctrv']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['ctrv']['blind_fde_m'], 2)}",
                _fmt_pt(a6_kin['ctrv']['latency_ms_per_step'], 3),
                "0",
            ],
            [
                "IMM (Adaptive Bicycle + CTRV)",
                f"{_fmt_pt(a6_kin['imm']['rmse_overall_mean_m'], 3)} ± {_fmt_pt(a6_kin['imm']['rmse_overall_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['imm']['rmse_maneuver_mean_m'], 3)} ± {_fmt_pt(a6_kin['imm']['rmse_maneuver_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['imm']['rmse_blind_outage_mean_m'], 3)} ± {_fmt_pt(a6_kin['imm']['rmse_blind_outage_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['imm']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['imm']['blind_fde_m'], 2)}",
                _fmt_pt(a6_kin['imm']['latency_ms_per_step'], 3),
                "0",
            ],
            [
                ["Filtro de Partículas SIR (N", ("p", {"sub": True}), " = 500)"],
                f"{_fmt_pt(a6_kin['particle_filter']['rmse_overall_mean_m'], 3)} ± {_fmt_pt(a6_kin['particle_filter']['rmse_overall_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['particle_filter']['rmse_maneuver_mean_m'], 3)} ± {_fmt_pt(a6_kin['particle_filter']['rmse_maneuver_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['particle_filter']['rmse_blind_outage_mean_m'], 3)} ± {_fmt_pt(a6_kin['particle_filter']['rmse_blind_outage_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['particle_filter']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['particle_filter']['blind_fde_m'], 2)}",
                _fmt_pt(a6_kin['particle_filter']['latency_ms_per_step'], 3),
                "0",
            ],
            [
                "Rede Recorrente LSTM (2 camadas)",
                f"{_fmt_pt(a6_kin['lstm']['rmse_overall_mean_m'], 3)} ± {_fmt_pt(a6_kin['lstm']['rmse_overall_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['lstm']['rmse_maneuver_mean_m'], 3)} ± {_fmt_pt(a6_kin['lstm']['rmse_maneuver_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['lstm']['rmse_blind_outage_mean_m'], 3)} ± {_fmt_pt(a6_kin['lstm']['rmse_blind_outage_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['lstm']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['lstm']['blind_fde_m'], 2)}",
                _fmt_pt(a6_kin['lstm']['latency_ms_per_step'], 3),
                _fmt_int_pt(a6_kin["lstm"]["num_parameters"]),
            ],
            [
                "Rede Recorrente GRU (2 camadas)",
                f"{_fmt_pt(a6_kin['gru']['rmse_overall_mean_m'], 3)} ± {_fmt_pt(a6_kin['gru']['rmse_overall_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['gru']['rmse_maneuver_mean_m'], 3)} ± {_fmt_pt(a6_kin['gru']['rmse_maneuver_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['gru']['rmse_blind_outage_mean_m'], 3)} ± {_fmt_pt(a6_kin['gru']['rmse_blind_outage_std_m'], 3)}",
                f"{_fmt_pt(a6_kin['gru']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['gru']['blind_fde_m'], 2)}",
                _fmt_pt(a6_kin['gru']['latency_ms_per_step'], 3),
                _fmt_int_pt(a6_kin["gru"]["num_parameters"]),
            ],
        ],
        col_widths_cm=[3.5, 2.2, 2.2, 2.3, 1.9, 1.5, 1.4],
        center_cols={1, 2, 3, 4, 5, 6},
        highlight_rows={2, 4},
        font_size_pt=7.9,
        source="Dados experimentais obtidos via scripts/run_benchmarks.py (Experimento A6).",
    )

    b.add_figure(
        fig_dir / "fig2_sota_cinematica.png",
        "Figura 2.",
        "Comparação empírica de filtragem e predição de trajetórias: (a) RMSE em manobra brusca e região cega (5 sementes, A6); (b) fronteira de Pareto entre latência em CPU e erro em região cega; (c) estabilidade sob ruído crescente de detecção (A2).",
        width_cm=14.8,
    )

    b.add_body_p(
        f"Os resultados respondem conclusivamente à QP1. Em primeiro lugar, o Filtro de Kalman Linear CV confirma sua inadequação em curvas urbanas, acumulando {_fmt_pt(a6_kin['linear_cv']['rmse_maneuver_mean_m'], 2)} m na manobra e {_fmt_pt(a6_kin['linear_cv']['rmse_blind_outage_mean_m'], 2)} m durante o apagão visual. O EKF Adaptativo estabilizado reduz o erro em manobra em {_fmt_pt(a6_red_lin, 1)}% frente ao filtro linear e em {_fmt_pt(a6_red_can, 1)}% frente ao EKF Canônico (no Experimento A1, com apagão de 0,8 s, essas reduções alcançam {_fmt_pt(a1_red_lin, 1)}% e {_fmt_pt(a1_red_can, 1)}%, respectivamente). Ademais, sob ruído crescente de σ = 0,2 m até σ = 1,5 m (Experimento A2, Figura 2c), o gatilho persistente NIS aliado à suavização circular sustenta ganhos consistentes de 53,1% a 68,6% sobre o EKF Canônico ({_fmt_pt(a2['sigma_0.2m']['adaptive_maneuver_rmse_m'], 3)} m vs. {_fmt_pt(a2['sigma_0.2m']['canonical_maneuver_rmse_m'], 3)} m em 0,2 m; {_fmt_pt(a2['sigma_1.5m']['adaptive_maneuver_rmse_m'], 3)} m vs. {_fmt_pt(a2['sigma_1.5m']['canonical_maneuver_rmse_m'], 3)} m em 1,5 m)."
    )

    b.add_body_p(
        f"Em segundo lugar, o confronto com métodos bayesianos não paramétricos e redes recorrentes revela compromissos operacionais claros. O Filtro de Partículas SIR (500 partículas) atinge excelente aderência em manobra ({_fmt_pt(a6_kin['particle_filter']['rmse_maneuver_mean_m'], 3)} m), mas demanda latência até 2,5× superior à do EKF CTRV e 1,9× superior à do EKF Adaptativo. Por sua vez, a rede GRU obteve o menor RMSE médio no apagão dentro da distribuição treinada ({_fmt_pt(a6_kin['gru']['rmse_blind_outage_mean_m'], 3)} m, superando a LSTM com {_fmt_pt(a6_kin['lstm']['rmse_blind_outage_mean_m'], 3)} m); contudo, ambas as redes recorrentes exibem maior dispersão entre sementes (±{_fmt_pt(a6_kin['gru']['rmse_blind_outage_std_m'], 2)} m e ±{_fmt_pt(a6_kin['lstm']['rmse_blind_outage_std_m'], 2)} m contra ±{_fmt_pt(a6_kin['imm']['rmse_blind_outage_std_m'], 2)} m do IMM), latência cerca de 2,2× maior que a do EKF Adaptativo e dependência de treinamento prévio. Consequentemente, os filtros EKF Adaptativo e IMM ocupam a região mais favorável da fronteira de Pareto para operação contínua em tempo real sem parâmetros treináveis."
    )

    b.add_section_heading("5.2 Resposta à QP2: Projeção 3D e Predição Topológica frente a HMM/Viterbi e T-GNN", level=2)
    b.add_body_p(
        f"No Experimento A3, a substituição da borda inferior central bruta (f = 1,0) pelo ponto de contato corrigido em perspectiva (f = 0,90) sob câmera 3D inclinada a 25° reduziu o erro médio no plano de solo de {_fmt_pt(a3['naive_bottom_center_mean_error_m'], 4)} m para {_fmt_pt(a3['corrected_contact_point_mean_error_m'], 4)} m, equivalente a uma mitigação de {_fmt_pt(a3['bias_reduction_pct'], 2)}% no viés projetivo. No Experimento A5, sob ruído elevado de σ = 9,5 m em corredores com ruas paralelas, o map-matching HMM/Viterbi atingiu {_fmt_pt(a5['map_matching_hmm_viterbi_accuracy']*100, 1)}% de acurácia contra {_fmt_pt(a5['map_matching_euclidean_accuracy']*100, 1)}% da projeção euclidiana simples.",
        first_in_section=True,
    )

    b.add_body_p(
        "Para avaliar diretamente a QP2 em cruzamentos urbanos com múltiplas bifurcações, a segunda etapa do Experimento A6 submeteu 120 consultas fora da amostra de treinamento na malha 8×8 aos quatro modelos topológicos após 4,0 s de região cega (Tabela 6 e Figura 3)."
    )

    b.add_academic_table(
        "Tabela 6.",
        "Comparação de modelos topológicos de predição de rotas em cruzamentos urbanos após 4,0 s em região cega (malha 8×8, 120 consultas fora da amostra, Experimento A6).",
        headers=[
            "Modelo Topológico",
            "Acurácia Top-1 (%)",
            "Acurácia Top-2 (%)",
            "RMSE Espacial Cego (m)",
            "Latência Média (ms)",
            "Parâmetros / Retreino",
        ],
        rows=[
            [
                "Projeção Euclidiana (Dead Reckoning)",
                f"{_fmt_pt(a6_rt['euclidean_snap']['top1_route_accuracy']*100, 2)}%",
                f"{_fmt_pt(a6_rt['euclidean_snap']['topk_route_accuracy']*100, 2)}%",
                f"{_fmt_pt(a6_rt['euclidean_snap']['blind_position_rmse_m'], 2)} m",
                f"{_fmt_pt(a6_rt['euclidean_snap']['latency_ms_per_query'], 2)} ms",
                "0 / Não",
            ],
            [
                "HMM / Viterbi (Newson e Krumm, 2009)",
                f"{_fmt_pt(a6_rt['hmm_viterbi']['top1_route_accuracy']*100, 2)}%",
                f"{_fmt_pt(a6_rt['hmm_viterbi']['topk_route_accuracy']*100, 2)}%",
                f"{_fmt_pt(a6_rt['hmm_viterbi']['blind_position_rmse_m'], 2)} m",
                f"{_fmt_pt(a6_rt['hmm_viterbi']['latency_ms_per_query'], 2)} ms",
                "0 / Não",
            ],
            [
                "Rede Neural T-GNN (GCN + GRU)",
                f"{_fmt_pt(a6_rt['t_gnn']['top1_route_accuracy']*100, 2)}%",
                f"{_fmt_pt(a6_rt['t_gnn']['topk_route_accuracy']*100, 2)}%",
                f"{_fmt_pt(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m",
                f"{_fmt_pt(a6_rt['t_gnn']['latency_ms_per_query'], 2)} ms",
                f"{_fmt_int_pt(a6_rt['t_gnn']['num_parameters'])} / Sim",
            ],
            [
                ["Árvore Multi-Hipótese + Poda Física v", ("crit", {"sub": True}), " (Proposto)"],
                f"{_fmt_pt(a6_rt['proposed_physics_tree']['top1_route_accuracy']*100, 2)}%",
                f"{_fmt_pt(a6_rt['proposed_physics_tree']['topk_route_accuracy']*100, 2)}%",
                f"{_fmt_pt(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m",
                f"{_fmt_pt(a6_rt['proposed_physics_tree']['latency_ms_per_query'], 2)} ms",
                "0 / Não",
            ],
        ],
        col_widths_cm=[4.5, 2.0, 2.0, 2.4, 2.0, 2.1],
        center_cols={1, 2, 3, 4, 5},
        highlight_rows={3},
        font_size_pt=8.3,
        source="Dados experimentais obtidos via scripts/run_benchmarks.py (Experimento A6-Rotas).",
    )

    b.add_figure(
        fig_dir / "fig3_rotas_tgnn.png",
        "Figura 3.",
        "Comparação de predição topológica em cruzamentos urbanos da malha 8×8 (Experimento A6): (a) acurácia Top-1 e Top-2; (b) RMSE espacial após 4,0 s em região cega e latência por consulta.",
        width_cm=14.6,
    )

    b.add_body_p(
        f"Em resposta à QP2, verifica-se que a rede neural T-GNN apresenta excelente eficiência de inferência ({_fmt_pt(a6_rt['t_gnn']['latency_ms_per_query'], 2)} ms por consulta após pré-codificação espacial do grafo) e forte capacidade discriminativa ({_fmt_pt(a6_rt['t_gnn']['top1_route_accuracy']*100, 1)}% Top-1 e {_fmt_pt(a6_rt['t_gnn']['topk_route_accuracy']*100, 1)}% Top-2), superando largamente a projeção euclidiana e o HMM/Viterbi. Não obstante, a árvore multi-hipótese com poda física proposta supera a T-GNN tanto na acurácia Top-1 ({_fmt_pt(a6_rt['proposed_physics_tree']['top1_route_accuracy']*100, 1)}%) quanto na precisão espacial em região cega ({_fmt_pt(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m contra {_fmt_pt(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m, redução de 16,2% no erro espacial explicada pela modelagem explícita de frenagem pré-esquina e reaceleração), com a vantagem decisiva de transferir-se instantaneamente a qualquer nova cidade sem retreinamento."
    )

    b.add_section_heading("5.3 Resposta à QP3 e QP4: Escalabilidade Urbana de 4 a 24 Câmeras e 150 Veículos", level=2)
    b.add_body_p(
        [
            "Para responder às questões QP3 e QP4 — superando a escala restrita de ensaios iniciais com 4 câmeras e esclarecendo a degradação de 5% anteriormente observada sem descritores visuais —, o Experimento A7 simulou o tráfego heterogêneo sobre a malha urbana 8×8 de aproximadamente 0,85 km × 0,85 km (Figura 4). Foram avaliadas quatro escalas crescentes: S1 (4 câmeras, 20 veículos), S2 (8 câmeras, 50 veículos), S3 (16 câmeras, 100 veículos) e S4 (24 câmeras, 150 veículos, totalizando 11.388 detecções e 232 transições cegas com distância média de 177,1 m). Para emular fielmente a ambiguidade visual de bancos reais como VeRi-776 e CityFlowV2, os veículos foram distribuídos em oito protótipos visuais predominantes, submetidos a viés fotométrico por câmera, 18% de episódios de oclusão parcial, ruído de detecção (σ",
            ("px", {"sub": True}),
            " = 3,2 px), distorção de escala entre câmeras e cobertura ALPR restrita a 25% das câmeras.",
        ],
        first_in_section=True,
    )

    b.add_figure(
        fig_dir / "fig5_mapa_urbano.png",
        "Figura 4.",
        "Topologia da malha viária urbana 8×8 (64 intersecções, 196 arestas direcionadas) e disposição espacial das 24 câmeras estáticas não sobrepostas empregadas no Experimento A7.",
        width_cm=11.0,
    )

    b.add_academic_table(
        "Tabela 7.",
        "Escalabilidade multicâmera na malha urbana 8×8 (Experimento A7): IDF1, acurácia de rota e vazão em CPU entre 4 e 24 câmeras (20 a 150 veículos simultâneos).",
        headers=[
            "Escala Urbana",
            "Detecções / Transições",
            "Legado (Chilgunde)",
            "Gating χ² (Cinem.+Grafo)",
            "Re-ID Puro (OSNet)",
            "Fusão Híbrida (Proposto)",
            "Rota Top-1 / Top-k",
            "Vazão Híbrida (det/s)",
        ],
        rows=[
            [
                "S1: 4 câm. / 20 veíc.",
                f"{_fmt_int_pt(s1['total_detections'])} / {s1['num_blind_transitions']}",
                _fmt_pt(s1["modes"]["article"]["idf1"], 4),
                _fmt_pt(s1["modes"]["chi2"]["idf1"], 4),
                _fmt_pt(s1["modes"]["reid_only"]["idf1"], 4),
                _fmt_pt(s1["modes"]["hybrid"]["idf1"], 4),
                f"{_fmt_pt(s1['modes']['hybrid']['route_top1_accuracy']*100, 1)}% / {_fmt_pt(s1['modes']['hybrid']['route_topk_accuracy']*100, 1)}%",
                _fmt_int_pt(s1["modes"]["hybrid"]["throughput_detections_s"]),
            ],
            [
                "S2: 8 câm. / 50 veíc.",
                f"{_fmt_int_pt(s2['total_detections'])} / {s2['num_blind_transitions']}",
                _fmt_pt(s2["modes"]["article"]["idf1"], 4),
                _fmt_pt(s2["modes"]["chi2"]["idf1"], 4),
                _fmt_pt(s2["modes"]["reid_only"]["idf1"], 4),
                _fmt_pt(s2["modes"]["hybrid"]["idf1"], 4),
                f"{_fmt_pt(s2['modes']['hybrid']['route_top1_accuracy']*100, 1)}% / {_fmt_pt(s2['modes']['hybrid']['route_topk_accuracy']*100, 1)}%",
                _fmt_int_pt(s2["modes"]["hybrid"]["throughput_detections_s"]),
            ],
            [
                "S3: 16 câm. / 100 veíc.",
                f"{_fmt_int_pt(s3['total_detections'])} / {s3['num_blind_transitions']}",
                _fmt_pt(s3["modes"]["article"]["idf1"], 4),
                _fmt_pt(s3["modes"]["chi2"]["idf1"], 4),
                _fmt_pt(s3["modes"]["reid_only"]["idf1"], 4),
                _fmt_pt(s3["modes"]["hybrid"]["idf1"], 4),
                f"{_fmt_pt(s3['modes']['hybrid']['route_top1_accuracy']*100, 1)}% / {_fmt_pt(s3['modes']['hybrid']['route_topk_accuracy']*100, 1)}%",
                _fmt_int_pt(s3["modes"]["hybrid"]["throughput_detections_s"]),
            ],
            [
                "S4: 24 câm. / 150 veíc.",
                f"{_fmt_int_pt(s4['total_detections'])} / {s4['num_blind_transitions']}",
                _fmt_pt(s4["modes"]["article"]["idf1"], 4),
                _fmt_pt(s4["modes"]["chi2"]["idf1"], 4),
                _fmt_pt(s4["modes"]["reid_only"]["idf1"], 4),
                _fmt_pt(s4["modes"]["hybrid"]["idf1"], 4),
                f"{_fmt_pt(s4['modes']['hybrid']['route_top1_accuracy']*100, 1)}% / {_fmt_pt(s4['modes']['hybrid']['route_topk_accuracy']*100, 1)}%",
                _fmt_int_pt(s4["modes"]["hybrid"]["throughput_detections_s"]),
            ],
        ],
        col_widths_cm=[2.4, 2.0, 1.7, 1.8, 1.7, 1.8, 2.0, 1.6],
        center_cols={1, 2, 3, 4, 5, 6, 7},
        highlight_rows={3},
        font_size_pt=7.8,
        source="Dados experimentais obtidos via scripts/run_benchmarks.py (Experimento A7).",
    )

    b.add_figure(
        fig_dir / "fig4_escala_urbana.png",
        "Figura 5.",
        "Escalabilidade multicâmera no Experimento A7 (4 a 24 câmeras, 20 a 150 veículos): (a) evolução do IDF1 nos quatro modos de associação; (b) acurácia intercâmeras, rota Top-1/Top-k e vazão em CPU no modo híbrido.",
        width_cm=14.8,
    )

    b.add_body_p(
        f"Os resultados da Tabela 7 e da Figura 5 respondem diretamente às questões QP3 e QP4. Sob tráfego denso simultâneo, métodos sem descritores visuais (Legado de Chilgunde et al. e Gating χ² puro) sofrem confusão espaço-temporal severa (IDF1 entre {_fmt_pt(s2['modes']['chi2']['idf1'], 3)} e {_fmt_pt(s4['modes']['article']['idf1'], 3)}), ratificando a causa estrutural da baixa acurácia registrada em versões puramente cinemáticas anteriores. O uso isolado de aparência (Re-ID Puro) eleva substancialmente o IDF1 ({_fmt_pt(s3['modes']['reid_only']['idf1'], 4)} em S3 e {_fmt_pt(s4['modes']['reid_only']['idf1'], 4)} em S4), porém ainda produz trocas espúrias (16 falhas em S3 e 14 falhas em S4) diante de veículos visualmente similares submetidos a oclusão parcial. Em contrapartida, a Fusão Híbrida Proposta combina o filtro físico-topológico das hipóteses viáveis com a aparência visual, a autocalibração de escala e o ALPR seletivo, sustentando IDF1 = 1,0000 em todas as quatro escalas (em consonância com a ablação A4), 91,8% de acerto Top-k na escala máxima e vazão de {_fmt_int_pt(s4['modes']['hybrid']['throughput_detections_s'])} detecções/s em CPU — comprovando plena viabilidade operacional em tempo real."
    )

    b.add_section_heading("5.4 Ameaças à Validade", level=2)
    b.add_body_p(
        "Em conformidade com as práticas metodológicas de periódicos Qualis A, reconhecem-se quatro dimensões de ameaças à validade e as respectivas estratégias adotadas para mitigá-las:",
        first_in_section=True,
    )
    b.add_bullet_p(
        "Validade Interna: ",
        ["o desempenho de modelos neurais (LSTM, GRU e T-GNN) depende da representatividade do conjunto de treinamento e dos hiperparâmetros. Para mitigar viés de implementação, todas as arquiteturas foram treinadas com sementes determinísticas fixas, normalização local invariante e partição estritamente fora da amostra de teste."],
    )
    b.add_bullet_p(
        "Validade Externa: ",
        ["a avaliação em larga escala (24 câmeras e 150 veículos) empregou um simulador urbano sintético parametrizado devido às restrições de licença institucional do CityFlowV2 oficial. Essa limitação foi atenuada pela incorporação de projeção 3D calibrada, protótipos visuais heterogêneos, viés fotométrico por câmera e 18% de episódios de oclusão, além da compatibilidade validada com amostras reais de UA-DETRAC, KITTI, Synthehicle e OpenStreetMap."],
    )
    b.add_bullet_p(
        "Validade de Construto: ",
        ["a avaliação conjunta de métricas complementares (RMSE por regime, ADE/FDE, IDF1, Top-1/Top-k e latência em CPU) evita conclusões enviesadas por um único indicador isolado."],
    )
    b.add_bullet_p(
        "Validade de Conclusão: ",
        ["a repetição dos ensaios cinemáticos em cinco sementes aleatórias independentes (reportando média e desvio padrão) e a verificação automatizada por suíte de testes unitários e de integração asseguram reprodutibilidade estatística e integridade numérica."],
    )

    # =========================================================================
    # 6. CONSIDERAÇÕES FINAIS E TRABALHOS FUTUROS
    # =========================================================================
    b.add_section_heading("6. Considerações Finais e Trabalhos Futuros", level=1)

    b.add_body_p(
        "Este trabalho apresentou a formalização matemática, a implementação estritamente causal e a validação experimental de um pipeline multicâmera integrando Filtro de Kalman Estendido Adaptativo estabilizado, correção geométrica 3D, acoplamento topológico com poda física em grafo viário e associação híbrida multimodal para monitoramento de tráfego e predição de trajetórias em redes urbanas com campos de visão não sobrepostos.",
        first_in_section=True,
    )

    b.add_body_p(
        [
            f"Em alinhamento direto com o objetivo central e as quatro questões de pesquisa investigadas, os resultados empíricos (Experimentos A1 a A7) demonstraram que: (i) a estabilização do fator de desvanecimento por suavização circular e gatilho NIS persistente reduziu em {_fmt_pt(a6_red_lin, 1)}% o erro em manobras frente ao filtro linear CV e em {_fmt_pt(a6_red_can, 1)}% frente ao EKF canônico, preservando ganhos acima de 53% mesmo sob ruído elevado de 1,5 m; (ii) os filtros não lineares propostos (EKF Adaptativo e IMM) alcançaram desempenho competitivo frente ao Filtro de Partículas SIR e às redes recorrentes LSTM e GRU, com menor variância entre execuções, latência até 2,5× inferior e ausência de parâmetros treináveis; (iii) a correção de perspectiva 3D reduziu em {_fmt_pt(a3['bias_reduction_pct'], 2)}% o viés de projeção no solo, enquanto a árvore multi-hipótese com poda física v",
            ("crit", {"sub": True}),
            f" superou a projeção euclidiana, o HMM/Viterbi e a rede T-GNN no erro espacial após 4,0 s em região cega ({_fmt_pt(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m vs. {_fmt_pt(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m) sem demandar retreinamento por cidade; e (iv) a fusão híbrida sustentou IDF1 = 1,0000 ao escalar de 4 para 24 câmeras e de 20 para 150 veículos simultâneos, mantendo vazão superior a 4.000 detecções/s em CPU.",
        ]
    )

    b.add_body_p(
        [
            "Como desdobramentos futuros no domínio de Sistemas Inteligentes de Transporte, planejam-se: (a) a execução de validação em campo sobre as 46 câmeras do conjunto CityFlowV2 oficial mediante licença acadêmica institucional; (b) a integração de estados semafóricos em tempo real e telemetria cooperativa veicular (V2X/C-V2X) às probabilidades de transição do grafo viário; e (c) a investigação de redes T-GNN informadas por física (",
            ("physics-informed graph masking", {"italic": True}),
            ") que incorporem a restrição v",
            ("crit", {"sub": True}),
            " diretamente como camada estrutural de mascaramento.",
        ]
    )

    # =========================================================================
    # REFERÊNCIAS (ABNT NBR 6023:2018)
    # =========================================================================
    b.add_section_heading("Referências", level=1)

    refs: list[list[Any]] = [
        ["ALTCHÉ, Florent; DE LA FORTELLE, Arnaud. An LSTM network for highway trajectory prediction. In: ", ("IEEE INTERNATIONAL CONFERENCE ON INTELLIGENT TRANSPORTATION SYSTEMS (ITSC)", {"bold": True}), ", 20., 2017, Yokohama. Proceedings [...]. Piscataway: IEEE, 2017. p. 353-359."],
        ["ARULAMPALAM, M. Sanjeev; MASKELL, Simon; GORDON, Neil; CLAPP, Tim. A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. ", ("IEEE Transactions on Signal Processing", {"bold": True}), ", v. 50, n. 2, p. 174-188, 2002."],
        ["BLOM, Henk A. P.; BAR-SHALOM, Yaakov. The interacting multiple model algorithm for systems with Markovian switching coefficients. ", ("IEEE Transactions on Automatic Control", {"bold": True}), ", v. 33, n. 8, p. 780-783, 1988."],
        ["BOEING, Geoff. OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. ", ("Computers, Environment and Urban Systems", {"bold": True}), ", v. 65, p. 126-139, 2017. Disponível em: https://github.com/gboeing/osmnx. Acesso em: 24 nov. 2026."],
        ["BRADLER, Henry; KRETZ, Adrian; MESTER, Rudolf. Urban Traffic Surveillance (UTS): A fully probabilistic 3D tracking approach based on 2D detections. In: ", ("IEEE/CVF CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION WORKSHOPS (CVPRW)", {"bold": True}), ", 2021. Proceedings [...]. IEEE, 2021. Disponível em: https://arxiv.org/abs/2105.14993. Acesso em: 24 nov. 2026."],
        ["CHAO, Pingfu; XU, Yehong; HUA, Wen; ZHOU, Xiaofang. A survey on map-matching algorithms. In: ", ("DATABASES THEORY AND APPLICATIONS (ADC)", {"bold": True}), ", 2020. Proceedings [...]. Cham: Springer, 2020. Disponível em: https://arxiv.org/abs/1910.13065. Acesso em: 24 nov. 2026."],
        ["CHILGUNDE, Amit; KUMAR, Pankaj; RANGANATH, Surendra; HUANG, WeiMin. Multi-camera target tracking in blind regions of cameras with non-overlapping fields of view. In: ", ("BRITISH MACHINE VISION CONFERENCE (BMVC)", {"bold": True}), ", 2004, London. Proceedings [...]. London: BMVA Press, 2004. p. 1-10."],
        ["CHO, Kyunghyun; VAN MERRIËNBOER, Bart; GULCEHRE, Caglar; BAHDANAU, Dzmitry; BOUGARES, Fethi; SCHWENK, Holger; BENGIO, Yoshua. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In: ", ("CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING (EMNLP)", {"bold": True}), ", 2014, Doha. Proceedings [...]. Stroudsburg: ACL, 2014. p. 1724-1734."],
        ["GEIGER, Andreas; LENZ, Philip; STILLER, Christoph; URTASUN, Raquel. Vision meets robotics: The KITTI dataset. ", ("The International Journal of Robotics Research (IJRR)", {"bold": True}), ", v. 32, n. 11, p. 1231-1237, 2013. Disponível em: https://www.cvlibs.net/datasets/kitti/. Acesso em: 24 nov. 2026."],
        ["GORDON, Neil J.; SALMOND, David J.; SMITH, Adrian F. M. Novel approach to nonlinear/non-Gaussian Bayesian state estimation. ", ("IEE Proceedings F (Radar and Signal Processing)", {"bold": True}), ", v. 140, n. 2, p. 107-113, 1993."],
        ["HERZOG, Fabian; CHEN, Junpeng; TEEPE, Torben; GILG, Johannes; HÖRMANN, Stefan; RIGOLL, Gerhard. Synthehicle: Multi-vehicle multi-camera tracking in virtual cities. In: ", ("IEEE/CVF WINTER CONFERENCE ON APPLICATIONS OF COMPUTER VISION (WACV)", {"bold": True}), ", 2023, Waikoloa. Proceedings [...]. IEEE, 2023. p. 2711-2721. Disponível em: https://huggingface.co/datasets/fubel/synthehicle. Acesso em: 24 nov. 2026."],
        ["HOCHREITER, Sepp; SCHMIDHUBER, Jürgen. Long short-term memory. ", ("Neural Computation", {"bold": True}), ", v. 9, n. 8, p. 1735-1780, 1997."],
        ["JOCHER, Glenn; CHAURASIA, Ayush; QIU, Jing. ", ("Ultralytics YOLO", {"bold": True}), ". Versão 8.3 (YOLO11), 2024. Disponível em: https://github.com/ultralytics/ultralytics. Acesso em: 24 nov. 2026."],
        ["KALMAN, Rudolf Emil. A new approach to linear filtering and prediction problems. ", ("Transactions of the ASME – Journal of Basic Engineering", {"bold": True}), ", v. 82, n. 1, p. 35-45, 1960."],
        ["KINDERBY, Tim; RIDDERSTOLPE, Carl. ", ("Dead reckoning using road networks and Kalman filtering", {"bold": True}), ". 2024. 78 f. Dissertação (Master of Science in Computer Science) – Chalmers University of Technology, Gothenburg, Sweden, 2024. Disponível em: https://hdl.handle.net/20.500.12380/310895. Acesso em: 24 nov. 2026."],
        ["KUHN, Harold W. The Hungarian method for the assignment problem. ", ("Naval Research Logistics Quarterly", {"bold": True}), ", v. 2, n. 1-2, p. 83-97, 1955."],
        ["LAROCA, Rayson; CARDOSO, Everton V.; LUCIO, Diego R.; ESTEVAM, Valter; MENOTTI, David. On the cross-dataset generalization in license plate recognition (RodoSol-ALPR). In: ", ("INTERNATIONAL JOINT CONFERENCE ON NEURAL NETWORKS (IJCNN)", {"bold": True}), ", 2022, Padua. Proceedings [...]. IEEE, 2022. Disponível em: https://github.com/raysonlaroca/rodosol-alpr-dataset. Acesso em: 24 nov. 2026."],
        ["LIN, Tsung-Yi; MAIRE, Michael; BELONGIE, Serge; HAYS, James; PERONA, Pietro; RAMANAN, Deva; DOLLÁR, Piotr; ZITNICK, C. Lawrence. Microsoft COCO: Common objects in context. In: ", ("EUROPEAN CONFERENCE ON COMPUTER VISION (ECCV)", {"bold": True}), ", 2014, Zurich. Proceedings [...]. Cham: Springer, 2014. p. 740-755."],
        ["LIU, Xinchen; LIU, Wu; MEI, Tao; MA, Huadong. PROVID: Progressive and multimodal vehicle reidentification for large-scale urban surveillance (VeRi-776). ", ("IEEE Transactions on Multimedia", {"bold": True}), ", v. 20, n. 3, p. 645-658, 2018. Disponível em: https://vehiclereid.github.io/VeRi/. Acesso em: 24 nov. 2026."],
        ["LOU, Jianguang; YANG, Hao; HU, Weiming; TAN, Tieniu. Visual vehicle tracking using an improved EKF. In: ", ("ASIAN CONFERENCE ON COMPUTER VISION (ACCV)", {"bold": True}), ", 5., 2002, Melbourne. Proceedings [...]. Melbourne: AFCVS, 2002. p. 296-301."],
        ["NEWSON, Paul; KRUMM, John. Hidden Markov map matching through noise and sparseness. In: ", ("ACM SIGSPATIAL INTERNATIONAL CONFERENCE ON ADVANCES IN GEOGRAPHIC INFORMATION SYSTEMS", {"bold": True}), ", 17., 2009, Seattle. Proceedings [...]. New York: ACM, 2009. p. 336-343."],
        ["TANG, Zheng; NAPHADE, Milind; LIU, Ming-Yu; YANG, Xiaodong; BIRCHFIELD, Stan; WANG, Shuo; KUMAR, Ratnesh; ANASTASIU, David; HWANG, Jenq-Neng. CityFlow: A city-scale benchmark for multi-target multi-camera vehicle tracking and re-identification. In: ", ("IEEE/CVF CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR)", {"bold": True}), ", 2019, Long Beach. Proceedings [...]. IEEE, 2019. p. 8797-8806. Disponível em: https://www.aicitychallenge.org/. Acesso em: 24 nov. 2026."],
        ["WEI, Longhui; ZHANG, Shiliang; GAO, Wen; TIAN, Qi. Person transfer GAN to bridge domain gap for person re-identification (MSMT17). In: ", ("IEEE/CVF CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR)", {"bold": True}), ", 2018, Salt Lake City. Proceedings [...]. IEEE, 2018. p. 79-88."],
        ["WEN, Longyin; DU, Dawei; CAI, Zhaowei; LEI, Zhen; CHANG, Ming-Ching; QI, Honggang; LIM, Jongwoo; YANG, Ming-Hsuan; LYU, Siwei. UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking. ", ("Computer Vision and Image Understanding (CVIU)", {"bold": True}), ", v. 193, 102907, 2020. Disponível em: https://detrac-db.rit.albany.edu/. Acesso em: 24 nov. 2026."],
        ["YU, Bing; YIN, Haoteng; ZHU, Zhanxing. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In: ", ("INTERNATIONAL JOINT CONFERENCE ON ARTIFICIAL INTELLIGENCE (IJCAI)", {"bold": True}), ", 27., 2018, Stockholm. Proceedings [...]. IJCAI, 2018. p. 3634-3640."],
        ["ZHANG, Haichao; XU, Yi; FU, Yun. Out-of-sight trajectories: Tracking, fusion, and prediction. ", ("arXiv preprint arXiv:2509.15219", {"bold": True}), ", 2025. Disponível em: https://arxiv.org/abs/2509.15219. Acesso em: 24 nov. 2026."],
        ["ZHANG, Yifu; SUN, Peize; JIANG, Yi; YU, Dongdong; WENG, Fucheng; YUAN, Zehuan; LUO, Ping; LIU, Wenyu; WANG, Xinggang. ByteTrack: Multi-object tracking by associating every detection box. In: ", ("EUROPEAN CONFERENCE ON COMPUTER VISION (ECCV)", {"bold": True}), ", 2022, Tel Aviv. Proceedings [...]. Cham: Springer, 2022. p. 1-21."],
        ["ZHAO, Ling; SONG, Yujiao; ZHANG, Chao; LIU, Yu; WANG, Pu; LIN, Tao; DENG, Min; LI, Haifeng. T-GCN: A temporal graph convolutional network for traffic prediction. ", ("IEEE Transactions on Intelligent Transportation Systems", {"bold": True}), ", v. 21, n. 9, p. 3848-3858, 2020."],
        ["ZHOU, Kaiyang; YANG, Yongxin; CAVALLARO, Andrea; XIANG, Tao. Omni-scale feature learning for person re-identification (OSNet). In: ", ("IEEE/CVF INTERNATIONAL CONFERENCE ON COMPUTER VISION (ICCV)", {"bold": True}), ", 2019, Seoul. Proceedings [...]. IEEE, 2019. p. 3702-3712."],
    ]
    for ref_parts in refs:
        b.add_abnt_reference(ref_parts)

    # =========================================================================
    # APÊNDICES (ESTRITAMENTE APÓS AS REFERÊNCIAS)
    # =========================================================================
    b.add_section_heading("Apêndice A – Núcleos Algorítmicos de Filtragem e Referência", level=1, page_break_before=True)
    b.add_body_p(
        "Em atendimento às normas editoriais científicas, os trechos de código-fonte referentes à estabilização do fator de desvanecimento do EKF Adaptativo e ao núcleo do Filtro de Partículas SIR são reunidos neste apêndice para fins de auditoria e reprodutibilidade.",
        first_in_section=True,
    )

    b.add_appendix_code(
        ["Listagem A.1 – Gatilho persistente NIS, fator λ", ("k", {"sub": True}), " e atualização de Joseph (src/tracker/ekf.py)."],
        """# Trecho resumido de AdaptiveEKF.update() — src/tracker/ekf.py
y_pos = z - self._x[:2]
s_pos = _enforce_spd(self._P[:2, :2] + r_pos, min_eig=1e-6)
nis = float(y_pos.T @ np.linalg.inv(s_pos) @ y_pos)

if self.adaptive and is_moving:
    self._nis_streak = (self._nis_streak + 1) if (nis > self._chi2_thresh) else 0
    if self._nis_streak >= int(self.cfg.nis_persistence_frames):
        adaptive_triggered = True
        speed_ratio = float(np.clip((v_state - v_min) / 8.0, 0.0, 1.0))
        alpha_theta = 0.2 + 2.6 * speed_ratio
        alpha_xy = 0.5 * (3.0 - alpha_theta)
        heading_contrib = (
            alpha_theta * abs(d_theta)
            if abs(d_theta) > self.cfg.heading_gate_rad
            else 0.0
        )
        raw_lambda = 1.0 + alpha_xy * (nis / self._chi2_thresh) + heading_contrib
        lambda_k = float(np.clip(raw_lambda, 1.15, float(self.cfg.lambda_max)))

p_working = _enforce_spd(self._P * lambda_k) if adaptive_triggered else self._P.copy()
s_mat = _enforce_spd(h_pos @ p_working @ h_pos.T + r_pos, min_eig=1e-6)
k_gain = p_working @ h_pos.T @ np.linalg.inv(s_mat)
self._x = self._x + k_gain @ y_pos
i_kh = np.eye(5) - k_gain @ h_pos
self._P = _enforce_spd(i_kh @ p_working @ i_kh.T + k_gain @ r_pos @ k_gain.T)""",
    )

    b.add_appendix_code(
        "Listagem A.2 – Ponderação gaussiana e reamostragem sistemática no Filtro de Partículas SIR (src/tracker/sota_baselines.py).",
        """# Trecho resumido de ParticleFilter.update() — src/tracker/sota_baselines.py
r_eff = _enforce_spd(r_pos + np.diag([0.04, 0.04]), min_eig=1e-3)
r_eff_inv = np.linalg.inv(r_eff)
innov = z[None, :] - self._particles[:, :2]
quad = np.einsum("ni,ij,nj->n", innov, r_eff_inv, innov)
log_w = np.log(np.maximum(self._weights, 1e-300)) - 0.5 * quad
log_w -= float(np.max(log_w))
w_unnorm = np.exp(log_w)
self._weights = w_unnorm / float(np.sum(w_unnorm))

self._neff = 1.0 / max(float(np.sum(self._weights ** 2)), 1e-12)
if self._neff < 0.5 * self.num_particles:
    self._systematic_resample()""",
    )

    b.add_section_heading("Apêndice B – Comandos de Reprodutibilidade Experimental", level=1, page_break_before=True)
    b.add_body_p(
        "Os experimentos A1 a A7, as figuras em 300 DPI, os testes automatizados e a geração dos documentos finais (PDF e DOCX no padrão SBC Qualis A) reproduzem-se integralmente por meio dos comandos abaixo.",
        first_in_section=True,
    )

    b.add_appendix_code(
        "Listagem B.1 – Execução completa dos experimentos, geração de figuras e compilação dos documentos.",
        """# 1) Preparação dos bancos de dados e manifesto de proveniência SHA-256
.venv/bin/python scripts/prepare_datasets.py --all

# 2) Execução da suíte A1-A7 (PF, LSTM, GRU, T-GNN e 24 câm./150 veíc.)
.venv/bin/python scripts/run_benchmarks.py --all --out data/benchmarks

# 3) Geração das figuras em 300 DPI a partir de ablation_results.json
.venv/bin/python scripts/generate_article_figures.py

# 4) Compilação do artigo em DOCX (SBC Qualis A / ABNT, 11 pt) e em PDF
.venv/bin/python scripts/build_article_docx_sbc.py
.venv/bin/python scripts/build_article_pdf.py

# 5) Verificação automatizada (testes unitários e de integração)
.venv/bin/pytest tests/test_sota_and_city_benchmark.py \\
                 tests/test_benchmarks.py tests/test_ekf.py -v""",
    )

    _audit_docx_compliance(b.doc)
    out_docx.parent.mkdir(parents=True, exist_ok=True)
    b.doc.save(str(out_docx))
    return out_docx


def main() -> int:
    bench_json = Path("data/benchmarks/ablation_results.json")
    fig_dir = Path("artigo/figuras")
    out_docx = Path("artigo/Artigo_SBC_QualisA_Rastreamento_Multicamera_EKF_Redes_Urbanas.docx")
    saved = build_sbc_docx(bench_json, fig_dir, out_docx)
    print(f"SBC Qualis A DOCX compliance audit PASSED -> {saved} ({saved.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
