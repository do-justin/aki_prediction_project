"""Portable Word and PDF layouts for the manuscript tables."""
from __future__ import annotations
from functools import partial
import json
import textwrap
from pathlib import Path
from typing import Mapping, Sequence
from docx import Document
from docx.enum.section import WD_ORIENT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '05_CODE'))
from publication_navigation import bookmark, word_section_index, table_footer, add_pdf_navigation
from publication_style import apply_document_style, WORD_TABLE_PT, color, style_context

from table_rendering import (
    _add_dataframe_to_doc, _clean_cell, _table_widths, dataframe_to_html,
    dataframe_to_markdown as _dataframe_to_markdown,
    export_dataframe_bundle as _export_dataframe_bundle, numeric_display_columns,
    symbol_display_columns, write_markdown_sections as _write_markdown_sections,
)


dataframe_to_markdown = partial(_dataframe_to_markdown, bold_empty=False)
write_markdown_sections = partial(_write_markdown_sections, bold_empty=False)


def write_docx_sections(
    path: Path,
    *,
    title: str,
    sections: Sequence[Mapping[str, object]],
) -> Path:
    """Write a DOCX document containing one or more table sections."""

    path.parent.mkdir(parents=True, exist_ok=True)
    document = apply_document_style(Document(), body_size=WORD_TABLE_PT)
    for section in document.sections:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = section.page_height, section.page_width

    document.add_heading(title, level=0)
    navigable=[section for section in sections if isinstance(section.get('dataframe'),pd.DataFrame)]
    headings=[str(section['heading']) for section in navigable]
    word_section_index(document,headings)
    section_number=0
    for section in sections:
        heading_paragraph=None
        if str(section["heading"]) != title:
            heading_paragraph=document.add_heading(str(section["heading"]), level=1)
        if isinstance(section.get('dataframe'),pd.DataFrame):
            if heading_paragraph is None:heading_paragraph=document.paragraphs[0]
            bookmark(heading_paragraph,f'TableSection_{section_number}',section_number+1)
            section_number+=1
        text = section.get("text")
        if text:
            document.add_paragraph(str(text))
        df = section.get("dataframe")
        if isinstance(df, pd.DataFrame):
            _add_dataframe_to_doc(document, df, section.get("styles"))
        if section.get("notes"):
            document.add_paragraph(str(section["notes"]))
        document.add_paragraph()
    document.save(path)
    return path

def write_pdf_sections(
    path: Path,
    *,
    title: str,
    sections: Sequence[Mapping[str, object]],
) -> Path:
    """Write a PDF document containing one or more table sections."""

    path.parent.mkdir(parents=True, exist_ok=True)
    headings=[str(section['heading']) for section in sections]
    section_pages=[];links=[]
    with style_context(), PdfPages(path) as pdf:
        pdf._publication_navigation={'headings':headings,'links':links}
        for section_number,section in enumerate(sections):
            pdf._publication_navigation['current']=section_number
            section_pages.append((str(section['heading']),pdf.get_pagecount()+1))
            heading = str(section["heading"])
            df = section.get("dataframe")
            text = section.get("text")
            title_text = str(section.get("pdf_title") or (title if heading == title else f"{title} | {heading}"))
            if text and not isinstance(df, pd.DataFrame):
                fig, ax = plt.subplots(figsize=(12, 3.5))
                ax.axis("off")
                ax.set_title(title_text, fontsize=14, fontweight="bold", pad=18)
                ax.text(0.02, 0.95, str(text), va="top", ha="left", wrap=True)
                pdf.savefig(fig, bbox_inches="tight", pad_inches=.2)
                plt.close(fig)
                continue
            if isinstance(df, pd.DataFrame):
                _render_pdf_table_page(pdf, title_text, df, notes=section.get("notes"), styles=section.get("styles"))
    if section_pages:
        record=add_pdf_navigation(path,title,section_pages,links)
        # Sidecars keep link destinations inspectable without a GUI dependency.
        path.with_suffix('.navigation.json').write_text(json.dumps(record,indent=2)+'\n')
    return path

export_dataframe_bundle = partial(
    _export_dataframe_bundle, write_docx=write_docx_sections, write_pdf=write_pdf_sections,
    bold_empty=False,
)


def _render_pdf_table_page(pdf, title, df, *, notes=None, styles=None):
    """Fixed A4 landscape pages, readable type and balanced continuation blocks."""
    width,height=297/25.4,210/25.4
    if len(df.columns)>7:
        ids=[c for c in df if str(c).lower() in {'model','outcome','branch','feature set'}]
        ids=ids[:2] or list(df.columns[:1]);values=[c for c in df if c not in ids]
        for start in range(0,len(values),5):
            block=values[start:start+5]
            _render_pdf_table_page(pdf,title+'\n'+block[0]+' to '+block[-1],df[ids+block],notes=notes,styles=styles)
        return
    font=9
    ids=[c for c in df if str(c).lower() in {'model','outcome','branch','feature set'}]
    weights=[2.7 if c=='Feature set' else 1.25 if c=='Model' else 1.55 for c in df]
    fractions=[v/sum(weights) for v in weights]
    available_width=width-.65
    def wrap(value,j):
        value=_clean_cell(value).replace('<br>','\n').replace(' (','\n(')
        n=max(13,int(available_width*fractions[j]*72/(font*.49)))
        return '\n'.join(textwrap.fill(line,n,break_long_words=False) for line in value.splitlines())
    headers=[wrap(c,j) for j,c in enumerate(df.columns)]
    rows=[[wrap(v,j) for j,v in enumerate(row)] for row in df.itertuples(index=False,name=None)]
    title_lines='\n'.join(textwrap.fill(line,110,break_long_words=False) for line in title.splitlines())
    note_lines=textwrap.wrap(str(notes or ''),165)
    title_h=(title_lines.count('\n')+1)*.19+.17
    note_h=len(note_lines)*.135+.12
    head_h=max(v.count('\n')+1 for v in headers)*font*1.18/72+.08
    heights=[max(v.count('\n')+1 for v in row)*font*1.18/72+.08 for row in rows]
    navigation=getattr(pdf,"_publication_navigation",None)
    has_footer=navigation and len(navigation["headings"])>=6
    available=height-.55-title_h-note_h-(.47 if has_footer else .25)
    if sum(heights)+head_h>available and len(df)>1:
        pages=max(2,int((sum(heights)+available-head_h-1e-8)//max(.1,available-head_h)))
        size=max(1,(len(df)+pages-1)//pages)
        for start in range(0,len(df),size):
            _render_pdf_table_page(pdf,title+f'\nRows {start+1}–{min(start+size,len(df))} of {len(df)}',df.iloc[start:start+size],notes=notes,styles=styles)
        return
    total=sum(heights)+head_h;top=height-.30-title_h;bottom=top-total
    assert bottom-note_h>.12,(title,bottom,note_h)
    fig=plt.figure(figsize=(width,height));fig.patch.set_facecolor('white')
    fig.text(.325/width,1-.20/height,title_lines,fontsize=11,weight='bold',va='top')
    ax=fig.add_axes([.325/width,bottom/height,available_width/width,total/height]);ax.axis('off')
    t=ax.table(cellText=rows,colLabels=headers,colWidths=fractions,cellLoc='right',bbox=[0,0,1,1]);t.auto_set_font_size(False);t.set_fontsize(font)
    numeric=numeric_display_columns(df)
    for (r,c),cell in t.get_celld().items():
        cell.set_height(([head_h]+heights)[r]/total);cell.set_linewidth(0);cell.PAD=.035
        cell.set_facecolor(color('table_header') if r==0 else 'white')
        cell_style=(styles or {}).get(df.index[r-1],{}).get(df.columns[c],{}) if r else {}
        cell.set_text_props(ha='right' if df.columns[c] in numeric else 'left',weight='bold' if r==0 or cell_style.get('bold') else 'normal')
    y=1
    for r,h in enumerate([head_h]+heights):
        ax.plot([0,1],[y,y],transform=ax.transAxes,color=color('reference') if r<2 else color('table_rule'),lw=.6 if r<2 else .25,clip_on=False);y-=h/total
    ax.plot([0,1],[0,0],transform=ax.transAxes,color=color('reference'),lw=.6,clip_on=False)
    if notes:fig.text(.325/width,(bottom-.14)/height,'\n'.join(note_lines),fontsize=8,va='top',linespacing=1.2)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for cell in t.get_celld().values():
        a=cell.get_text().get_window_extent(renderer);b=cell.get_window_extent(renderer)
        assert b.x0-.2<=a.x0 and a.x1<=b.x1+.2 and b.y0-.2<=a.y0 and a.y1<=b.y1+.2,(title,cell.get_text().get_text())
    if has_footer:
        for link in table_footer(fig,navigation['headings'],navigation['current']):
            navigation['links'].append(dict(link,page=pdf.get_pagecount()+1))
    pdf.savefig(fig);plt.close(fig)
