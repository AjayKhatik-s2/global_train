"""Per-camera "TRAIN INSPECTION REPORT" PDF (A4 landscape).

Reproduces the per-camera inspection-report layout in reportlab -- the same
stack the rest of `reporting/` already uses, so no new runtime dependency is
introduced.

Page order (a section is omitted entirely when it has no content):

    1.  Title page          TRAIN INSPECTION REPORT
                            Station / Camera / Rake Type / Direction /
                            Raw Video / Generated / Trimmed Video
                            Total Wagons: N    Damaged: M
    2.  Summary page(s)     WAGON STATUS SUMMARY (Page i of n)
                            [DATE-TIME | TOTAL WAGONS | DAMAGED WAGONS | STATUS]
                            [SR.NO | WAGON ID | STATUS]  -- 20 rows per page,
                            STATUS cell colour-coded
    3.  Divider             DAMAGE / DOOR DETECTED           (if any problem frame)
    4.  Problem pages       Wagon <id> - Problem Frames      (3 snapshots per page,
                            captioned "<class> | frame <n>")
    5.  Locomotive page     LOCOMOTIVES                      (if any ENGINE segment)
    6.  Per-wagon pages     <TYPE> #<id> - <STATUS>
                            problem frames when present, else three cache frames
                            sampled at 25% / 55% / 80% of the wagon span

Renders only from the view model built by `_inspection_adapter` -- it never
loads a model, opens a video, or reads GlobalTrainState directly.  Every image
is a JPEG already on disk (wagon_cache / evidence); a missing file degrades to
a placeholder rather than failing the page.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from core.logging_setup import get_logger

from ._inspection_adapter import CameraInspectionModel, ProblemFrame, STATUS_COLOR

log = get_logger("reporting.inspection")

PAGE_SIZE = landscape(A4)
PAGE_W, PAGE_H = PAGE_SIZE
MARGIN = 0.4 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

WAGON_ROWS_PER_PAGE = 20
PROBLEM_FRAMES_PER_PAGE = 3
DEFAULT_POSITIONS = (0.25, 0.55, 0.80)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _styles() -> dict:
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "IRTitle", parent=ss["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=26, alignment=TA_CENTER, spaceAfter=18),
        "section": ParagraphStyle(
            "IRSection", parent=ss["Title"], fontName="Helvetica-Bold",
            fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=12),
        "divider": ParagraphStyle(
            "IRDivider", parent=ss["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=28, alignment=TA_CENTER,
            textColor=colors.red),
        "label": ParagraphStyle(
            "IRLabel", parent=ss["Normal"], fontName="Helvetica-Bold",
            fontSize=14, leading=20),
        "value": ParagraphStyle(
            "IRValue", parent=ss["Normal"], fontName="Helvetica",
            fontSize=14, leading=20),
        "url": ParagraphStyle(
            "IRUrl", parent=ss["Normal"], fontName="Helvetica",
            fontSize=9, leading=12, textColor=colors.blue),
        "total": ParagraphStyle(
            "IRTotal", parent=ss["Normal"], fontName="Helvetica-Bold",
            fontSize=16, leading=22, spaceBefore=18),
        "caption": ParagraphStyle(
            "IRCaption", parent=ss["Normal"], fontName="Helvetica",
            fontSize=10, leading=12, alignment=TA_CENTER),
        "wagontitle": ParagraphStyle(
            "IRWagonTitle", parent=ss["Title"], fontName="Helvetica-Bold",
            fontSize=14, leading=18, alignment=TA_CENTER, spaceAfter=10),
        "placeholder": ParagraphStyle(
            "IRPlaceholder", parent=ss["Normal"], fontName="Helvetica-Oblique",
            fontSize=9, leading=11, alignment=TA_CENTER,
            textColor=colors.grey),
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _scaled_image(path: str, max_w: float, max_h: float) -> Optional[Image]:
    """Aspect-preserving Image flowable, or None if unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(path).getSize()
        if not iw or not ih:
            return None
        scale = min(max_w / float(iw), max_h / float(ih))
        return Image(path, width=iw * scale, height=ih * scale)
    except Exception as e:                       # unreadable / truncated JPEG
        log.debug("[INSPECTION] cannot load image %s: %s", path, e)
        return None


def _image_cell(path: Optional[str], caption: str, st: dict,
                max_w: float, max_h: float) -> Table:
    """One image + caption stacked in a borderless single-column table."""
    img = _scaled_image(path, max_w, max_h) if path else None
    body = img if img is not None else Paragraph("Snapshot not available",
                                                 st["placeholder"])
    t = Table([[body], [Paragraph(caption, st["caption"])]],
              colWidths=[max_w])
    t.setStyle(TableStyle([
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _image_row(cells: Sequence[Table], total_w: float) -> Table:
    """Lay image cells side by side across the page."""
    n = max(1, len(cells))
    t = Table([list(cells)], colWidths=[total_w / n] * n)
    t.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _title_page(model: CameraInspectionModel, st: dict,
                logo_path: Optional[str]) -> List:
    rake_type, rake_hex, arrow = model.style.rake_for(model.direction)
    story: List = []

    logo = _scaled_image(logo_path, 1.4 * inch, 0.7 * inch) if logo_path else None
    if logo is not None:
        lt = Table([[logo]], colWidths=[CONTENT_W])
        lt.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "LEFT")]))
        story.append(lt)

    story.append(Paragraph("TRAIN INSPECTION REPORT", st["title"]))
    story.append(Spacer(1, 0.25 * inch))

    rows = [
        ("Station",   model.style.station_name,      None),
        ("Camera",    model.style.camera_label,      None),
        ("Rake Type", f"{rake_type}  {arrow}",       rake_hex),
        ("Direction", model.direction,               None),
        ("Raw Video", model.raw_video_name,          None),
        ("Generated", model.generated_at.strftime("%d-%m-%Y %H:%M:%S"), None),
    ]
    data = []
    for label, value, hexcolor in rows:
        vstyle = st["value"]
        if hexcolor:
            vstyle = ParagraphStyle(f"v{label}", parent=st["value"],
                                    textColor=colors.HexColor(hexcolor),
                                    fontName="Helvetica-Bold")
        data.append([Paragraph(f"{label}:", st["label"]),
                     Paragraph(str(value), vstyle)])

    if model.trimmed_video_url:
        data.append([
            Paragraph("Trimmed Video:", st["label"]),
            Paragraph(
                f'<link href="{model.trimmed_video_url}">'
                f'{model.trimmed_video_url}</link>', st["url"]),
        ])

    t = Table(data, colWidths=[0.25 * CONTENT_W, 0.75 * CONTENT_W])
    t.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    story.append(Paragraph(
        f"Total Wagons: {model.total_wagons}    "
        f"Damaged: {model.damaged_wagons}", st["total"]))
    return story


def _summary_pages(model: CameraInspectionModel, st: dict) -> List:
    wagons = [s for s in model.segments if s.is_wagon]
    if not wagons:
        return []

    total = len(wagons)
    damaged = model.damaged_wagons
    stamp = model.generated_at.strftime("%d-%m-%Y %H:%M")
    pages = max(1, math.ceil(len(wagons) / WAGON_ROWS_PER_PAGE))

    story: List = []
    for page_idx in range(pages):
        story.append(PageBreak())
        story.append(Paragraph(
            f"WAGON STATUS SUMMARY (Page {page_idx + 1} of {pages})",
            st["section"]))

        top = Table(
            [["DATE-TIME", "TOTAL WAGONS", "DAMAGED WAGONS", "STATUS"],
             [stamp, str(total), str(damaged), "ISSUES" if damaged else "OK"]],
            colWidths=[CONTENT_W / 4.0] * 4)
        top.setStyle(TableStyle([
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(top)
        story.append(Spacer(1, 0.3 * inch))

        chunk = wagons[page_idx * WAGON_ROWS_PER_PAGE:
                       (page_idx + 1) * WAGON_ROWS_PER_PAGE]
        data = [["SR.NO", "WAGON ID", "STATUS"]]
        style_cmds = [
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ]
        for i, seg in enumerate(chunk):
            sr_no = page_idx * WAGON_ROWS_PER_PAGE + i + 1
            status = model.status_for(seg.segment_id)
            data.append([str(sr_no), str(seg.segment_id), status])
            style_cmds.append((
                "BACKGROUND", (2, i + 1), (2, i + 1),
                colors.HexColor(STATUS_COLOR.get(status, "#ffffff"))))

        tbl = Table(data, colWidths=[CONTENT_W / 3.0] * 3, repeatRows=1)
        tbl.setStyle(TableStyle(style_cmds))
        story.append(tbl)
    return story


def _problem_pages(model: CameraInspectionModel, st: dict) -> List:
    if not model.problem_frames:
        return []

    story: List = [PageBreak(), Spacer(1, 2.4 * inch),
                   Paragraph("DAMAGE / DOOR DETECTED", st["divider"])]

    # Preserve wagon order as the segments define it.
    order = [s.segment_id for s in model.segments]
    seen = {p.wagon_id for p in model.problem_frames}
    for wid in [w for w in order if w in seen]:
        frames = model.problem_frames_for(wid)
        for i in range(0, len(frames), PROBLEM_FRAMES_PER_PAGE):
            story.extend(_problem_chunk(wid, frames[i:i + PROBLEM_FRAMES_PER_PAGE], st))
    return story


def _problem_chunk(wagon_id: int, chunk: List[ProblemFrame], st: dict) -> List:
    cell_w = CONTENT_W / max(1, len(chunk))
    cell_h = (PAGE_H - 2 * MARGIN - 1.0 * inch)
    cells = [
        _image_cell(p.image_path,
                    f"{p.problem_type} | frame {p.frame_number}",
                    st, cell_w - 8, cell_h)
        for p in chunk
    ]
    return [
        PageBreak(),
        Paragraph(f"Wagon {wagon_id} &mdash; Problem Frames", st["wagontitle"]),
        _image_row(cells, CONTENT_W),
    ]


def _loco_pages(model: CameraInspectionModel, st: dict) -> List:
    if not model.locos:
        return []
    cols = min(3, len(model.locos))
    rows = math.ceil(len(model.locos) / cols)
    cell_w = CONTENT_W / cols
    cell_h = (PAGE_H - 2 * MARGIN - 1.0 * inch) / max(1, rows)

    story: List = [PageBreak(), Paragraph("LOCOMOTIVES", st["section"])]
    for r in range(rows):
        batch = model.locos[r * cols:(r + 1) * cols]
        story.append(_image_row(
            [_image_cell(l.frame_path, f"Loco #{l.loco_id}", st,
                         cell_w - 8, cell_h) for l in batch],
            CONTENT_W))
    return story


def _wagon_pages(model: CameraInspectionModel, st: dict,
                 positions: Sequence[float]) -> List:
    story: List = []
    cell_h = PAGE_H - 2 * MARGIN - 1.0 * inch

    for seg in model.segments:
        status = model.status_for(seg.segment_id)
        hexcolor = STATUS_COLOR.get(status, "#000000")
        title = ParagraphStyle(
            f"w{seg.segment_id}", parent=st["wagontitle"],
            textColor=colors.HexColor(hexcolor))

        problems = model.problem_frames_for(seg.segment_id)[:PROBLEM_FRAMES_PER_PAGE]
        if problems:
            cell_w = CONTENT_W / len(problems)
            cells = [
                _image_cell(p.image_path,
                            f"{p.problem_type} f{p.frame_number}",
                            st, cell_w - 8, cell_h)
                for p in problems
            ]
        else:
            cell_w = CONTENT_W / max(1, len(positions))
            cells = []
            span = max(0, seg.end_frame - seg.start_frame)
            for pos in positions:
                idx = int(seg.start_frame + span * pos)
                path = (os.path.join(seg.directory, f"frame_{idx:06d}.jpg")
                        if seg.directory else None)
                cells.append(_image_cell(path, f"pos={int(pos * 100)}%",
                                         st, cell_w - 8, cell_h))

        story.append(PageBreak())
        story.append(Paragraph(
            f"{seg.segment_type.upper()} #{seg.segment_id} &mdash; {status}",
            title))
        story.append(_image_row(cells, CONTENT_W))
    return story


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def build(
    *,
    model: CameraInspectionModel,
    output_path: str,
    logo_path: Optional[str] = None,
    positions: Optional[Sequence[float]] = None,
) -> str:
    """Render one camera's inspection PDF.  Returns `output_path`."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                exist_ok=True)
    st = _styles()
    positions = tuple(positions or DEFAULT_POSITIONS)

    story: List = []
    story.extend(_title_page(model, st, logo_path))
    story.extend(_summary_pages(model, st))
    story.extend(_problem_pages(model, st))
    story.extend(_loco_pages(model, st))
    story.extend(_wagon_pages(model, st, positions))

    doc = SimpleDocTemplate(
        output_path, pagesize=PAGE_SIZE,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"Train Inspection Report - {model.raw_video_name}",
        author="Automated CCTV Analytics System",
        subject=("Inspection report generated at "
                 f"{model.generated_at.isoformat()}"),
    )
    doc.build(story)
    log.info("[INSPECTION/%s] wrote %s (%d bytes)",
             model.style.camera_id, output_path,
             os.path.getsize(output_path) if os.path.isfile(output_path) else 0)
    return output_path
