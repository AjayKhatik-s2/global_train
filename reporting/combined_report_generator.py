"""
Combined Multi-Camera Wagon Eye Report Generator

Generates a single unified PDF report that combines
data from ALL 4 cameras: LEFT_UP, RIGHT_UP, RIGHT_UP_TOP, LEFT_UP_TOP.

Report structure:
    1. Header: Title + Date/Time
    2. RAW VIDEO section with clickable S3 links
    3. Processed Video section with tracked video links
    4. PARTIAL REPORT warning (if missing cameras)
    5. Detailed Reports links
    6. Summary table (status, wagon count, door counts)
    7. Wagon inspection table
    8. Damaged wagon images

Table format:
    SR.NO | WAGON NUMBER | LEFT DOORS | RIGHT DOORS | RIGHT TOP DAMAGES | LEFT TOP DAMAGES

- Wagon numbers come from RIGHT_UP (which has OCR)
- Wagons matched by sequence order (same train, same wagon order)
- Rows highlighted red if ANY camera detects open door / damage
"""

import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ─── Professional Color Palette ───
LIGHT_RED = colors.Color(1.0, 0.85, 0.85)
LIGHT_GREEN = colors.Color(0.85, 0.96, 0.85)
HEADER_GRAY = colors.Color(0.92, 0.92, 0.92)
HEADER_BLUE = colors.Color(0.2, 0.4, 0.6)

# Primary brand colors
NAVY_DARK = colors.HexColor('#0B1D3A')       # Deep navy for title banner
NAVY_MID = colors.HexColor('#162D50')        # Mid navy for section headers
TEAL_ACCENT = colors.HexColor('#1A8A7D')     # Teal accent for section labels
SLATE_BG = colors.HexColor('#F4F6F9')        # Light slate background
SLATE_BORDER = colors.HexColor('#C8CED8')    # Subtle border color
WHITE = colors.HexColor('#FFFFFF')
LINK_BLUE = colors.HexColor('#1565C0')       # Professional link blue
NO_FEED_RED = colors.HexColor('#C62828')      # Red for no feed
SECTION_TEAL_BG = colors.HexColor('#E0F2F1') # Light teal section background
WARN_BG = colors.HexColor('#FFF3E0')         # Warning background (amber)
WARN_BORDER = colors.HexColor('#E65100')     # Warning border


class CombinedReportGenerator:
    """
    Generate a unified wagon inspection PDF combining data from all 4 cameras.

    Report structure:
    1. Header with logo and title banner
    2. Video links section (Raw + Processed in a single structured table)
    3. Detailed report links
    4. Summary section (total wagons, open door counts per camera)
    5. Single unified wagon table with all camera columns
    6. Damaged wagon snapshot images
    """

    def __init__(self, output_path: str, logo_path: str,
                 source_video_url: str = None,
                 left_report_url: str = None,
                 right_report_url: str = None,
                 top_report_url: str = None,
                 left_top_report_url: str = None,
                 left_video_url: str = None,
                 right_video_url: str = None,
                 top_video_url: str = None,
                 left_top_video_url: str = None,
                 left_tracked_url: str = None,
                 right_tracked_url: str = None,
                 top_tracked_url: str = None,
                 left_top_tracked_url: str = None):
        self.output_path = output_path
        self.logo_path = logo_path
        self.source_video_url = source_video_url
        self.left_report_url = left_report_url
        self.right_report_url = right_report_url
        self.top_report_url = top_report_url
        self.left_top_report_url = left_top_report_url
        self.left_video_url = left_video_url
        self.right_video_url = right_video_url
        self.top_video_url = top_video_url
        self.left_top_video_url = left_top_video_url
        self.left_tracked_url = left_tracked_url
        self.right_tracked_url = right_tracked_url
        self.top_tracked_url = top_tracked_url
        self.left_top_tracked_url = left_top_tracked_url
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _add_page_logo(self, canvas, doc):
        """Add logo to every page."""
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                canvas.saveState()
                canvas.drawImage(
                    self.logo_path,
                    doc.leftMargin,
                    doc.height + doc.topMargin + 0.05 * inch,
                    width=1.0 * inch,
                    height=0.4 * inch,
                    preserveAspectRatio=True,
                    mask='auto'
                )
                canvas.restoreState()
            except Exception:
                pass

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for a professional report."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=0,
            spaceBefore=0,
            textColor=WHITE,
            fontName='Helvetica-Bold',
            leading=24,
        ))
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=0,
            spaceBefore=0,
            textColor=colors.HexColor('#B0BEC5'),
            fontName='Helvetica',
            leading=14,
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=NAVY_DARK,
            fontName='Helvetica-Bold'
        ))
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.black,
            fontName='Helvetica-Bold'
        ))
        self.styles.add(ParagraphStyle(
            name='TableCell',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.black,
            fontName='Helvetica'
        ))
        self.styles.add(ParagraphStyle(
            name='SmallNote',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_LEFT,
            textColor=colors.gray,
            fontName='Helvetica-Oblique'
        ))
        self.styles.add(ParagraphStyle(
            name='SmallNoteRight',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_RIGHT,
            textColor=colors.gray,
            fontName='Helvetica-Oblique'
        ))
        self.styles.add(ParagraphStyle(
            name='SectionLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_LEFT,
            textColor=NAVY_DARK,
            fontName='Helvetica-Bold',
            spaceAfter=4,
            spaceBefore=6
        ))
        self.styles.add(ParagraphStyle(
            name='LinkCell',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.black,
            fontName='Helvetica'
        ))
        # ── Additional professional styles ──
        self.styles.add(ParagraphStyle(
            name='BannerTitle',
            parent=self.styles['Normal'],
            fontSize=18,
            alignment=TA_CENTER,
            textColor=WHITE,
            fontName='Helvetica-Bold',
            leading=24,
        ))
        self.styles.add(ParagraphStyle(
            name='BannerDate',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#B0BEC5'),
            fontName='Helvetica',
            leading=14,
        ))
        self.styles.add(ParagraphStyle(
            name='SectionTitleWhite',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=WHITE,
            fontName='Helvetica-Bold',
            leading=13,
        ))
        self.styles.add(ParagraphStyle(
            name='CameraLabel',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#546E7A'),
            fontName='Helvetica-Bold',
            leading=11,
        ))
        self.styles.add(ParagraphStyle(
            name='LinkCellPro',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=LINK_BLUE,
            fontName='Helvetica-Bold',
        ))
        self.styles.add(ParagraphStyle(
            name='NoFeedCell',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=NO_FEED_RED,
            fontName='Helvetica-Oblique',
        ))
        self.styles.add(ParagraphStyle(
            name='WarningText',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=WARN_BORDER,
            fontName='Helvetica-Bold',
            leading=14,
        ))

    def _make_camera_link(self, url, camera_name, missing_cameras=None):
        """Create a clickable link or 'NO FEED' styled text for a camera."""
        if missing_cameras and camera_name in missing_cameras:
            return Paragraph(
                '<font color="#C62828"><i>NO FEED</i></font>',
                self.styles['NoFeedCell']
            )
        if url:
            return Paragraph(
                f'<a href="{url}" color="#1565C0"><b><u>Click to View</u></b></a>',
                self.styles['LinkCellPro']
            )
        else:
            return Paragraph(
                '<font color="#C62828"><i>NO FEED</i></font>',
                self.styles['NoFeedCell']
            )

    def _make_report_link(self, url, label, cam_id, missing_cameras):
        """Create a styled report link or disabled text."""
        if url and cam_id not in missing_cameras:
            return Paragraph(
                f'<a href="{url}" color="#1565C0"><b><u>{label}</u></b></a>',
                self.styles['LinkCellPro']
            )
        elif cam_id in missing_cameras:
            return Paragraph(
                '<font color="#C62828"><i>NO FEED</i></font>',
                self.styles['NoFeedCell']
            )
        else:
            return Paragraph(
                f'<font color="#78909C">{label}</font>',
                self.styles['LinkCell']
            )

    def _create_header(self, missing_cameras=None):
        """
        Create a professional header with:
        1. Dark navy title banner with report name + date
        2. VIDEO EVIDENCE section — structured table with Raw + Processed rows
        3. PARTIAL REPORT warning banner (if missing cameras)
        4. DETAILED REPORTS section — styled link cards
        5. Separator
        """
        elements = []
        if missing_cameras is None:
            missing_cameras = []

        IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(IST)
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H:%M IST")

        # ══════════════════════════════════════════════════════════════
        # TITLE BANNER — dark navy background, white text
        # ══════════════════════════════════════════════════════════════
        elements.append(Spacer(1, 0.25 * inch))

        banner_data = [[
            Paragraph("COMBINED WAGON EYE REPORT", self.styles['BannerTitle'])
        ], [
            Paragraph(f'{date_str}  |  {time_str}', self.styles['BannerDate'])
        ]]
        banner_table = Table(banner_data, colWidths=[10.0 * inch])
        banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), NAVY_DARK),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (0, 0), 14),
            ('BOTTOMPADDING', (0, 0), (0, 0), 2),
            ('TOPPADDING', (0, 1), (0, 1), 0),
            ('BOTTOMPADDING', (0, 1), (0, 1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            # Rounded-feel: thick outer box
            ('BOX', (0, 0), (-1, -1), 1.5, NAVY_DARK),
        ]))
        elements.append(banner_table)
        elements.append(Spacer(1, 0.18 * inch))

        # ══════════════════════════════════════════════════════════════
        # VIDEO EVIDENCE — single structured table
        # Row 0: Section header (navy)
        # Row 1: Camera labels
        # Row 2: Raw video links
        # Row 3: Camera labels (repeated for processed)
        # Row 4: Processed video links
        # ══════════════════════════════════════════════════════════════
        camera_names = ['LEFT_UP', 'RIGHT_UP', 'RIGHT_UP_TOP', 'LEFT_UP_TOP']
        camera_labels = [
            Paragraph(f'<b>{name}</b>', self.styles['CameraLabel'])
            for name in camera_names
        ]

        raw_video_urls = [
            (self.left_video_url, 'LEFT_UP'),
            (self.right_video_url, 'RIGHT_UP'),
            (self.top_video_url, 'RIGHT_UP_TOP'),
            (self.left_top_video_url, 'LEFT_UP_TOP'),
        ]
        raw_links = [self._make_camera_link(url, name, missing_cameras)
                     for url, name in raw_video_urls]

        tracked_urls = [
            (self.left_tracked_url, 'LEFT_UP'),
            (self.right_tracked_url, 'RIGHT_UP'),
            (self.top_tracked_url, 'RIGHT_UP_TOP'),
            (self.left_top_tracked_url, 'LEFT_UP_TOP'),
        ]
        tracked_links = [self._make_camera_link(url, name, missing_cameras)
                         for url, name in tracked_urls]

        # Build the video evidence table
        col_w = 2.5 * inch
        video_data = [
            # Row 0: Section header spanning all columns
            [Paragraph('<b>VIDEO EVIDENCE</b>', self.styles['SectionTitleWhite']),
             '', '', ''],
            # Row 1: "Raw Video" sub-header + camera labels
            [Paragraph('<b>Raw Video</b>', self.styles['CameraLabel'])] + camera_labels[:3],
            # But we need 4 camera columns — let me restructure:
        ]

        # Actually, let's use a 5-column layout: label column + 4 camera columns
        label_col_w = 1.4 * inch
        cam_col_w = 2.15 * inch

        video_data = [
            # Row 0: section title (spans all 5 cols)
            [Paragraph('<b>VIDEO EVIDENCE</b>', self.styles['SectionTitleWhite']),
             '', '', '', ''],
            # Row 1: camera names header
            [Paragraph('', self.styles['CameraLabel']),
             Paragraph(f'<b>{camera_names[0]}</b>', self.styles['CameraLabel']),
             Paragraph(f'<b>{camera_names[1]}</b>', self.styles['CameraLabel']),
             Paragraph(f'<b>{camera_names[2]}</b>', self.styles['CameraLabel']),
             Paragraph(f'<b>{camera_names[3]}</b>', self.styles['CameraLabel'])],
            # Row 2: Raw Video links
            [Paragraph('<b>Raw Video</b>', self.styles['CameraLabel'])] + raw_links,
            # Row 3: Processed Video links
            [Paragraph('<b>Processed Video</b>', self.styles['CameraLabel'])] + tracked_links,
        ]

        video_table = Table(video_data,
                            colWidths=[label_col_w, cam_col_w, cam_col_w, cam_col_w, cam_col_w])
        video_style = [
            # Section header row (navy background)
            ('SPAN', (0, 0), (-1, 0)),
            ('BACKGROUND', (0, 0), (-1, 0), NAVY_MID),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            # Camera names header row (light slate)
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#E8EAF0')),
            ('TOPPADDING', (0, 1), (-1, 1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 5),
            # Raw video row
            ('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#E8EAF0')),
            ('BACKGROUND', (1, 2), (-1, 2), WHITE),
            ('TOPPADDING', (0, 2), (-1, 2), 8),
            ('BOTTOMPADDING', (0, 2), (-1, 2), 8),
            # Processed video row
            ('BACKGROUND', (0, 3), (0, 3), colors.HexColor('#E8EAF0')),
            ('BACKGROUND', (1, 3), (-1, 3), SLATE_BG),
            ('TOPPADDING', (0, 3), (-1, 3), 8),
            ('BOTTOMPADDING', (0, 3), (-1, 3), 8),
            # General
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, SLATE_BORDER),
            ('INNERGRID', (0, 1), (-1, -1), 0.5, SLATE_BORDER),
            ('LINEBELOW', (0, 0), (-1, 0), 1, SLATE_BORDER),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]

        # Gray out missing camera columns
        for i, cam_name in enumerate(camera_names):
            if cam_name in missing_cameras:
                col_idx = i + 1  # +1 because col 0 is the label column
                video_style.append(('BACKGROUND', (col_idx, 1), (col_idx, -1),
                                    colors.HexColor('#ECEFF1')))

        video_table.setStyle(TableStyle(video_style))
        elements.append(video_table)
        elements.append(Spacer(1, 0.14 * inch))

        # ══════════════════════════════════════════════════════════════
        # PARTIAL REPORT WARNING BANNER
        # ══════════════════════════════════════════════════════════════
        if missing_cameras:
            missing_str = ', '.join(missing_cameras)
            warn_data = [[
                Paragraph(
                    f'<b>⚠  PARTIAL REPORT</b> — No feed received from: <b>{missing_str}</b>',
                    self.styles['WarningText']
                )
            ]]
            warn_table = Table(warn_data, colWidths=[10.0 * inch])
            warn_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), WARN_BG),
                ('BOX', (0, 0), (-1, -1), 1.2, WARN_BORDER),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(warn_table)
            elements.append(Spacer(1, 0.12 * inch))

        # ══════════════════════════════════════════════════════════════
        # DETAILED REPORTS — structured table with header + link row
        # ══════════════════════════════════════════════════════════════
        report_links = [
            (self.left_report_url, 'LEFT Detail Report', 'LEFT_UP'),
            (self.right_report_url, 'RIGHT Detail Report', 'RIGHT_UP'),
            (self.top_report_url, 'R-TOP Detail Report', 'RIGHT_UP_TOP'),
            (self.left_top_report_url, 'L-TOP Detail Report', 'LEFT_UP_TOP'),
        ]

        report_cells = [
            self._make_report_link(url, label, cam_id, missing_cameras)
            for url, label, cam_id in report_links
        ]

        report_data = [
            # Row 0: Section header
            [Paragraph('<b>DETAILED REPORTS</b>', self.styles['SectionTitleWhite']),
             '', '', ''],
            # Row 1: Camera labels
            [Paragraph(f'<b>{camera_names[0]}</b>', self.styles['CameraLabel']),
             Paragraph(f'<b>{camera_names[1]}</b>', self.styles['CameraLabel']),
             Paragraph(f'<b>{camera_names[2]}</b>', self.styles['CameraLabel']),
             Paragraph(f'<b>{camera_names[3]}</b>', self.styles['CameraLabel'])],
            # Row 2: Links
            report_cells,
        ]

        report_col_w = 2.5 * inch
        report_table = Table(report_data, colWidths=[report_col_w] * 4)
        report_style = [
            # Header row (teal accent)
            ('SPAN', (0, 0), (-1, 0)),
            ('BACKGROUND', (0, 0), (-1, 0), TEAL_ACCENT),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, 0), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 7),
            # Camera labels row
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#E8EAF0')),
            ('TOPPADDING', (0, 1), (-1, 1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 5),
            # Links row
            ('BACKGROUND', (0, 2), (-1, 2), WHITE),
            ('TOPPADDING', (0, 2), (-1, 2), 9),
            ('BOTTOMPADDING', (0, 2), (-1, 2), 9),
            # General
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, SLATE_BORDER),
            ('INNERGRID', (0, 1), (-1, -1), 0.5, SLATE_BORDER),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]

        # Gray out missing camera columns
        for i, cam_name in enumerate(camera_names):
            if cam_name in missing_cameras:
                report_style.append(('BACKGROUND', (i, 1), (i, -1),
                                     colors.HexColor('#ECEFF1')))

        report_table.setStyle(TableStyle(report_style))
        elements.append(report_table)

        # ── Thin separator line ──
        elements.append(Spacer(1, 0.18 * inch))

        return elements

    def _create_summary_section(self, total_wagons, left_open, right_open,
                                 top_open, left_top_open, left_partial, right_partial,
                                 top_partial, left_top_partial, status,
                                 loco_numbers=None, missing_cameras=None,
                                 rake_type=None):
        """Create professional summary table with navy header banner."""
        elements = []
        if missing_cameras is None:
            missing_cameras = []

        IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(IST)
        date_time_str = now.strftime("%d-%m-%Y - %H:%M:%S")

        # Partial closed display — show N/A if camera missing
        left_partial_str = "N/A" if "LEFT_UP" in missing_cameras else str(left_partial)
        right_partial_str = "N/A" if "RIGHT_UP" in missing_cameras else str(right_partial)
        partial_text = f"L {left_partial_str} / R {right_partial_str}"

        # Show N/A for missing cameras instead of 0
        left_open_str = "N/A" if "LEFT_UP" in missing_cameras else str(left_open)
        right_open_str = "N/A" if "RIGHT_UP" in missing_cameras else str(right_open)
        top_open_str = "N/A" if "RIGHT_UP_TOP" in missing_cameras else str(top_open)
        left_top_open_str = "N/A" if "LEFT_UP_TOP" in missing_cameras else str(left_top_open)

        # Rake type display
        rake_type_str = rake_type if rake_type else "N/A"

        # Loco number display — supports multiple loco numbers
        if isinstance(loco_numbers, list) and loco_numbers:
            loco_display = " / ".join(str(n).upper() for n in loco_numbers)
        elif loco_numbers:
            loco_display = str(loco_numbers).upper()
        else:
            loco_display = "Not Detected"

        # ── White-on-navy header style for table headers ──
        header_style = ParagraphStyle(
            'SummaryHeader', parent=self.styles['Normal'],
            fontSize=8, alignment=TA_CENTER,
            textColor=WHITE, fontName='Helvetica-Bold', leading=11
        )
        data_style = ParagraphStyle(
            'SummaryData', parent=self.styles['Normal'],
            fontSize=9, alignment=TA_CENTER,
            textColor=colors.HexColor('#1A1A2E'), fontName='Helvetica', leading=12
        )
        data_bold = ParagraphStyle(
            'SummaryDataBold', parent=self.styles['Normal'],
            fontSize=9, alignment=TA_CENTER,
            textColor=colors.HexColor('#1A1A2E'), fontName='Helvetica-Bold', leading=12
        )

        # Status display style
        if status == "NOT OK":
            status_color = '#C62828'
        else:
            status_color = '#2E7D32'

        # Rake type color
        if rake_type_str == "LOADED RAKE":
            rake_color = '#1565C0'  # blue
        elif rake_type_str == "EMPTY RAKE":
            rake_color = '#E65100'  # orange
        else:
            rake_color = '#1A1A2E'

        # Section title row
        title_row = [
            Paragraph('<b>INSPECTION SUMMARY</b>', self.styles['SectionTitleWhite']),
            '', '', '', '', '', '', '', '', ''
        ]

        # Header row
        header_row = [
            Paragraph("DATE-TIME", header_style),
            Paragraph("LOCO NUMBER", header_style),
            Paragraph("TOTAL<br/>WAGONS", header_style),
            Paragraph("LEFT OPEN<br/>DOORS", header_style),
            Paragraph("RIGHT OPEN<br/>DOORS", header_style),
            Paragraph("R-TOP<br/>DAMAGES", header_style),
            Paragraph("L-TOP<br/>DAMAGES", header_style),
            Paragraph("PARTIAL<br/>CLOSED", header_style),
            Paragraph("RAKE<br/>TYPE", header_style),
            Paragraph("STATUS", header_style),
        ]

        # Data row
        data_row = [
            Paragraph(date_time_str, data_style),
            Paragraph(f"<b>{loco_display}</b>", data_bold),
            Paragraph(f"<b>{total_wagons}</b>", data_bold),
            Paragraph(f"<b>{left_open_str}</b>", data_bold),
            Paragraph(f"<b>{right_open_str}</b>", data_bold),
            Paragraph(f"<b>{top_open_str}</b>", data_bold),
            Paragraph(f"<b>{left_top_open_str}</b>", data_bold),
            Paragraph(partial_text, data_style),
            Paragraph(f'<b><font color="{rake_color}">{rake_type_str}</font></b>', data_bold),
            Paragraph(f'<b><font color="{status_color}">{status}</font></b>', data_bold),
        ]

        summary_data = [title_row, header_row, data_row]

        # Column widths — must sum to 10.0 inches (same as all other tables)
        col_widths = [1.2*inch, 1.1*inch, 0.7*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.9*inch, 1.0*inch, 1.0*inch]
        summary_table = Table(summary_data, colWidths=col_widths)

        # Status background
        status_bg = colors.HexColor('#FFEBEE') if status == "NOT OK" else colors.HexColor('#E8F5E9')

        # Rake type background
        if rake_type_str == "LOADED RAKE":
            rake_bg = colors.HexColor('#E3F2FD')  # light blue
        elif rake_type_str == "EMPTY RAKE":
            rake_bg = colors.HexColor('#FFF3E0')  # light orange
        else:
            rake_bg = WHITE

        summary_style = [
            # Title row — navy banner
            ('SPAN', (0, 0), (-1, 0)),
            ('BACKGROUND', (0, 0), (-1, 0), NAVY_MID),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, 0), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 7),
            # Header row — dark navy
            ('BACKGROUND', (0, 1), (-1, 1), NAVY_DARK),
            ('TOPPADDING', (0, 1), (-1, 1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 8),
            # Data row — white
            ('BACKGROUND', (0, 2), (-1, 2), WHITE),
            ('TOPPADDING', (0, 2), (-1, 2), 10),
            ('BOTTOMPADDING', (0, 2), (-1, 2), 10),
            # Status cell background
            ('BACKGROUND', (-1, 2), (-1, 2), status_bg),
            # Rake type cell background
            ('BACKGROUND', (8, 2), (8, 2), rake_bg),
            # General
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, SLATE_BORDER),
            ('INNERGRID', (0, 1), (-1, -1), 0.5, SLATE_BORDER),
            ('LINEBELOW', (0, 0), (-1, 0), 1, SLATE_BORDER),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]

        # Gray out N/A cells for missing cameras
        na_bg = colors.HexColor('#ECEFF1')
        if "LEFT_UP" in missing_cameras:
            summary_style.append(('BACKGROUND', (3, 2), (3, 2), na_bg))
        if "RIGHT_UP" in missing_cameras:
            summary_style.append(('BACKGROUND', (4, 2), (4, 2), na_bg))
        if "RIGHT_UP_TOP" in missing_cameras:
            summary_style.append(('BACKGROUND', (5, 2), (5, 2), na_bg))
        if "LEFT_UP_TOP" in missing_cameras:
            summary_style.append(('BACKGROUND', (6, 2), (6, 2), na_bg))

        summary_table.setStyle(TableStyle(summary_style))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.12 * inch))

        return elements

    def _format_door_status(self, state: str) -> str:
        """Format door state to uppercase readable format."""
        if not state:
            return "UNKNOWN"
        state_upper = state.upper().replace('_', ' ')
        mapping = {
            'OPEN': 'OPEN',
            'CLOSED': 'CLOSED',
            'DAMAGE': 'DAMAGE',
            'PARTIAL CLOSED': 'PARTIAL CLOSED',
            'PARTIALLY CLOSED': 'PARTIAL CLOSED',
            # Top/Side camera damage states — normalize to 'DAMAGE'
            'FLOOR DAMAGE': 'DAMAGE',
            'INNER WALL DAMAGE': 'DAMAGE',
            'OUTER WALL DAMAGE': 'DAMAGE',
            'SIDE DAMAGE': 'DAMAGE',
            'FLOOR DMG': 'DAMAGE',
            'FLOOR DMG PROBABLE': 'DAMAGE',
            'BODY DMG': 'DAMAGE',
            'BODY DMG PROBABLE': 'DAMAGE',
        }
        return mapping.get(state_upper, state_upper)

    def _effective_door_state(self, door, apply_event_filter=True):
        """Get the effective door state, respecting the open_event_raised filter.
        
        When apply_event_filter=True (RIGHT_UP camera), a door classified as
        'open' but whose open_event_raised flag is False was filtered out by
        the event pipeline and should be shown as CLOSED.
        When apply_event_filter=False (LEFT_UP camera), all open doors are shown.
        """
        state = door.get('state', '').lower()
        is_open = 'open' in state and 'partial' not in state
        if False:  # open_event_raised filter disabled
            return 'closed'
        return door.get('state', '')

    def _build_doors_text(self, wagon_doors, apply_event_filter=True):
        """Build door status text from a list of doors for a wagon."""
        if not wagon_doors:
            return "NO DATA"

        door1_status = None
        door2_status = None

        sorted_doors = sorted(wagon_doors, key=lambda d: d.get('door_number', 0))

        for door in sorted_doors:
            door_num = door.get('door_number', 1)
            status = self._format_door_status(self._effective_door_state(door, apply_event_filter))
            if door_num == 1:
                door1_status = status
            elif door_num == 2:
                door2_status = status

        if door1_status is None and len(sorted_doors) > 0:
            door1_status = self._format_door_status(self._effective_door_state(sorted_doors[0], apply_event_filter))

        if door1_status is None:
            return "NO DOOR DETECTED"
        elif door2_status:
            return f"DOOR 1 {door1_status} / DOOR 2 {door2_status}"
        else:
            return f"DOOR 1 {door1_status}"

    def _has_open_door(self, wagon_doors, apply_event_filter=True):
        """Check if any door in the list has a confirmed open or damage event.
        
        When apply_event_filter=True (RIGHT_UP), only open doors with
        open_event_raised=True are counted.
        When apply_event_filter=False (LEFT_UP), all open doors are counted.
        """
        if not wagon_doors:
            return False
        for d in wagon_doors:
            state = d.get('state', '').lower()
            # Damage is always reported
            if 'damage' in state:
                return True
            # Open doors: apply filter only for RIGHT camera
            if 'open' in state and 'partial' not in state:
                if True:  # open_event_raised filter disabled
                    return True
        return False

    def _create_unified_wagon_table(self, merged_wagons, missing_cameras=None):
        """
        Create professional wagon inspection table with navy header banner.

        merged_wagons: List of dicts with keys:
            wagon_sr_no, ocr_wagon_number,
            left_doors_text, right_doors_text, top_doors_text, left_top_doors_text,
            has_open_left, has_open_right, has_open_top, has_open_left_top
        missing_cameras: List of camera IDs whose feeds were not available
        """
        elements = []
        if missing_cameras is None:
            missing_cameras = []

        NO_FEED_TEXT = "\u26a0 NO FEED"

        # Map camera IDs to their column indices
        # col 2 = LEFT_UP doors, col 3 = RIGHT_UP doors,
        # col 4 = RIGHT_UP_TOP damages, col 5 = LEFT_UP_TOP damages
        camera_col_map = {
            'LEFT_UP': 2,
            'RIGHT_UP': 3,
            'RIGHT_UP_TOP': 4,
            'LEFT_UP_TOP': 5,
        }
        # Build missing column set
        missing_cols = set()
        for cam in missing_cameras:
            if cam in camera_col_map:
                missing_cols.add(camera_col_map[cam])

        # ── Styles for this table ──
        col_header_style = ParagraphStyle(
            'WagonColHeader', parent=self.styles['Normal'],
            fontSize=8, alignment=TA_CENTER,
            textColor=WHITE, fontName='Helvetica-Bold', leading=11
        )
        cell_style = ParagraphStyle(
            'WagonCell', parent=self.styles['Normal'],
            fontSize=8, alignment=TA_CENTER,
            textColor=colors.HexColor('#263238'), fontName='Helvetica', leading=11
        )
        cell_bold = ParagraphStyle(
            'WagonCellBold', parent=self.styles['Normal'],
            fontSize=8, alignment=TA_CENTER,
            textColor=colors.HexColor('#1A1A2E'), fontName='Helvetica-Bold', leading=11
        )
        issue_style = ParagraphStyle(
            'WagonIssue', parent=self.styles['Normal'],
            fontSize=8, alignment=TA_CENTER,
            textColor=colors.HexColor('#C62828'), fontName='Helvetica-Bold', leading=11
        )
        no_feed_style = ParagraphStyle(
            'WagonNoFeed', parent=self.styles['Normal'],
            fontSize=7, alignment=TA_CENTER,
            textColor=colors.HexColor('#9E9E9E'), fontName='Helvetica-Oblique', leading=10
        )

        # ── Section title row (navy banner) ──
        title_row = [
            Paragraph('<b>WAGON INSPECTION DETAILS</b>', self.styles['SectionTitleWhite']),
            '', '', '', '', '', ''
        ]

        # ── Column headers (dark navy) ──
        header_row = [
            Paragraph("SR.NO", col_header_style),
            Paragraph("WAGON NUMBER", col_header_style),
            Paragraph("LEFT CAMERA<br/>DOORS", col_header_style),
            Paragraph("RIGHT CAMERA<br/>DOORS", col_header_style),
            Paragraph("R-TOP<br/>DAMAGES", col_header_style),
            Paragraph("L-TOP<br/>DAMAGES", col_header_style),
            Paragraph("WAGON<br/>TYPE", col_header_style),
        ]

        table_data = [title_row, header_row]

        # Track which rows need issue highlighting
        # Each entry: (row_idx_in_table, [col_indices])
        highlight_info = []

        for idx, wagon in enumerate(merged_wagons, start=1):
            sr_no = str(wagon['wagon_sr_no'])
            row_idx = idx + 1  # +1 for the title row offset (title=0, header=1, data starts at 2)

            # Wagon number from RIGHT camera (OCR)
            ocr_num = wagon.get('ocr_wagon_number')
            if ocr_num and str(ocr_num).strip() not in ('-', '', 'None'):
                wagon_num_display = str(ocr_num).upper()
            else:
                wagon_num_display = "-"

            # Determine issue status per camera
            has_left = wagon.get('has_open_left') and 2 not in missing_cols
            has_right = wagon.get('has_open_right') and 3 not in missing_cols
            has_top = wagon.get('has_open_top') and 4 not in missing_cols
            has_left_top = wagon.get('has_open_left_top') and 5 not in missing_cols

            # Choose text styles based on issues
            left_text = NO_FEED_TEXT if 2 in missing_cols else wagon.get('left_doors_text', 'NO DATA')
            right_text = NO_FEED_TEXT if 3 in missing_cols else wagon.get('right_doors_text', 'NO DATA')
            top_text = NO_FEED_TEXT if 4 in missing_cols else wagon.get('top_doors_text', 'NO DATA')
            left_top_text = NO_FEED_TEXT if 5 in missing_cols else wagon.get('left_top_doors_text', 'NO DATA')

            # Sanitize damage columns: any specific damage type → 'DAMAGE'
            _DAMAGE_VARIANTS = {'FLOOR DAMAGE', 'INNER WALL DAMAGE', 'OUTER WALL DAMAGE',
                                'SIDE DAMAGE', 'FLOOR DMG',
                                'FLOOR DMG PROBABLE', 'BODY DMG', 'BODY DMG PROBABLE'}
            if top_text.upper() in _DAMAGE_VARIANTS:
                top_text = "DAMAGE"
            if left_top_text.upper() in _DAMAGE_VARIANTS:
                left_top_text = "DAMAGE"

            # Wagon type for this wagon
            wagon_type_text = wagon.get('wagon_type', '-')
            # Style for wagon type
            if wagon_type_text == "LOADED":
                wtype_cell_style = ParagraphStyle(
                    'WagonLoaded', parent=cell_style,
                    textColor=colors.HexColor('#1565C0'), fontName='Helvetica-Bold'
                )
            elif wagon_type_text == "EMPTY":
                wtype_cell_style = ParagraphStyle(
                    'WagonEmpty', parent=cell_style,
                    textColor=colors.HexColor('#E65100'), fontName='Helvetica-Bold'
                )
            else:
                wtype_cell_style = cell_style

            # Style cells — red text for issues, gray for no feed
            left_cell_style = issue_style if has_left else (no_feed_style if 2 in missing_cols else cell_style)
            right_cell_style = issue_style if has_right else (no_feed_style if 3 in missing_cols else cell_style)
            top_cell_style = issue_style if has_top else (no_feed_style if 4 in missing_cols else cell_style)
            lt_cell_style = issue_style if has_left_top else (no_feed_style if 5 in missing_cols else cell_style)

            table_data.append([
                Paragraph(f"<b>{sr_no}</b>", cell_bold),
                Paragraph(f"<b>{wagon_num_display}</b>", cell_bold) if (wagon_num_display != "-" and not wagon.get('is_manipulated', False)) else Paragraph(wagon_num_display, cell_style),
                Paragraph(left_text, left_cell_style),
                Paragraph(right_text, right_cell_style),
                Paragraph(top_text, top_cell_style),
                Paragraph(left_top_text, lt_cell_style),
                Paragraph(wagon_type_text, wtype_cell_style),
            ])

            # Track issue columns for background highlighting
            issue_cols = []
            if has_left:
                issue_cols.append(2)
            if has_right:
                issue_cols.append(3)
            if has_top:
                issue_cols.append(4)
            if has_left_top:
                issue_cols.append(5)
            if issue_cols:
                highlight_info.append((row_idx, issue_cols))

        # Column widths: SR.NO | WAGON# | LEFT DOORS | RIGHT DOORS | R-TOP | L-TOP | WAGON TYPE = ~10.0 inches
        table = Table(table_data,
                      colWidths=[0.5*inch, 1.5*inch, 2.0*inch, 2.0*inch, 1.0*inch, 1.0*inch, 1.0*inch],
                      repeatRows=2)  # Repeat title + header rows on every page

        # ── Alternating row colors ──
        alt_even = WHITE
        alt_odd = SLATE_BG  # light slate

        table_style = [
            # Title row — navy banner
            ('SPAN', (0, 0), (-1, 0)),
            ('BACKGROUND', (0, 0), (-1, 0), NAVY_MID),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, 0), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 7),
            # Column header row — dark navy
            ('BACKGROUND', (0, 1), (-1, 1), NAVY_DARK),
            ('TOPPADDING', (0, 1), (-1, 1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 8),
            # General
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, SLATE_BORDER),
            ('INNERGRID', (0, 1), (-1, -1), 0.5, SLATE_BORDER),
            ('LINEBELOW', (0, 0), (-1, 0), 1, SLATE_BORDER),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 2), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 2), (-1, -1), 8),
        ]

        # Alternating row backgrounds for data rows (starting at row 2)
        for i in range(len(merged_wagons)):
            row_idx = i + 2  # data rows start at index 2
            bg = alt_even if i % 2 == 0 else alt_odd
            table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), bg))

        # Gray background for entire column of missing cameras
        no_feed_bg = colors.HexColor('#ECEFF1')
        num_data_rows = len(merged_wagons)
        for col in missing_cols:
            if num_data_rows > 0:
                table_style.append(('BACKGROUND', (col, 2), (col, num_data_rows + 1), no_feed_bg))

        # Issue highlighting — light red background
        issue_red_bg = colors.HexColor('#FFEBEE')
        for row_idx, issue_cols in highlight_info:
            if len(issue_cols) >= 2:
                # Multiple cameras have issues — highlight entire row
                table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), issue_red_bg))
            else:
                # Single camera issue — highlight SR.NO + WAGON NUMBER + specific column
                table_style.append(('BACKGROUND', (0, row_idx), (1, row_idx), issue_red_bg))
                for col in issue_cols:
                    table_style.append(('BACKGROUND', (col, row_idx), (col, row_idx), issue_red_bg))

        table.setStyle(TableStyle(table_style))
        elements.append(table)

        return elements

    def _create_open_door_images(self, left_doors, right_doors, top_doors=None,
                                     left_top_doors=None, merged_wagons=None, max_wagons=0,
                                     left_side_damages=None, right_side_damages=None):
        """Create section with open door snapshot images from all cameras.

        Layout:
        - "Damaged Wagon Report" title displayed ONCE (centered)
        - "Total Damaged Wagons: N" displayed ONCE (simple format)
        - Continuous table entries without repeating section headers
        - Each entry: table row (with Camera Angle column) followed by image
        """
        elements = []

        # Collect open door images from all cameras
        # For side cameras (LEFT/RIGHT), only include doors where the open
        # event was actually raised (passed confidence/persistence/edge filters).
        open_images = []

        for door in (left_doors or []):
            state = door.get('state', '').lower()
            is_open = 'open' in state and 'partial' not in state
            is_damage = 'damage' in state
            # LEFT camera: show ALL open doors (no event filter)
            if is_open or is_damage:
                snapshot = door.get('local_snapshot_path') or door.get('snapshot_path')
                if snapshot and os.path.exists(snapshot):
                    wagon_num = door.get('wagon_number', '?')
                    door_num = door.get('door_number', '?')
                    label_type = 'Damage' if is_damage else 'Door'
                    open_images.append({
                        'path': snapshot,
                        'label': f"Wagon {wagon_num} {label_type} {door_num}",
                        'camera': 'Left',
                        'wagon_number': wagon_num
                    })

        for door in (right_doors or []):
            state = door.get('state', '').lower()
            is_open = 'open' in state and 'partial' not in state
            is_damage = 'damage' in state
            # Skip open doors that did NOT pass the event filter
            if False:  # open_event_raised filter disabled
                continue
            if is_open or is_damage:
                snapshot = door.get('local_snapshot_path') or door.get('snapshot_path')
                if snapshot and os.path.exists(snapshot):
                    wagon_num = door.get('wagon_number', '?')
                    door_num = door.get('door_number', '?')
                    label_type = 'Damage' if is_damage else 'Door'
                    open_images.append({
                        'path': snapshot,
                        'label': f"Wagon {wagon_num} {label_type} {door_num}",
                        'camera': 'Right',
                        'wagon_number': wagon_num
                    })

        for door in (top_doors or []):
            # Top camera uses damage states, not 'open' door states
            state = door.get('state', '').lower()
            has_damage = state not in ('no_damage', 'closed', '', 'loaded')
            if has_damage:
                snapshot = door.get('local_snapshot_path') or door.get('snapshot_path')
                if snapshot and os.path.exists(snapshot):
                    wagon_num = door.get('wagon_number', '?')
                    door_num = door.get('damage_number', door.get('door_number', '?'))
                    open_images.append({
                        'path': snapshot,
                        'label': f"Wagon {wagon_num} Damage {door_num}",
                        'camera': 'Right-Top',
                        'wagon_number': wagon_num
                    })

        for door in (left_top_doors or []):
            # LEFT_UP_TOP camera uses damage states
            state = door.get('state', '').lower()
            has_damage = state not in ('no_damage', 'closed', '', 'loaded')
            if has_damage:
                snapshot = door.get('local_snapshot_path') or door.get('snapshot_path')
                if snapshot and os.path.exists(snapshot):
                    wagon_num = door.get('wagon_number', '?')
                    door_num = door.get('damage_number', door.get('door_number', '?'))
                    open_images.append({
                        'path': snapshot,
                        'label': f"Wagon {wagon_num} Damage {door_num}",
                        'camera': 'Left-Top',
                        'wagon_number': wagon_num
                    })

        # Side camera damages (from side_damage.pt model)
        for damage in (left_side_damages or []):
            snapshot_path = damage.get('_local_snapshot_path')
            if snapshot_path and os.path.exists(snapshot_path):
                wagon_num = damage.get('wagon_number', '?')
                damage_num = damage.get('damage_number', '?')
                damage_class = damage.get('state', 'damage').replace('_', ' ').title()
                open_images.append({
                    'path': snapshot_path,
                    'label': f"Wagon {wagon_num} {damage_class} {damage_num}",
                    'camera': 'Left-Side',
                    'wagon_number': wagon_num
                })

        for damage in (right_side_damages or []):
            snapshot_path = damage.get('_local_snapshot_path')
            if snapshot_path and os.path.exists(snapshot_path):
                wagon_num = damage.get('wagon_number', '?')
                damage_num = damage.get('damage_number', '?')
                damage_class = damage.get('state', 'damage').replace('_', ' ').title()
                open_images.append({
                    'path': snapshot_path,
                    'label': f"Wagon {wagon_num} {damage_class} {damage_num}",
                    'camera': 'Right-Side',
                    'wagon_number': wagon_num
                })

        if not open_images:
            return elements

        # Sort by wagon number first, then by camera type priority
        _CAMERA_PRIORITY = {
            'Left': 0, 'Right': 1,
            'Left-Side': 2, 'Right-Side': 3,
            'Left-Top': 4, 'Right-Top': 5,
        }

        def _wagon_sort_key(img):
            wn = img.get('wagon_number', 0)
            try:
                wn_int = int(wn)
            except (ValueError, TypeError):
                wn_int = 999999
            cam_priority = _CAMERA_PRIORITY.get(img.get('camera', ''), 99)
            return (wn_int, cam_priority)
        open_images.sort(key=_wagon_sort_key)

        elements.append(PageBreak())

        # Build lookup from merged_wagons
        wagon_lookup = {}
        if merged_wagons:
            for w in merged_wagons:
                wagon_lookup[w['wagon_sr_no']] = w

        # Collect unique damaged wagon numbers for total count
        damaged_wagon_nums = sorted(set(
            img.get('wagon_number', '?') for img in open_images
            if img.get('wagon_number', '?') != '?'
        ))
        total_damaged = len(damaged_wagon_nums)

        MAX_IMG_WIDTH = 4.2 * inch
        MAX_IMG_HEIGHT = 2.8 * inch

        # =====================================================================
        # TITLE — displayed ONCE, centered
        # =====================================================================
        elements.append(Paragraph(
            "<b>Damaged Wagon Report</b>",
            self.styles['ReportTitle']
        ))
        elements.append(Spacer(1, 0.05 * inch))

        # Total count — simple format, no fraction
        elements.append(Paragraph(
            f"<b>Total Damaged Wagons: {total_damaged}</b>",
            self.styles['ReportSubtitle']
        ))
        elements.append(Spacer(1, 0.2 * inch))

        # =====================================================================
        # GROUP IMAGES BY WAGON NUMBER
        # =====================================================================
        from collections import OrderedDict
        wagon_images = OrderedDict()
        for img_info in open_images:
            wn = img_info.get('wagon_number', '?')
            if wn not in wagon_images:
                wagon_images[wn] = []
            wagon_images[wn].append(img_info)

        IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(IST)
        date_time_str = now.strftime("%d-%m-%Y %H:%M:%S IST")

        # Camera label mapping for clear display
        _CAMERA_LABELS = {
            'Left': 'Side Camera (Left) – Open Door',
            'Right': 'Side Camera (Right) – Open Door',
            'Left-Side': 'Side Camera (Left) – Side Damage',
            'Right-Side': 'Side Camera (Right) – Side Damage',
            'Left-Top': 'Top Camera (Left) – Damage',
            'Right-Top': 'Top Camera (Right) – Damage',
        }

        sn_counter = 0

        for wagon_num, images in wagon_images.items():
            try:
                sn_counter += 1

                # Get OCR wagon number from merged_wagons lookup
                wagon_info = wagon_lookup.get(wagon_num, {})
                ocr_num = wagon_info.get('ocr_wagon_number', '-')
                if not ocr_num or str(ocr_num).strip() in ('', 'None'):
                    ocr_num = '-'

                # Camera angles for this wagon
                camera_angles = ', '.join(sorted(set(
                    img.get('camera', '-') for img in images
                )))
                num_issues = len(images)

                # ── INFO TABLE ──
                header_row = [
                    Paragraph("<b>SN</b>", self.styles['TableHeader']),
                    Paragraph("<b>Wagon ID</b>", self.styles['TableHeader']),
                    Paragraph("<b>Wagon No.</b>", self.styles['TableHeader']),
                    Paragraph("<b>Camera Angles</b>", self.styles['TableHeader']),
                    Paragraph("<b>Issues</b>", self.styles['TableHeader']),
                    Paragraph("<b>Date &amp; Time</b>", self.styles['TableHeader']),
                ]

                data_row = [
                    Paragraph(f"{sn_counter}.", self.styles['TableCell']),
                    Paragraph(str(wagon_num), self.styles['TableCell']),
                    Paragraph(str(ocr_num), self.styles['TableCell']),
                    Paragraph(f"<b>{camera_angles}</b>", self.styles['TableCell']),
                    Paragraph(f"<b>{num_issues}</b>", self.styles['TableCell']),
                    Paragraph(date_time_str, self.styles['TableCell']),
                ]

                info_col_widths = [0.5*inch, 0.8*inch, 2.2*inch, 1.8*inch, 0.7*inch, 2.6*inch]
                info_table = Table([header_row, data_row], colWidths=info_col_widths)
                info_table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 0), (-1, 0), HEADER_GRAY),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ]))

                # ── IMAGE GRID (2 columns) ──
                image_cells = []
                for img_info in images:
                    try:
                        camera = img_info.get('camera', '-')
                        img_label = img_info.get('label', '')
                        # Dynamic label: use 'Damage' if damage, else 'Open Door'
                        if 'damage' in img_label.lower():
                            issue_type = 'Damage'
                        else:
                            issue_type = 'Open Door'
                        display_label = f"{camera} Camera – {issue_type}"

                        img = Image(img_info['path'])
                        img_w, img_h = img.drawWidth, img.drawHeight

                        if img_w > MAX_IMG_WIDTH:
                            scale = MAX_IMG_WIDTH / img_w
                            img_w *= scale
                            img_h *= scale
                        if img_h > MAX_IMG_HEIGHT:
                            scale = MAX_IMG_HEIGHT / img_h
                            img_w *= scale
                            img_h *= scale

                        img.drawWidth = img_w
                        img.drawHeight = img_h

                        # Label style
                        label_style = ParagraphStyle(
                            'SnapLabel',
                            parent=self.styles['TableCell'],
                            fontSize=8,
                            leading=10,
                            alignment=1,  # CENTER
                            textColor=colors.HexColor('#1A1A2E'),
                            fontName='Helvetica-Bold',
                        )

                        cell_content = [
                            Paragraph(f"<b>{display_label}</b>", label_style),
                            Spacer(1, 0.05 * inch),
                            img,
                        ]
                        image_cells.append(cell_content)
                    except Exception as e:
                        print(f"  ⚠ Failed to load image {img_info.get('path', '?')}: {e}")

                if not image_cells:
                    continue

                # Build image grid — adaptive layout:
                #   1 image  → single centered column (full width)
                #   2+ images → 2-column grid, last odd image centered
                full_grid_width = 9.6 * inch
                grid_col_width = 4.8 * inch

                if len(image_cells) == 1:
                    # Single image: center it across the full width
                    grid_rows = [image_cells]
                    image_grid = Table(grid_rows, colWidths=[full_grid_width])
                else:
                    # 2-column grid for paired images
                    grid_rows = []
                    paired_count = len(image_cells) - (len(image_cells) % 2)
                    for i in range(0, paired_count, 2):
                        grid_rows.append([image_cells[i], image_cells[i + 1]])

                    # If odd number of images, center the last one across both columns
                    if len(image_cells) % 2 == 1:
                        last_cell = image_cells[-1]
                        grid_rows.append([last_cell])

                    image_grid = Table(grid_rows, colWidths=[grid_col_width, grid_col_width])

                    # Merge the last row's single cell across both columns if odd
                    if len(image_cells) % 2 == 1:
                        last_row_idx = len(grid_rows) - 1
                        image_grid.setStyle(TableStyle([
                            ('SPAN', (0, last_row_idx), (1, last_row_idx)),
                        ]))

                image_grid.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#E0E0E0')),
                ]))

                # KeepTogether: info table + image grid on same page
                entry_block = KeepTogether([
                    info_table,
                    Spacer(1, 0.15 * inch),
                    image_grid,
                    Spacer(1, 0.3 * inch),
                ])
                elements.append(entry_block)

            except Exception as e:
                print(f"  ⚠ Failed to add wagon {wagon_num} images: {e}")

        return elements

    # ══════════════════════════════════════════════════════════════════════
    # CROSS-CAMERA WAGON ALIGNMENT (±3 drift fix)
    # ══════════════════════════════════════════════════════════════════════

    def _find_best_offset(self, side_doors, top_doors, pair_label=""):
        """
        Find the best wagon number offset between a side camera (open doors)
        and its paired top camera (inner_wall_damage).

        Only uses inner_wall_damage for corroboration — floor_damage is ignored.

        Returns:
            int: offset to ADD to top camera wagon numbers to align with side camera.
                 0 = already aligned, -1 = top numbers are +1 too high, +1 = too low.
        """
        # Side signal = OPEN doors (non-partial) ∪ DAMAGE detections
        # Both classes are in side_doors (door_state.pt outputs all 4 classes)
        side_open_wagons = set()
        side_damage_wagons = set()
        for d in (side_doors or []):
            state = d.get('state', '').lower()
            wn = d.get('wagon_number')
            if wn is None:
                continue
            if 'open' in state and 'partial' not in state:
                side_open_wagons.add(wn)
            elif 'damage' in state:
                side_damage_wagons.add(wn)

        side_signal_wagons = side_open_wagons | side_damage_wagons

        if not side_signal_wagons:
            print(f"     {pair_label}: no open doors or damages on side — skipping alignment")
            return 0

        # Find top camera wagons with inner_wall_damage ONLY (ignore floor_damage)
        INNER_WALL_LABELS = {'inner_wall_damage', 'inner wall damage', 'inner_wall'}
        top_damage_wagons = set()
        for d in (top_doors or []):
            state = d.get('state', '').lower().strip()
            if state in INNER_WALL_LABELS:
                wn = d.get('wagon_number')
                if wn is not None:
                    top_damage_wagons.add(wn)

        if not top_damage_wagons:
            print(f"     {pair_label}: no inner_wall_damage in top camera — skipping alignment")
            return 0

        print(f"     {pair_label}: side open doors at wagons {sorted(side_open_wagons)}")
        if side_damage_wagons:
            print(f"     {pair_label}: side damages at wagons {sorted(side_damage_wagons)}")
        print(f"     {pair_label}: top inner_wall_damage at wagons {sorted(top_damage_wagons)}")

        # Test offsets -3 to +3 and pick the one with most matches
        best_offset = 0
        best_matches = 0

        for test_offset in [-3, -2, -1, 0, 1, 2, 3]:
            matches = 0
            for side_wn in side_signal_wagons:
                # After applying test_offset: top_wn + test_offset = side_wn
                # So raw top_wn that would match = side_wn - test_offset
                if (side_wn - test_offset) in top_damage_wagons:
                    matches += 1
            if matches > best_matches:
                best_matches = matches
                best_offset = test_offset

        if best_offset != 0:
            print(f"     ✅ {pair_label}: OFFSET = {best_offset:+d} "
                  f"({best_matches} open↔damage pairs corroborated)")
        else:
            print(f"     ✅ {pair_label}: ALIGNED (offset=0, {best_matches} pairs matched)")

        return best_offset

    def _apply_wagon_offset(self, items, offset):
        """Apply wagon number offset to all items in a doors/damages list."""
        if offset == 0 or not items:
            return items

        adjusted = []
        for item in items:
            item_copy = dict(item)
            wn = item_copy.get('wagon_number', 1)
            new_wn = wn + offset
            if new_wn >= 1:  # Don't allow wagon number < 1
                item_copy['wagon_number'] = new_wn
            adjusted.append(item_copy)

        return adjusted

    def _fuzzy_merge_top_to_side(self, side_doors, top_doors, pair_label=""):
        """
        Per-wagon fuzzy merge: adjust top camera wagon numbers by ±1 to match
        side camera open doors when exact match doesn't exist.

        After global offset alignment, some wagons may still be misaligned due
        to LOCAL counting drift (e.g. side=54, top=53). This method:
        1. Finds exact matches (same wagon number on both) → no change
        2. For unmatched top wagons, checks ±1 for an unmatched side wagon
        3. Adjusts the top wagon number to match the side wagon number

        This ensures all corroborated damages get merged in the combined report.

        Args:
            side_doors: Side camera doors list (reference, open doors)
            top_doors: Top camera doors/damages list (to be adjusted)
            pair_label: Label for logging

        Returns:
            Adjusted top_doors list with wagon numbers corrected
        """
        if not side_doors or not top_doors:
            return top_doors, {}

        # Side signal wagons = OPEN doors (non-partial) ∪ DAMAGE detections
        # Both classes are in side_doors (door_state.pt outputs all 4 classes)
        side_signal_wagons = set()
        for d in (side_doors or []):
            state = d.get('state', '').lower()
            wn = d.get('wagon_number')
            if wn is None:
                continue
            if 'open' in state and 'partial' not in state:
                side_signal_wagons.add(wn)
            elif 'damage' in state:
                side_signal_wagons.add(wn)

        # Get top camera wagon numbers with inner_wall_damage
        INNER_WALL_LABELS = {'inner_wall_damage', 'inner wall damage', 'inner_wall'}
        top_damage_wagons = set()
        for d in (top_doors or []):
            state = d.get('state', '').lower().strip()
            if state in INNER_WALL_LABELS:
                wn = d.get('wagon_number')
                if wn is not None:
                    top_damage_wagons.add(wn)

        if not side_signal_wagons or not top_damage_wagons:
            return top_doors, {}

        # Step 1: Find exact matches — these are already aligned, no change needed
        exact_matches = side_signal_wagons & top_damage_wagons

        # Step 2: Find unmatched wagons on each side
        unmatched_side = side_signal_wagons - exact_matches
        unmatched_top = top_damage_wagons - exact_matches

        if not unmatched_side or not unmatched_top:
            return top_doors, {}

        # Step 3: For each unmatched top wagon, try ±1 to match an unmatched side wagon
        remap = {}  # old_top_wn → new_top_wn (adjusted to match side)
        used_side = set()

        for top_wn in sorted(unmatched_top):
            for delta in [+1, -1]:  # Check +1 first (top is often 1 behind side)
                candidate_side = top_wn + delta
                if candidate_side in unmatched_side and candidate_side not in used_side:
                    remap[top_wn] = candidate_side
                    used_side.add(candidate_side)
                    print(f"     [FuzzyMerge] {pair_label}: top wagon {top_wn} → {candidate_side} "
                          f"(±1 match with side signal)")
                    break

        if not remap:
            return top_doors, {}

        # Step 4: Apply remap to ALL top door entries at those wagon numbers
        adjusted = []
        for d in top_doors:
            wn = d.get('wagon_number')
            if wn in remap:
                d_copy = dict(d)
                d_copy['wagon_number'] = remap[wn]
                adjusted.append(d_copy)
            else:
                adjusted.append(d)

        print(f"     [FuzzyMerge] {pair_label}: {len(remap)} wagon(s) adjusted by ±1")
        return adjusted, remap

    def _align_cross_camera_wagons(self, left_doors, right_doors, top_doors, left_top_doors,
                                    left_summary, right_summary, top_summary, left_top_summary):
        """
        Align wagon numbers across paired cameras to fix ±1 counting drift.

        Uses open doors from side cameras and inner_wall_damage from top cameras
        as corroboration signals. Floor damage is ignored for alignment.

        Camera pairs:
          - RIGHT_UP (side, open doors) ↔ LEFT_UP_TOP (top, inner_wall_damage)
          - LEFT_UP (side, open doors) ↔ RIGHT_UP_TOP (top, inner_wall_damage)

        Returns:
            Tuple of (left_doors, right_doors, top_doors, left_top_doors, offsets_dict)
            with corrected wagon numbers in top camera data.
            offsets_dict contains the per-camera offsets applied.
        """
        print(f"\n  {'='*55}")
        print(f"  🔧 CROSS-CAMERA WAGON ALIGNMENT")
        print(f"  {'='*55}")

        # Pair 1: RIGHT_UP (open doors) ↔ LEFT_UP_TOP (inner_wall_damage)
        left_top_offset = self._find_best_offset(
            side_doors=right_doors,
            top_doors=left_top_doors,
            pair_label="RIGHT_UP ↔ LEFT_UP_TOP"
        )
        if left_top_offset != 0:
            left_top_doors = self._apply_wagon_offset(left_top_doors, left_top_offset)
            print(f"     → LEFT_UP_TOP wagon numbers adjusted by {left_top_offset:+d}")

        # Pair 2: LEFT_UP (open doors) ↔ RIGHT_UP_TOP (inner_wall_damage)
        top_offset = self._find_best_offset(
            side_doors=left_doors,
            top_doors=top_doors,
            pair_label="LEFT_UP ↔ RIGHT_UP_TOP"
        )
        if top_offset != 0:
            top_doors = self._apply_wagon_offset(top_doors, top_offset)
            print(f"     → RIGHT_UP_TOP wagon numbers adjusted by {top_offset:+d}")

        print(f"  {'='*55}\n")

        offsets_dict = {
            'LEFT_UP_TOP': left_top_offset,
            'RIGHT_UP_TOP': top_offset,
        }

        return left_doors, right_doors, top_doors, left_top_doors, offsets_dict

    def generate(self, left_data: Dict, right_data: Dict, top_data: Dict = None,
                 left_top_data: Dict = None, missing_cameras: list = None,
                 **kwargs) -> str:
        """
        Generate the unified combined PDF report with all 4 cameras.

        Args:
            left_data: Dict with keys: wagon_summary, doors, state_counts, wagon_numbers,
                       source_video_url, main_report_url
            right_data: Same structure as left_data
            top_data: Same structure as left_data (RIGHT_UP_TOP camera)
            left_top_data: Same structure as left_data (LEFT_UP_TOP camera)
            missing_cameras: List of camera IDs whose feeds were not available from S3

        Returns:
            Path to generated PDF
        """
        if top_data is None:
            top_data = {}
        if left_top_data is None:
            left_top_data = {}
        if missing_cameras is None:
            missing_cameras = []

        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=landscape(A4),
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        elements = []

        # Header (with PARTIAL warning if cameras missing)
        elements.extend(self._create_header(missing_cameras=missing_cameras))

        # Extract data from all 4 cameras
        left_summary = left_data.get('wagon_summary') or []
        right_summary = right_data.get('wagon_summary') or []
        top_summary = top_data.get('wagon_summary') or []
        left_top_summary = left_top_data.get('wagon_summary') or []
        left_doors = left_data.get('doors') or []
        right_doors = right_data.get('doors') or []
        top_doors = top_data.get('damages') or top_data.get('doors') or []
        left_top_doors = left_top_data.get('damages') or left_top_data.get('doors') or []

        # ── CROSS-CAMERA WAGON ALIGNMENT ──
        # Fix ±3 wagon number drift between side↔top camera pairs
        # Uses open doors (side) ↔ inner_wall_damage (top) as corroboration
        left_doors, right_doors, top_doors, left_top_doors, alignment_offsets = self._align_cross_camera_wagons(
            left_doors=left_doors,
            right_doors=right_doors,
            top_doors=top_doors,
            left_top_doors=left_top_doors,
            left_summary=left_summary,
            right_summary=right_summary,
            top_summary=top_summary,
            left_top_summary=left_top_summary
        )

        # Store alignment offsets so caller can access them
        self.alignment_offsets = alignment_offsets

        # ── PER-WAGON FUZZY MERGE (±1) ──
        # After global offset, handle LOCAL drift: if a top camera wagon has no
        # exact side match but has a ±1 neighbor with an open door, adjust the
        # top wagon number so they merge in the report.
        # Pair 1: RIGHT_UP (side) ↔ LEFT_UP_TOP (top)
        left_top_doors, left_top_fuzzy_remap = self._fuzzy_merge_top_to_side(
            side_doors=right_doors,
            top_doors=left_top_doors,
            pair_label="RIGHT_UP ↔ LEFT_UP_TOP"
        )
        # Pair 2: LEFT_UP (side) ↔ RIGHT_UP_TOP (top)
        top_doors, top_fuzzy_remap = self._fuzzy_merge_top_to_side(
            side_doors=left_doors,
            top_doors=top_doors,
            pair_label="LEFT_UP ↔ RIGHT_UP_TOP"
        )

        # Store fuzzy remaps in alignment_offsets for JSON patching
        # Format: {old_wagon_num: new_wagon_num, ...}
        if left_top_fuzzy_remap:
            self.alignment_offsets['LEFT_UP_TOP_fuzzy_remap'] = left_top_fuzzy_remap
        if top_fuzzy_remap:
            self.alignment_offsets['RIGHT_UP_TOP_fuzzy_remap'] = top_fuzzy_remap

        # Group doors by wagon number for each camera
        left_doors_by_wagon = {}
        for door in left_doors:
            wn = door.get('wagon_number', 1)
            left_doors_by_wagon.setdefault(wn, []).append(door)

        right_doors_by_wagon = {}
        for door in right_doors:
            wn = door.get('wagon_number', 1)
            right_doors_by_wagon.setdefault(wn, []).append(door)

        top_doors_by_wagon = {}
        for door in top_doors:
            wn = door.get('wagon_number', 1)
            top_doors_by_wagon.setdefault(wn, []).append(door)

        left_top_doors_by_wagon = {}
        for door in left_top_doors:
            wn = door.get('wagon_number', 1)
            left_top_doors_by_wagon.setdefault(wn, []).append(door)

        # Use the longest wagon list as the reference
        # RIGHT_UP has OCR, so prefer its wagon numbers
        max_wagons = max(len(left_summary), len(right_summary), len(top_summary), len(left_top_summary))
        if max_wagons == 0:
            max_wagons = max(
                max(left_doors_by_wagon.keys()) if left_doors_by_wagon else 0,
                max(right_doors_by_wagon.keys()) if right_doors_by_wagon else 0,
                max(top_doors_by_wagon.keys()) if top_doors_by_wagon else 0,
                max(left_top_doors_by_wagon.keys()) if left_top_doors_by_wagon else 0
            )

        # Build merged wagon list — matched by sequence order
        merged_wagons = []
        left_open_count = 0
        right_open_count = 0
        top_open_count = 0
        left_top_open_count = 0

        left_partial_count = 0
        right_partial_count = 0
        top_partial_count = 0
        left_top_partial_count = 0
        loaded_wagon_count = 0
        empty_wagon_count = 0

        for i in range(max_wagons):
            wagon_num = i + 1  # 1-based wagon number

            # Get LEFT camera data for this wagon
            left_wagon_doors = left_doors_by_wagon.get(wagon_num, [])
            left_text = self._build_doors_text(left_wagon_doors, apply_event_filter=False) if left_wagon_doors else "NO DOOR DETECTED"
            has_open_left = self._has_open_door(left_wagon_doors, apply_event_filter=False)

            # Get RIGHT camera data for this wagon
            right_wagon_doors = right_doors_by_wagon.get(wagon_num, [])
            right_text = self._build_doors_text(right_wagon_doors, apply_event_filter=True) if right_wagon_doors else "NO DOOR DETECTED"
            has_open_right = self._has_open_door(right_wagon_doors, apply_event_filter=True)

            # Get RIGHT_UP_TOP camera data for this wagon (damage/loaded detection)
            top_wagon_doors = top_doors_by_wagon.get(wagon_num, [])

            # Check is_loaded from top_summary
            is_loaded = False
            if i < len(top_summary):
                is_loaded = top_summary[i].get('is_loaded', False)

            # Track loaded status for rake type calculation
            if is_loaded:
                loaded_wagon_count += 1
            else:
                empty_wagon_count += 1

            # Check if any actual damage detected (exclude no_damage/loaded states)
            has_actual_damage = any(
                d.get('state', '').lower() not in ('no_damage', 'closed', '', 'loaded')
                for d in top_wagon_doors
            ) if top_wagon_doors else False

            if has_actual_damage:
                top_text = "DAMAGE"
                has_open_top = True
            else:
                top_text = "OK"
                has_open_top = False

            # Get LEFT_UP_TOP camera data for this wagon
            left_top_wagon_doors = left_top_doors_by_wagon.get(wagon_num, [])

            # Check is_loaded from left_top_summary
            is_loaded_lt = False
            if i < len(left_top_summary):
                is_loaded_lt = left_top_summary[i].get('is_loaded', False)

            has_actual_damage_lt = any(
                d.get('state', '').lower() not in ('no_damage', 'closed', '', 'loaded')
                for d in left_top_wagon_doors
            ) if left_top_wagon_doors else False

            if has_actual_damage_lt:
                left_top_text = "DAMAGE"
                has_open_left_top = True
            else:
                left_top_text = "OK"
                has_open_left_top = False

            # Get OCR wagon number from RIGHT camera's wagon_summary
            ocr_wagon_number = None
            is_manipulated = False
            if i < len(right_summary):
                ocr_obj = right_summary[i].get('ocr_wagon_number')
                is_manipulated = right_summary[i].get('is_manipulated', False)
                if ocr_obj is not None:
                    if hasattr(ocr_obj, 'is_valid') and ocr_obj.is_valid:
                        ocr_wagon_number = (f"{ocr_obj.wagon_type}-"
                                            f"{ocr_obj.owning_railway}-"
                                            f"{ocr_obj.wagon_number}-"
                                            f"{ocr_obj.check_digit}")
                        is_manipulated = getattr(ocr_obj, 'is_manipulated', False)
                    elif hasattr(ocr_obj, 'full_number'):
                        ocr_wagon_number = ocr_obj.full_number
                    elif isinstance(ocr_obj, str):
                        ocr_wagon_number = ocr_obj

            # Count open doors per camera
            if has_open_left:
                left_open_count += 1
            if has_open_right:
                right_open_count += 1
            if has_open_top:
                top_open_count += 1
            if has_open_left_top:
                left_top_open_count += 1

            # Check partial
            has_partial_left = any(
                'partial' in d.get('state', '').lower()
                for d in left_wagon_doors
            )
            has_partial_right = any(
                'partial' in d.get('state', '').lower()
                for d in right_wagon_doors
            )
            has_partial_top = any(
                'partial' in d.get('state', '').lower()
                for d in top_wagon_doors
            )
            has_partial_left_top = any(
                'partial' in d.get('state', '').lower()
                for d in left_top_wagon_doors
            )
            if has_partial_left:
                left_partial_count += 1
            if has_partial_right:
                right_partial_count += 1
            if has_partial_top:
                top_partial_count += 1
            if has_partial_left_top:
                left_top_partial_count += 1

            merged_wagons.append({
                'wagon_sr_no': wagon_num,
                'ocr_wagon_number': ocr_wagon_number,
                'is_manipulated': is_manipulated,
                'left_doors_text': left_text,
                'right_doors_text': right_text,
                'top_doors_text': top_text,
                'left_top_doors_text': left_top_text,
                'has_open_left': has_open_left,
                'has_open_right': has_open_right,
                'has_open_top': has_open_top,
                'has_open_left_top': has_open_left_top,
            })

        # Determine overall status
        total_open = (left_open_count + right_open_count + top_open_count +
                      left_top_open_count)
        status = "NOT OK" if total_open > 0 else "OK"

        # Determine rake type based on loaded vs empty wagon count
        if loaded_wagon_count > empty_wagon_count:
            rake_type = "LOADED RAKE"
        else:
            rake_type = "EMPTY RAKE"

        # Set wagon_type on every merged wagon based on rake type and damage
        # Only populate wagon_type when at least one top camera feed is available
        has_any_top_feed = ('RIGHT_UP_TOP' not in missing_cameras or
                           'LEFT_UP_TOP' not in missing_cameras)
        for w in merged_wagons:
            w['rake_type'] = rake_type
            if has_any_top_feed:
                if rake_type == "LOADED RAKE":
                    # Loaded rake: damaged wagon → EMPTY, otherwise → LOADED
                    has_damage = w.get('has_open_top') or w.get('has_open_left_top')
                    w['wagon_type'] = "EMPTY" if has_damage else "LOADED"
                else:
                    # Empty rake: always EMPTY
                    w['wagon_type'] = "EMPTY"
            else:
                # No top camera feed — cannot determine wagon type
                w['wagon_type'] = "-"

        # Extract loco numbers from RIGHT camera data (only RIGHT_UP runs OCR)
        # Supports multiple locomotives (e.g. dual-loco trains)
        loco_numbers = right_data.get('loco_numbers', []) if right_data else []
        if not loco_numbers:
            # Backward compatible: fall back to single loco_number
            single_loco = right_data.get('loco_number') if right_data else None
            if single_loco:
                loco_numbers = [single_loco]

        # Summary section
        elements.extend(self._create_summary_section(
            total_wagons=max_wagons,
            left_open=left_open_count,
            right_open=right_open_count,
            top_open=top_open_count,
            left_top_open=left_top_open_count,
            left_partial=left_partial_count,
            right_partial=right_partial_count,
            top_partial=top_partial_count,
            left_top_partial=left_top_partial_count,
            status=status,
            loco_numbers=loco_numbers,
            missing_cameras=missing_cameras,
            rake_type=rake_type
        ))

        # Unified wagon table
        if merged_wagons:
            elements.extend(self._create_unified_wagon_table(merged_wagons, missing_cameras=missing_cameras))

        # ──────────────────────────────────────────────────────────────
        # SAVE SIDE DAMAGE SNAPSHOTS TO TEMP FILES
        # ──────────────────────────────────────────────────────────────
        # Side damages from side_damage.pt have numpy array snapshots.
        # Save them to temp files so the image section can display them.
        import cv2 as _cv2
        import tempfile as _tempfile

        left_side_damages = kwargs.get('left_side_damages') or []
        right_side_damages = kwargs.get('right_side_damages') or []

        _side_snap_dir = _tempfile.mkdtemp(prefix='side_damage_snaps_')

        for idx, damage in enumerate(left_side_damages):
            snapshot = damage.get('snapshot')
            if snapshot is not None and hasattr(snapshot, 'shape'):
                try:
                    local_path = os.path.join(_side_snap_dir, f'left_side_dmg_{idx}.jpg')
                    _cv2.imwrite(local_path, snapshot, [_cv2.IMWRITE_JPEG_QUALITY, 85])
                    damage['_local_snapshot_path'] = local_path
                except Exception:
                    pass

        for idx, damage in enumerate(right_side_damages):
            snapshot = damage.get('snapshot')
            if snapshot is not None and hasattr(snapshot, 'shape'):
                try:
                    local_path = os.path.join(_side_snap_dir, f'right_side_dmg_{idx}.jpg')
                    _cv2.imwrite(local_path, snapshot, [_cv2.IMWRITE_JPEG_QUALITY, 85])
                    damage['_local_snapshot_path'] = local_path
                except Exception:
                    pass

        _side_dmg_count = sum(1 for d in left_side_damages if d.get('_local_snapshot_path')) + \
                          sum(1 for d in right_side_damages if d.get('_local_snapshot_path'))
        if _side_dmg_count:
            print(f"  Side damage snapshots saved: {_side_dmg_count}")

        # Open door images from all cameras (with Damaged Wagon Report table)
        elements.extend(self._create_open_door_images(
            left_doors, right_doors, top_doors, left_top_doors,
            merged_wagons=merged_wagons, max_wagons=max_wagons,
            left_side_damages=left_side_damages,
            right_side_damages=right_side_damages
        ))

        # Build PDF
        doc.build(elements, onFirstPage=self._add_page_logo,
                  onLaterPages=self._add_page_logo)

        print(f"✓ Combined report generated: {self.output_path}")
        print(f"  Total wagons: {max_wagons}")
        print(f"  LEFT open: {left_open_count}, RIGHT open: {right_open_count}, "
              f"R-TOP damages: {top_open_count}, L-TOP damages: {left_top_open_count}")
        print(f"  Status: {status}")

        return self.output_path, self.alignment_offsets
