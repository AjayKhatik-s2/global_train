"""
Hazaribagh Wagon Inspection PDF Report Generator

Generates a professional, audit-friendly PDF report with:
- Enlarged logo at top-left, centered title
- Table-format summary (Date-Time, Total Wagons, Open Door Wagons, Status, Video Link)
- 3-column table: SR.NO, WAGON NUMBER, DOORS
- Conditional highlighting: only OPEN doors highlighted in red
- All text in UPPERCASE for professional appearance
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from PIL import Image as PILImage

import cv2
import numpy as np

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# Light red color for highlighting open doors
LIGHT_RED = colors.Color(1.0, 0.8, 0.8)
# Light gray for table headers
HEADER_GRAY = colors.Color(0.85, 0.85, 0.85)
# Professional blue for header underline
HEADER_BLUE = colors.Color(0.2, 0.4, 0.6)


class HazaribagReportGenerator:
    """
    Generate professional wagon inspection PDF for Hazaribagh.
    
    Report structure:
    1. Header with enlarged logo and title
    2. Summary table (Date-Time, Total Wagons, Open Door Wagons, Status, Video Link)
    3. Wagon table with SR.NO, WAGON NUMBER, DOORS columns
    """
    
    def __init__(
        self,
        output_path: str,
        logo_path: str,
        source_video_url: str = None,
        video_file_path: str = None,
        main_report_url: str = None
    ):
        self.output_path = output_path
        self.logo_path = logo_path
        self.source_video_url = source_video_url or video_file_path
        self.main_report_url = main_report_url
        
        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _add_page_logo(self, canvas, doc):
        """Add logo to every page (called by ReportLab for each page)."""
        if self.logo_path and os.path.exists(self.logo_path):
            canvas.saveState()
            try:
                # Position: top-left corner
                # A4 portrait: width=595, height=842 points
                logo_width = 1.2 * inch
                logo_height = 0.6 * inch
                x_pos = 0.3 * inch
                y_pos = doc.pagesize[1] - logo_height - 0.2 * inch
                canvas.drawImage(
                    self.logo_path,
                    x_pos,
                    y_pos,
                    width=logo_width,
                    height=logo_height,
                    preserveAspectRatio=True,
                    mask='auto'
                )
            except Exception as e:
                print(f"Warning: Could not add logo to page: {e}")
            canvas.restoreState()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='TitleCenter',
            parent=self.styles['Title'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=10,
            fontName='Helvetica-Bold',
            textColor=HEADER_BLUE
        ))
        
        self.styles.add(ParagraphStyle(
            name='SummaryLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT
        ))
        
        self.styles.add(ParagraphStyle(
            name='SummaryValue',
            parent=self.styles['Normal'],
            fontSize=10,
            fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            name='StatusOk',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.green,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='StatusNotOk',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.red,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='TableCell',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Helvetica',
            alignment=TA_CENTER
        ))
    
    def _create_header(self) -> List:
        """Create header with logo, centered title, and video link above line."""
        elements = []
        
        # Logo - enlarged for better visibility (top-left)
        if os.path.exists(self.logo_path):
            logo = Image(self.logo_path, width=1.8*inch, height=1.0*inch)
        else:
            logo = Paragraph("", self.styles['Normal'])
        
        # Title on one line - larger font for landscape
        title = Paragraph(
            "WAGON AND DOOR INSPECTION REPORT - HAZARIBAGH",
            self.styles['TitleCenter']
        )
        
        # Empty cell for balance
        empty = Paragraph("", self.styles['Normal'])
        
        header_data = [[logo, title, empty]]
        
        # Landscape A4: ~11 inches wide, use 10 inches for table
        header_table = Table(header_data, colWidths=[2.0*inch, 6.5*inch, 1.5*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
            ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        
        elements.append(header_table)
        
        # Video link row - right aligned, just above the line
        if self.source_video_url:
            video_link_style = ParagraphStyle(
                'VideoLinkRight',
                parent=self.styles['Normal'],
                fontSize=10,
                fontName='Helvetica-Bold',
                alignment=2  # TA_RIGHT = 2
            )
            video_link = Paragraph(
                f'<b>VIDEO LINK:</b> <link href="{self.source_video_url}" color="blue"><u>CLICK TO VIEW</u></link>',
                video_link_style
            )
            elements.append(video_link)
            elements.append(Spacer(1, 5))
        
        # Main detailed report link - right aligned
        if self.main_report_url:
            report_link_style = ParagraphStyle(
                'ReportLinkRight',
                parent=self.styles['Normal'],
                fontSize=10,
                fontName='Helvetica-Bold',
                alignment=2  # TA_RIGHT = 2
            )
            report_link = Paragraph(
                f'<b>DETAILED REPORT:</b> <link href="{self.main_report_url}" color="blue"><u>CLICK TO VIEW</u></link>',
                report_link_style
            )
            elements.append(report_link)
            elements.append(Spacer(1, 5))
        
        # Add decorative line under header - full width for landscape
        line_table = Table([[""], [""]], colWidths=[10.5*inch])
        line_table.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, 0), 2, HEADER_BLUE),
        ]))
        elements.append(line_table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _create_summary_section(
        self,
        total_wagons: int,
        open_door_wagons: int,
        partial_closed_wagons: int,
        closed_door_wagons: int,
        status: str
    ) -> List:
        """Create professional table-format summary section."""
        elements = []
        
        # Use IST timezone (UTC+5:30)
        IST = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime.now(IST).strftime("%d-%m-%Y - %H:%M:%S")
        
        # Determine status style
        status_text = status.upper()
        if status.upper() == "OK":
            status_para = Paragraph(f"<b>{status_text}</b>", self.styles['StatusOk'])
        else:
            status_para = Paragraph(f"<b>{status_text}</b>", self.styles['StatusNotOk'])
        
        # Build summary table data with 6 columns
        summary_data = [
            [
                Paragraph("<b>DATE-TIME</b>", self.styles['SummaryLabel']),
                Paragraph("<b>TOTAL WAGONS</b>", self.styles['SummaryLabel']),
                Paragraph("<b>OPEN DOOR WAGONS</b>", self.styles['SummaryLabel']),
                Paragraph("<b>PARTIAL CLOSED DOORS</b>", self.styles['SummaryLabel']),
                Paragraph("<b>CLOSED DOORS</b>", self.styles['SummaryLabel']),
                Paragraph("<b>STATUS</b>", self.styles['SummaryLabel']),
            ],
            [
                Paragraph(timestamp, self.styles['SummaryValue']),
                Paragraph(str(total_wagons), self.styles['SummaryValue']),
                Paragraph(str(open_door_wagons), self.styles['SummaryValue']),
                Paragraph(str(partial_closed_wagons), self.styles['SummaryValue']),
                Paragraph(str(closed_door_wagons), self.styles['SummaryValue']),
                status_para,
            ]
        ]
        
        # Landscape: use adjusted column widths for 6 columns
        summary_table = Table(summary_data, colWidths=[2.0*inch, 1.5*inch, 1.8*inch, 2.0*inch, 1.8*inch, 1.4*inch])
        summary_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), HEADER_GRAY),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            # Padding
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(summary_table)
        
        elements.append(Spacer(1, 25))
        
        return elements
    
    def _format_door_status(self, state: str) -> str:
        """Format door state to uppercase readable format."""
        state_lower = state.lower()
        if 'partial' in state_lower:
            return "PARTIAL CLOSED"
        elif 'open' in state_lower:
            return "OPEN"
        else:
            return "CLOSED"
    
    def _create_wagon_table(
        self,
        wagon_data: List[Dict]
    ) -> List:
        """
        Create professional wagon inspection table.
        
        Format: SR.NO | WAGON NUMBER | DOORS
        - SR.NO shows sequential number (1, 2, 3...)
        - WAGON NUMBER shows OCR-detected wagon number below SR.NO
        - Doors column shows: DOOR 1 CLOSED / DOOR 2 OPEN etc.
        Only OPEN doors are highlighted in red.
        """
        elements = []
        
        # Table header - all UPPERCASE
        table_data = [
            [
                Paragraph("<b>SR.NO</b>", self.styles['TableHeader']),
                Paragraph("<b>WAGON NUMBER</b>", self.styles['TableHeader']),
                Paragraph("<b>DOORS</b>", self.styles['TableHeader'])
            ]
        ]
        
        # Track which rows need highlighting
        highlight_rows = []
        
        for idx, wagon in enumerate(wagon_data, start=1):
            wagon_sr_no = wagon['wagon_sr_no']  # Sequential number (1, 2, 3...)
            ocr_wagon_number = wagon.get('ocr_wagon_number', None)  # OCR-detected number
            door1_status = wagon.get('door1_status', None)
            door2_status = wagon.get('door2_status', None)
            has_open_door = wagon['has_open_door']
            
            # Build doors text
            if door1_status is None:
                doors_text = "NO DOOR DETECTED"
            elif door2_status:
                doors_text = f"DOOR 1 {door1_status} / DOOR 2 {door2_status}"
            else:
                doors_text = f"DOOR 1 {door1_status}"
            
            # SR.NO is sequential (1, 2, 3...)
            sr_no_text = str(wagon_sr_no)
            
            # WAGON NUMBER is OCR-detected number or '-' if not available
            if ocr_wagon_number and str(ocr_wagon_number).strip() != '-':
                wagon_num_display = str(ocr_wagon_number).upper()
            else:
                wagon_num_display = "-"
            
            table_data.append([
                Paragraph(f"<b>{sr_no_text}</b>", self.styles['TableCell']),
                Paragraph(f"<b>{wagon_num_display}</b>", self.styles['TableCell']),
                Paragraph(doors_text, self.styles['TableCell'])
            ])
            
            # Mark row for highlighting if has open door
            if has_open_door:
                highlight_rows.append(idx)  # idx is 1-based, table row is idx (0 is header)
        
        # Create table - wider for landscape format
        # Column widths: SR.NO (compact), WAGON NUMBER (wider for OCR numbers), DOORS (expanded for status text)
        table = Table(table_data, colWidths=[1.0*inch, 2.5*inch, 6.5*inch])
        
        # Base table style
        table_style = [
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), HEADER_GRAY),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            
            # All cells
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            
            # Padding
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
        ]
        
        # Add light red highlighting for rows with OPEN doors
        # Highlight full row (SR.NO, WAGON NUMBER, DOORS)
        for row_idx in highlight_rows:
            table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), LIGHT_RED))
        
        table.setStyle(TableStyle(table_style))
        
        elements.append(table)
        
        return elements
    
    def _create_open_door_images(
        self,
        doors: List[Dict]
    ) -> List:
        """
        Create section with snapshot images of wagons that have open doors.
        
        Only includes doors with state='open' (excluding partial_closed).
        Images are displayed in a grid layout with labels.
        """
        elements = []
        
        # Filter only doors with OPEN state (not partial) that have snapshots
        open_doors = [
            d for d in doors
            if d.get('snapshot') is not None
            and 'open' in d.get('state', '').lower()
            and 'partial' not in d.get('state', '').lower()
        ]
        
        if not open_doors:
            return elements
        
        # Section title
        elements.append(Spacer(1, 20))
        title_style = ParagraphStyle(
            'OpenDoorTitle',
            parent=self.styles['Title'],
            fontSize=14,
            fontName='Helvetica-Bold',
            textColor=colors.red,
            alignment=TA_CENTER,
            spaceAfter=15
        )
        elements.append(Paragraph(
            f"OPEN DOOR WAGON IMAGES ({len(open_doors)} DETECTED)",
            title_style
        ))
        
        # Create temp directory for snapshot images
        temp_dir = tempfile.mkdtemp(prefix='hazaribagh_snapshots_')
        self._temp_dirs = getattr(self, '_temp_dirs', [])
        self._temp_dirs.append(temp_dir)
        
        # Caption style
        caption_style = ParagraphStyle(
            'ImageCaption',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            textColor=colors.red,
            spaceBefore=5,
            spaceAfter=10
        )
        
        # Target image size in points for PDF (fits landscape A4)
        img_width = 4.5 * inch
        img_height = 3.0 * inch
        
        # Display images in pairs (2 per row) for landscape layout
        row_images = []
        for i, door in enumerate(open_doors):
            snapshot = door['snapshot']
            wagon_num = door.get('wagon_number', '?')
            door_num = door.get('door_number', i + 1)
            state = door.get('state', 'OPEN').upper()
            confidence = door.get('confidence', 0.0)
            
            # Save snapshot as temporary image file using PIL for proper DPI
            temp_path = os.path.join(temp_dir, f'open_door_{i}.png')
            try:
                if not isinstance(snapshot, np.ndarray):
                    continue
                
                # Resize to reasonable pixel dimensions
                max_w, max_h = 800, 600
                h, w = snapshot.shape[:2]
                if h > max_h or w > max_w:
                    scale = min(max_w / w, max_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    snapshot = cv2.resize(snapshot, (new_w, new_h),
                                          interpolation=cv2.INTER_AREA)
                
                # Convert BGR (OpenCV) to RGB (PIL) and save with explicit DPI
                rgb_snapshot = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb_snapshot)
                pil_img.save(temp_path, 'PNG', dpi=(72, 72))
                
            except Exception as e:
                print(f"Warning: Failed to save snapshot for door {door_num}: {e}")
                continue
            
            if not os.path.exists(temp_path):
                continue
            
            # Create image element with forced dimensions
            try:
                img = Image(temp_path, width=img_width, height=img_height)
                # Force the draw dimensions to prevent ReportLab
                # from using native pixel size
                img.drawWidth = img_width
                img.drawHeight = img_height
                img.hAlign = 'CENTER'
                
                # Sanity check: skip if ReportLab computed absurd size
                if getattr(img, 'imageHeight', 0) > 10000:
                    print(f"Warning: Skipping oversized image for door {door_num}")
                    continue
                    
            except Exception as e:
                print(f"Warning: Failed to load snapshot image: {e}")
                continue
            
            caption = Paragraph(
                f"WAGON {wagon_num} - DOOR {door_num} - {state} (CONF: {confidence:.2f})",
                caption_style
            )
            
            # Add image and caption separately (no KeepTogether to avoid layout issues)
            row_images.append([img, caption])
            
            # Add pair of images as a row
            if len(row_images) == 2:
                img_table_data = [[row_images[0][0], row_images[1][0]],
                                  [row_images[0][1], row_images[1][1]]]
                img_table = Table(
                    img_table_data,
                    colWidths=[5.0 * inch, 5.0 * inch],
                    rowHeights=[img_height + 5, 20]
                )
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
                    ('VALIGN', (0, 1), (-1, 1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ]))
                elements.append(img_table)
                elements.append(Spacer(1, 10))
                row_images = []
        
        # Handle remaining single image
        if row_images:
            empty_cell = Paragraph('', self.styles['Normal'])
            img_table_data = [[row_images[0][0], empty_cell],
                              [row_images[0][1], empty_cell]]
            img_table = Table(
                img_table_data,
                colWidths=[5.0 * inch, 5.0 * inch],
                rowHeights=[img_height + 5, 20]
            )
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
                ('VALIGN', (0, 1), (-1, 1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(img_table)
        
        return elements
    
    def generate(
        self,
        wagon_summary: List[Dict],
        doors: List[Dict],
        state_counts: Dict[str, int],
        wagon_numbers: Dict = None
    ) -> str:
        """
        Generate the Hazaribagh wagon inspection PDF report.
        
        Args:
            wagon_summary: List of wagon info from pipeline
            doors: List of door dictionaries from pipeline
            state_counts: Dict mapping state to count
            wagon_numbers: Optional dict of OCR-detected wagon numbers
            
        Returns:
            Path to generated PDF
        """
        # Create document in landscape (same as main report)
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=landscape(A4),
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        elements = []
        
        # Header
        elements.extend(self._create_header())
        
        # Calculate summary metrics
        total_wagons = len(wagon_summary) if wagon_summary else 0
        
        # Group doors by wagon
        doors_by_wagon = {}
        for door in doors:
            wagon_num = door.get('wagon_number', 1)
            if wagon_num not in doors_by_wagon:
                doors_by_wagon[wagon_num] = []
            doors_by_wagon[wagon_num].append(door)
        
        # Determine wagon counts by door state
        open_door_wagons = 0
        partial_closed_wagons = 0
        closed_door_wagons = 0
        
        for wagon_num, wagon_doors in doors_by_wagon.items():
            # Check for open doors (excluding partial)
            has_open = any('open' in d['state'].lower() and 'partial' not in d['state'].lower() for d in wagon_doors)
            # Check for partial closed doors
            has_partial = any('partial' in d['state'].lower() for d in wagon_doors)
            # Check for closed doors (all doors are closed, not open or partial)
            all_closed = all('closed' in d['state'].lower() and 'partial' not in d['state'].lower() for d in wagon_doors)
            
            if has_open:
                open_door_wagons += 1
            elif has_partial:
                partial_closed_wagons += 1
            elif all_closed and len(wagon_doors) > 0:
                closed_door_wagons += 1
        
        # Determine overall status
        has_any_open = any(
            'open' in d['state'].lower() and 'partial' not in d['state'].lower()
            for d in doors
        )
        status = "NOT OK" if has_any_open else "OK"
        
        # Summary section
        elements.extend(self._create_summary_section(
            total_wagons=total_wagons,
            open_door_wagons=open_door_wagons,
            partial_closed_wagons=partial_closed_wagons,
            closed_door_wagons=closed_door_wagons,
            status=status
        ))
        
        # Prepare wagon table data
        wagon_table_data = []
        
        # Sort wagons by wagon_number
        if wagon_summary:
            sorted_wagons = sorted(wagon_summary, key=lambda w: w.get('wagon_number', 0))
        else:
            # Fallback: create wagon entries from doors
            unique_wagons = sorted(set(d.get('wagon_number', 1) for d in doors))
            sorted_wagons = [{'wagon_number': wn} for wn in unique_wagons]
        
        for idx, wagon in enumerate(sorted_wagons, start=1):
            wagon_num = wagon.get('wagon_number', idx)
            wagon_doors = doors_by_wagon.get(wagon_num, [])
            
            # Check if wagon has any open door (excluding partial)
            has_open = any(
                'open' in d['state'].lower() and 'partial' not in d['state'].lower()
                for d in wagon_doors
            )
            
            # Handle case where no doors detected for this wagon
            if len(wagon_doors) == 0:
                door1_status = None  # Will show "NO DOOR DETECTED"
                door2_status = None
            else:
                # Determine door statuses
                door1_status = None
                door2_status = None
                
                # Sort doors by door_number (now per-wagon: 1, 2, etc.)
                wagon_doors_sorted = sorted(wagon_doors, key=lambda d: d.get('door_number', 0))
                
                for door in wagon_doors_sorted:
                    wagon_door_num = door.get('door_number', 1)
                    door_status = self._format_door_status(door['state'])
                    
                    if wagon_door_num == 1:
                        door1_status = door_status
                    elif wagon_door_num == 2:
                        door2_status = door_status
                
                # If only one door detected and no door1_status set, use first door
                if door1_status is None and len(wagon_doors_sorted) > 0:
                    door1_status = self._format_door_status(wagon_doors_sorted[0]['state'])
            
            # Get OCR wagon number from wagon_summary (same as report_generator.py)
            ocr_number = None
            ocr_wagon_num = wagon.get('ocr_wagon_number')  # WagonNumber object from wagon_timeline
            
            if ocr_wagon_num is not None and hasattr(ocr_wagon_num, 'is_valid') and ocr_wagon_num.is_valid:
                # Format as XX-XX-XX-XXXX-X (same format as main PDF)
                ocr_number = (f"{ocr_wagon_num.wagon_type}-"
                             f"{ocr_wagon_num.owning_railway}-"
                             f"{ocr_wagon_num.year_of_manufacture}-"
                             f"{ocr_wagon_num.individual_number}-"
                             f"{ocr_wagon_num.check_digit}")
            
            wagon_table_data.append({
                'wagon_sr_no': idx,  # Sequential number (1, 2, 3...)
                'ocr_wagon_number': ocr_number,  # OCR-detected wagon number
                'door1_status': door1_status,
                'door2_status': door2_status,
                'has_open_door': has_open
            })
        
        # Create wagon table
        if wagon_table_data:
            elements.extend(self._create_wagon_table(wagon_table_data))
        
        # Add open door images section (if any open doors with snapshots)
        elements.extend(self._create_open_door_images(doors))
        
        # Build PDF (logo is already in header via _create_header, no callback needed)
        doc.build(elements)
        
        # Cleanup temporary snapshot files
        for temp_dir in getattr(self, '_temp_dirs', []):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        
        print(f"✓ Hazaribagh report generated: {self.output_path}")
        
        return self.output_path

