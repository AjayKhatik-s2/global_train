import os
import tempfile
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
import numpy as np
import cv2
import requests

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# State/damage color mapping for PDF
STATE_COLORS_RGB = {
    'crack': colors.red,
    'dent': colors.orange,
    'broken': colors.red,
    'broken_component': colors.red,
    'corrosion': colors.Color(0.8, 0.6, 0),
    'damage': colors.red,
    'no_damage': colors.green,
    'closed': colors.green,
    'unknown': colors.grey,
}


class DamageReportGenerator:
    """
    Generate PDF report for top-view damage inspection results.
    
    Report structure:
    1. Summary page with video info and damage statistics
    2. One page per damage detection with annotated snapshot and details
    """
    
    # API Configuration
    UPLOAD_API_URL = "https://reports-api.suvidhaen.com/api/upload-pdf"
    EMAIL_API_URL = "https://railopsapi.suvidhaen.com/notification_microservice/send-email"
    PRODUCT_NAME = "CCTV-Top_Damage_Detection-Reports"
    EMAIL_RECEIVER = ["atul.nitt.cse@gmail.com"]
    EMAIL_RECEIVER_CC = [
        "ayushkum012@gmail.com",
        "rithish.sheru.s2@gmail.com",
        "kumarankitiitps2@gmail.com",
        "ajaykhatik6367s2@gmail.com",
        "admedhn@gmail.com",
        "sushantju63@gmail.com",
        "rajchaudhary01.official@gmail.com",
        "shyambabugupt.s2@gmail.com",
        "contact@suvidhaen.com"
    ]
    
    def __init__(
        self,
        output_path: str,
        video_path: str,
        temp_dir: str = None,
        source_video_url: str = None,
        region: str = "ap-south-1",
        logo_path: str = None
    ):
        self.output_path = output_path
        self.video_path = video_path
        # Portable temp dir (old code hardcoded /tmp which fails on Windows).
        # Staging location only; has no bearing on the rendered PDF.
        self.temp_dir = temp_dir or os.path.join(
            tempfile.gettempdir(), "damage_report_images")
        self.source_video_url = source_video_url
        self.region = region
        self.logo_path = logo_path

        # Create temp directory for snapshot images
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def get_s3_console_url(self, bucket: str, key: str) -> str:
        """Generate S3 console URL for a given bucket and key."""
        encoded_key = quote(key, safe='')
        return f"https://s3.console.aws.amazon.com/s3/object/{bucket}?region={self.region}&prefix={encoded_key}"
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='Title_Custom',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='Heading_Custom',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='StateOpen',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.red,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='StateClosed',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.green,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='StateOther',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.orange,
            fontName='Helvetica-Bold'
        ))
    
    def _add_logo_to_page(self, canvas, doc):
        """Add logo to top-left of every page."""
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                # Logo position: top-left corner
                logo_width = 1.2 * inch
                logo_height = 0.6 * inch
                x = 0.5 * inch
                y = doc.pagesize[1] - 0.8 * inch  # From top
                canvas.drawImage(
                    self.logo_path, x, y,
                    width=logo_width, height=logo_height,
                    preserveAspectRatio=True, mask='auto'
                )
            except Exception as e:
                print(f"⚠ Logo draw failed: {e}")
    
    def _save_snapshot_image(
        self, 
        frame: np.ndarray, 
        door_id: int,
        prefix: str = ""
    ) -> str:
        """Save snapshot to temp file and return path. Returns '' on failure."""
        try:
            if frame is None or not hasattr(frame, 'shape') or frame.size == 0:
                return ""
            
            tag = f"{prefix}_" if prefix else ""
            filename = f"damage_{tag}{door_id}_snapshot.jpg"
            filepath = os.path.join(self.temp_dir, filename)
            
            # Ensure temp dir exists (may be cleaned between calls)
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Write frame directly — snapshot is already in BGR (OpenCV native)
            success = cv2.imwrite(filepath, frame)
            
            if not success or not os.path.exists(filepath):
                print(f"⚠ Failed to write snapshot: {filepath}")
                return ""
            
            return filepath
        except Exception as e:
            print(f"⚠ Snapshot save failed for damage {door_id}: {e}")
            return ""
    
    def _get_state_style(self, state: str) -> str:
        """Get paragraph style name for damage state."""
        state_lower = state.lower()
        if state_lower in ('crack', 'dent', 'broken', 'broken_component', 'corrosion', 'damage'):
            return 'StateOpen'  # Red for damage
        elif state_lower in ('no_damage', 'closed'):
            return 'StateClosed'  # Green for no damage
        else:
            return 'StateOther'
    
    def _create_priority_alert_page(
        self,
        doors: List[dict],
        wagon_summary: List[dict],
        wagon_damage_map: Optional[Dict] = None,
        damage_frames: Optional[Dict] = None  # NEW: frames dict for damage snapshots
    ) -> List:
        """
        Create priority alert summary page showing critical findings.
        
        Shows OPEN DOOR wagon images first, then DAMAGE wagon images.
        Only created if open doors or damage exist.
        
        Args:
            doors: List of door dictionaries
            wagon_summary: List of wagon information
            wagon_damage_map: Optional damage map (wagon_number -> damages)
            
        Returns:
            List of reportlab elements for priority alert page
        """
        elements = []
        
        # Check if we have any alerts to show
        has_damage = wagon_damage_map and any(
            len(damages) > 0 for wagon_num, damages in wagon_damage_map.items() if wagon_num != 0
        )
        
        # Skip if no damage alerts
        if not has_damage:
            return []
        
        # Title
        elements.append(Paragraph(
            "🚨 PRIORITY ALERTS 🚨",
            ParagraphStyle('AlertTitle', parent=self.styles['Title'],
                         fontSize=28, alignment=TA_CENTER, 
                         textColor=colors.HexColor('#d9534f'),
                         spaceAfter=20)
        ))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # SECTION 2: DAMAGE DETECTED (second priority)
        if has_damage:
            elements.append(Paragraph(
                "🔧 DAMAGE DETECTED",
                ParagraphStyle('DamageSection', parent=self.styles['Heading1'],
                             fontSize=20, textColor=colors.HexColor('#d9534f'),
                             spaceAfter=12)
            ))
            
            # Find wagons with damage (exclude wagon 0 = unassociated)
            damage_wagon_nums = sorted([
                wagon_num for wagon_num in wagon_damage_map.keys() 
                if wagon_num != 0 and wagon_damage_map[wagon_num]
            ])
            
            # Calculate total damages
            total_damages = sum(len(wagon_damage_map[w]) for w in damage_wagon_nums)
            
            # Create summary text
            damage_summary = f"<b>Total Damages: {total_damages}</b>"
            elements.append(Paragraph(damage_summary, self.styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
            
            # Show damage images and detection summary for each wagon
            from damage_detector import DamageDetection
            
            for wagon_num in damage_wagon_nums:
                damages = wagon_damage_map[wagon_num]
                
                # Wagon header
                elements.append(Paragraph(
                    f"<b>Wagon {wagon_num}</b> - {len(damages)} damage(s) detected",
                    ParagraphStyle('WagonDamageHeader', parent=self.styles['Heading2'],
                                 fontSize=14, textColor=colors.HexColor('#d9534f'))
                ))
                elements.append(Spacer(1, 0.1*inch))
                
                # Show damage images (max 2 per wagon on priority alert page)
                image_row = []
                for idx, damage in enumerate(damages[:2], start=1):  # Show first 2 damages
                    if damage_frames and damage.frame_idx in damage_frames:
                        frame = damage_frames[damage.frame_idx]
                        
                        # Use FULL wagon frame with bounding box (not cropped)
                        full_frame = frame.copy()
                        x1, y1, x2, y2 = [int(v) for v in damage.bbox]
                        
                        # Draw thick red bounding box on full frame
                        cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        
                        # Add label above bounding box
                        label = f"{damage.class_name} ({damage.confidence:.0%})"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        label_y = max(y1 - 10, text_h + 10)
                        # Draw background for label
                        cv2.rectangle(full_frame, (x1, label_y - text_h - 10),
                                    (x1 + text_w + 10, label_y + 5), (0, 0, 255), -1)
                        cv2.putText(full_frame, label, (x1 + 5, label_y),
                                  font, font_scale, (255, 255, 255), thickness)
                        
                        # Save snapshot
                        snapshot_filename = f"priority_damage_w{wagon_num}_d{idx}.jpg"
                        snapshot_path = os.path.join(self.temp_dir, snapshot_filename)
                        cv2.imwrite(snapshot_path, full_frame)
                        
                        # Create image with caption
                        from io import BytesIO
                        with open(snapshot_path, 'rb') as img_file:
                            img_data = BytesIO(img_file.read())
                        img = Image(img_data, width=3.5*inch, height=2.5*inch)
                        caption = Paragraph(
                            f"<b>Damage #{idx}</b><br/>Frame: {damage.frame_idx}",
                            ParagraphStyle('ImgCaption', parent=self.styles['Normal'],
                                         fontSize=9, alignment=TA_CENTER)
                        )
                        image_row.append([img, caption])
                
                # Add images to page (2 per row)
                if image_row:
                    # Pad if needed
                    while len(image_row) < 2:
                        image_row.append([Paragraph("", self.styles['Normal'])])
                    
                    img_table = Table([image_row], colWidths=[4*inch, 4*inch])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 5),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                    ]))
                    elements.append(img_table)
                
                elements.append(Spacer(1, 0.2*inch))
        
        elements.append(PageBreak())
        
        return elements
    
    def _create_summary_page(
        self,
        total_doors: int,
        state_counts: Dict[str, int],
        processing_time: float,
        door_open_events: int = 0,
        total_wagons: int = 0,
        doors: Optional[List[dict]] = None,  # NEW: For priority alerts
        wagon_summary: Optional[List[dict]] = None,  # NEW: For priority alerts
        wagon_damage_map: Optional[Dict] = None,  # NEW: For damage summary
        damage_frames: Optional[Dict] = None  # NEW: For damage frames for priority alert images
    ) -> List:
        """Create combined summary page with damage detection information."""
        elements = []
        
        # Title
        elements.append(Paragraph(
            "Top-View Damage Inspection Report",
            self.styles['Title_Custom']
        ))
        
        # Timestamp
        IST = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime.now(IST).strftime("%d-%m-%Y %H:%M:%S")
        elements.append(Paragraph(
            f"Generated: {timestamp}",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 20))
        
        # Video info
        elements.append(Paragraph(
            "Video Information",
            self.styles['Heading_Custom']
        ))
        
        video_name = os.path.basename(self.video_path)
        # Convert processing time to minutes
        processing_minutes = processing_time / 60.0
        
        video_table_data = [
            ["Source Video:", video_name],
            ["Processing Time:", f"{processing_minutes:.2f} minutes"],
        ]
        
        video_table = Table(video_table_data, colWidths=[2*inch, 6*inch])
        
        video_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(video_table)
        elements.append(Spacer(1, 20))
        
        # Total Number of Wagons - Prominent display
        elements.append(Paragraph(
            f"<b>Total Number of Wagons: {total_wagons}</b>",
            ParagraphStyle('WagonCount', parent=self.styles['Heading1'],
                         fontSize=22, alignment=TA_CENTER, textColor=colors.black,
                         spaceAfter=20)
        ))
        elements.append(Spacer(1, 20))
        
        # ==================================================================
        # COMBINED DETECTION SUMMARY (DOOR + DAMAGE)
        # ==================================================================
        elements.append(Paragraph(
            "Detection Summary",
            self.styles['Heading_Custom']
        ))
        
        # State counts table
        stats_data = [["Metric", "Count"]]
        
        # Add total wagons
        stats_data.append(["TOTAL WAGONS", str(total_wagons)])
        
        # Add loaded/empty wagon counts from wagon_summary
        if wagon_summary:
            loaded_count = sum(1 for w in wagon_summary if w.get('is_loaded', False))
            empty_count = total_wagons - loaded_count
            stats_data.append(["LOADED WAGONS", str(loaded_count)])
            stats_data.append(["EMPTY WAGONS", str(empty_count)])
        
        # Add damage states (excluding non-relevant ones)
        excluded_states = {'other', 'unknown', 'others', 'open', 'closed', 'no_damage'}
        for state, count in sorted(state_counts.items()):
            if state.lower() not in excluded_states:
                stats_data.append([state.upper().replace('_', ' '), str(count)])
        
        # Add damage info to same table
        if wagon_damage_map is not None:
            # Count unique wagons with damage (excluding wagon 0 = unassociated)
            wagons_with_damage = sum(1 for wagon_num, damages in wagon_damage_map.items() 
                                    if wagon_num != 0 and len(damages) > 0)
            stats_data.append(["DAMAGED WAGONS", str(wagons_with_damage)])
            
            # Total individual damage detections
            total_damage_count = sum(len(damages) for wagon_num, damages in wagon_damage_map.items() 
                                    if wagon_num != 0)
            stats_data.append(["TOTAL DAMAGES", str(total_damage_count)])
        
        stats_table = Table(stats_data, colWidths=[3*inch, 1.5*inch])
        
        # Table styling with color coding
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]
        
        # Color code rows based on state
        for i, row in enumerate(stats_data[1:], start=1):
            state = row[0].lower()
            if 'loaded' in state:
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.Color(0.85, 0.92, 1.0)))
            elif 'open' in state:
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.Color(1, 0.8, 0.8)))
            elif 'closed' in state and 'partial' not in state:
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.Color(0.8, 1, 0.8)))
            elif 'damage' in state:
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.Color(1, 0.95, 0.95)))
        
        stats_table.setStyle(TableStyle(table_style))
        elements.append(stats_table)
        elements.append(Spacer(1, 15))
        
        # ==================================================================
        # PRIORITY ALERTS WITH IMAGES (NEW)
        # ==================================================================
        if doors is not None and wagon_summary is not None:
            # Check for damage
            has_damage = wagon_damage_map and any(
                len(damages) > 0 for wagon_num, damages in wagon_damage_map.items() if wagon_num != 0
            )
            
            if has_damage:
                elements.append(Paragraph(
                    "🚨 PRIORITY ALERTS",
                    ParagraphStyle('AlertHeader', parent=self.styles['Heading_Custom'],
                                 fontSize=18, textColor=colors.red)
                ))
                
                
                # DAMAGE DETECTED - Show centered with wagon numbers
                if has_damage:
                    elements.append(Spacer(1, 0.2*inch))
                    
                    elements.append(Paragraph(
                        "🔧 DAMAGE DETECTED",
                        ParagraphStyle('DamageHeader', parent=self.styles['Normal'],
                                     fontSize=16, textColor=colors.HexColor('#d9534f'),
                                     fontName='Helvetica-Bold', alignment=TA_CENTER)
                    ))
                    
                    # Find wagons with damage (exclude wagon 0 = unassociated)
                    damage_wagon_nums = sorted([
                        wagon_num for wagon_num in wagon_damage_map.keys() 
                        if wagon_num != 0 and wagon_damage_map[wagon_num]
                    ])
                    
                    # Total damages already shown in summary table, no need to repeat
                    elements.append(Spacer(1, 0.1*inch))
                    
                    # Show damage images for each wagon
                    from damage_detector import DamageDetection
                    
                    for wagon_num in damage_wagon_nums:
                        damages = wagon_damage_map[wagon_num]
                        
                        # Wagon number only - centered
                        elements.append(Paragraph(
                            f"<b>Wagon {wagon_num}</b>",
                            ParagraphStyle('WagonDamageHeader', parent=self.styles['Heading2'],
                                         fontSize=14, textColor=colors.HexColor('#d9534f'),
                                         alignment=TA_CENTER)
                        ))
                        elements.append(Spacer(1, 0.1*inch))
                        
                        # Show damage images (max 2 per wagon)
                        image_row = []
                        for idx, damage in enumerate(damages[:2], start=1):
                            if damage_frames and damage.frame_idx in damage_frames:
                                frame = damage_frames[damage.frame_idx]
                                
                                # Use FULL wagon frame with bounding box
                                full_frame = frame.copy()
                                x1, y1, x2, y2 = [int(v) for v in damage.bbox]
                                
                                # Draw thick red bounding box
                                cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                
                                # Add label above bounding box
                                label = f"{damage.class_name} ({damage.confidence:.0%})"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.8
                                thickness = 2
                                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                                
                                label_y = max(y1 - 10, text_h + 10)
                                cv2.rectangle(full_frame, (x1, label_y - text_h - 10),
                                            (x1 + text_w + 10, label_y + 5), (0, 0, 255), -1)
                                cv2.putText(full_frame, label, (x1 + 5, label_y),
                                          font, font_scale, (255, 255, 255), thickness)
                                
                                # Save snapshot
                                snapshot_filename = f"summary_damage_w{wagon_num}_d{idx}.jpg"
                                snapshot_path = os.path.join(self.temp_dir, snapshot_filename)
                                cv2.imwrite(snapshot_path, full_frame)
                                
                                # Create image with caption
                                from io import BytesIO
                                with open(snapshot_path, 'rb') as img_file:
                                    img_data = BytesIO(img_file.read())
                                img = Image(img_data, width=3.5*inch, height=2.5*inch)
                                caption = Paragraph(
                                    f"<b>Damage #{idx}</b><br/>Frame: {damage.frame_idx}",
                                    ParagraphStyle('ImgCaption', parent=self.styles['Normal'],
                                                 fontSize=9, alignment=TA_CENTER)
                                )
                                image_row.append([img, caption])
                        
                        # Add images to page (2 per row)
                        if image_row:
                            # Pad if needed
                            while len(image_row) < 2:
                                image_row.append([Paragraph("", self.styles['Normal'])])
                            
                            img_table = Table([image_row], colWidths=[4*inch, 4*inch])
                            img_table.setStyle(TableStyle([
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ]))
                            elements.append(img_table)
                        
                        elements.append(Spacer(1, 0.2*inch))
        
        elements.append(PageBreak())
        return elements
    
    def _create_damage_page(
        self,
        door_number: int,
        door_id: int,
        state: str,
        confidence: float,
        snapshot_path: str,
        wagon_number: Optional[int] = None,
        ocr_wagon_id: Optional[str] = None  # NEW: OCR-detected wagon number
    ) -> List:
        """Create decorative page for a single damage detection - all on one page."""
        page_elements = []  # Elements for this door's page
        
        # Get state color for styling
        state_lower = state.lower()
        if state_lower in ('crack', 'dent', 'broken', 'broken_component', 'corrosion', 'damage'):
            state_color = colors.Color(1, 0.2, 0.2)  # Red
            bg_color = colors.Color(1, 0.9, 0.9)  # Light red
        elif state_lower in ('no_damage', 'closed'):
            state_color = colors.Color(0, 0.6, 0)  # Green
            bg_color = colors.Color(0.9, 1, 0.9)  # Light green
        else:
            state_color = colors.Color(0.9, 0.6, 0)  # Orange
            bg_color = colors.Color(1, 0.95, 0.85)  # Light orange
        
        state_display = state.upper().replace('_', ' ')
        
        # Build header text with wagon number and OCR ID if available
        if wagon_number is not None:
            header_text = f"<b>Wagon No: {wagon_number} | Damage #{door_number}</b>"
            
            # Add OCR wagon number if available
            if ocr_wagon_id is not None:
                header_text += f"<br/><font size=14 color='green'>Wagon ID: {ocr_wagon_id}</font>"
        else:
            header_text = f"<b>Damage #{door_number}</b>"
        
        # Create header table with door info
        header_data = [
            [Paragraph(header_text, 
                      ParagraphStyle('DoorHeader', parent=self.styles['Heading1'], 
                                   fontSize=20, alignment=TA_CENTER, textColor=colors.black))],
        ]
        
        header_table = Table(header_data, colWidths=[10*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        page_elements.append(header_table)
        page_elements.append(Spacer(1, 8))
        
        # Create info table with state and confidence
        info_data = [
            [Paragraph("<b>Final State</b>", self.styles['Normal']),
             Paragraph(f"<b>{state_display}</b>", 
                      ParagraphStyle('StateValue', parent=self.styles['Normal'],
                                   fontSize=12, textColor=state_color, fontName='Helvetica-Bold'))],
            [Paragraph("<b>Confidence</b>", self.styles['Normal']),
             Paragraph(f"<b>{confidence:.1%}</b>", 
                      ParagraphStyle('ConfValue', parent=self.styles['Normal'],
                                   fontSize=12, fontName='Helvetica-Bold'))],
        ]
        
        info_table = Table(info_data, colWidths=[1.5*inch, 2.5*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 2, state_color),
            ('LINEBELOW', (0, 0), (-1, 0), 1, state_color),
        ]))
        
        # Center the info table
        info_wrapper = Table([[info_table]], colWidths=[10*inch])
        info_wrapper.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        page_elements.append(info_wrapper)
        page_elements.append(Spacer(1, 10))
        
        # Snapshot image with border
        # Read image into memory buffer so it survives temp file cleanup
        # (ReportLab Image stores only the path; if the file is deleted before
        #  doc.build() renders it, you get FileNotFoundError)
        if snapshot_path and os.path.exists(snapshot_path):
            try:
                from io import BytesIO
                with open(snapshot_path, 'rb') as img_file:
                    img_data = BytesIO(img_file.read())
                img = Image(img_data, width=9*inch, height=4.5*inch)
                img.hAlign = 'CENTER'
                
                # Wrap image in a table for border effect
                img_table = Table([[img]], colWidths=[9.2*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOX', (0, 0), (-1, -1), 2, colors.grey),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 3),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ]))
                
                # Center the image table
                img_wrapper = Table([[img_table]], colWidths=[10*inch])
                img_wrapper.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                page_elements.append(img_wrapper)
            except Exception as e:
                print(f"⚠ Failed to embed snapshot image: {e}")
                page_elements.append(Paragraph(
                    "[Snapshot image could not be loaded]",
                    ParagraphStyle('NoImage', parent=self.styles['Normal'], 
                                 alignment=TA_CENTER, textColor=colors.grey)
                ))
        else:
            page_elements.append(Paragraph(
                "[Snapshot not available]",
                ParagraphStyle('NoImage', parent=self.styles['Normal'], 
                             alignment=TA_CENTER, textColor=colors.grey)
            ))
        
        # Wrap everything in KeepTogether to ensure all on same page
        return [KeepTogether(page_elements), PageBreak()]
    
    def _create_no_damage_wagon_page(
        self,
        wagon_number: int,
        wagon_start_frame: int,
        wagon_end_frame: int
    ) -> List:
        """
        Create a page for a wagon with no detected damage.
        
        Extracts a representative snapshot at the wagon's temporal midpoint.
        
        Args:
            wagon_number: Wagon number (1-indexed)
            wagon_start_frame: First frame of wagon
            wagon_end_frame: Last frame of wagon
            
        Returns:
            List of reportlab elements
        """
        page_elements = []
        
        # Gray color for "no door" pages
        bg_color = colors.Color(0.9, 0.9, 0.9)  # Light gray
        text_color = colors.Color(0.4, 0.4, 0.4)  # Dark gray
        
        # Header with wagon number
        header_text = f"<b>Wagon No: {wagon_number}</b>"
        header_data = [
            [Paragraph(header_text, 
                      ParagraphStyle('WagonHeader', parent=self.styles['Heading1'], 
                                   fontSize=24, alignment=TA_CENTER, textColor=colors.black))],
        ]
        
        header_table = Table(header_data, colWidths=[10*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        page_elements.append(header_table)
        page_elements.append(Spacer(1, 8))
        
        # "NO DAMAGE DETECTED" subtitle
        no_damage_style = ParagraphStyle(
            'NoDamage',
            parent=self.styles['Heading2'],
            fontSize=18,
            alignment=TA_CENTER,
            textColor=text_color,
            spaceAfter=5
        )
        page_elements.append(Paragraph("NO DAMAGE DETECTED", no_damage_style))
        
        # Frame range info
        mid_frame = (wagon_start_frame + wagon_end_frame) // 2
        frame_info = f"Frames: {wagon_start_frame} - {wagon_end_frame} | Snapshot at frame: {mid_frame}"
        page_elements.append(Paragraph(
            frame_info,
            ParagraphStyle('FrameInfo', parent=self.styles['Normal'],
                         fontSize=10, alignment=TA_CENTER, textColor=text_color)
        ))
        page_elements.append(Spacer(1, 10))
        
        # Extract wagon snapshot at midpoint frame
        wagon_snapshot_path = self._extract_wagon_snapshot(wagon_number, mid_frame)
        
        if wagon_snapshot_path and os.path.exists(wagon_snapshot_path):
            # Display wagon snapshot image
            try:
                from io import BytesIO
                with open(wagon_snapshot_path, 'rb') as img_file:
                    img_data = BytesIO(img_file.read())
                img = Image(img_data, width=9*inch, height=4.5*inch)
                img.hAlign = 'CENTER'
                
                # Create bordered image container
                img_table = Table([[img]], colWidths=[9.2*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOX', (0, 0), (-1, -1), 2, colors.gray),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                page_elements.append(img_table)
            except Exception as e:
                print(f"Error adding wagon snapshot image: {e}")
                page_elements.append(self._create_no_snapshot_placeholder())
        else:
            # No snapshot available - show placeholder
            page_elements.append(self._create_no_snapshot_placeholder())
        
        return [KeepTogether(page_elements), PageBreak()]
    
    def _create_loaded_wagon_page(
        self,
        wagon_number: int,
        wagon_start_frame: int,
        wagon_end_frame: int
    ) -> List:
        """
        Create a page for a wagon classified as loaded.
        
        Shows a representative snapshot with 'LOADED' overlay.
        
        Args:
            wagon_number: Wagon number (1-indexed)
            wagon_start_frame: First frame of wagon
            wagon_end_frame: Last frame of wagon
            
        Returns:
            List of reportlab elements
        """
        page_elements = []
        
        # Blue color scheme for loaded wagons
        bg_color = colors.Color(0.85, 0.92, 1.0)  # Light blue
        text_color = colors.Color(0.2, 0.4, 0.7)  # Dark blue
        
        # Header with wagon number
        header_text = f"<b>Wagon No: {wagon_number}</b>"
        header_data = [
            [Paragraph(header_text, 
                      ParagraphStyle('WagonHeader', parent=self.styles['Heading1'], 
                                   fontSize=24, alignment=TA_CENTER, textColor=colors.black))],
        ]
        
        header_table = Table(header_data, colWidths=[10*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        page_elements.append(header_table)
        page_elements.append(Spacer(1, 8))
        
        # "LOADED" status subtitle
        loaded_style = ParagraphStyle(
            'LoadedStatus',
            parent=self.styles['Heading2'],
            fontSize=18,
            alignment=TA_CENTER,
            textColor=text_color,
            spaceAfter=5
        )
        page_elements.append(Paragraph("LOADED \u2013 FLOOR NOT VISIBLE", loaded_style))
        
        # Frame range info
        mid_frame = (wagon_start_frame + wagon_end_frame) // 2
        frame_info = f"Frames: {wagon_start_frame} - {wagon_end_frame} | Snapshot at frame: {mid_frame}"
        page_elements.append(Paragraph(
            frame_info,
            ParagraphStyle('FrameInfo', parent=self.styles['Normal'],
                         fontSize=10, alignment=TA_CENTER, textColor=text_color)
        ))
        page_elements.append(Spacer(1, 10))
        
        # Extract wagon snapshot at midpoint frame
        wagon_snapshot_path = self._extract_loaded_wagon_snapshot(wagon_number, mid_frame)
        
        if wagon_snapshot_path and os.path.exists(wagon_snapshot_path):
            try:
                from io import BytesIO
                with open(wagon_snapshot_path, 'rb') as img_file:
                    img_data = BytesIO(img_file.read())
                img = Image(img_data, width=9*inch, height=4.5*inch)
                img.hAlign = 'CENTER'
                
                img_table = Table([[img]], colWidths=[9.2*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOX', (0, 0), (-1, -1), 2, colors.Color(0.2, 0.4, 0.7)),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                page_elements.append(img_table)
            except Exception as e:
                print(f"Error adding loaded wagon snapshot image: {e}")
                page_elements.append(self._create_no_snapshot_placeholder())
        else:
            page_elements.append(self._create_no_snapshot_placeholder())
        
        return [KeepTogether(page_elements), PageBreak()]
    
    def _extract_loaded_wagon_snapshot(self, wagon_number: int, frame_idx: int) -> Optional[str]:
        """
        Extract a frame from the video with a 'Loaded' overlay label.
        
        Args:
            wagon_number: Wagon number for filename
            frame_idx: Frame index to extract
            
        Returns:
            Path to saved snapshot image, or None
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return None
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return None
            
            # Add loaded label overlay
            label = f"Wagon {wagon_number} - Loaded"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw semi-transparent blue background for label
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (text_width + 30, text_height + 30), (180, 100, 30), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw text
            cv2.putText(frame, label, (20, text_height + 20), font, font_scale, (255, 255, 255), thickness)
            
            # Save to temp file
            filename = f"wagon_{wagon_number}_loaded_snapshot.jpg"
            filepath = os.path.join(self.temp_dir, filename)
            cv2.imwrite(filepath, frame)
            
            return filepath
            
        except Exception as e:
            print(f"Error extracting loaded wagon snapshot: {e}")
            return None
    
    def _extract_wagon_snapshot(self, wagon_number: int, frame_idx: int) -> Optional[str]:
        """
        Extract a frame from the video at the specified index.
        
        Args:
            wagon_number: Wagon number for filename
            frame_idx: Frame index to extract
            
        Returns:
            Path to saved snapshot image, or None if extraction failed
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Could not open video: {self.video_path}")
                return None
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print(f"Could not read frame {frame_idx}")
                return None
            
            # Add wagon label overlay
            label = f"Wagon {wagon_number} - No Damage Detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw semi-transparent background for label
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (text_width + 30, text_height + 30), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw text
            cv2.putText(frame, label, (20, text_height + 20), font, font_scale, (255, 255, 255), thickness)
            
            # Save to temp file
            filename = f"wagon_{wagon_number}_snapshot.jpg"
            filepath = os.path.join(self.temp_dir, filename)
            cv2.imwrite(filepath, frame)
            
            return filepath
            
        except Exception as e:
            print(f"Error extracting wagon snapshot: {e}")
            return None
    
    def _create_no_snapshot_placeholder(self) -> Table:
        """Create a placeholder for when snapshot is not available."""
        text_color = colors.Color(0.4, 0.4, 0.4)
        bg_color = colors.Color(0.95, 0.95, 0.95)
        
        placeholder_data = [
            [Paragraph("Wagon Snapshot Not Available",
                      ParagraphStyle('Placeholder', parent=self.styles['Normal'],
                                   fontSize=16, alignment=TA_CENTER, textColor=text_color))]
        ]
        
        placeholder_table = Table(placeholder_data, colWidths=[9*inch], rowHeights=[4*inch])
        placeholder_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 2, colors.gray),
        ]))
        
        return placeholder_table
    
    def _create_non_wagon_page(
        self,
        segment_type: str,
        wagon_start_frame: int,
        wagon_end_frame: int
    ) -> List:
        """
        Create a page for a non-wagon segment (engine or breakvan).
        Shows a snapshot with the segment type label. NOT counted as wagon.
        """
        page_elements = []
        
        if segment_type == 'engine':
            text_color = colors.Color(0.4, 0.2, 0.6)
            border_color = colors.Color(0.5, 0.3, 0.7)
            label = "ENGINE"
            overlay_label_text = "Engine"
            overlay_bg = (120, 50, 160)
        else:
            text_color = colors.Color(0.2, 0.45, 0.45)
            border_color = colors.Color(0.3, 0.55, 0.55)
            label = "BREAKVAN"
            overlay_label_text = "Breakvan"
            overlay_bg = (115, 115, 50)
        
        header_text = f"<b>{label}</b>"
        header_data = [
            [Paragraph(header_text, 
                      ParagraphStyle('NonWagonHeader', parent=self.styles['Heading1'], 
                                   fontSize=24, alignment=TA_CENTER, textColor=text_color))],
        ]
        header_table = Table(header_data, colWidths=[10*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        page_elements.append(header_table)
        page_elements.append(Spacer(1, 8))
        
        subtitle_style = ParagraphStyle(
            'NonWagonSubtitle', parent=self.styles['Heading2'],
            fontSize=16, alignment=TA_CENTER, textColor=text_color, spaceAfter=5
        )
        page_elements.append(Paragraph("NOT COUNTED AS WAGON", subtitle_style))
        
        mid_frame = (wagon_start_frame + wagon_end_frame) // 2
        frame_info = f"Frames: {wagon_start_frame} - {wagon_end_frame} | Snapshot at frame: {mid_frame}"
        page_elements.append(Paragraph(
            frame_info,
            ParagraphStyle('FrameInfoNW', parent=self.styles['Normal'],
                         fontSize=10, alignment=TA_CENTER, textColor=colors.Color(0.5, 0.5, 0.5))
        ))
        page_elements.append(Spacer(1, 10))
        
        snapshot_path = self._extract_non_wagon_snapshot(
            segment_type, mid_frame, overlay_label_text, overlay_bg
        )
        
        if snapshot_path and os.path.exists(snapshot_path):
            try:
                from io import BytesIO
                with open(snapshot_path, 'rb') as img_file:
                    img_data = BytesIO(img_file.read())
                img = Image(img_data, width=9*inch, height=4.5*inch)
                img.hAlign = 'CENTER'
                img_table = Table([[img]], colWidths=[9.2*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOX', (0, 0), (-1, -1), 2, border_color),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                page_elements.append(img_table)
            except Exception as e:
                print(f"Error adding {segment_type} snapshot: {e}")
                page_elements.append(self._create_no_snapshot_placeholder())
        else:
            page_elements.append(self._create_no_snapshot_placeholder())
        
        return [KeepTogether(page_elements), PageBreak()]
    
    def _extract_non_wagon_snapshot(self, segment_type, frame_idx, overlay_text, overlay_bg_color):
        """Extract a frame from video with a segment type overlay label."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            label = f"{overlay_text} (Frame {frame_idx})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (text_width + 30, text_height + 30), overlay_bg_color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, label, (20, text_height + 20), font, font_scale, (255, 255, 255), thickness)
            filename = f"{segment_type}_snapshot_{frame_idx}.jpg"
            filepath = os.path.join(self.temp_dir, filename)
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"Error extracting {segment_type} snapshot: {e}")
            return None
    
    def generate_report(
        self,
        damages: List[dict],
        state_counts: Dict[str, int],
        processing_time: float,
        damage_events: int = 0,
        wagon_summary: List[dict] = None,
        damage_elements: List = None,
        wagon_damage_map: Optional[Dict] = None,
        damage_frames: Optional[Dict] = None,
        full_wagon_summary: Optional[List[dict]] = None
    ) -> str:
        """
        Generate complete PDF report.
        
        Args:
            doors: List of door dictionaries with keys:
                - door_id: int
                - door_number: int (sequential)
                - state: str
                - confidence: float
                - snapshot: np.ndarray (annotated frame)
                - wagon_number: int (optional)
            state_counts: Dict mapping state to count
            processing_time: Total processing time in seconds
            door_open_events: Number of door open events detected
            wagon_summary: Optional list of wagon info dicts with keys:
                - wagon_number: int
                - start_frame: int
                - end_frame: int
                - door_count: int
                - door_ids: List[int]
            damage_elements: Optional list of reportlab elements for damage section
            wagon_damage_map: Optional damage map for priority alert page
            
        Returns:
            Path to generated PDF
        """
        # Create document in landscape
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=landscape(A4),
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        elements = []
        
        # Combined Summary page (includes door stats, damage stats, and priority alerts with images)
        total_wagons = len(wagon_summary) if wagon_summary else 0
        elements.extend(self._create_summary_page(
            total_doors=len(damages),
            state_counts=state_counts,
            processing_time=processing_time,
            door_open_events=damage_events,
            total_wagons=total_wagons,
            doors=damages,
            wagon_summary=wagon_summary,
            wagon_damage_map=wagon_damage_map,
            damage_frames=damage_frames
        ))
        
        # Priority Alerts are now embedded in summary page, no separate page needed
        
        # Build damage lookup by wagon_number
        damages_by_wagon = {}
        for damage in damages:
            wagon_num = damage.get('wagon_number', 1)
            if wagon_num not in damages_by_wagon:
                damages_by_wagon[wagon_num] = []
            damages_by_wagon[wagon_num].append(damage)
        
        # Build OCR wagon ID lookup from wagon_summary
        wagon_ocr_ids = {}  # wagon_number -> formatted OCR ID string
        if wagon_summary:
            for wagon in wagon_summary:
                wagon_num = wagon['wagon_number']
                ocr_wagon_num = wagon.get('ocr_wagon_number')  # WagonNumber object or None
                
                if ocr_wagon_num and ocr_wagon_num.is_valid:
                    # Format as XX-XX-XX-XXXX-X
                    formatted_id = (f"{ocr_wagon_num.wagon_type}-"
                                   f"{ocr_wagon_num.owning_railway}-"
                                   f"{ocr_wagon_num.year_of_manufacture}-"
                                   f"{ocr_wagon_num.individual_number}-"
                                   f"{ocr_wagon_num.check_digit}")
                    wagon_ocr_ids[wagon_num] = formatted_id
        
        # WAGON-DRIVEN GENERATION
        if wagon_summary:
            # Find wagons with damage for highlight section
            wagons_with_damage = []
            for wagon in wagon_summary:
                wagon_num = wagon['wagon_number']
                wagon_damages = damages_by_wagon.get(wagon_num, [])
                if wagon_damages:
                    wagons_with_damage.append(wagon)
            
            # SECTION 1: Damaged Wagons (Highlights) - show first for quick review
            if wagons_with_damage:
                
                for wagon in sorted(wagons_with_damage, key=lambda w: w['wagon_number']):
                    wagon_num = wagon['wagon_number']
                    wagon_damages = damages_by_wagon.get(wagon_num, [])
                    
                    for damage in sorted(wagon_damages, key=lambda d: d.get('damage_number', 0)):
                        if damage.get('snapshot') is not None:
                            snapshot_path = self._save_snapshot_image(
                                damage['snapshot'],
                                damage.get('damage_id', 0),
                                prefix="highlight"
                            )
                        else:
                            snapshot_path = ""
                        
                        elements.extend(self._create_damage_page(
                            door_number=damage.get('damage_number', 0),
                            door_id=damage.get('damage_id', 0),
                            state=damage.get('state', 'unknown'),
                            confidence=damage.get('confidence', 0.0),
                            snapshot_path=snapshot_path,
                            wagon_number=damage.get('wagon_number'),
                            ocr_wagon_id=wagon_ocr_ids.get(wagon_num)
                        ))

            
            # SECTION 2: All wagons in serial order (1, 2, 3, ...)
            # Use full_wagon_summary if available (includes engine/breakvan segments)
            serial_summary = full_wagon_summary if full_wagon_summary else wagon_summary
            for wagon in sorted(serial_summary, key=lambda w: w['start_frame']):
                # Check if this is a non-wagon segment (engine/breakvan)
                if wagon.get('is_non_wagon'):
                    seg_type = wagon.get('segment_type', 'engine')
                    elements.extend(self._create_non_wagon_page(
                        segment_type=seg_type,
                        wagon_start_frame=wagon.get('start_frame', 0),
                        wagon_end_frame=wagon.get('end_frame', 0)
                    ))
                    continue
                
                wagon_num = wagon['wagon_number']
                wagon_damages = damages_by_wagon.get(wagon_num, [])
                
                if wagon_damages:
                    wagon_damages_sorted = sorted(
                        wagon_damages,
                        key=lambda d: d.get('damage_number', 0)
                    )
                    
                    for damage in wagon_damages_sorted:
                        if damage.get('snapshot') is not None:
                            snapshot_path = self._save_snapshot_image(
                                damage['snapshot'],
                                damage.get('damage_id', 0),
                                prefix="detail"
                            )
                        else:
                            snapshot_path = ""
                        
                        elements.extend(self._create_damage_page(
                            door_number=damage.get('damage_number', 0),
                            door_id=damage.get('damage_id', 0),
                            state=damage.get('state', 'unknown'),
                            confidence=damage.get('confidence', 0.0),
                            snapshot_path=snapshot_path,
                            wagon_number=damage.get('wagon_number'),
                            ocr_wagon_id=wagon_ocr_ids.get(wagon_num)
                        ))
                else:
                    # Check if wagon is loaded
                    if wagon.get('is_loaded', False):
                        # Loaded wagon — show "LOADED" page
                        elements.extend(self._create_loaded_wagon_page(
                            wagon_number=wagon_num,
                            wagon_start_frame=wagon.get('start_frame', 0),
                            wagon_end_frame=wagon.get('end_frame', 0)
                        ))
                    else:
                        # No damage in this empty wagon - add "NO DAMAGE" page
                        elements.extend(self._create_no_damage_wagon_page(
                            wagon_number=wagon_num,
                            wagon_start_frame=wagon.get('start_frame', 0),
                            wagon_end_frame=wagon.get('end_frame', 0)
                        ))
        else:
            # Fallback to damage-driven (legacy behavior)
            sorted_damages = sorted(
                damages,
                key=lambda d: d.get('damage_number', 0)
            )
            
            for damage in sorted_damages:
                if damage.get('snapshot') is not None:
                    snapshot_path = self._save_snapshot_image(
                        damage['snapshot'],
                        damage.get('damage_id', 0),
                        prefix="legacy"
                    )
                else:
                    snapshot_path = ""
                
                elements.extend(self._create_damage_page(
                    door_number=damage.get('damage_number', 0),
                    door_id=damage.get('damage_id', 0),
                    state=damage.get('state', 'unknown'),
                    confidence=damage.get('confidence', 0.0),
                    snapshot_path=snapshot_path,
                    wagon_number=damage.get('wagon_number')
                ))
        
        # NEW: Append damage section if provided
        if damage_elements:
            print(f"  Adding {len(damage_elements)} damage report elements to PDF")
            elements.extend(damage_elements)
        
        # Build PDF with logo on all pages
        doc.build(
            elements,
            onFirstPage=self._add_logo_to_page,
            onLaterPages=self._add_logo_to_page
        )
        
        return self.output_path
    
    def upload_and_send_email(self, pdf_path: str) -> dict:
        
        IST = timezone(timedelta(hours=5, minutes=30))
        today_str = datetime.now(IST).strftime("%d-%m-%Y")
        
        # ================= UPLOAD GENERATED PDF =================
        print("\n" + "="*60)
        print("Uploading to upload-pdf microservice...")
        print("="*60)
        
        try:
            with open(pdf_path, "rb") as f:
                files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
                data = {
                    "product_name": self.PRODUCT_NAME,
                    "folder_name": today_str
                }
                
                upload_response = requests.post(
                    self.UPLOAD_API_URL, 
                    data=data, 
                    files=files,
                    timeout=60  # 60 second timeout
                )
            
            if upload_response.status_code == 200:
                report_url = upload_response.json().get("url")
                print(f"✔ Upload success: {report_url}")
            else:
                error_msg = f"Upload failed with status {upload_response.status_code}: {upload_response.text}"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during upload: {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Upload error: {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        
        # ================= SEND EMAIL NOTIFICATION =================
        print("\n" + "="*60)
        print("Sending email notification...")
        print("="*60)
        
        email_data = {
            "to": self.EMAIL_RECEIVER,
            "cc": self.EMAIL_RECEIVER_CC,
            "context": {
                "report_date": today_str,
                "report_url": report_url,
                "generated_by": "Automated CCTV Analytics System",
                "status": "Completed"
            },
            "attachment_url": report_url,
            "mail_from_name": "WagonEye Report V1",
            "template_name": "rake_inspection_report_v1.txt"
        }
        
        max_retries = 3
        retry_delay = 15  # Base delay in seconds, doubles each retry
        current_delay = retry_delay
        
        for attempt in range(1, max_retries + 1):
            try:
                email_response = requests.post(
                    self.EMAIL_API_URL, 
                    json=email_data,
                    timeout=60
                )
                
                # Retry on server errors (5xx — e.g. 504 Gateway Timeout)
                if email_response.status_code >= 500:
                    print(f"⚠ Email attempt {attempt}/{max_retries} got server error (HTTP {email_response.status_code})")
                    if attempt < max_retries:
                        print(f"  Retrying in {current_delay}s...")
                        import time as _time
                        _time.sleep(current_delay)
                        current_delay *= 2
                        continue
                    else:
                        print(f"❌ Email failed after {max_retries} attempts (last status: {email_response.status_code})")
                        return {
                            "success": True, 
                            "report_url": report_url, 
                            "email_warning": f"Server error {email_response.status_code} after {max_retries} attempts",
                            "email_status": "failed"
                        }
                
                print(f"Email Response Status: {email_response.status_code}")
                
                if email_response.status_code == 200:
                    print("✔ Email sent successfully")
                    return {
                        "success": True, 
                        "report_url": report_url,
                        "email_status": "sent"
                    }
                else:
                    print(f"⚠ Email may have failed with status {email_response.status_code}")
                    return {
                        "success": True, 
                        "report_url": report_url, 
                        "email_warning": email_response.text,
                        "email_status": "failed"
                    }
                    
            except requests.exceptions.Timeout:
                print(f"⚠ Email attempt {attempt}/{max_retries} timed out")
            except requests.exceptions.ConnectionError:
                print(f"⚠ Email attempt {attempt}/{max_retries} - connection error")
            except Exception as e:
                print(f"⚠ Email attempt {attempt}/{max_retries} failed: {e}")
            
            if attempt < max_retries:
                print(f"  Retrying in {current_delay}s...")
                import time as _time
                _time.sleep(current_delay)
                current_delay *= 2
        
        # All retries exhausted
        print(f"❌ Email notification failed after {max_retries} attempts")
        return {
            "success": True, 
            "report_url": report_url, 
            "email_error": f"Email failed after {max_retries} attempts",
            "email_status": "error"
        }
    
    def cleanup_temp_files(self):
        """Remove temporary snapshot images."""
        import shutil
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"✔ Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                print(f"⚠ Could not clean up temp directory: {e}")
