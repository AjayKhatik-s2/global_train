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


# State color mapping (BGR to RGB for PDF)
STATE_COLORS_RGB = {
    'open_door': colors.red,
    'open': colors.red,
    'closed_door': colors.green,
    'closed': colors.green,
    'closed_with_wire': colors.yellow,
    'partial_closed': colors.orange,
    'partially_closed': colors.orange,
}


class DoorReportGenerator:
    """
    Generate PDF report for door inspection results.
    
    Report structure:
    1. Summary page with video info and door statistics
    2. One page per door with annotated snapshot and details
    """
    
    # API Configuration
    UPLOAD_API_URL = "https://reports-api.suvidhaen.com/api/upload-pdf"
    EMAIL_API_URL = "https://railopsapi.suvidhaen.com/notification_microservice/send-email"
    PRODUCT_NAME = "CCTV-Door_Detection-Reports"
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
        logo_path: str = None,
        require_open_event: bool = False
    ):
        self.output_path = output_path
        self.video_path = video_path
        # Portable temp dir (old code hardcoded /tmp which fails on Windows).
        # Location only affects where intermediate snapshot JPEGs are staged;
        # it has no bearing on the rendered PDF, so this is fidelity-neutral.
        self.temp_dir = temp_dir or os.path.join(
            tempfile.gettempdir(), "door_report_images")
        self.source_video_url = source_video_url
        self.region = region
        self.logo_path = logo_path
        # LEFT_UP required an explicit open-event to count a door OPEN; RIGHT_UP
        # treated any 'open' state as open.  Parameterized so one class serves
        # both side cameras.
        self.require_open_event = require_open_event

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
            if frame is None or frame.size == 0:
                print(f"  ⚠ Empty snapshot frame for door {door_id}, skipping")
                return ""
            
            tag = f"{prefix}_" if prefix else ""
            filename = f"door_{tag}{door_id}_snapshot.jpg"
            filepath = os.path.join(self.temp_dir, filename)
            
            # Ensure temp dir exists
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Write frame directly — snapshot is already in BGR (OpenCV native)
            success = cv2.imwrite(filepath, frame)
            
            if not success or not os.path.exists(filepath):
                print(f"  ⚠ Failed to write snapshot for door {door_id}")
                return ""
            
            return filepath
        except Exception as e:
            print(f"  ⚠ Snapshot save error for door {door_id}: {e}")
            return ""
    
    def _get_state_style(self, state: str) -> str:
        """Get paragraph style name for door state."""
        state_lower = state.lower()
        if 'open' in state_lower or 'damage' in state_lower:
            return 'StateOpen'
        elif 'closed' in state_lower and 'partial' not in state_lower:
            return 'StateClosed'
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
        has_open_doors = any(
            ('open' in d.get('state', '').lower()
             and (not self.require_open_event or d.get('open_event_raised', False)))
            or 'damage' in d.get('state', '').lower()
            for d in doors
        )
        has_damage = wagon_damage_map and any(
            len(damages) > 0 for wagon_num, damages in wagon_damage_map.items() if wagon_num != 0
        )
        
        # Skip if no alerts
        if not has_open_doors and not has_damage:
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
        
        # SECTION 1: OPEN DOORS (highest priority)
        if has_open_doors:
            elements.append(Paragraph(
                "⚠️ OPEN DOORS DETECTED",
                ParagraphStyle('OpenDoorSection', parent=self.styles['Heading1'],
                             fontSize=20, textColor=colors.red,
                             spaceAfter=12)
            ))
            
            # Find wagons with open doors
            open_door_wagons = {}
            for door in doors:
                state = door.get('state', '').lower()
                is_open = 'open' in state and (
                    not self.require_open_event or door.get('open_event_raised', False))
                is_damage = 'damage' in state
                if is_open or is_damage:
                    wagon_num = door.get('wagon_number', 1)
                    if wagon_num not in open_door_wagons:
                        open_door_wagons[wagon_num] = []
                    open_door_wagons[wagon_num].append(door)
            
            # Create image grid for open door wagons (2 per row)
            image_data = []
            row = []
            for wagon_num in sorted(open_door_wagons.keys()):
                wagon_doors = open_door_wagons[wagon_num]
                # Use first door's snapshot as representative image
                first_door = wagon_doors[0]
                
                if first_door.get('snapshot') is not None:
                    snapshot_path = self._save_snapshot_image(
                        first_door['snapshot'],
                        first_door['door_id']
                    )
                    
                    if snapshot_path and os.path.exists(snapshot_path):
                        # Create image with label
                        from io import BytesIO
                        with open(snapshot_path, 'rb') as img_file:
                            img_data = BytesIO(img_file.read())
                        img = Image(img_data, width=4*inch, height=2.5*inch)
                        label = Paragraph(
                            f"<b>Wagon {wagon_num}</b><br/>{len(wagon_doors)} open door(s)",
                            ParagraphStyle('ImgLabel', parent=self.styles['Normal'],
                                         fontSize=10, alignment=TA_CENTER)
                        )
                        
                        cell = [img, label]
                        row.append(cell)
                    
                    # Add row when we have 2 images
                    if len(row) == 2:
                        image_data.append(row)
                        row = []
            
            # Add remaining images
            if row:
                # Pad with empty cell if odd number
                while len(row) < 2:
                    row.append([""])
                image_data.append(row)
            
            if image_data:
                img_table = Table(image_data, colWidths=[4.5*inch, 4.5*inch])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                ]))
                elements.append(img_table)
            
            elements.append(Spacer(1, 0.3*inch))
        
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
        damage_frames: Optional[Dict] = None,  # NEW: For damage frames for priority alert images
        loco_numbers=None  # Loco numbers list from OCR
    ) -> List:
        """Create combined summary page with door and damage information."""
        elements = []
        
        # Title
        elements.append(Paragraph(
            "Door Inspection Report",
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
        video_table_data = [
            ["Source Video:", video_name],
            ["Processing Time:", f"{processing_time:.2f} seconds"],
        ]
        
        # Add source video URL if provided
        if self.source_video_url:
            # Create a clickable link with "Click to view video" text
            link_paragraph = Paragraph(
                f'<link href="{self.source_video_url}" color="blue"><u>Click to view video</u></link>',
                self.styles['Normal']
            )
            video_table_data.append(["Video Link:", link_paragraph])
        
        video_table = Table(video_table_data, colWidths=[2*inch, 6*inch])
        
        video_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(video_table)
        elements.append(Spacer(1, 20))
        
        # Total Number of Wagons + Loco Number - Prominent display
        if isinstance(loco_numbers, list) and loco_numbers:
            loco_display = " / ".join(str(n).upper() for n in loco_numbers)
        elif loco_numbers:
            loco_display = str(loco_numbers).upper()
        else:
            loco_display = "Not Detected"
        elements.append(Paragraph(
            f"<b>Total Number of Wagons: {total_wagons}  |  Loco Number: {loco_display}</b>",
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
        
        # Add open doors first (safety critical) - always show even if 0
        open_count = sum(v for k, v in state_counts.items() if 'open' in k.lower() or 'damage' in k.lower())
        stats_data.append(["OPEN DOORS / DAMAGE", str(open_count)])
        
        # Add other states (excluding 'other' and 'unknown')
        excluded_states = {'other', 'unknown', 'others'}
        for state, count in sorted(state_counts.items()):
            if 'open' not in state.lower() and 'damage' not in state.lower() and state.lower() not in excluded_states:
                stats_data.append([state.upper(), str(count)])
        
        stats_data.append(["TOTAL DOORS", str(total_doors)])
        
        # Add damage info to same table
        if wagon_damage_map is not None:
            # Count wagons with damage (excluding wagon 0 = unassociated)
            wagons_with_damage = sum(1 for wagon_num, damages in wagon_damage_map.items() 
                                    if wagon_num != 0 and len(damages) > 0)
            stats_data.append(["TOTAL DAMAGES", str(wagons_with_damage)])
        
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
            if 'open' in state:
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
            # Check for open doors
            has_open_doors = any(
                ('open' in d.get('state', '').lower()
                 and (not self.require_open_event or d.get('open_event_raised', False)))
                or 'damage' in d.get('state', '').lower()
                for d in doors
            )
            has_damage = wagon_damage_map and any(
                len(damages) > 0 for wagon_num, damages in wagon_damage_map.items() if wagon_num != 0
            )
            
            if has_open_doors or has_damage:
                elements.append(Paragraph(
                    "🚨 PRIORITY ALERTS",
                    ParagraphStyle('AlertHeader', parent=self.styles['Heading_Custom'],
                                 fontSize=18, textColor=colors.red)
                ))
                
                # OPEN DOOR IMAGES
                if has_open_doors:
                    elements.append(Paragraph(
                        "⚠️ OPEN DOORS / DAMAGE DETECTED",
                        ParagraphStyle('OpenDoorAlert', parent=self.styles['Normal'],
                                     fontSize=14, textColor=colors.red, fontName='Helvetica-Bold')
                    ))
                    
                    # Find wagons with open doors
                    open_door_wagons = {}
                    for door in doors:
                        state = door.get('state', '').lower()
                        is_open = 'open' in state and (
                            not self.require_open_event or door.get('open_event_raised', False))
                        is_damage = 'damage' in state
                        if is_open or is_damage:
                            wagon_num = door.get('wagon_number', 1)
                            if wagon_num not in open_door_wagons:
                                open_door_wagons[wagon_num] = []
                            open_door_wagons[wagon_num].append(door)
                    
                    # Create image row (max 2 per row)
                    image_row = []
                    for wagon_num in sorted(open_door_wagons.keys())[:2]:  # Show max 2 images
                        wagon_doors = open_door_wagons[wagon_num]
                        first_door = wagon_doors[0]
                        
                        if first_door.get('snapshot') is not None:
                            snapshot_path = self._save_snapshot_image(
                                first_door['snapshot'],
                                first_door['door_id']
                            )
                            
                            if snapshot_path and os.path.exists(snapshot_path):
                                from io import BytesIO
                                with open(snapshot_path, 'rb') as img_file:
                                    img_data = BytesIO(img_file.read())
                                img = Image(img_data, width=3.5*inch, height=2.2*inch)
                            label = Paragraph(
                                f"<b>Wagon {wagon_num}</b>",
                                ParagraphStyle('ImgLabel', parent=self.styles['Normal'],
                                             fontSize=9, alignment=TA_CENTER)
                            )
                            image_row.append([img, label])
                    
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
                        elements.append(Spacer(1, 10))
                
                
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
    
    def _create_door_page(
        self,
        door_number: int,
        door_id: int,
        state: str,
        confidence: float,
        snapshot_path: str,
        wagon_number: Optional[int] = None,
        ocr_wagon_id: Optional[str] = None  # NEW: OCR-detected wagon number
    ) -> List:
        """Create decorative page for a single door - all on one page."""
        page_elements = []  # Elements for this door's page
        
        # Get state color for styling
        state_lower = state.lower()
        if 'open' in state_lower:
            state_color = colors.Color(1, 0.2, 0.2)  # Red
            bg_color = colors.Color(1, 0.9, 0.9)  # Light red
        elif 'closed' in state_lower and 'partial' not in state_lower:
            state_color = colors.Color(0, 0.6, 0)  # Green
            bg_color = colors.Color(0.9, 1, 0.9)  # Light green
        else:
            state_color = colors.Color(0.9, 0.6, 0)  # Orange
            bg_color = colors.Color(1, 0.95, 0.85)  # Light orange
        
        state_display = state.upper().replace('_', ' ')
        
        # Build header text with wagon number and OCR ID if available
        if wagon_number is not None:
            header_text = f"<b>Wagon No: {wagon_number} | Door #{door_number}</b>"
            
            # Add OCR wagon number if available
            if ocr_wagon_id is not None:
                header_text += f"<br/><font size=14 color='green'>Wagon ID: {ocr_wagon_id}</font>"
        else:
            header_text = f"<b>Door #{door_number}</b>"
        
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
    
    def _extract_wagon_snapshots_quad(
        self,
        wagon_number: int,
        start_frame: int,
        end_frame: int
    ) -> List[Optional[str]]:
        """
        Extract 4 snapshots from a wagon's frame range at quartile positions.
        
        Positions (center of each quartile):
          Snapshot 1: ~12.5%  (start portion)
          Snapshot 2: ~37.5%  (first middle portion)
          Snapshot 3: ~62.5%  (second middle portion)
          Snapshot 4: ~87.5%  (end portion)
        
        Args:
            wagon_number: Wagon number for filenames
            start_frame: First frame of wagon
            end_frame: Last frame of wagon
            
        Returns:
            List of 4 file paths (some may be None if extraction failed)
        """
        duration = end_frame - start_frame
        if duration <= 0:
            return [None, None, None, None]
        
        # Calculate 4 quartile center positions
        positions = [
            start_frame + int(duration * 0.125),   # 12.5% — Start
            start_frame + int(duration * 0.375),   # 37.5% — Middle-1
            start_frame + int(duration * 0.625),   # 62.5% — Middle-2
            start_frame + int(duration * 0.875),   # 87.5% — End
        ]
        
        labels = ["Start", "Middle-1", "Middle-2", "End"]
        paths = []
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Could not open video: {self.video_path}")
                return [None, None, None, None]
            
            for idx, (frame_idx, label) in enumerate(zip(positions, labels)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    print(f"Could not read frame {frame_idx} for wagon {wagon_number} ({label})")
                    paths.append(None)
                    continue
                
                # Add label overlay
                overlay_label = f"Wagon {wagon_number} | {label} (Frame {frame_idx})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(overlay_label, font, font_scale, thickness)
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (text_w + 30, text_h + 30), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, overlay_label, (20, text_h + 20), font, font_scale, (255, 255, 255), thickness)
                
                # Save to temp file
                filename = f"wagon_{wagon_number}_snap_{idx}_{label.lower()}.jpg"
                filepath = os.path.join(self.temp_dir, filename)
                success = cv2.imwrite(filepath, frame)
                if success and os.path.exists(filepath):
                    paths.append(filepath)
                else:
                    print(f"  ⚠ Failed to save wagon snapshot: {filepath}")
                    paths.append(None)
            
            cap.release()
            return paths
            
        except Exception as e:
            print(f"Error extracting wagon snapshots: {e}")
            return [None, None, None, None]
    
    def _create_wagon_overview_page(
        self,
        wagon_number: int,
        wagon_start_frame: int,
        wagon_end_frame: int,
        ocr_wagon_id: Optional[str] = None,
        subtitle: Optional[str] = None
    ) -> List:
        """
        Create a wagon overview page with 4 snapshots in a 2×2 grid.
        
        Used for ALL wagons (with or without doors) to provide full
        visual coverage of the wagon body.
        
        Args:
            wagon_number: Wagon number (1-indexed)
            wagon_start_frame: First frame of wagon
            wagon_end_frame: Last frame of wagon
            ocr_wagon_id: Optional OCR-detected wagon ID string
            subtitle: Optional subtitle (e.g., 'NO DOOR DETECTED')
            
        Returns:
            List of reportlab elements
        """
        page_elements = []
        
        # Header with wagon number
        header_text = f"<b>Wagon No: {wagon_number}</b>"
        if ocr_wagon_id:
            header_text += f"<br/><font size=12 color='green'>Wagon ID: {ocr_wagon_id}</font>"
        
        header_data = [
            [Paragraph(header_text, 
                      ParagraphStyle('WagonOverviewHeader', parent=self.styles['Heading1'], 
                                   fontSize=20, alignment=TA_CENTER, textColor=colors.black))],
        ]
        header_table = Table(header_data, colWidths=[10*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
        ]))
        page_elements.append(header_table)
        
        # Optional subtitle (e.g., "NO DOOR DETECTED")
        if subtitle:
            text_color = colors.Color(0.4, 0.4, 0.4)
            page_elements.append(Paragraph(
                subtitle,
                ParagraphStyle('WagonSubtitle', parent=self.styles['Heading2'],
                             fontSize=14, alignment=TA_CENTER, textColor=text_color,
                             spaceAfter=2)
            ))
        
        # Frame range info
        frame_info = f"Frames: {wagon_start_frame} - {wagon_end_frame}"
        page_elements.append(Paragraph(
            frame_info,
            ParagraphStyle('FrameRangeInfo', parent=self.styles['Normal'],
                         fontSize=9, alignment=TA_CENTER,
                         textColor=colors.Color(0.5, 0.5, 0.5))
        ))
        page_elements.append(Spacer(1, 6))
        
        # Extract 4 snapshots at quartile positions
        snapshot_paths = self._extract_wagon_snapshots_quad(
            wagon_number, wagon_start_frame, wagon_end_frame
        )
        
        labels = ["Start", "Middle-1", "Middle-2", "End"]
        
        # Build 2×2 grid of snapshot images
        # Landscape A4 usable width ~10", height ~6.5" after header
        # Each image: ~4.8" × 2.7" to maximize space
        img_width = 4.8 * inch
        img_height = 2.7 * inch
        col_width = 5.0 * inch
        
        # Row 1: Start + Middle-1
        row1_cells = []
        for i in range(2):
            path = snapshot_paths[i]
            label = labels[i]
            if path and os.path.exists(path):
                try:
                    from io import BytesIO
                    with open(path, 'rb') as img_file:
                        img_data = BytesIO(img_file.read())
                    img = Image(img_data, width=img_width, height=img_height)
                    caption = Paragraph(
                        f"<b>{label}</b>",
                        ParagraphStyle('SnapCaption', parent=self.styles['Normal'],
                                     fontSize=9, alignment=TA_CENTER)
                    )
                    row1_cells.append([img, caption])
                except Exception:
                    row1_cells.append([self._create_small_placeholder(label)])
            else:
                row1_cells.append([self._create_small_placeholder(label)])
        
        # Row 2: Middle-2 + End
        row2_cells = []
        for i in range(2, 4):
            path = snapshot_paths[i]
            label = labels[i]
            if path and os.path.exists(path):
                try:
                    from io import BytesIO
                    with open(path, 'rb') as img_file:
                        img_data = BytesIO(img_file.read())
                    img = Image(img_data, width=img_width, height=img_height)
                    caption = Paragraph(
                        f"<b>{label}</b>",
                        ParagraphStyle('SnapCaption2', parent=self.styles['Normal'],
                                     fontSize=9, alignment=TA_CENTER)
                    )
                    row2_cells.append([img, caption])
                except Exception:
                    row2_cells.append([self._create_small_placeholder(label)])
            else:
                row2_cells.append([self._create_small_placeholder(label)])
        
        # Build 2×2 table
        grid_data = [row1_cells, row2_cells]
        grid_table = Table(grid_data, colWidths=[col_width, col_width])
        grid_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('BOX', (0, 0), (0, 0), 1, colors.Color(0.85, 0.85, 0.85)),
            ('BOX', (1, 0), (1, 0), 1, colors.Color(0.85, 0.85, 0.85)),
            ('BOX', (0, 1), (0, 1), 1, colors.Color(0.85, 0.85, 0.85)),
            ('BOX', (1, 1), (1, 1), 1, colors.Color(0.85, 0.85, 0.85)),
        ]))
        page_elements.append(grid_table)
        
        return [KeepTogether(page_elements), PageBreak()]
    
    def _create_no_door_wagon_page(
        self,
        wagon_number: int,
        wagon_start_frame: int,
        wagon_end_frame: int
    ) -> List:
        """
        Create a page for a wagon with no detected doors.
        Shows 4 snapshots in a 2×2 grid with 'NO DOOR DETECTED' subtitle.
        """
        return self._create_wagon_overview_page(
            wagon_number=wagon_number,
            wagon_start_frame=wagon_start_frame,
            wagon_end_frame=wagon_end_frame,
            subtitle="NO DOOR DETECTED"
        )
    
    def _create_small_placeholder(self, label: str = "") -> Table:
        """Create a small placeholder for a missing snapshot in the 2×2 grid."""
        text_color = colors.Color(0.5, 0.5, 0.5)
        bg_color = colors.Color(0.95, 0.95, 0.95)
        
        placeholder_data = [
            [Paragraph(f"<b>{label}</b><br/>Snapshot Not Available",
                      ParagraphStyle('SmallPlaceholder', parent=self.styles['Normal'],
                                   fontSize=11, alignment=TA_CENTER, textColor=text_color))]
        ]
        
        placeholder_table = Table(placeholder_data, colWidths=[4.5*inch], rowHeights=[2.5*inch])
        placeholder_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 1, colors.gray),
        ]))
        
        return placeholder_table
    
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
    
    def generate_report(
        self,
        doors: List[dict],
        state_counts: Dict[str, int],
        processing_time: float,
        door_open_events: int = 0,
        wagon_summary: List[dict] = None,
        damage_elements: List = None,  # Accept damage report elements
        wagon_damage_map: Optional[Dict] = None,  # NEW: Accept damage map for priority alerts
        damage_frames: Optional[Dict] = None,  # NEW: Accept damage frames for priority alert images
        loco_numbers=None,  # Loco numbers list from OCR
        full_wagon_summary: Optional[List[dict]] = None  # Includes engine/brakevan segments; accepted for API parity with TOP cameras
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
            total_doors=len(doors),
            state_counts=state_counts,
            processing_time=processing_time,
            door_open_events=door_open_events,
            total_wagons=total_wagons,
            doors=doors,  # Pass for priority alerts
            wagon_summary=wagon_summary,  # Pass for priority alerts
            wagon_damage_map=wagon_damage_map,  # Pass for damage summary
            damage_frames=damage_frames,  # Pass damage frames for priority alert images
            loco_numbers=loco_numbers  # Pass loco numbers for summary display
        ))
        
        # Priority Alerts are now embedded in summary page, no separate page needed
        
        # Build door lookup by wagon_number
        doors_by_wagon = {}
        for door in doors:
            wagon_num = door.get('wagon_number', 1)
            if wagon_num not in doors_by_wagon:
                doors_by_wagon[wagon_num] = []
            doors_by_wagon[wagon_num].append(door)
        
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
            # Find wagons with open doors for highlight section
            wagons_with_open_doors = []
            for wagon in wagon_summary:
                wagon_num = wagon['wagon_number']
                wagon_doors = doors_by_wagon.get(wagon_num, [])
                if any(
                    ('open' in d['state'].lower()
                     and (not self.require_open_event or d.get('open_event_raised', False)))
                    or 'damage' in d['state'].lower()
                    for d in wagon_doors
                ):
                    wagons_with_open_doors.append(wagon)
            
            # SECTION 1: Open Door Wagons (Highlights) - show first for quick review
            if wagons_with_open_doors:
                
                for wagon in sorted(wagons_with_open_doors, key=lambda w: w['wagon_number']):
                    wagon_num = wagon['wagon_number']
                    wagon_doors = doors_by_wagon.get(wagon_num, [])
                    
                    # Only show open doors that passed event filters, plus damage
                    open_doors = [
                        d for d in wagon_doors
                        if ('open' in d['state'].lower()
                            and (not self.require_open_event or d.get('open_event_raised', False)))
                        or 'damage' in d['state'].lower()
                    ]
                    for door in sorted(open_doors, key=lambda d: d['door_number']):
                        if door.get('snapshot') is not None:
                            snapshot_path = self._save_snapshot_image(
                                door['snapshot'],
                                door['door_id'],
                                prefix="highlight"
                            )
                        else:
                            snapshot_path = ""
                        
                        elements.extend(self._create_door_page(
                            door_number=door['door_number'],
                            door_id=door['door_id'],
                            state=door['state'],
                            confidence=door.get('confidence', 0.0),
                            snapshot_path=snapshot_path,
                            wagon_number=door.get('wagon_number'),
                            ocr_wagon_id=wagon_ocr_ids.get(wagon_num)  # Add OCR wagon ID
                        ))

            
            # SECTION 2: All wagons in serial order (1, 2, 3, ...)
            # Each wagon gets a 4-snapshot overview page first, then door detail pages
            for wagon in sorted(wagon_summary, key=lambda w: w['wagon_number']):
                wagon_num = wagon['wagon_number']
                wagon_doors = doors_by_wagon.get(wagon_num, [])
                
                if wagon_doors:
                    # Add wagon overview page with 4 snapshots FIRST
                    elements.extend(self._create_wagon_overview_page(
                        wagon_number=wagon_num,
                        wagon_start_frame=wagon.get('start_frame', 0),
                        wagon_end_frame=wagon.get('end_frame', 0),
                        ocr_wagon_id=wagon_ocr_ids.get(wagon_num)
                    ))
                    
                    # Then add individual door detail pages
                    wagon_doors_sorted = sorted(
                        wagon_doors,
                        key=lambda d: (0 if 'open' in d['state'].lower() else 1, d['door_number'])
                    )
                    
                    for door in wagon_doors_sorted:
                        # Save snapshot
                        if door.get('snapshot') is not None:
                            snapshot_path = self._save_snapshot_image(
                                door['snapshot'],
                                door['door_id'],
                                prefix="detail"
                            )
                        else:
                            snapshot_path = ""
                        
                        elements.extend(self._create_door_page(
                            door_number=door['door_number'],
                            door_id=door['door_id'],
                            state=door['state'],
                            confidence=door.get('confidence', 0.0),
                            snapshot_path=snapshot_path,
                            wagon_number=door.get('wagon_number'),
                            ocr_wagon_id=wagon_ocr_ids.get(wagon_num)  # Add OCR wagon ID
                        ))
                else:
                    # No doors in this wagon - add overview page with 'NO DOOR DETECTED'
                    elements.extend(self._create_no_door_wagon_page(
                        wagon_number=wagon_num,
                        wagon_start_frame=wagon.get('start_frame', 0),
                        wagon_end_frame=wagon.get('end_frame', 0)
                    ))
        else:
            # Fallback to door-driven (legacy behavior)
            sorted_doors = sorted(
                doors,
                key=lambda d: (0 if 'open' in d['state'].lower() else 1, d['door_number'])
            )
            
            for door in sorted_doors:
                if door.get('snapshot') is not None:
                    snapshot_path = self._save_snapshot_image(
                        door['snapshot'],
                        door['door_id'],
                        prefix="legacy"
                    )
                else:
                    snapshot_path = ""
                
                elements.extend(self._create_door_page(
                    door_number=door['door_number'],
                    door_id=door['door_id'],
                    state=door['state'],
                    confidence=door.get('confidence', 0.0),
                    snapshot_path=snapshot_path,
                    wagon_number=door.get('wagon_number')
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
