"""
PDF Utilities for Door Detection Pipeline

Provides functions for:
- Merging multiple PDFs into one (with Hazaribagh as first page)
- Adding logo to all pages of a PDF
"""

import os
from typing import List, Optional


def merge_pdfs(pdf_paths: List[str], output_path: str) -> Optional[str]:
    """
    Merge multiple PDFs into a single PDF file.
    
    Args:
        pdf_paths: List of paths to PDF files to merge (in order)
        output_path: Path for the merged output PDF
        
    Returns:
        Path to the merged PDF file, or None if failed
    """
    try:
        from PyPDF2 import PdfMerger
        
        merger = PdfMerger()
        
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                print(f"  Adding to merge: {os.path.basename(pdf_path)}")
                merger.append(pdf_path)
            else:
                print(f"  ⚠ PDF not found, skipping: {pdf_path}")
        
        merger.write(output_path)
        merger.close()
        
        print(f"✓ Merged PDF created: {output_path}")
        return output_path
        
    except ImportError:
        print("⚠ PyPDF2 not installed. Attempting with pypdf...")
        try:
            from pypdf import PdfMerger
            
            merger = PdfMerger()
            
            for pdf_path in pdf_paths:
                if os.path.exists(pdf_path):
                    print(f"  Adding to merge: {os.path.basename(pdf_path)}")
                    merger.append(pdf_path)
            
            merger.write(output_path)
            merger.close()
            
            print(f"✓ Merged PDF created: {output_path}")
            return output_path
            
        except ImportError:
            print("⚠ Neither PyPDF2 nor pypdf installed. Cannot merge PDFs.")
            print("  Install with: pip install PyPDF2")
            return None
    except Exception as e:
        print(f"⚠ PDF merge failed: {e}")
        return None


def create_combined_report(
    hazaribagh_pdf_path: str,
    main_report_pdf_path: str,
    output_path: str,
    skip_main_first_page: bool = True
) -> Optional[str]:
    """
    Create a combined report with Hazaribagh as first page(s), 
    followed by the main report pages.
    
    Args:
        hazaribagh_pdf_path: Path to Hazaribagh PDF (will be first)
        main_report_pdf_path: Path to main report PDF
        output_path: Path for the combined output PDF
        skip_main_first_page: If True, skip the first page of main report
                             (useful if main report has a summary page to replace)
        
    Returns:
        Path to combined PDF, or None if failed
    """
    try:
        try:
            from PyPDF2 import PdfReader, PdfWriter
        except ImportError:
            from pypdf import PdfReader, PdfWriter
        
        writer = PdfWriter()
        
        # Add all pages from Hazaribagh PDF first
        if os.path.exists(hazaribagh_pdf_path):
            print(f"  Adding Hazaribagh pages: {os.path.basename(hazaribagh_pdf_path)}")
            hazaribagh_reader = PdfReader(hazaribagh_pdf_path)
            for page in hazaribagh_reader.pages:
                writer.add_page(page)
            print(f"    Added {len(hazaribagh_reader.pages)} page(s) from Hazaribagh report")
        else:
            print(f"  ⚠ Hazaribagh PDF not found: {hazaribagh_pdf_path}")
        
        # Add pages from main report (optionally skip first page)
        if os.path.exists(main_report_pdf_path):
            print(f"  Adding main report pages: {os.path.basename(main_report_pdf_path)}")
            main_reader = PdfReader(main_report_pdf_path)
            
            start_page = 1 if skip_main_first_page else 0
            pages_added = 0
            
            for i, page in enumerate(main_reader.pages):
                if i >= start_page:
                    writer.add_page(page)
                    pages_added += 1
            
            if skip_main_first_page:
                print(f"    Skipped first page of main report (replaced by Hazaribagh)")
            print(f"    Added {pages_added} page(s) from main report")
        else:
            print(f"  ⚠ Main report PDF not found: {main_report_pdf_path}")
        
        # Write combined PDF
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"✓ Combined report created: {output_path}")
        return output_path
        
    except ImportError:
        print("⚠ Neither PyPDF2 nor pypdf installed. Cannot create combined PDF.")
        print("  Install with: pip install PyPDF2")
        return None
    except Exception as e:
        print(f"⚠ Combined PDF creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_logo_to_pdf(input_pdf_path: str, logo_path: str, output_path: str = None) -> Optional[str]:
    """
    Add logo to all pages of a PDF.
    
    Note: This is a placeholder - adding logo overlay requires more complex
    PDF manipulation. For now, logo is added during PDF generation instead.
    
    Args:
        input_pdf_path: Path to input PDF
        logo_path: Path to logo image
        output_path: Path for output PDF (defaults to overwrite input)
        
    Returns:
        Path to output PDF, or None if failed
    """
    # Logo is added during PDF generation via ReportLab callbacks
    # This function is kept for future enhancement if needed
    print("  Note: Logo is added during PDF generation")
    return input_pdf_path

