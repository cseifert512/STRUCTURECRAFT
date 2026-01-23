# app/services/pdf_service.py
"""
PDF Report Generation Service - creates downloadable design summaries.
"""

from io import BytesIO
from typing import Dict, Any
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def is_pdf_available() -> bool:
    """Check if PDF generation is available (reportlab installed)."""
    return REPORTLAB_AVAILABLE


def generate_design_pdf(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    deflection_check: Dict[str, Any],
    utilization_check: Dict[str, Any],
) -> bytes:
    """
    Generate a one-page PDF design summary.
    
    Args:
        params: Design parameters dictionary
        metrics: Analysis metrics dictionary
        deflection_check: Result from compute_deflection_check()
        utilization_check: Result from compute_utilization()
    
    Returns:
        bytes: PDF file contents
    
    Raises:
        ImportError: If reportlab is not installed
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "PDF generation requires reportlab. Install with: pip install reportlab"
        )
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=20,
        alignment=TA_CENTER,
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("StructureCraft Design Summary", title_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle('Date', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.gray)
    ))
    elements.append(Spacer(1, 20))
    
    # Geometry Table
    elements.append(Paragraph("Geometry", heading_style))
    geom_data = [
        ['Parameter', 'Value'],
        ['Width', f"{params.get('width', 0):.1f} m"],
        ['Depth', f"{params.get('depth', 0):.1f} m"],
        ['Grid', f"{params.get('nx', 0)} × {params.get('ny', 0)}"],
        ['Height Range', f"{params.get('min_height', 0):.2f} - {params.get('max_height', 0):.2f} m"],
        ['Shape', params.get('heightfield', 'N/A')],
        ['Topology', params.get('topology', 'N/A')],
        ['Supports', params.get('support_layout', 'N/A')],
    ]
    geom_table = Table(geom_data, colWidths=[2.5*inch, 2.5*inch])
    geom_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    elements.append(geom_table)
    elements.append(Spacer(1, 15))
    
    # Structure Stats
    elements.append(Paragraph("Structure", heading_style))
    struct_data = [
        ['Property', 'Value'],
        ['Nodes', str(metrics.get('n_nodes', 0))],
        ['Members', str(metrics.get('n_bars', 0))],
        ['Supports', str(metrics.get('n_supports', 0))],
        ['Material Volume', f"{metrics.get('volume', 0):.4f} m³"],
        ['Length Bins', str(metrics.get('n_length_bins', 0))],
    ]
    struct_table = Table(struct_data, colWidths=[2.5*inch, 2.5*inch])
    struct_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    elements.append(struct_table)
    elements.append(Spacer(1, 15))
    
    # Design Checks Table
    elements.append(Paragraph("Design Checks", heading_style))
    
    # Determine status colors/text
    defl_pass = deflection_check.get('overall_pass', False)
    defl_ratio = deflection_check.get('checks', {}).get('L/240', {}).get('ratio', 0)
    
    util_pass = utilization_check.get('all_pass', False)
    worst_util = utilization_check.get('worst_util', 0)
    
    max_len = metrics.get('max_member_length', 0)
    shipping_pass = max_len <= 6.0
    
    checks_data = [
        ['Check', 'Value', 'Status'],
        ['Deflection (L/240)', f"{defl_ratio:.0%}", 'PASS' if defl_pass else 'FAIL'],
        ['Member Utilization', f"{worst_util*100:.0f}%", 'PASS' if util_pass else 'FAIL'],
        ['Max Displacement', f"{metrics.get('max_displacement', 0)*1000:.2f} mm", '-'],
        ['Max Tension', f"{metrics.get('max_tension', 0)/1000:.1f} kN", '-'],
        ['Max Compression', f"{metrics.get('max_compression', 0)/1000:.1f} kN", '-'],
        ['Longest Member', f"{max_len:.2f} m", 'PASS' if shipping_pass else 'FAIL'],
    ]
    
    checks_table = Table(checks_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    
    # Build table style with conditional colors for status column
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a4a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]
    
    # Color status cells
    for i, row in enumerate(checks_data[1:], start=1):
        if row[2] == 'PASS':
            table_style.append(('TEXTCOLOR', (2, i), (2, i), colors.HexColor('#28a745')))
            table_style.append(('FONTNAME', (2, i), (2, i), 'Helvetica-Bold'))
        elif row[2] == 'FAIL':
            table_style.append(('TEXTCOLOR', (2, i), (2, i), colors.HexColor('#dc3545')))
            table_style.append(('FONTNAME', (2, i), (2, i), 'Helvetica-Bold'))
    
    checks_table.setStyle(TableStyle(table_style))
    elements.append(checks_table)
    elements.append(Spacer(1, 20))
    
    # Overall Status
    all_pass = defl_pass and util_pass and shipping_pass
    if all_pass:
        status_text = "✓ ALL CHECKS PASS — Design is acceptable"
        status_color = colors.HexColor('#28a745')
    else:
        status_text = "✗ DESIGN NEEDS REVISION — Some checks fail"
        status_color = colors.HexColor('#dc3545')
    
    status_style = ParagraphStyle(
        'Status',
        parent=styles['Normal'],
        fontSize=14,
        textColor=status_color,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceBefore=10,
    )
    elements.append(Paragraph(status_text, status_style))
    
    # Footer
    elements.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        alignment=TA_CENTER,
    )
    elements.append(Paragraph(
        "Generated by StructureCraft — For preliminary design only. "
        "Final designs must be verified by a licensed structural engineer.",
        footer_style
    ))
    
    # Build PDF
    doc.build(elements)
    
    return buffer.getvalue()

