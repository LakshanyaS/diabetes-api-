from __future__ import annotations

import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf_report(
    patient_name: str,
    inputs: dict[str, Any],
    prediction_label: str,
    risk_pct: float,
    risk_level: str,
    suggestions: list[str],
) -> bytes:
    """Generate a downloadable PDF report for the user inputs."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=1 * inch,
        bottomMargin=0.8 * inch,
    )
    styles = getSampleStyleSheet()
    story: list[Any] = []

    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1e1e2e"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=16,
    )
    heading_style = ParagraphStyle(
        "heading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1e1e2e"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
    )

    cmap = {"Low": colors.HexColor("#22c55e"), "Medium": colors.HexColor("#f59e0b"), "High": colors.HexColor("#ef4444")}
    risk_color = cmap.get(risk_level, colors.grey)

    story += [
        Paragraph("Diabetes Risk Assessment Report", title_style),
        Paragraph(f"Prepared for: <b>{patient_name}</b>", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")),
        Spacer(1, 0.2 * inch),
    ]

    # Summary table
    summary = Table(
        [["Prediction", "Risk Score", "Risk Level"], [prediction_label, f"{risk_pct:.1f}%", risk_level]],
        colWidths=[2.1 * inch, 2.0 * inch, 1.6 * inch],
    )
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e1e2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9fafb")]),
                ("TEXTCOLOR", (2, 1), (2, 1), risk_color),
                ("FONTNAME", (2, 1), (2, 1), "Helvetica-Bold"),
                ("FONTSIZE", (2, 1), (2, 1), 13),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ]
        )
    )
    story += [summary, Spacer(1, 0.2 * inch)]

    # Patient inputs
    story.append(Paragraph("Patient Input Values", heading_style))
    t_inputs = Table([["Parameter", "Value"]] + [[k, str(v)] for k, v in inputs.items()], colWidths=[3.0 * inch, 3.0 * inch])
    t_inputs.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ]
        )
    )
    story += [t_inputs, Spacer(1, 0.2 * inch)]

    # Suggestions
    story.append(Paragraph("Health Suggestions", heading_style))
    for s in suggestions:
        story += [Paragraph(f"• {s}", body_style), Spacer(1, 2)]

    story += [
        Spacer(1, 0.3 * inch),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")),
        Spacer(1, 0.15 * inch),
        Paragraph(
            "<i>Disclaimer: AI-generated report. Not a substitute for professional medical advice.</i>",
            ParagraphStyle("disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey, alignment=1),
        ),
    ]

    doc.build(story)
    return buf.getvalue()

