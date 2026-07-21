"""
Report generator: creates styled PDF + CSV session summaries.
Professional design — clean white background, dark readable text,
clear color-coded severity, no overlapping, print-ready layout.
"""

import io
import os
import csv
import time
import math
from datetime import datetime

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics import renderPDF

# ── Professional colour palette ──────────────────────────────────────────────
C_WHITE      = colors.HexColor("#FFFFFF")
C_BG_LIGHT   = colors.HexColor("#F8F9FA")   # very light grey for alternating rows
C_BG_HEADER  = colors.HexColor("#1E293B")   # dark slate for table headers
C_SURFACE    = colors.HexColor("#F1F5F9")   # card background
C_BORDER     = colors.HexColor("#CBD5E1")   # subtle border
C_BORDER_MED = colors.HexColor("#94A3B8")

C_TEXT       = colors.HexColor("#0F172A")   # near black — main text
C_TEXT_SOFT  = colors.HexColor("#475569")   # secondary text
C_TEXT_MUTED = colors.HexColor("#94A3B8")   # captions/footer

C_GREEN      = colors.HexColor("#16A34A")   # attentive / good
C_GREEN_BG   = colors.HexColor("#DCFCE7")
C_AMBER      = colors.HexColor("#D97706")   # warning
C_AMBER_BG   = colors.HexColor("#FEF3C7")
C_RED        = colors.HexColor("#DC2626")   # alert / high severity
C_RED_BG     = colors.HexColor("#FEE2E2")
C_BLUE       = colors.HexColor("#1D4ED8")   # accent / header
C_BLUE_LIGHT = colors.HexColor("#EFF6FF")

SEV_HIGH   = C_RED
SEV_MED    = C_AMBER
SEV_LOW    = C_GREEN
SEV_HIGH_BG   = C_RED_BG
SEV_MED_BG    = C_AMBER_BG
SEV_LOW_BG    = C_GREEN_BG


def _score_color(score: float):
    if score >= 70:  return C_GREEN
    elif score >= 45: return C_AMBER
    return C_RED

def _score_bg(score: float):
    if score >= 70:  return C_GREEN_BG
    elif score >= 45: return C_AMBER_BG
    return C_RED_BG


def _timeline_chart(timeline: list, width=460, height=72) -> Drawing:
    """Clean, readable strip chart — white bg, colour fill, crisp lines."""
    d = Drawing(width, height)
    # White background with border
    d.add(Rect(0, 0, width, height, fillColor=C_WHITE, strokeColor=C_BORDER, strokeWidth=0.8))

    if not timeline or len(timeline) < 2:
        d.add(String(width/2 - 40, height/2, "No data", fontSize=8,
                     fillColor=C_TEXT_MUTED, fontName="Helvetica"))
        return d

    times = [t[0] for t in timeline]
    vals  = [1.0 if t[1] else 0.0 for t in timeline]
    t_max = max(times) or 1

    bar_top    = height - 20
    bar_bottom = 16
    bar_h      = bar_top - bar_bottom
    step       = width / max(len(times), 1)

    # Fill bands for each frame
    for i in range(len(times) - 1):
        x0 = i * step
        x1 = (i + 1) * step
        fill = C_GREEN_BG if vals[i] else C_RED_BG
        d.add(Rect(x0, bar_bottom, x1 - x0, bar_h, fillColor=fill, strokeColor=None))

    # Horizontal midline
    mid_y = bar_bottom + bar_h / 2
    d.add(Line(0, mid_y, width, mid_y,
               strokeColor=C_BORDER, strokeWidth=0.4, strokeDashArray=[2, 4]))

    # Stepped state line on top
    for i in range(len(times) - 1):
        x0 = i * step
        x1 = (i + 1) * step
        y0 = bar_top - 4 if vals[i] else bar_bottom + 4
        y1 = bar_top - 4 if vals[i + 1] else bar_bottom + 4
        col = C_GREEN if vals[i] else C_RED
        d.add(Line(x0, y0, x1, y0, strokeColor=col, strokeWidth=1.8))
        if vals[i] != vals[i + 1]:
            d.add(Line(x1, y0, x1, y1, strokeColor=C_BORDER_MED, strokeWidth=0.8))

    # Time axis labels
    label_y = 3
    d.add(String(3, label_y, "0:00", fontSize=7, fillColor=C_TEXT_MUTED, fontName="Helvetica"))
    mid_s = int(t_max / 2)
    mid_label = f"{mid_s // 60}:{mid_s % 60:02d}"
    d.add(String(width/2 - 10, label_y, mid_label, fontSize=7, fillColor=C_TEXT_MUTED, fontName="Helvetica"))
    end_s = int(t_max)
    end_label = f"{end_s // 60}:{end_s % 60:02d}"
    d.add(String(width - 28, label_y, end_label, fontSize=7, fillColor=C_TEXT_MUTED, fontName="Helvetica"))

    # Legend labels inside the chart
    d.add(String(4, bar_top + 3, "Attentive", fontSize=6, fillColor=C_GREEN, fontName="Helvetica-Bold"))
    d.add(String(4, bar_bottom - 10 + 3, "Distracted", fontSize=6, fillColor=C_RED, fontName="Helvetica-Bold"))

    return d


def _make_styles():
    title_style = ParagraphStyle(
        "CVTitle", fontName="Helvetica-Bold", fontSize=20,
        textColor=C_TEXT, spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "CVSub", fontName="Helvetica", fontSize=10,
        textColor=C_TEXT_SOFT, spaceAfter=16,
    )
    section_style = ParagraphStyle(
        "CVSection", fontName="Helvetica-Bold", fontSize=11,
        textColor=C_BLUE, spaceBefore=20, spaceAfter=8,
        borderPad=0,
    )
    body_style = ParagraphStyle(
        "CVBody", fontName="Helvetica", fontSize=9,
        textColor=C_TEXT, leading=14,
    )
    muted_style = ParagraphStyle(
        "CVMuted", fontName="Helvetica", fontSize=8,
        textColor=C_TEXT_MUTED, leading=12,
    )
    label_style = ParagraphStyle(
        "CVLabel", fontName="Helvetica", fontSize=8,
        textColor=C_TEXT_SOFT, leading=11,
    )
    # Used specifically for header cells on the dark navy (C_BG_HEADER) background —
    # needs white text since Table's TEXTCOLOR command doesn't override Paragraph cells.
    th_style = ParagraphStyle(
        "CVTableHeader", fontName="Helvetica-Bold", fontSize=8,
        textColor=C_WHITE, leading=11,
    )
    return title_style, sub_style, section_style, body_style, muted_style, label_style, th_style


def _stat_table(rows: list) -> Table:
    """Render key/value pairs as a clean two-column stats grid."""
    tbl = Table(rows, colWidths=[130, 110],
                style=TableStyle([
                    ("FONTNAME",   (0, 0), (0, -1), "Helvetica"),
                    ("FONTNAME",   (1, 0), (1, -1), "Helvetica-Bold"),
                    ("FONTSIZE",   (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR",  (0, 0), (0, -1), C_TEXT_SOFT),
                    ("TEXTCOLOR",  (1, 0), (1, -1), C_TEXT),
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_WHITE, C_BG_LIGHT]),
                    ("GRID",       (0, 0), (-1, -1), 0.4, C_BORDER),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING",   (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
                ]))
    return tbl


def generate_attention_pdf(summary: dict, session_name: str = "Session") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2.2*cm,
    )

    title_style, sub_style, section_style, body_style, muted_style, label_style, th_style = _make_styles()
    story = []
    score = summary.get("attentiveness_score", 0)

    # ── Header bar ──────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("<font color='#FFFFFF'><b>ClassPulse</b></font>", body_style),
        Paragraph(f"<font color='#FFFFFF'>Attentiveness Report</font>", muted_style),
        Paragraph(f"<font color='#FFFFFF'>{datetime.now().strftime('%d %b %Y, %H:%M')}</font>", muted_style),
    ]]
    header_tbl = Table(header_data, colWidths=[120, 200, 130],
                       style=TableStyle([
                           ("BACKGROUND", (0,0), (-1,-1), C_BG_HEADER),
                           ("PADDING", (0,0), (-1,-1), 10),
                           ("ALIGN", (2,0), (2,0), "RIGHT"),
                           ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                       ]))
    story.append(header_tbl)
    story.append(Spacer(1, 16))

    # ── Score hero + stats ─────────────────────────────────────────────────
    score_color = _score_color(score)
    score_bg    = _score_bg(score)
    score_label = "Good" if score >= 70 else "Average" if score >= 45 else "Low"

    score_cell = Table(
        [[Paragraph(f"<font size='40' color='{score_color.hexval()}'><b>{score}%</b></font>", body_style)],
         [Paragraph(f"<font size='9' color='{score_color.hexval()}'>{score_label} attentiveness</font>", body_style)]],
        style=TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), score_bg),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 14),
            ("BOTTOMPADDING", (0,0), (-1,-1), 14),
            ("BOX", (0,0), (-1,-1), 1, score_color),
        ])
    )

    stats_tbl = _stat_table([
        ["Duration",          f"{summary.get('duration_minutes', 0):.1f} min"],
        ["Total Frames",      f"{summary.get('total_frames', 0):,}"],
        ["Attentive Frames",  f"{summary.get('attentive_frames', 0):,}"],
        ["Distracted Frames", f"{summary.get('distracted_frames', 0):,}"],
        ["Students Detected", f"{summary.get('num_students', 0)}"],
    ])

    hero = Table([[score_cell, Spacer(8,1), stats_tbl]],
                 colWidths=[160, 12, 260],
                 style=TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
    story.append(hero)
    story.append(Spacer(1, 18))

    # ── Event summary ──────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    story.append(Paragraph("Event Summary", section_style))

    sev_styles = {
        "High":   (SEV_HIGH,   SEV_HIGH_BG),
        "Medium": (SEV_MED,    SEV_MED_BG),
        "Low":    (SEV_LOW,    SEV_LOW_BG),
    }

    events_data = [
        [Paragraph("Event", th_style),
         Paragraph("Count", th_style),
         Paragraph("Severity", th_style)],
    ]
    event_rows = [
        ("Eyes Closed / Drowsy", summary.get("eyes_closed_events", 0), "High"),
        ("Head Turned Away",     summary.get("head_turned_events", 0), "Medium"),
        ("Head Down / Up",       summary.get("head_down_events",   0), "Medium"),
        ("No Face Detected",     summary.get("no_face_events",     0), "Low"),
    ]
    for name, count, sev in event_rows:
        col, bg = sev_styles[sev]
        events_data.append([
            Paragraph(name, body_style),
            Paragraph(str(count), body_style),
            Paragraph(f"<font color='{col.hexval()}'><b>{sev}</b></font>", body_style),
        ])

    events_tbl = Table(events_data, colWidths=[230, 80, 100],
                       style=TableStyle([
                           ("BACKGROUND",    (0, 0), (-1, 0), C_BG_HEADER),
                           ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
                           ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                           ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_BG_LIGHT]),
                           ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
                           ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                           ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                           ("TOPPADDING",    (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                           ("ALIGN",         (1, 0), (1, -1), "CENTER"),
                           ("ALIGN",         (2, 0), (2, -1), "CENTER"),
                       ]))
    story.append(events_tbl)
    story.append(Spacer(1, 18))

    # ── Per-student breakdown ───────────────────────────────────────────────
    students = summary.get("students", [])
    if students:
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
        story.append(Paragraph("Per-Student Breakdown", section_style))

        stu_data = [[
            Paragraph("Student", th_style),
            Paragraph("Attentiveness", th_style),
            Paragraph("Frames Seen", th_style),
            Paragraph("Eye Events", th_style),
            Paragraph("Head Turns", th_style),
            Paragraph("Head Down/Up", th_style),
        ]]
        for stu in students:
            stu_score = stu.get("attentiveness_score", 0)
            stu_color = _score_color(stu_score)
            stu_data.append([
                Paragraph(stu.get("label", "—"), body_style),
                Paragraph(f"<font color='{stu_color.hexval()}'><b>{stu_score}%</b></font>", body_style),
                Paragraph(str(stu.get("frames_seen", 0)), body_style),
                Paragraph(str(stu.get("eyes_closed_events", 0)), body_style),
                Paragraph(str(stu.get("head_turned_events", 0)), body_style),
                Paragraph(str(stu.get("head_down_events", 0)), body_style),
            ])

        stu_tbl = Table(stu_data, colWidths=[60, 80, 75, 70, 75, 80],
                        style=TableStyle([
                            ("BACKGROUND",    (0, 0), (-1, 0), C_BG_HEADER),
                            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
                            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_BG_LIGHT]),
                            ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
                            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                            ("TOPPADDING",    (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
                        ]))
        story.append(stu_tbl)
        story.append(Spacer(1, 18))

    # ── Timeline ───────────────────────────────────────────────────────────
    timeline = summary.get("timeline", [])
    if timeline:
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
        story.append(Paragraph("Attentiveness Timeline", section_style))
        story.append(Paragraph(
            "Green = Attentive  ·  Red = Distracted  ·  Horizontal axis = session time",
            muted_style))
        story.append(Spacer(1, 6))
        chart = _timeline_chart(timeline, width=460, height=72)
        story.append(chart)

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 28))
    story.append(HRFlowable(width="100%", thickness=0.4, color=C_BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated by ClassPulse  ·  Session: {session_name}  ·  {datetime.now().strftime('%d %b %Y, %H:%M')}",
        ParagraphStyle("foot", fontName="Helvetica", fontSize=7,
                       textColor=C_TEXT_MUTED, alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()


def generate_phone_pdf(summary: dict, session_name: str = "Session") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2.2*cm,
    )
    title_style, sub_style, section_style, body_style, muted_style, label_style, th_style = _make_styles()
    story = []

    incidents = summary.get("incident_count", 0)
    rate      = summary.get("detection_rate", 0)
    sev_color = C_RED if incidents > 3 else C_AMBER if incidents > 0 else C_GREEN
    sev_bg    = C_RED_BG if incidents > 3 else C_AMBER_BG if incidents > 0 else C_GREEN_BG
    sev_label = "High Risk" if incidents > 3 else "Moderate" if incidents > 0 else "Clean"

    # ── Header bar ──────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("<font color='#FFFFFF'><b>ClassPulse</b></font>", body_style),
        Paragraph("<font color='#FFFFFF'>Phone Detection Report</font>", muted_style),
        Paragraph(f"<font color='#FFFFFF'>{datetime.now().strftime('%d %b %Y, %H:%M')}</font>", muted_style),
    ]]
    header_tbl = Table(header_data, colWidths=[120, 200, 130],
                       style=TableStyle([
                           ("BACKGROUND", (0,0), (-1,-1), C_BG_HEADER),
                           ("PADDING", (0,0), (-1,-1), 10),
                           ("ALIGN", (2,0), (2,0), "RIGHT"),
                           ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                       ]))
    story.append(header_tbl)
    story.append(Spacer(1, 16))

    # ── Hero ──────────────────────────────────────────────────────────────
    hero_cell = Table(
        [[Paragraph(f"<font size='40' color='{sev_color.hexval()}'><b>{incidents}</b></font>", body_style)],
         [Paragraph(f"<font size='9' color='{sev_color.hexval()}'>{sev_label} · phone incidents</font>", body_style)]],
        style=TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), sev_bg),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 14),
            ("BOTTOMPADDING", (0,0), (-1,-1), 14),
            ("BOX", (0,0), (-1,-1), 1, sev_color),
        ])
    )
    stats_tbl = _stat_table([
        ["Session Duration",   f"{summary.get('duration_minutes', 0):.1f} min"],
        ["Detection Rate",     f"{rate}%"],
        ["Frames with Phone",  f"{summary.get('frames_with_phone', 0):,}"],
        ["Detection Backend",  summary.get('backend', 'N/A').upper()],
    ])
    hero = Table([[hero_cell, Spacer(8,1), stats_tbl]],
                 colWidths=[160, 12, 260],
                 style=TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
    story.append(hero)
    story.append(Spacer(1, 18))

    # ── Incident log ──────────────────────────────────────────────────────
    incident_list = summary.get("incidents", [])
    if incident_list:
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
        story.append(Paragraph("Incident Log", section_style))

        inc_data = [[
            Paragraph("#",          th_style),
            Paragraph("Timestamp",  th_style),
            Paragraph("Confidence", th_style),
            Paragraph("Severity",   th_style),
        ]]
        for idx, (ts, conf) in enumerate(incident_list, 1):
            m, s = int(ts // 60), int(ts % 60)
            sev = "High" if conf > 0.75 else "Medium" if conf > 0.5 else "Low"
            sc, _ = {"High": (SEV_HIGH, None), "Medium": (SEV_MED, None), "Low": (SEV_LOW, None)}[sev]
            inc_data.append([
                Paragraph(str(idx), body_style),
                Paragraph(f"{m:02d}:{s:02d}", body_style),
                Paragraph(f"{int(conf*100)}%", body_style),
                Paragraph(f"<font color='{sc.hexval()}'><b>{sev}</b></font>", body_style),
            ])
        inc_tbl = Table(inc_data, colWidths=[40, 130, 100, 100],
                        style=TableStyle([
                            ("BACKGROUND",    (0, 0), (-1, 0), C_BG_HEADER),
                            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
                            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_BG_LIGHT]),
                            ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
                            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                            ("TOPPADDING",    (0, 0), (-1, -1), 7),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ]))
        story.append(inc_tbl)

    # ── Timeline ──────────────────────────────────────────────────────────
    timeline = summary.get("timeline", [])
    if timeline:
        story.append(Spacer(1, 18))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
        story.append(Paragraph("Detection Timeline", section_style))
        story.append(Paragraph(
            "Green = No phone detected  ·  Red = Phone visible  ·  Horizontal axis = session time",
            muted_style))
        story.append(Spacer(1, 6))
        chart = _timeline_chart(timeline, width=460, height=72)
        story.append(chart)

    story.append(Spacer(1, 28))
    story.append(HRFlowable(width="100%", thickness=0.4, color=C_BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated by ClassPulse  ·  Session: {session_name}  ·  {datetime.now().strftime('%d %b %Y, %H:%M')}",
        ParagraphStyle("foot", fontName="Helvetica", fontSize=7,
                       textColor=C_TEXT_MUTED, alignment=TA_CENTER)
    ))
    doc.build(story)
    return buf.getvalue()


def summary_to_csv(summary: dict, mode: str = "attention") -> str:
    """Return clean, well-structured CSV string from a session summary dict."""
    buf = io.StringIO()
    w = csv.writer(buf)

    if mode == "attention":
        # Section header
        w.writerow(["ClassPulse Attentiveness Report"])
        w.writerow(["Generated", datetime.now().strftime("%d %b %Y %H:%M")])
        w.writerow([])
        # Summary block
        w.writerow(["SUMMARY"])
        w.writerow(["Metric", "Value", "Unit"])
        w.writerow(["Duration",              summary.get("duration_minutes", 0),      "minutes"])
        w.writerow(["Attentiveness Score",   summary.get("attentiveness_score", 0),   "%"])
        w.writerow(["Total Frames",          summary.get("total_frames", 0),          "frames"])
        w.writerow(["Attentive Frames",      summary.get("attentive_frames", 0),      "frames"])
        w.writerow(["Distracted Frames",     summary.get("distracted_frames", 0),     "frames"])
        w.writerow(["Students Detected",     summary.get("num_students", 0),          "students"])
        w.writerow([])
        # Events block
        w.writerow(["EVENTS"])
        w.writerow(["Event Type", "Count", "Severity"])
        w.writerow(["Eyes Closed / Drowsy",  summary.get("eyes_closed_events", 0),   "High"])
        w.writerow(["Head Turned Away",      summary.get("head_turned_events", 0),    "Medium"])
        w.writerow(["Head Down / Up",        summary.get("head_down_events", 0),      "Medium"])
        w.writerow(["No Face Detected",      summary.get("no_face_events", 0),        "Low"])
        w.writerow([])
        # Per-student block
        students = summary.get("students", [])
        if students:
            w.writerow(["PER-STUDENT BREAKDOWN"])
            w.writerow(["Student", "Attentiveness %", "Frames Seen", "Eye Events", "Head Turns", "Head Down/Up"])
            for stu in students:
                w.writerow([
                    stu.get("label", ""),
                    stu.get("attentiveness_score", 0),
                    stu.get("frames_seen", 0),
                    stu.get("eyes_closed_events", 0),
                    stu.get("head_turned_events", 0),
                    stu.get("head_down_events", 0),
                ])
            w.writerow([])
        # Timeline block
        w.writerow(["TIMELINE"])
        w.writerow(["Timestamp (s)", "Status"])
        for ts, att in summary.get("timeline", []):
            w.writerow([round(ts, 2), "Attentive" if att else "Distracted"])
    else:
        w.writerow(["ClassPulse Phone Detection Report"])
        w.writerow(["Generated", datetime.now().strftime("%d %b %Y %H:%M")])
        w.writerow([])
        w.writerow(["SUMMARY"])
        w.writerow(["Metric", "Value", "Unit"])
        w.writerow(["Duration",          summary.get("duration_minutes", 0),  "minutes"])
        w.writerow(["Detection Rate",    summary.get("detection_rate", 0),    "%"])
        w.writerow(["Total Frames",      summary.get("total_frames", 0),      "frames"])
        w.writerow(["Frames with Phone", summary.get("frames_with_phone", 0), "frames"])
        w.writerow(["Incident Count",    summary.get("incident_count", 0),    "incidents"])
        w.writerow([])
        w.writerow(["INCIDENTS"])
        w.writerow(["Incident #", "Timestamp (s)", "Confidence (%)", "Severity"])
        for i, (ts, conf) in enumerate(summary.get("incidents", []), 1):
            sev = "High" if conf > 0.75 else "Medium" if conf > 0.5 else "Low"
            w.writerow([i, round(ts, 2), round(conf * 100, 1), sev])

    return buf.getvalue()