"""
Strip-chart renderer — the signature visual element of ClassPulse.

Renders a recent window of boolean readings (attentive/distracted, or
phone-detected/clear) as a stepped trace, in the style of a seismograph or
EKG strip rather than a smooth analytics line chart. This is rendered live
under the camera feed during a session, not just in the post-session
report, so it doubles as the at-a-glance instrument reading.
"""

from utils import theme


def render_strip(values: list, width: int = 760, height: int = 64,
                  good_color: str = None, bad_color: str = None) -> str:
    """
    values: list of bool, most recent last. True = good/attentive/clear.
    Returns raw SVG markup (string) ready for st.markdown(..., unsafe_allow_html=True).
    """
    good_color = good_color or theme.CHALK
    bad_color  = bad_color or theme.PEN_RED

    if not values:
        return (
            f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}" '
            f'stroke="{theme.RULE}" stroke-width="1"/></svg>'
        )

    n = len(values)
    step = width / max(n - 1, 1)
    top, bot = height * 0.22, height * 0.78

    pts = []
    for i, v in enumerate(values):
        x = i * step
        y = top if v else bot
        pts.append((x, y))

    # Build a stepped polyline (horizontal-then-vertical) so transitions
    # look like instrument readings, not a smoothed curve.
    path_d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f} "
    for i in range(1, len(pts)):
        x_prev, y_prev = pts[i - 1]
        x_cur, y_cur = pts[i]
        path_d += f"L {x_cur:.1f} {y_prev:.1f} L {x_cur:.1f} {y_cur:.1f} "

    # Color segments individually so a run of "bad" reads in red, "good" in
    # chalk-green, rather than one flat color for the whole trace.
    segments = []
    seg_start = 0
    for i in range(1, n + 1):
        if i == n or values[i] != values[seg_start]:
            x0 = seg_start * step
            x1 = (i - 1) * step if i - 1 > seg_start else x0 + step * 0.001
            y = top if values[seg_start] else bot
            color = good_color if values[seg_start] else bad_color
            segments.append((x0, x1, y, color))
            seg_start = i

    seg_markup = "".join(
        f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
        f'stroke="{color}" stroke-width="2"/>'
        for x0, x1, y, color in segments
    )
    # vertical connectors at transitions
    vert_markup = "".join(
        f'<line x1="{pts[i][0]:.1f}" y1="{top:.1f}" x2="{pts[i][0]:.1f}" y2="{bot:.1f}" '
        f'stroke="{theme.RULE_DARK}" stroke-width="1"/>'
        for i in range(1, n) if values[i] != values[i - 1]
    )

    baseline_good = f'<line x1="0" y1="{top:.1f}" x2="{width}" y2="{top:.1f}" stroke="{theme.RULE}" stroke-width="0.5" stroke-dasharray="2,3"/>'
    baseline_bad  = f'<line x1="0" y1="{bot:.1f}" x2="{width}" y2="{bot:.1f}" stroke="{theme.RULE}" stroke-width="0.5" stroke-dasharray="2,3"/>'

    return (
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
        f'{baseline_good}{baseline_bad}{vert_markup}{seg_markup}'
        f'</svg>'
    )
