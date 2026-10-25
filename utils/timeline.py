"""Small helper to keep session timelines compact."""


def bucket_timeline(timeline, bucket_s: float = 1.0, any_mode: bool = False):
    """
    Collapse a per-frame timeline [(ts, bool), ...] into one point per
    `bucket_s` seconds so long sessions stay light in the PDF, CSV and charts.

    any_mode=False -> bucket is True when at least half the frames are True
    any_mode=True  -> bucket is True if ANY frame in it is True (phone events)
    """
    if not timeline:
        return []
    buckets = {}
    for ts, val in timeline:
        k = int(ts // bucket_s)
        b = buckets.setdefault(k, [0, 0])
        b[0] += 1 if val else 0
        b[1] += 1
    out = []
    for k in sorted(buckets):
        true_n, total = buckets[k]
        v = true_n > 0 if any_mode else true_n * 2 >= total
        out.append((round(k * bucket_s, 3), bool(v)))
    return out


def fmt_mmss(seconds: float) -> str:
    """72.4 -> '1:12'"""
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"
