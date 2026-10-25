"""Synthetic session data so the report screens can be explored without a camera."""
import random


def make_demo_summaries(seed: int = 7):
    rng = random.Random(seed)
    secs = 12 * 60                      # a 12-minute demo lesson
    fps = 10

    # Attentiveness: mostly focused, with a dip mid-lesson
    timeline = []
    for t in range(secs):
        base = 0.92 if not (360 <= t <= 480) else 0.55
        timeline.append((float(t), rng.random() < base))
    attentive_s = sum(1 for _, v in timeline if v)

    names = ["S01", "S02", "S03", "S04", "S05"]
    scores = [93.4, 88.1, 81.7, 72.5, 90.2]
    students = []
    for i, (lab, sc) in enumerate(zip(names, scores), start=1):
        students.append({
            "student_id": i, "label": lab,
            "frames_seen": secs * fps - rng.randint(0, 300),
            "attentiveness_score": sc,
            "eyes_closed_events": rng.randint(0, 4),
            "head_turned_events": rng.randint(2, 11),
            "head_down_events": rng.randint(1, 8),
        })

    attention = {
        "is_demo": True,
        "duration_minutes": secs / 60,
        "total_frames": secs * fps,
        "attentiveness_score": round(attentive_s / secs * 100, 1),
        "attentive_frames": attentive_s * fps,
        "distracted_frames": (secs - attentive_s) * fps,
        "eyes_closed_events": sum(s["eyes_closed_events"] for s in students),
        "head_turned_events": sum(s["head_turned_events"] for s in students),
        "head_down_events": sum(s["head_down_events"] for s in students),
        "no_face_events": 3,
        "no_face_frames": 140,
        "timeline": timeline,
        "num_students": len(students),
        "students": students,
    }

    # Phone: sightings attributed to specific students
    sightings = [(95, 8, 2), (262, 14, 2), (415, 6, 4), (520, 11, 1)]   # (start_s, seconds, student id)
    ptl = [(float(t_), any(s <= t_ < s + d for s, d, _ in sightings)) for t_ in range(secs)]
    log, per = [], {i: [] for i in range(1, 6)}
    for start, dur, sid in sightings:
        conf = round(rng.uniform(0.55, 0.88), 2)
        log.append({"student": f"S{sid:02d}", "start": float(start), "end": float(start + dur),
                    "seconds": float(dur), "confidence": conf})
        per[sid].append((float(start), float(start + dur), conf))
    pstudents = []
    for sid in range(1, 6):
        incs = per[sid]
        pstudents.append({
            "student_id": sid, "label": f"S{sid:02d}", "first_seen": 0.0, "last_seen": float(secs - 1),
            "frames_seen": secs * fps - rng.randint(0, 300), "phone_frames": int(sum(i[1] - i[0] for i in incs) * fps),
            "incident_count": len(incs), "phone_seconds": float(sum(i[1] - i[0] for i in incs)), "incidents": incs,
        })
    frames_with_phone = sum(1 for _, v in ptl if v) * fps
    phone = {
        "is_demo": True,
        "duration_minutes": secs / 60,
        "total_frames": secs * fps,
        "detection_rate": round(frames_with_phone / (secs * fps) * 100, 1),
        "frames_with_phone": frames_with_phone,
        "incident_count": len(log),
        "incidents": [(r["start"], r["confidence"]) for r in log],
        "incident_log": log,
        "students": pstudents,
        "num_students": len(pstudents),
        "unattributed_incidents": 0,
        "timeline": ptl,
        "backend": "yolov8",
        "identity_backend": "ready",
    }
    return attention, phone
