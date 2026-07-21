"""
Optional AI narrative layer powered by Groq.

This module is entirely additive: if no GROQ_API_KEY is configured, every
public function returns a clean "not configured" result instead of raising,
so the rest of the app works exactly as before without it.

Setup:
    1. pip install groq
    2. Set the key one of two ways:
       - Environment variable:   export GROQ_API_KEY="gsk_..."
       - Streamlit secrets:      .streamlit/secrets.toml -> GROQ_API_KEY = "gsk_..."
    3. Restart the app. A "Generate AI Summary" button will appear on the
       Session Reports page automatically once a key is detected.
"""

import os
from typing import Optional

_MODEL = "llama-3.3-70b-versatile"


def _get_api_key() -> Optional[str]:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")  # type: ignore[union-attr]
    except Exception:
        return None


def is_configured() -> bool:
    return bool(_get_api_key())


def _client():
    from groq import Groq  # imported lazily so the package is optional
    return Groq(api_key=_get_api_key())


def _format_attention_facts(summary: dict) -> str:
    return (
        f"Duration: {summary.get('duration_minutes', 0)} minutes\n"
        f"Attentiveness score: {summary.get('attentiveness_score', 0)}%\n"
        f"Total frames analysed: {summary.get('total_frames', 0)}\n"
        f"Attentive frames: {summary.get('attentive_frames', 0)}\n"
        f"Distracted frames: {summary.get('distracted_frames', 0)}\n"
        f"Eyes-closed / drowsy events: {summary.get('eyes_closed_events', 0)}\n"
        f"Head-turned-away events: {summary.get('head_turned_events', 0)}\n"
        f"Head-down/up events: {summary.get('head_down_events', 0)}\n"
        f"No-face-detected events: {summary.get('no_face_events', 0)}\n"
        f"Distinct students detected: {summary.get('num_students', 0)}\n"
    )


def _format_phone_facts(summary: dict) -> str:
    return (
        f"Duration: {summary.get('duration_minutes', 0)} minutes\n"
        f"Phone detection rate: {summary.get('detection_rate', 0)}%\n"
        f"Total frames analysed: {summary.get('total_frames', 0)}\n"
        f"Frames containing a phone: {summary.get('frames_with_phone', 0)}\n"
        f"Distinct incidents: {summary.get('incident_count', 0)}\n"
    )


def generate_session_narrative(
    attn_summary: Optional[dict] = None,
    phone_summary: Optional[dict] = None,
) -> str:
    """
    Returns a short, teacher-facing paragraph summarising the session.
    Raises RuntimeError with a human-readable message on failure — callers
    should catch this and display it rather than letting it propagate.
    """
    if not is_configured():
        raise RuntimeError(
            "Groq API key not found. Set GROQ_API_KEY as an environment "
            "variable or in .streamlit/secrets.toml to enable AI summaries."
        )

    facts = []
    if attn_summary:
        facts.append("ATTENTIVENESS DATA:\n" + _format_attention_facts(attn_summary))
    if phone_summary:
        facts.append("PHONE DETECTION DATA:\n" + _format_phone_facts(phone_summary))
    if not facts:
        raise RuntimeError("No session data available to summarise.")

    prompt = (
        "You are an assistant that writes a short, plain-language summary of a "
        "classroom monitoring session for a teacher who is not technical. "
        "Use only the numbers given below — do not invent any. Write 3-5 "
        "sentences, no headers, no bullet points, a neutral and constructive "
        "tone. Call out the most notable pattern (e.g. when distraction was "
        "highest) only if the data supports it.\n\n"
        + "\n\n".join(facts)
    )

    try:
        client = _client()
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        raise RuntimeError(
            "The 'groq' package isn't installed. Run: pip install groq"
        )
    except Exception as e:
        raise RuntimeError(f"Groq request failed: {e}")
