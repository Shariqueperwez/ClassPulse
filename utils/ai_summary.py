"""
Optional AI narrative layer — Google Gemini (free tier) with Groq as a fallback.

Entirely additive: with no key configured, every public function fails
politely and the rest of the app works exactly as before.

Setup (Gemini, free — no billing needed):
    1. Create a key at https://aistudio.google.com/apikey
    2. Put it in .streamlit/secrets.toml:   GEMINI_API_KEY = "AIza..."
       (or set the GEMINI_API_KEY environment variable)
    3. Restart the app. "Generate Summary" works on the Session Reports page.

Only aggregate numbers (scores, counts, durations) are sent — never video,
frames or images. Uses the plain REST API, so no extra package is required.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Optional

# Tried in order until one works. "-latest" aliases follow Google's newest
# Flash models, so the app keeps working when specific versions are retired.
_DEFAULT_GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


class _ModelNotFound(Exception):
    pass


def _secret(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    try:
        import streamlit as st
        for n in names:
            v = st.secrets.get(n)
            if v:
                return str(v)
    except Exception:
        pass
    return None


def _gemini_key() -> Optional[str]:
    return _secret("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _groq_key() -> Optional[str]:
    return _secret("GROQ_API_KEY")


def provider() -> Optional[str]:
    """'gemini', 'groq', or None when nothing is configured."""
    if _gemini_key():
        return "gemini"
    if _groq_key():
        return "groq"
    return None


def is_configured() -> bool:
    return provider() is not None


def _gemini_models() -> list:
    custom = _secret("GEMINI_MODEL")
    return ([custom] if custom else []) + _DEFAULT_GEMINI_MODELS


def _call_gemini(prompt: str, key: str, model: str) -> str:
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1500},
    }).encode("utf-8")
    req = urllib.request.Request(
        _GEMINI_URL.format(model=model), data=body, method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            msg = ""
        if e.code == 404:
            raise _ModelNotFound(msg or model)
        if e.code == 429:
            raise RuntimeError("Gemini free-tier rate limit reached. Wait a minute and try again.")
        if e.code in (400, 401, 403):
            raise RuntimeError(f"Gemini rejected the request ({e.code}). Check your GEMINI_API_KEY. {msg}".strip())
        raise RuntimeError(f"Gemini error {e.code}. {msg}".strip())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Gemini (check your internet connection): {e.reason}")

    cands = data.get("candidates") or []
    if not cands:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
        raise RuntimeError(f"Gemini returned no summary ({reason}).")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty summary. Try again.")
    return text


def _generate_gemini(prompt: str) -> str:
    key = _gemini_key()
    last = ""
    for model in _gemini_models():
        try:
            return _call_gemini(prompt, key, model)
        except _ModelNotFound as e:
            last = str(e)
            continue
    raise RuntimeError(f"No available Gemini model worked. Last error: {last}")


def _generate_groq(prompt: str) -> str:
    try:
        from groq import Groq  # optional package
    except ImportError:
        raise RuntimeError("The 'groq' package isn't installed. Run: pip install groq")
    try:
        resp = Groq(api_key=_groq_key()).chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Groq request failed: {e}")


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
        f"No-face events (face absent for 1s+): {summary.get('no_face_events', 0)}\n"
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
            "No AI key found. Add GEMINI_API_KEY to .streamlit/secrets.toml "
            "(free key: https://aistudio.google.com/apikey)."
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

    if provider() == "gemini":
        return _generate_gemini(prompt)
    return _generate_groq(prompt)
