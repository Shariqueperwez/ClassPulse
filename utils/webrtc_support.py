"""
Optional live-webcam support (streamlit-webrtc).

Live WebRTC works when you run the app on your own machine. Some hosts
(e.g. Streamlit Community Cloud) block it, so the pages import it through
this module: if it is missing or broken the app still runs, using
"Webcam (snapshots)" and "Upload video" instead.
"""
import os

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    HAS_WEBRTC = True
except Exception:  # not installed, or platform can't load it
    av = None
    webrtc_streamer = None
    WebRtcMode = None
    HAS_WEBRTC = False

    class VideoProcessorBase:  # stub so page modules can still define processors
        pass


def _secret(name: str):
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _build_rtc_configuration() -> dict:
    ice = [{"urls": ["stun:stun.l.google.com:19302"]}]
    # Optional TURN relay for restrictive networks. Provide your own via
    # env vars or .streamlit/secrets.toml (TURN_URL, TURN_USERNAME, TURN_CREDENTIAL).
    url = _secret("TURN_URL")
    if url:
        ice.append({
            "urls": [url],
            "username": _secret("TURN_USERNAME") or "",
            "credential": _secret("TURN_CREDENTIAL") or "",
        })
    return {"iceServers": ice}


RTC_CONFIGURATION = _build_rtc_configuration()
