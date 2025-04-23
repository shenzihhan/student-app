from datetime import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import time
import numpy as np
from deepface import DeepFace
import requests
from streamlit_extras.add_vertical_space import add_vertical_space

# ========== API Settings ==========
API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# ========== Session State Setup ==========
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "recording" not in st.session_state:
    st.session_state.recording = False
if "frames" not in st.session_state:
    st.session_state.frames = []
if "done" not in st.session_state:
    st.session_state.done = False
if "result" not in st.session_state:
    st.session_state.result = None

# ========== Emotion Processor ==========
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_capture_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if st.session_state.recording and len(st.session_state.frames) < 6:
            if now - self.last_capture_time >= 5:
                try:
                    analysis = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
                    result = analysis[0] if isinstance(analysis, list) else analysis
                    emotion = result["dominant_emotion"]
                    region = result.get("region", {})
                    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                    if w > 0 and h > 0:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    st.session_state.frames.append(img)
                except Exception:
                    st.session_state.frames.append(img)
                self.last_capture_time = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========== Analysis Functions ==========
def analyze_emotions_and_attention(frames):
    emotions = []
    attention_scores = []
    for img in frames:
        try:
            analysis = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
            emotion = analysis[0]["dominant_emotion"] if isinstance(analysis, list) else analysis["dominant_emotion"]
            emotions.append(emotion)
            attention_scores.append(1 if emotion in ["happy", "neutral", "surprise"] else 0)
        except:
            emotions.append("error")
            attention_scores.append(0)
    avg_attention = round(np.mean(attention_scores), 2)
    return emotions, avg_attention

def upload_to_api(emotions, attention):
    payload = {
        "student_id": "student_001",
        "emotions": emotions,
        "attention": attention,
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        res = requests.post(API_ENDPOINT, json=payload)
        return res.status_code == 200
    except:
        return False

# ========== Streamlit UI ==========
st.set_page_config(page_title="Emotion Detection - Student")
st.title("Emotion Detection - Student")
st.markdown("Select webcam device below and press **Start Emotion & Attention Analysis**.")

add_vertical_space(1)

webrtc_ctx = webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("▶Start Emotion & Attention Analysis") and webrtc_ctx.state.playing:
    st.session_state.start_time = time.time()
    st.session_state.recording = True
    st.session_state.frames = []
    st.session_state.done = False
    st.session_state.result = None

# ========== Timer & Progress ==========
if st.session_state.recording:
    elapsed = time.time() - st.session_state.start_time
    if elapsed < 30:
        st.progress(elapsed / 30.0, f"⏳ {30 - int(elapsed)} seconds remaining")
    else:
        st.session_state.recording = False
        st.session_state.done = True
        st.success("Recording complete. Analyzing...")

# ========== After Analysis ==========
if st.session_state.done and st.session_state.result is None:
    with st.spinner("Analyzing emotions..."):
        emotions, attention = analyze_emotions_and_attention(st.session_state.frames)
        success = upload_to_api(emotions, attention)
        st.session_state.result = (emotions, attention, success)

if st.session_state.result:
    emotions, attention, success = st.session_state.result
    st.markdown(f"**Average Attention Score:** {attention}")
    st.markdown(f"**Detected Emotions:** {emotions}")
    if success:
        st.success("Uploaded to teacher dashboard.")
    else:
        st.error("Upload failed. Try again.")

