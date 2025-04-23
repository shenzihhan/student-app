from datetime import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import time
import numpy as np
from deepface import DeepFace
import requests

# ========== API ==========
API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# ========== Video Processor ==========
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.last_capture_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        if now - self.last_capture_time >= 5 and len(self.frames) < 6:
            self.frames.append(img)
            self.last_capture_time = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========== Emotion & Attention Analysis ==========
def analyze_emotions_and_attention(frames):
    emotions = []
    attention_scores = []
    for img in frames:
        try:
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
            emotions.append(emotion)
            attention = 1 if emotion in ['happy', 'neutral', 'surprise'] else 0
            attention_scores.append(attention)
        except:
            emotions.append("error")
            attention_scores.append(0)
    avg_attention = round(np.mean(attention_scores), 2) if attention_scores else 0
    return emotions, avg_attention

# ========== Upload to API ==========
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
st.markdown("The system will use your webcam to analyze your emotion and attention over 30 seconds. Please stay visible on camera.")

if "start" not in st.session_state:
    st.session_state.start = False
if "processor" not in st.session_state:
    st.session_state.processor = EmotionProcessor()

if st.button("Start Emotion & Attention Analysis") and not st.session_state.start:
    st.session_state.start = True
    st.session_state.start_time = time.time()
    
    webrtc_ctx = webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: st.session_state.processor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
    countdown = st.empty()
    status = st.empty()

    while time.time() - st.session_state.start_time < 30:
        remaining = int(30 - (time.time() - st.session_state.start_time))
        countdown.markdown(f"⏳ Time remaining: **{remaining} seconds**")
        time.sleep(1)

    countdown.empty()
    status.markdown("⏹️ Stopping recording and analyzing...")

    with st.spinner("Analyzing emotions and attention..."):
        frames = st.session_state.processor.frames
        emotions, attention = analyze_emotions_and_attention(frames)
        success = upload_to_api(emotions, attention)

        if success:
            st.success("Emotions and attention uploaded successfully!")
        else:
            st.error("Failed to upload data.")
