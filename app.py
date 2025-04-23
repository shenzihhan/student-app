from datetime import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import time
import numpy as np
from deepface import DeepFace
import requests

# API settings
API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Emotion Processor
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.last_capture_time = time.time()
        self.recording = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Only record frames if recording is True
        if self.recording:
            now = time.time()
            if now - self.last_capture_time >= 5 and len(self.frames) < 6:
                try:
                    analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                    result = analysis[0] if isinstance(analysis, list) else analysis
                    emotion = result['dominant_emotion']
                    region = result.get('region', {})
                    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                    if w > 0 and h > 0:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    self.frames.append(img)
                except:
                    self.frames.append(img)
                self.last_capture_time = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Analyze and Upload
def analyze_emotions_and_attention(frames):
    emotions, attention_scores = [], []
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

# ===== Streamlit UI =====
st.set_page_config(page_title="Emotion Detection - Student")
st.title("Emotion Detection - Student")
st.markdown("The system will use your webcam to analyze your emotion and attention over 30 seconds. Please stay visible on camera.")

# Initialize session state
if "start" not in st.session_state:
    st.session_state.start = False
if "start_time" not in st.session_state:
    st.session_state.start_time = 0
if "processor" not in st.session_state:
    st.session_state.processor = EmotionProcessor()

# WebRTC should always be initialized once and reused
webrtc_ctx = webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: st.session_state.processor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# UI
progress_bar = st.empty()
status_text = st.empty()

if st.button("Start Emotion & Attention Analysis"):
    if webrtc_ctx.state.playing:
        st.session_state.processor.frames.clear()
        st.session_state.processor.recording = True
        st.session_state.start_time = time.time()
        st.session_state.start = True

# Record and display progress
if st.session_state.start:
    elapsed = time.time() - st.session_state.start_time
    if elapsed < 30:
        progress_bar.progress(int((elapsed / 30) * 100))
    else:
        st.session_state.start = False
        st.session_state.processor.recording = False
        progress_bar.empty()
        status_text.markdown("⏹️ Stopping recording and analyzing...")

        with st.spinner("Analyzing emotions and attention..."):
            frames = st.session_state.processor.frames
            emotions, attention = analyze_emotions_and_attention(frames)
            success = upload_to_api(emotions, attention)

            if success:
                st.success("Emotions and attention uploaded successfully!")
            else:
                st.error("Failed to upload data.")
