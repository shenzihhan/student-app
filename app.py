import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
from deepface import DeepFace
import requests
import cv2

# API Endpoint
API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"

# WebRTC Configuration (STUN + TURN)
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {
            "urls": "turn:openrelay.metered.ca:80",
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
}

# Emotion Video Processor
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

# Emotion Analysis
def analyze_emotions(frames):
    results = []
    for img in frames:
        try:
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            results.append(emotion)
        except:
            results.append("error")
    return results

# API Upload
def upload_to_api(emotion_results):
    data = {"emotions": emotion_results}
    try:
        response = requests.post(API_ENDPOINT, json=data)
        return response.status_code == 200
    except Exception:
        return False

# Page Setup
st.set_page_config(page_title="Emotion Detection - Student")
st.title("Emotion Detection - Student")
st.markdown("The system will use your webcam to analyze your emotion over 30 seconds. Please stay visible on camera.")

# Initialize session states
if "recording" not in st.session_state:
    st.session_state.recording = False
if "processor" not in st.session_state:
    st.session_state.processor = EmotionProcessor()

# Start button
if st.button("Start Emotion Analysis"):
    st.session_state.recording = True
    st.session_state.start_time = time.time()
    st.session_state.processor = EmotionProcessor()

    webrtc_ctx = webrtc_streamer(
        key="emotion",
        video_processor_factory=lambda: st.session_state.processor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True
    )

    status_placeholder = st.empty()
    countdown = st.empty()
    debug_state = st.empty()

    while st.session_state.recording:
        elapsed = time.time() - st.session_state.start_time
        remaining = int(30 - elapsed)
        debug_state.text(f"WebRTC status: {webrtc_ctx.state}")
        if remaining > 0:
            countdown.markdown(f"⏳ Time remaining: **{remaining} seconds**")
            time.sleep(1)
        else:
            st.session_state.recording = False
            countdown.empty()
            status_placeholder.markdown("⏹️ Stopping recording and analyzing...")
            break

    # Analyze and upload
    with st.spinner("Analyzing emotions..."):
        frames = st.session_state.processor.frames
        emotions = analyze_emotions(frames)
        success = upload_to_api(emotions)

        if success:
            st.success("🎉 Emotions uploaded successfully!")
        else:
            st.error("❌ Failed to upload emotions.")
