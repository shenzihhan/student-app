import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import time
import numpy as np
from deepface import DeepFace
import requests
from datetime import datetime

API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.records = []
        self.last_capture_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        if now - self.last_capture_time >= 5 and len(self.records) < 6:
            attention = self.estimate_attention(img)
            timestamp = datetime.now().isoformat()
            self.records.append({"frame": img, "timestamp": timestamp, "attention": attention})
            self.last_capture_time = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def estimate_attention(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return min(len(faces), 1)  # 0 = inattentive, 1 = attentive

def analyze_and_upload(records):
    payload = []
    for r in records:
        try:
            analysis = DeepFace.analyze(r["frame"], actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
        except:
            emotion = "error"
        payload.append({
            "timestamp": r["timestamp"],
            "attention": r["attention"],
            "emotion": emotion
        })

    try:
        res = requests.post(API_ENDPOINT, json={"student": "student_001", "records": payload})
        return res.status_code == 200
    except:
        return False

# Streamlit UI
st.set_page_config(page_title="Emotion Detection - Student")
st.title("Emotion Detection - Student")
st.markdown("The system will use your webcam to analyze your emotion and attention over 30 seconds.")

if "start" not in st.session_state:
    st.session_state.start = False

if st.button("Start Emotion Analysis") and not st.session_state.start:
    st.session_state.processor = EmotionProcessor()
    st.session_state.start = True
    st.session_state.start_time = time.time()

    webrtc_ctx = webrtc_streamer(
        key="emotion",
        mode="SENDRECV",
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

    with st.spinner("Analyzing..."):
        success = analyze_and_upload(st.session_state.processor.records)
        if success:
            st.success("Uploaded emotion + attention successfully!")
        else:
            st.error("Failed to upload data.")
