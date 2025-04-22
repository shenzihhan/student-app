import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2
import numpy as np
import time
import requests
from deepface import DeepFace

API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"

# RTC Config to avoid ICE/stun errors
rtc_config = RTCConfiguration({"iceServers": []})

# Store emotions to upload later
emotion_counts = {}
record_seconds = 30
frame_interval = 5

class EmotionAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.start_time = time.time()
        self.last_capture_time = 0
        self.frame_count = 0

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        global emotion_counts
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # Capture and analyze every frame_interval seconds
        if now - self.last_capture_time > frame_interval:
            self.last_capture_time = now
            try:
                result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                dominant = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
                emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
                print(f"Captured emotion: {dominant}")
            except Exception as e:
                print("Analysis error:", e)

        # Stop after record_seconds
        if now - self.start_time > record_seconds:
            st.session_state.finished = True

        return img

st.set_page_config(page_title="Student Emotion App", layout="centered")
st.title("Emotion Detection - Student")

st.markdown("The system will use your webcam to analyze your emotion over 30 seconds. Please stay visible on camera.")

if "finished" not in st.session_state:
    st.session_state.finished = False

if st.button("Start Emotion Analysis"):
    emotion_counts = {}  # Reset
    webrtc_streamer(
        key="emotion-demo",
        video_transformer_factory=EmotionAnalyzer,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    with st.spinner("Recording for 30 seconds. Please stay visible..."):
        while not st.session_state.finished:
            time.sleep(1)

    # Upload results
    requests.post(API_ENDPOINT, json={"student": "student_001", "emotions": emotion_counts})
    st.success("Analysis complete and sent to server.")
