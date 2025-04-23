import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import time
import numpy as np
from deepface import DeepFace
import requests

# ========== API 目標 ==========
API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"

# ========== WebRTC 設定 ==========
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# ========== 視訊處理器 ==========
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

# ========== 分析情緒 ==========
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

# ========== 傳送到 API ==========
def upload_to_api(emotion_results):
    try:
        res = requests.post(API_ENDPOINT, json={"emotions": emotion_results})
        return res.status_code == 200
    except:
        return False

# ========== Streamlit 介面 ==========
st.set_page_config(page_title="Emotion Detection - Student")
st.title("Emotion Detection - Student")
st.markdown("The system will use your webcam to analyze your emotion over 30 seconds. Please stay visible on camera.")

# 初始化狀態變數
if "start" not in st.session_state:
    st.session_state.start = False
if "processor" not in st.session_state:
    st.session_state.processor = EmotionProcessor()

# 按鈕觸發
if st.button("Start Emotion Analysis") and not st.session_state.start:
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

    # 分析並上傳結果
    countdown.empty()
    status.markdown("⏹️ Stopping recording and analyzing...")

    with st.spinner("Analyzing emotions..."):
        frames = st.session_state.processor.frames
        emotions = analyze_emotions(frames)
        success = upload_to_api(emotions)

        if success:
            st.success("✅ Emotions uploaded successfully!")
        else:
            st.error("❌ Failed to upload emotions.")
