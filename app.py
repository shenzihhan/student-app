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
        self.capture_started = False
        self.start_time = None
        self.emotion_results = []

    def analyze_and_draw(self, img):
        try:
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            result = analysis[0] if isinstance(analysis, list) else analysis
            emotion = result['dominant_emotion']
            region = result.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            if w > 0 and h > 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return img, emotion
        except Exception as e:
            return img, "error"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if self.capture_started:
            if self.start_time is None:
                self.start_time = now

            elapsed = now - self.start_time
            if elapsed <= 30:
                if now - self.last_capture_time >= 5 and len(self.frames) < 6:
                    img, emotion = self.analyze_and_draw(img)
                    self.frames.append(img)
                    self.emotion_results.append(emotion)
                    self.last_capture_time = now
            else:
                self.capture_started = False  # Stop capture after 30s

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========== Emotion & Attention Analysis ==========
def analyze_emotions_and_attention(emotion_list):
    attention_scores = [1 if e in ['happy', 'neutral', 'surprise'] else 0 for e in emotion_list if e != "error"]
    avg_attention = round(np.mean(attention_scores), 2) if attention_scores else 0
    return emotion_list, avg_attention

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
st.markdown("Select your webcam device (if required), then click **Start Emotion & Attention Analysis**.")

# Session state setup
if "processor" not in st.session_state:
    st.session_state.processor = EmotionProcessor()
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False
if "progress" not in st.session_state:
    st.session_state.progress = 0

# Streamlit WebRTC
webrtc_ctx = webrtc_streamer(
    key="emotion-detect",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: st.session_state.processor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# UI Components
progress_bar = st.empty()
status_text = st.empty()

# Start button logic
if st.button("Start Emotion & Attention Analysis") and webrtc_ctx.state.playing:
    st.session_state.processor.frames.clear()
    st.session_state.processor.emotion_results.clear()
    st.session_state.processor.capture_started = True
    st.session_state.processor.start_time = None
    st.session_state.analysis_started = True
    st.session_state.progress = 0

# Progress and post-analysis
if st.session_state.analysis_started and webrtc_ctx.state.playing:
    elapsed = time.time() - (st.session_state.processor.start_time or time.time())
    if elapsed <= 30:
        progress = int((elapsed / 30) * 100)
        st.session_state.progress = progress
        progress_bar.progress(progress)
    else:
        st.session_state.analysis_started = False
        progress_bar.empty()
        status_text.markdown("⏹️ Analyzing captured frames...")

        with st.spinner("Analyzing emotions and attention..."):
            emotions, attention = analyze_emotions_and_attention(st.session_state.processor.emotion_results)
            success = upload_to_api(emotions, attention)

            if success:
                st.success("Emotions and attention uploaded successfully!")
                st.write("Detected emotions:", emotions)
                st.write("Average attention score:", attention)
            else:
                st.error("Failed to upload data.")
