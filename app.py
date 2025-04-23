from datetime import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import time
import numpy as np
from deepface import DeepFace
import requests
import threading

# API settings
API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Session init
if "start" not in st.session_state:
    st.session_state.start = False
if "frames" not in st.session_state:
    st.session_state.frames = []
if "progress" not in st.session_state:
    st.session_state.progress = 0

# Emotion Processor
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_capture_time = time.time()
        self.capture_frames = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if (
            st.session_state.get("start", False)
            and self.capture_frames
            and len(st.session_state.frames) < 6
            and now - self.last_capture_time >= 5
        ):
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
            except Exception as e:
                st.session_state.frames.append(img)
            self.last_capture_time = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Analyze emotions
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
    avg_attention = round(np.mean(attention_scores), 2) if attention_scores else 0
    return emotions, avg_attention

# Upload to API
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

# Countdown worker
def run_countdown_and_process():
    for i in range(30):
        st.session_state.progress = i + 1
        time.sleep(1)

    st.session_state.start = False
    processor.capture_frames = False
    emotions, attention = analyze_emotions_and_attention(st.session_state.frames)
    success = upload_to_api(emotions, attention)
    st.session_state.result = ("success", emotions, attention) if success else ("fail", [], 0)

# Page setup
st.set_page_config(page_title="Emotion Detection - Student")
st.title("Emotion Detection - Student")
st.markdown("""
🎥 The system will use your webcam to analyze your **emotion and attention** over 30 seconds.  
✅ After selecting webcam, press **Start Emotion & Attention Analysis** button to begin.
""")

# Streamlit component
processor = EmotionProcessor()
webrtc_ctx = webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: processor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# UI elements
progress_bar = st.progress(0, text="⌛ Waiting to start...")
start_button = st.button("▶️ Start Emotion & Attention Analysis")

# On start
if start_button and webrtc_ctx.state.playing:
    st.session_state.frames = []
    st.session_state.progress = 0
    st.session_state.start = True
    processor.capture_frames = True
    threading.Thread(target=run_countdown_and_process, daemon=True).start()

# Show progress
if st.session_state.start:
    progress_bar.progress(st.session_state.progress / 30.0, text=f"⏳ {30 - st.session_state.progress} seconds remaining")

# Show results
if "result" in st.session_state and not st.session_state.start:
    outcome, emotions, attention = st.session_state.result
    if outcome == "success":
        st.success(f"Uploaded successfully! Attention Score: **{attention}**")
        st.write("Detected Emotions:", emotions)
    else:
        st.error("Failed to upload emotion data.")
