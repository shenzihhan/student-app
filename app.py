import streamlit as st
from utils import analyze_emotion_and_upload

st.set_page_config(page_title="Student Emotion App", layout="centered")
st.title("Emotion Detection - Student")

st.markdown("Please turn on your camera. The system will record for 30 seconds and analyze one frame every 5 seconds.")

if st.button("Start Emotion Analysis"):
    analyze_emotion_and_upload(record_seconds=30, frame_interval=5)
    st.success("Analysis complete and sent to server.")
