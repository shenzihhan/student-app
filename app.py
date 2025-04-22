import streamlit as st
from utils import analyze_emotion_and_upload

st.set_page_config(page_title="Student Emotion App", layout="centered")
st.title("Emotion Detection - Student")

st.markdown("Please turn on your camera. The system will analyze your emotion and upload the result.")

if st.button("Start Emotion Analysis"):
    analyze_emotion_and_upload()
    st.success("Analysis complete and sent to server.")
