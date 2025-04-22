import streamlit as st
import cv2
import requests
from deepface import DeepFace

st.title("Student Emotion Uploader")

api_url = st.text_input("Enter API endpoint (e.g., http://xxxxxx.ngrok.io/upload)", "")

frame_window = st.image([])
camera = cv2.VideoCapture(0)

if st.button("Capture and Upload"):
    ret, frame = camera.read()
    if ret:
        frame_window.image(frame, channels="BGR")
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            student_id = st.text_input("Your Student ID", "student_001")

            data = {"student_id": student_id, "emotion": emotion}
            res = requests.post(f"{api_url}/upload", json=data)
            st.success(f"Uploaded: {emotion}, Response: {res.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Failed to capture frame.")
