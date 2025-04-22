import cv2
import os
import time
import requests
from deepface import DeepFace

API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"

def analyze_emotion_and_upload():
    duration = 30
    interval = 5
    start_time = time.time()
    emotions = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
            emotions[dominant] = emotions.get(dominant, 0) + 1
            print(f"Detected: {dominant}")
        except Exception as e:
            print("Error:", e)

        time.sleep(interval)

    cap.release()
    requests.post(API_ENDPOINT, json={"student": "student_001", "emotions": emotions})
