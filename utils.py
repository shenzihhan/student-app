import cv2
import os
import time
import requests
import numpy as np
from deepface import DeepFace
from PIL import Image

SAVE_DIR = "frames"
API_ENDPOINT = "https://your-render-api-url.com/upload" 
os.makedirs(SAVE_DIR, exist_ok=True)

def analyze_emotion_and_upload():
    emotion_counts = {}
    capture_duration = 30  # seconds
    interval = 5  # seconds
    start_time = time.time()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_id = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed >= capture_duration:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        if int(elapsed) % interval == 0:
            frame_path = os.path.join(SAVE_DIR, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)

            try:
                result = DeepFace.analyze(frame_path, actions=['emotion'], enforce_detection=False)
                dominant = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
                emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
                print(f"Frame {frame_id}: {dominant}")
            except Exception as e:
                print(f"Emotion analysis failed: {e}")

            frame_id += 1
            time.sleep(1)  # slight buffer to avoid double captures

    cap.release()
    cv2.destroyAllWindows()

    # Upload to API
    payload = {"student_id": "student_001", "emotions": emotion_counts}
    try:
        res = requests.post(API_ENDPOINT, json=payload)
        if res.status_code == 200:
            print("Upload successful.")
        else:
            print(f"Upload failed: {res.status_code}")
    except Exception as e:
        print(f"Failed to connect to API: {e}")
