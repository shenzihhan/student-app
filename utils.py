from datetime import datetime
import os
import time
import cv2
import requests
from deepface import DeepFace

SAVE_DIR = "frames"
API_ENDPOINT = "https://student-api-emk4.onrender.com/upload"
os.makedirs(SAVE_DIR, exist_ok=True)

def analyze_emotion_and_upload(record_seconds=30, frame_interval=5):
    emotion_counts = {}
    start_time = time.time()
    frame_id = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        elapsed = time.time() - start_time
        if elapsed >= record_seconds:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        if int(elapsed) % frame_interval == 0:
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
            time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

    payload = {
        "student": "student_001",
        "timestamp": datetime.now().isoformat(),
        "emotions": emotion_counts,
    }
    try:
        res = requests.post(API_ENDPOINT, json=payload)
        if res.status_code == 200:
            print("Upload successful.")
        else:
            print(f"Upload failed: {res.status_code}")
    except Exception as e:
        print(f"Failed to connect to API: {e}")
