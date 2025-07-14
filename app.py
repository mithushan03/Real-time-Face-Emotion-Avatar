import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("😊 Real-Time Face Emotion Detector")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

if run:
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to read from camera.")
            break

        # Flip and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect emotion
        try:
            result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(rgb_frame, f'Emotion: {emotion}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(rgb_frame, 'No face detected', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display frame
        FRAME_WINDOW.image(rgb_frame)

    camera.release()
else:
    st.info("Turn on the checkbox to start the webcam.")
