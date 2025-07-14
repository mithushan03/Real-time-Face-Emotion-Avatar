import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import base64
import time

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils
LIPS = mp_face_mesh.FACEMESH_LIPS
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# Emotion settings
emotions = {
    "happy": {"emoji": "😊", "color": (0, 255, 0)},
    "sad": {"emoji": "😢", "color": (255, 0, 0)},
    "angry": {"emoji": "😠", "color": (0, 0, 255)},
    "surprise": {"emoji": "😲", "color": (0, 255, 255)},
    "neutral": {"emoji": "😐", "color": (255, 255, 255)},
    "fear": {"emoji": "😨", "color": (255, 140, 0)},
    "disgust": {"emoji": "🤢", "color": (138, 43, 226)}
}

def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_emotion(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    iris_left = landmarks[468]
    eye_center = landmarks[33]

    face_width = distance(landmarks[234], landmarks[454])

    # Features
    mouth_open = distance(top_lip, bottom_lip) / face_width
    mouth_stretch = distance(left_mouth, right_mouth) / face_width
    eye_open = (distance(left_eye_top, left_eye_bottom) + distance(right_eye_top, right_eye_bottom)) / (2 * face_width)
    
    # SAD: Is iris significantly lower than middle of eyelid?
    eye_top_avg = (left_eye_top.y + right_eye_top.y) / 2
    eye_bottom_avg = (left_eye_bottom.y + right_eye_bottom.y) / 2
    iris_avg = iris_left.y
    eye_center_y = (eye_top_avg + eye_bottom_avg) / 2
    sad_offset = iris_avg - eye_center_y  # How far iris moved down from center

    # Emotion rules (tuned)
    # Emotion rules (tuned for better detection)
    if mouth_stretch > 0.45 and mouth_open < 0.05: # Increased stretch for happy
        return "happy"
    elif mouth_open >= 0.15: # Increased open for surprise
        return "surprise"
    elif 0.07 < mouth_open < 0.15: # Adjusted range for fear
        return "fear"
    elif sad_offset > 0.015 and eye_open < 0.035: # Adjusted offset and eye open for sad
        return "sad"
    elif mouth_open < 0.025 and eye_open < 0.07 and mouth_stretch < 0.35: # Adjusted thresholds for disgust
        return "disgust"
    elif eye_open > 0.1 and mouth_open < 0.05: # Adjusted eye open for angry
        return "angry"
    else:
        return "neutral"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    data = request.get_json()
    img_data = data['image'].split(',')[1] # Remove "data:image/jpeg;base64,"
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    avatar_canvas = np.zeros_like(frame)
    emotion = "neutral"
    mesh_image_base64 = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_list = face_landmarks.landmark
            emotion = get_emotion(landmark_list)
            print(f"Detected emotion: {emotion}") # Add this line for debugging
            color = emotions[emotion]["color"]
            emoji = emotions[emotion]["emoji"]

            # Draw avatar
            drawing_utils.draw_landmarks(
                avatar_canvas, face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION, None,
                drawing_utils.DrawingSpec(color=color, thickness=1, circle_radius=1)
            )
            drawing_utils.draw_landmarks(
                avatar_canvas, face_landmarks, LIPS, None,
                drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            for idx in LEFT_IRIS + RIGHT_IRIS:
                pt = landmark_list[idx]
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(avatar_canvas, (cx, cy), 2, (0, 255, 255), -1)
        
        # Encode mesh image to base64
        _, buffer = cv2.imencode('.jpg', avatar_canvas)
        mesh_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "emotion": emotion,
        "emoji": emotions[emotion]["emoji"],
        "mesh_image": mesh_image_base64
    })


