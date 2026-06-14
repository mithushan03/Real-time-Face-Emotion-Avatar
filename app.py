import os
import base64
from html import escape

# Reduce TensorFlow console noise. Keep CPU-friendly behavior for Streamlit Cloud.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="EmotiAvatar",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from deepface import DeepFace
except ValueError as exc:
    if "requires tf-keras package" in str(exc):
        st.title("EmotiAvatar")
        st.error(
            "DeepFace could not start because `tf-keras` is missing for the installed "
            "TensorFlow version. Install dependencies again with `pip install -r requirements.txt`."
        )
        st.stop()
    raise
except ModuleNotFoundError:
    st.title("EmotiAvatar")
    st.error(
        "DeepFace is not installed. Install your dependencies with `pip install -r requirements.txt`."
    )
    st.stop()


EMOTION_STYLES = {
    "happy": {"color": "#22c55e", "emoji": "😊", "accent": "Radiant and upbeat"},
    "neutral": {"color": "#3b82f6", "emoji": "😐", "accent": "Steady and composed"},
    "surprise": {"color": "#8b5cf6", "emoji": "😮", "accent": "Alert and reactive"},
    "sad": {"color": "#f59e0b", "emoji": "😢", "accent": "Quiet and reflective"},
    "angry": {"color": "#ef4444", "emoji": "😠", "accent": "High energy and intense"},
    "fear": {"color": "#06b6d4", "emoji": "😨", "accent": "Tense and cautious"},
    "disgust": {"color": "#ec4899", "emoji": "🤢", "accent": "Strong aversion detected"},
}


def inject_ui_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
            --cream: #fff7ed;
            --cream-2: #fff1df;
            --ink: #111827;
            --muted: #64748b;
            --orange: #f97316;
            --orange-dark: #c2410c;
            --navy: #07101d;
            --panel: rgba(255,255,255,0.95);
            --line: rgba(249,115,22,0.18);
            --shadow: 0 22px 55px rgba(120, 72, 20, 0.13);
            --shadow-soft: 0 12px 30px rgba(17, 24, 39, 0.08);
        }

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(251, 191, 36, 0.22), transparent 24%),
                radial-gradient(circle at 95% 18%, rgba(249, 115, 22, 0.13), transparent 28%),
                linear-gradient(180deg, #fffaf4 0%, #fff1df 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1600px;
            padding: 1.15rem 1.4rem 1.1rem 1.4rem;
        }

        #MainMenu,
        footer,
        header[data-testid="stHeader"] {
            visibility: hidden;
            display: none;
        }

        .hero-card {
            position: relative;
            min-height: 230px;
            border-radius: 28px;
            padding: 2.15rem 2.4rem;
            background:
                radial-gradient(circle at 82% 45%, rgba(255,177,95,0.35), transparent 20%),
                linear-gradient(115deg, #07101d 0%, #151c2b 56%, #ff7a1a 100%);
            color: white;
            box-shadow: var(--shadow);
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(360px, 0.85fr);
            align-items: center;
            gap: 1.2rem;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at 88% 18%, rgba(255,255,255,0.12), transparent 16%),
                radial-gradient(circle at 73% 70%, rgba(255,255,255,0.07), transparent 18%);
            pointer-events: none;
        }

        .hero-title {
            position: relative;
            z-index: 2;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: clamp(2.5rem, 4vw, 4.25rem);
            font-weight: 900;
            letter-spacing: -0.07em;
            line-height: 0.95;
        }

        .hero-logo {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: linear-gradient(180deg, #ffe58f, #f59e0b);
            color: #111827;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 2.1rem;
            box-shadow: inset 0 4px 10px rgba(255,255,255,0.35), 0 15px 28px rgba(0,0,0,0.18);
            flex: 0 0 auto;
        }

        .hero-accent {
            color: #ffb454;
        }

        .hero-copy {
            position: relative;
            z-index: 2;
            max-width: 650px;
            color: rgba(255,255,255,0.9);
            font-size: 1.08rem;
            line-height: 1.6;
            margin: 1rem 0 1.25rem 0;
        }

        .hero-chips {
            position: relative;
            z-index: 2;
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.78rem 1rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
            color: rgba(255,255,255,0.96);
            font-size: 0.92rem;
            font-weight: 700;
        }

        .hero-visual {
            position: relative;
            z-index: 2;
            min-height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .avatar-orbit {
            position: absolute;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            border: 1px dashed rgba(255,224,178,0.38);
        }

        .avatar-orbit::before {
            content: "";
            position: absolute;
            inset: -24px;
            border-radius: 50%;
            border: 1px dashed rgba(255,224,178,0.18);
        }

        .avatar-face {
            position: relative;
            width: 205px;
            height: 205px;
            border-radius: 50%;
            background:
                radial-gradient(circle at 38% 30%, #ffe1bd 0 18%, #ffb47e 42%, #dc6f2c 75%, #5c2f1e 100%);
            box-shadow: 0 26px 50px rgba(0,0,0,0.28);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 7.2rem;
        }

        .face-bracket {
            position: absolute;
            inset: 34px;
            pointer-events: none;
        }

        .face-bracket::before,
        .face-bracket::after,
        .face-bracket span::before,
        .face-bracket span::after {
            content: "";
            position: absolute;
            width: 30px;
            height: 30px;
            border-color: rgba(255,255,255,0.95);
            border-style: solid;
        }

        .face-bracket::before {
            top: 0;
            left: 0;
            border-width: 6px 0 0 6px;
            border-top-left-radius: 14px;
        }

        .face-bracket::after {
            top: 0;
            right: 0;
            border-width: 6px 6px 0 0;
            border-top-right-radius: 14px;
        }

        .face-bracket span::before {
            bottom: 0;
            left: 0;
            border-width: 0 0 6px 6px;
            border-bottom-left-radius: 14px;
        }

        .face-bracket span::after {
            bottom: 0;
            right: 0;
            border-width: 0 6px 6px 0;
            border-bottom-right-radius: 14px;
        }

        .mood-bubble {
            position: absolute;
            width: 62px;
            height: 62px;
            border-radius: 50%;
            background: rgba(255,255,255,0.92);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            box-shadow: 0 14px 26px rgba(0,0,0,0.17);
            border: 4px solid rgba(255,255,255,0.45);
        }

        .bubble-happy { top: 6px; left: 44px; }
        .bubble-sad { top: 6px; right: 38px; }
        .bubble-angry { bottom: 28px; left: 12px; }
        .bubble-surprise { bottom: 32px; right: 2px; }

        .engine-card {
            position: relative;
            margin-top: 1.25rem;
            padding: 1.25rem 1.35rem;
            border-radius: 26px;
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,250,245,0.95));
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
            overflow: hidden;
        }

        .engine-card::before {
            content: "";
            position: absolute;
            right: -3%;
            bottom: -36px;
            width: 47%;
            height: 150px;
            background:
                radial-gradient(circle at 10px 10px, rgba(249,115,22,0.18) 1.4px, transparent 2px) 0 0 / 10px 10px,
                repeating-radial-gradient(ellipse at 0% 100%, transparent 0 28px, rgba(249,115,22,0.26) 29px 30px);
            mask-image: linear-gradient(90deg, transparent 0%, #000 24%);
            opacity: 0.82;
        }

        .engine-content,
        .section-head {
            display: flex;
            align-items: center;
            gap: 1rem;
            position: relative;
            z-index: 2;
        }

        .icon-box {
            width: 68px;
            height: 68px;
            border-radius: 18px;
            background: linear-gradient(180deg, #fff8ef, #fff1e2);
            border: 1px solid rgba(249,115,22,0.18);
            box-shadow: var(--shadow-soft);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            flex: 0 0 auto;
        }

        .label {
            color: var(--orange-dark);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.75rem;
            font-weight: 900;
            margin-bottom: 0.35rem;
        }

        .title {
            margin: 0;
            color: var(--ink);
            font-size: 1.28rem;
            font-weight: 900;
            letter-spacing: -0.03em;
        }

        .copy {
            margin: 0.45rem 0 0 0;
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.55;
            max-width: 760px;
        }

        .panel-card {
            padding: 1.35rem;
            border-radius: 26px;
            background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
            min-height: 100%;
        }

        .camera-box {
            margin-top: 1rem;
            padding: 0.55rem;
            border-radius: 23px;
            background: linear-gradient(180deg, #fff, #fff7ef);
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: var(--shadow-soft);
            position: relative;
            overflow: hidden;
        }

        .live-pill {
            position: absolute;
            top: 1.05rem;
            right: 1.05rem;
            z-index: 4;
            padding: 0.36rem 0.68rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.78);
            color: white;
            font-size: 0.78rem;
            font-weight: 800;
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            letter-spacing: 0.03em;
        }

        .live-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #7ee787;
            box-shadow: 0 0 12px rgba(126,231,135,0.9);
            display: inline-block;
        }

        .result-layout {
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(330px, 0.95fr);
            gap: 1rem;
            margin-top: 1rem;
        }

        .preview-card,
        .result-card,
        .prob-card,
        .placeholder-card {
            border-radius: 22px;
            background: linear-gradient(180deg, #ffffff, #fffaf5);
            border: 1px solid rgba(15,23,42,0.10);
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(17, 24, 39, 0.04);
        }

        .card-title {
            margin: 0 0 0.8rem 0;
            font-size: 0.98rem;
            font-weight: 900;
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        .selfie-img {
            width: 100%;
            height: 260px;
            object-fit: cover;
            border-radius: 17px;
            border: 1px solid rgba(15,23,42,0.08);
            display: block;
        }

        .result-card {
            background:
                radial-gradient(circle at 88% 10%, rgba(251,191,36,0.22), transparent 32%),
                linear-gradient(180deg, #fffdfb, #fff7ef);
            border-color: rgba(249,115,22,0.20);
        }

        .result-small {
            color: var(--orange-dark);
            text-transform: uppercase;
            letter-spacing: 0.11em;
            font-size: 0.75rem;
            font-weight: 900;
            margin-bottom: 0.75rem;
        }

        .emotion-row {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .emotion-emoji {
            font-size: 3.6rem;
            line-height: 1;
        }

        .emotion-text {
            margin: 0;
            font-size: clamp(2rem, 3.2vw, 3.2rem);
            line-height: 0.95;
            color: var(--orange-dark);
            font-weight: 900;
            letter-spacing: -0.06em;
        }

        .confidence {
            margin-top: 1rem;
            padding: 0.95rem 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #111827, #1f2937);
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .confidence-label {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.92rem;
            font-weight: 800;
        }

        .confidence-score {
            color: #62e38f;
            font-size: clamp(1.65rem, 2.4vw, 2.3rem);
            font-weight: 900;
            letter-spacing: -0.04em;
            white-space: nowrap;
        }

        .result-note {
            color: #5f4f43;
            margin: 0.95rem 0 0 0;
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .prob-card {
            margin-top: 1rem;
        }

        .prob-grid {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 1rem;
        }

        .prob-item {
            text-align: center;
        }

        .prob-name {
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .prob-track {
            height: 9px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.08);
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: inherit;
        }

        .prob-value {
            margin-top: 0.45rem;
            color: var(--ink);
            font-size: 0.86rem;
            font-weight: 800;
        }

        .placeholder-card {
            margin-top: 1rem;
            min-height: 235px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: var(--muted);
            background:
                radial-gradient(circle at 30% 25%, rgba(249,115,22,0.15), transparent 22%),
                linear-gradient(180deg, #fffdf9, #fff7ef);
            border-style: dashed;
        }

        .placeholder-icon {
            font-size: 4.8rem;
            margin-bottom: 0.7rem;
        }

        .placeholder-title {
            color: var(--ink);
            font-weight: 900;
            font-size: 1.25rem;
            margin-bottom: 0.35rem;
        }

        .placeholder-text {
            max-width: 520px;
            line-height: 1.55;
        }

        .app-footer {
            text-align: center;
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 1.05rem;
            font-weight: 600;
        }

        [data-testid="stCameraInput"] {
            padding: 0 !important;
        }

        [data-testid="stCameraInput"] > div {
            gap: 0.85rem;
        }

        [data-testid="stCameraInput"] label {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }

        [data-testid="stCameraInput"] video,
        [data-testid="stCameraInput"] img {
            transform: scaleX(-1);
            border-radius: 19px;
            min-height: 380px;
            object-fit: cover;
            background: #000;
        }

        [data-testid="stCameraInput"] button {
            min-height: 52px;
            width: 100%;
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #ff7a1a, #d64b08);
            color: white;
            font-size: 0.98rem;
            font-weight: 900;
            box-shadow: 0 14px 26px rgba(214,75,8,0.24);
            transition: all 0.2s ease;
        }

        [data-testid="stCameraInput"] button:hover {
            transform: translateY(-1px);
            color: white;
            box-shadow: 0 18px 32px rgba(214,75,8,0.32);
        }

        [data-testid="stAlert"] {
            border-radius: 18px;
        }

        @media (max-width: 1100px) {
            .hero-card {
                grid-template-columns: 1fr;
            }

            .hero-visual {
                min-height: 190px;
            }

            .prob-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .result-layout {
                grid-template-columns: 1fr;
            }

            [data-testid="stCameraInput"] video,
            [data-testid="stCameraInput"] img {
                min-height: 320px;
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 0.75rem;
            }

            .hero-card {
                padding: 1.3rem;
                border-radius: 22px;
            }

            .hero-title {
                font-size: 2.35rem;
            }

            .hero-logo {
                width: 50px;
                height: 50px;
                font-size: 1.6rem;
            }

            .avatar-face {
                width: 165px;
                height: 165px;
                font-size: 5.5rem;
            }

            .avatar-orbit {
                width: 210px;
                height: 210px;
            }

            .mood-bubble {
                width: 48px;
                height: 48px;
                font-size: 1.45rem;
            }

            .engine-content,
            .section-head {
                align-items: flex-start;
            }

            .prob-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def emotion_theme(emotion: str) -> dict:
    return EMOTION_STYLES.get(
        str(emotion).lower(),
        {"color": "#64748b", "emoji": "🙂", "accent": "Expression captured"},
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-card">
            <div>
                <h1 class="hero-title">
                    <span class="hero-logo">😊</span>
                    <span>Emoti<span class="hero-accent">Avatar</span></span>
                </h1>
                <p class="hero-copy">
                    Capture a selfie, detect your facial emotion using AI, and instantly review the dominant expression.
                </p>
                <div class="hero-chips">
                    <span class="hero-chip">⚡ Real-time Capture</span>
                    <span class="hero-chip">🧠 DeepFace AI Engine</span>
                    <span class="hero-chip">🎭 Emotion Detection</span>
                    <span class="hero-chip">📸 Selfie Analysis</span>
                </div>
            </div>

            <div class="hero-visual">
                <div class="avatar-orbit"></div>
                <div class="mood-bubble bubble-happy">😊</div>
                <div class="mood-bubble bubble-sad">😭</div>
                <div class="mood-bubble bubble-angry">😠</div>
                <div class="mood-bubble bubble-surprise">😮</div>
                <div class="avatar-face">🙂</div>
                <div class="face-bracket"><span></span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_engine_card() -> None:
    st.markdown(
        """
        <section class="engine-card">
            <div class="engine-content">
                <div class="icon-box">🤖</div>
                <div>
                    <div class="label">Inference Engine</div>
                    <h2 class="title">DeepFace Emotion Analysis</h2>
                    <p class="copy">
                        The system captures a selfie and predicts the dominant facial emotion such as happy,
                        sad, angry, neutral, surprised, fear, or disgust.
                    </p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, label: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="section-head">
                <div class="icon-box">{icon}</div>
                <div>
                    <div class="label">{escape(label)}</div>
                    <h3 class="title">{escape(title)}</h3>
                    <p class="copy">{escape(copy)}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_breakdown(scores: dict) -> str:
    order = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]
    items = []

    for name in order:
        theme = emotion_theme(name)
        score = float(scores.get(name, 0.0) or 0.0)
        items.append(
            f"""
            <div class="prob-item">
                <div class="prob-name">{escape(name.title())}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{min(score, 100):.2f}%; background:{theme["color"]};"></div>
                </div>
                <div class="prob-value">{score:.2f}%</div>
            </div>
            """
        )

    return '<div class="prob-grid">' + "".join(items) + "</div>"


def rgb_to_data_uri(rgb_image: np.ndarray) -> str:
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        return ""
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def read_camera_image(uploaded_photo) -> np.ndarray | None:
    file_bytes = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    # Make output match the mirrored camera preview.
    frame = cv2.flip(frame, 1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def analyze_emotion(rgb_frame: np.ndarray) -> tuple[str, dict]:
    analysis = DeepFace.analyze(
        img_path=rgb_frame,
        actions=["emotion"],
        enforce_detection=False,
    )

    if isinstance(analysis, list):
        analysis = analysis[0]

    emotion = str(analysis.get("dominant_emotion", "neutral")).lower()
    scores = analysis.get("emotion", {}) or {}
    return emotion, scores


def render_waiting_state() -> None:
    st.markdown(
        f"""
        <div class="result-layout">
            <div class="placeholder-card">
                <div>
                    <div class="placeholder-icon">📷</div>
                    <div class="placeholder-title">Captured Selfie</div>
                    <div class="placeholder-text">
                        Your captured selfie preview will appear here after you take a photo.
                    </div>
                </div>
            </div>

            <div class="result-card">
                <div class="result-small">Detected Emotion</div>
                <div class="emotion-row">
                    <div class="emotion-emoji">🙂</div>
                    <h3 class="emotion-text">Waiting</h3>
                </div>
                <div class="confidence">
                    <div class="confidence-label">🛡️ Confidence Score</div>
                    <div class="confidence-score">0.00%</div>
                </div>
                <p class="result-note">Capture a selfie to start the DeepFace emotion analysis.</p>
            </div>
        </div>

        <div class="prob-card">
            <h4 class="card-title">Emotion Probability Breakdown</h4>
            {render_probability_breakdown({})}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_state(rgb_frame: np.ndarray, emotion: str, scores: dict) -> None:
    theme = emotion_theme(emotion)
    confidence = float(scores.get(emotion, 0.0) or 0.0)
    data_uri = rgb_to_data_uri(rgb_frame)

    st.markdown(
        f"""
        <div class="result-layout">
            <div class="preview-card">
                <h4 class="card-title">Captured Selfie</h4>
                <img class="selfie-img" src="{data_uri}" alt="Captured selfie">
            </div>

            <div class="result-card">
                <div class="result-small">Detected Emotion</div>
                <div class="emotion-row">
                    <div class="emotion-emoji">{theme["emoji"]}</div>
                    <h3 class="emotion-text">{escape(emotion.title())}</h3>
                </div>
                <div class="confidence">
                    <div class="confidence-label">🛡️ Confidence Score</div>
                    <div class="confidence-score">{confidence:.2f}%</div>
                </div>
                <p class="result-note">
                    {escape(theme["accent"])}. This is the dominant emotion detected from your selfie using
                    the DeepFace emotion analysis model.
                </p>
            </div>
        </div>

        <div class="prob-card">
            <h4 class="card-title">Emotion Probability Breakdown</h4>
            {render_probability_breakdown(scores)}
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_ui_css()
render_hero()
render_engine_card()

left_col, right_col = st.columns([0.90, 1.35], gap="large")

with left_col:
    render_section_header(
        icon="📷",
        label="Camera Input",
        title="Take Selfie",
        copy="Capture a clear front-facing selfie in good lighting for better results.",
    )

    st.markdown(
        """
        <div class="camera-box">
            <div class="live-pill"><span class="live-dot"></span> LIVE</div>
        """,
        unsafe_allow_html=True,
    )
    photo = st.camera_input("Take Photo")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    render_section_header(
        icon="📊",
        label="Output",
        title="Emotion Result",
        copy="Your captured selfie preview and AI emotion result will appear here.",
    )

    if photo is None:
        render_waiting_state()
    else:
        rgb_frame = read_camera_image(photo)

        if rgb_frame is None:
            st.warning("Failed to decode the captured image. Please retake the photo.")
        else:
            with st.spinner("Analyzing your facial emotion..."):
                try:
                    emotion, scores = analyze_emotion(rgb_frame)
                    render_result_state(rgb_frame, emotion, scores)
                except Exception as exc:
                    st.markdown(
                        """
                        <div class="placeholder-card">
                            <div>
                                <div class="placeholder-icon">⚠️</div>
                                <div class="placeholder-title">Face not detected</div>
                                <div class="placeholder-text">
                                    Retake the selfie with your face larger in the frame and use more even lighting.
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Technical detail: {exc}")

st.markdown(
    """
    <div class="app-footer">
        © 2024 EmotiAvatar &nbsp;•&nbsp; Powered by DeepFace AI ❤️
    </div>
    """,
    unsafe_allow_html=True,
)
