import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import cv2
import numpy as np
from html import escape

try:
    from deepface import DeepFace
except ValueError as exc:
    if "requires tf-keras package" in str(exc):
        st.set_page_config(page_title="EmotiAvatar", layout="centered")
        st.title("EmotiAvatar")
        st.error(
            "DeepFace could not start because `tf-keras` is missing for the installed "
            "TensorFlow version. Install dependencies again with "
            "`pip install -r requirements.txt`."
        )
        st.stop()
    raise

st.set_page_config(page_title="EmotiAvatar", layout="wide")

EMOTION_STYLES = {
    "happy": {"color": "#f59e0b", "accent": "Radiant and upbeat", "emoji": "😊"},
    "sad": {"color": "#4f7cff", "accent": "Quiet and reflective", "emoji": "😢"},
    "angry": {"color": "#ef4444", "accent": "High energy and intense", "emoji": "😠"},
    "surprise": {"color": "#8b5cf6", "accent": "Alert and reactive", "emoji": "😮"},
    "fear": {"color": "#06b6d4", "accent": "Tense and cautious", "emoji": "😨"},
    "disgust": {"color": "#ec4899", "accent": "Strong aversion detected", "emoji": "🤢"},
    "neutral": {"color": "#38bdf8", "accent": "Steady and composed", "emoji": "😐"},
}


def render_shell() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

        :root {
            --bg: #fff7ed;
            --ink: #172033;
            --muted: #5e6777;
            --panel: rgba(255, 255, 255, 0.96);
            --line: rgba(247, 149, 51, 0.18);
            --shadow: 0 22px 60px rgba(240, 124, 32, 0.14);
            --shadow-soft: 0 12px 30px rgba(31, 41, 55, 0.08);
            --accent: #f97316;
            --accent-deep: #c2410c;
            --green: #22c55e;
            --navy: #0f172a;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(251, 191, 36, 0.18), transparent 22%),
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 24%),
                linear-gradient(180deg, #fffaf4 0%, #fff1df 100%);
            color: var(--ink);
            font-family: "Manrope", sans-serif;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 0.9rem;
            padding-left: 1.1rem;
            padding-right: 1.1rem;
            max-width: 1600px;
            width: 100%;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
        }

        .hero {
            position: relative;
            padding: 1.8rem 2rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at 78% 50%, rgba(251, 146, 60, 0.32), transparent 18%),
                linear-gradient(115deg, #07101d 0%, #141b2b 56%, #f97316 100%);
            box-shadow: var(--shadow);
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
            gap: 1rem;
            align-items: center;
            color: white;
        }

        .hero-copy-wrap {
            max-width: 760px;
        }

        .hero-title {
            margin: 0;
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: clamp(2.4rem, 4vw, 4.1rem);
            line-height: 0.95;
        }

        .hero-title-emoji {
            width: 56px;
            height: 56px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: linear-gradient(180deg, #fde68a, #f59e0b);
            box-shadow: inset 0 3px 10px rgba(255, 255, 255, 0.35);
            font-size: 1.85rem;
            flex: 0 0 auto;
        }

        .hero-title-accent {
            color: #ffb454;
        }

        .hero-copy {
            color: rgba(255, 255, 255, 0.88);
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0.95rem 0 1.15rem 0;
            max-width: 640px;
        }

        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.8rem 1rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.20);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
            font-size: 0.96rem;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.95);
        }

        .hero-visual {
            position: relative;
            min-height: 210px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .hero-avatar-ring {
            position: absolute;
            inset: 18px;
            border-radius: 50%;
            border: 1px dashed rgba(255, 196, 120, 0.34);
        }

        .hero-avatar-ring::before,
        .hero-avatar-ring::after {
            content: "";
            position: absolute;
            inset: -18px;
            border-radius: 50%;
            border: 1px dashed rgba(255, 196, 120, 0.18);
        }

        .hero-avatar {
            position: relative;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background:
                radial-gradient(circle at 35% 30%, #ffddb8 0%, #ffbd8e 24%, #f28a4a 60%, #5c2f1e 100%);
            box-shadow: 0 24px 50px rgba(0, 0, 0, 0.28);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 7rem;
        }

        .hero-bubble {
            position: absolute;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.9rem;
        }

        .bubble-1 { top: 4px; left: 20px; }
        .bubble-2 { top: 8px; right: 20px; }
        .bubble-3 { bottom: 24px; left: 6px; }
        .bubble-4 { bottom: 26px; right: 8px; }

        .hero-bracket {
            position: absolute;
            inset: 20px;
            pointer-events: none;
        }

        .hero-bracket::before,
        .hero-bracket::after,
        .hero-bracket-bottom::before,
        .hero-bracket-bottom::after {
            content: "";
            position: absolute;
            width: 28px;
            height: 28px;
        }

        .hero-bracket::before {
            top: 0;
            left: 0;
            border-top: 6px solid rgba(255, 255, 255, 0.95);
            border-left: 6px solid rgba(255, 255, 255, 0.95);
            border-top-left-radius: 14px;
        }

        .hero-bracket::after {
            top: 0;
            right: 0;
            border-top: 6px solid rgba(255, 255, 255, 0.95);
            border-right: 6px solid rgba(255, 255, 255, 0.95);
            border-top-right-radius: 14px;
        }

        .hero-bracket-bottom::before {
            bottom: 0;
            left: 0;
            border-bottom: 6px solid rgba(255, 255, 255, 0.95);
            border-left: 6px solid rgba(255, 255, 255, 0.95);
            border-bottom-left-radius: 14px;
        }

        .hero-bracket-bottom::after {
            bottom: 0;
            right: 0;
            border-bottom: 6px solid rgba(255, 255, 255, 0.95);
            border-right: 6px solid rgba(255, 255, 255, 0.95);
            border-bottom-right-radius: 14px;
        }

        .feature-band {
            position: relative;
            margin-top: 1.15rem;
            padding: 1.2rem 1.3rem;
            border-radius: 26px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(255, 251, 246, 0.95));
            border: 1px solid rgba(247, 149, 51, 0.18);
            box-shadow: var(--shadow-soft);
            overflow: hidden;
        }

        .feature-band::after {
            content: "";
            position: absolute;
            right: -60px;
            bottom: -42px;
            width: 42%;
            height: 140px;
            background:
                radial-gradient(circle at 10px 10px, rgba(249, 115, 22, 0.18) 1.5px, transparent 2px) 0 0 / 10px 10px,
                linear-gradient(180deg, transparent 0%, rgba(249, 115, 22, 0.06) 100%);
            mask-image: linear-gradient(90deg, transparent 0%, black 18%);
            opacity: 0.95;
        }

        .feature-band::before {
            content: "";
            position: absolute;
            right: 0;
            bottom: 10px;
            width: 44%;
            height: 90px;
            background:
                radial-gradient(120px 36px at 0 100%, transparent 68%, rgba(249, 115, 22, 0.32) 69%, transparent 71%) 0 0 / 140px 36px repeat-x;
            opacity: 0.55;
        }

        .feature-band-content {
            display: flex;
            align-items: center;
            gap: 1rem;
            position: relative;
            z-index: 1;
        }

        .feature-icon,
        .section-icon {
            width: 68px;
            height: 68px;
            border-radius: 18px;
            background: linear-gradient(180deg, #fff8ef, #fff1e2);
            border: 1px solid rgba(249, 115, 22, 0.16);
            box-shadow: var(--shadow-soft);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            flex: 0 0 auto;
        }

        .section-icon {
            width: 62px;
            height: 62px;
            font-size: 1.8rem;
        }

        .feature-label,
        .section-label {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.75rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .feature-title,
        .section-title {
            margin: 0;
            font-size: 1.1rem;
        }

        .feature-copy,
        .section-copy {
            margin: 0.45rem 0 0 0;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .workspace-grid {
            display: grid;
            grid-template-columns: minmax(360px, 0.95fr) minmax(0, 1.45fr);
            gap: 1.2rem;
            align-items: start;
            margin-top: 1rem;
        }

        .panel {
            height: 100%;
            padding: 1.25rem;
            border-radius: 26px;
            border: 1px solid var(--line);
            background: var(--panel);
            box-shadow: var(--shadow-soft);
        }

        .section-head {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .camera-shell {
            padding: 0.45rem;
            border-radius: 22px;
            background: linear-gradient(180deg, #ffffff, #fff7ef);
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
            position: relative;
        }

        .live-pill {
            position: absolute;
            top: 1rem;
            right: 1rem;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.74);
            color: white;
            font-size: 0.82rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
        }

        .live-pill::before {
            content: "";
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #7ee787;
            box-shadow: 0 0 10px rgba(126, 231, 135, 0.9);
        }

        .preview-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
            gap: 1rem;
            margin-top: 0.2rem;
        }

        .preview-card,
        .result-card,
        .probability-card {
            padding: 0.95rem;
            border-radius: 22px;
            border: 1px solid rgba(15, 23, 42, 0.10);
            background: linear-gradient(180deg, #ffffff, #fffaf5);
        }

        .image-shell {
            border-radius: 18px;
            overflow: hidden;
        }

        .subcard-title {
            margin: 0 0 0.7rem 0;
            font-size: 0.95rem;
            font-weight: 800;
        }

        .result-card {
            padding: 1rem 1.1rem;
            background:
                radial-gradient(circle at top right, rgba(251, 191, 36, 0.18), transparent 34%),
                linear-gradient(180deg, #fffdfa, #fff7ef);
        }

        .result-header {
            color: var(--ink);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            font-weight: 800;
            opacity: 0.76;
        }

        .result-emotion {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 0.7rem;
        }

        .result-emoji {
            font-size: 3.3rem;
            line-height: 1;
        }

        .result-value {
            margin: 0;
            font-family: "Space Grotesk", sans-serif;
            font-size: clamp(2rem, 4vw, 3.2rem);
            font-weight: 700;
            color: var(--accent-deep);
            line-height: 0.95;
        }

        .confidence-bar {
            margin-top: 1rem;
            padding: 0.95rem 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #111827, #1f2937);
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
        }

        .confidence-label {
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            font-weight: 700;
        }

        .confidence-score {
            font-family: "Space Grotesk", sans-serif;
            font-size: clamp(1.8rem, 3vw, 2.4rem);
            color: #62e38f;
            font-weight: 700;
            margin: 0;
        }

        .result-copy {
            margin: 0.9rem 0 0 0;
            color: #5f4f43;
            line-height: 1.5;
        }

        .placeholder-box {
            min-height: 240px;
            border-radius: 20px;
            border: 1px dashed rgba(15, 23, 42, 0.14);
            background:
                radial-gradient(circle at 30% 30%, rgba(249, 115, 22, 0.14), transparent 22%),
                linear-gradient(180deg, #fffdf9, #fff7ef);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--muted);
            text-align: center;
            padding: 1.4rem;
        }

        .probability-card {
            margin-top: 1rem;
        }

        .probability-grid {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 1rem;
        }

        .prob-item {
            text-align: center;
        }

        .prob-name {
            font-size: 0.92rem;
            font-weight: 700;
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
            font-size: 0.9rem;
            color: var(--ink);
            font-weight: 700;
        }

        .app-footer {
            text-align: center;
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: 1.1rem;
        }

        [data-testid="stImage"] img {
            border-radius: 18px;
            border: 1px solid rgba(31, 41, 51, 0.08);
            object-fit: cover;
        }

        .preview-card [data-testid="stImage"] img,
        .camera-shell [data-testid="stImage"] img {
            border: none;
        }

        .stCameraInput > label {
            font-family: "Space Grotesk", sans-serif;
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

        [data-testid="stCameraInput"] {
            padding-top: 0;
        }

        [data-testid="stCameraInput"] button {
            border-radius: 999px;
            background: linear-gradient(135deg, #ff7a1a, #d64b08);
            color: white;
            border: none;
            font-weight: 700;
            box-shadow: 0 12px 24px rgba(214, 75, 8, 0.24);
            min-height: 50px;
            padding: 0.8rem 1.2rem;
            width: 100%;
        }

        [data-testid="stCameraInput"] video,
        [data-testid="stCameraInput"] img {
            transform: scaleX(-1);
            border-radius: 20px;
        }

        [data-testid="stCaptionContainer"] {
            margin-top: 0.35rem;
        }

        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
            width: 100%;
        }

        [data-testid="column"] {
            width: 100%;
        }

        [data-testid="column"] > div {
            height: 100%;
        }

        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] {
            gap: 0.9rem;
        }

        .stAppViewContainer,
        .main,
        .main > div {
            width: 100%;
        }

        footer, header[data-testid="stHeader"] {
            display: none;
        }

        @media (max-width: 900px) {
            .hero,
            .workspace-grid,
            .preview-grid,
            .probability-grid {
                grid-template-columns: 1fr;
            }

            .hero {
                padding: 1.2rem;
            }

            .hero-title {
                font-size: 2.3rem;
                gap: 0.75rem;
            }

            .hero-avatar {
                width: 180px;
                height: 180px;
                font-size: 5.8rem;
            }

            .feature-band::after {
                width: 70%;
            }

            .block-container {
                padding-top: 0.75rem;
            }

            .stApp {
                overflow-y: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def emotion_theme(emotion: str) -> dict:
    return EMOTION_STYLES.get(
        emotion.lower(),
        {"color": "#355070", "accent": "Expression captured", "emoji": "🙂"},
    )


def render_probability_breakdown(emotion_scores: dict) -> str:
    order = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]
    items = []
    for name in order:
        theme = emotion_theme(name)
        score = float(emotion_scores.get(name, 0.0))
        items.append(
            f"""
            <div class="prob-item">
                <div class="prob-name">{name.title()}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {min(score, 100):.2f}%; background: {theme["color"]};"></div>
                </div>
                <div class="prob-value">{score:.2f}%</div>
            </div>
            """
        )
    return '<div class="probability-grid">' + "".join(items) + "</div>"


render_shell()

st.markdown(
    """
    <section class="hero">
        <div class="hero-copy-wrap">
            <h1 class="hero-title">
                <span class="hero-title-emoji">😊</span>
                <span>Emoti<span class="hero-title-accent">Avatar</span></span>
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
            <div class="hero-avatar-ring"></div>
            <div class="hero-bubble bubble-1">😊</div>
            <div class="hero-bubble bubble-2">😭</div>
            <div class="hero-bubble bubble-3">😠</div>
            <div class="hero-bubble bubble-4">😮</div>
            <div class="hero-avatar">🙂</div>
            <div class="hero-bracket"></div>
            <div class="hero-bracket hero-bracket-bottom"></div>
        </div>
    </section>
    <section class="feature-band">
        <div class="feature-band-content">
            <div class="feature-icon">🧠</div>
            <div>
                <div class="feature-label">Inference Engine</div>
                <h2 class="feature-title">DeepFace Emotion Analysis</h2>
                <p class="feature-copy">
                    The system captures a selfie and predicts the dominant facial emotion such as happy, sad,
                    angry, neutral, surprised, fear, or disgust.
                </p>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.88, 1.12], gap="large")

with left_col:
    st.markdown(
        """
        <div class="panel">
            <div class="section-head">
                <div class="section-icon">📷</div>
                <div>
                    <div class="section-label">Camera Input</div>
                    <h3 class="section-title">Take Selfie</h3>
                    <p class="section-copy">Capture a clear front-facing selfie in good lighting for better results.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="camera-shell">', unsafe_allow_html=True)
    st.markdown('<div class="live-pill">LIVE</div>', unsafe_allow_html=True)
    photo = st.camera_input("Take Selfie")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown(
        """
        <div class="panel">
            <div class="section-head">
                <div class="section-icon">📊</div>
                <div>
                    <div class="section-label">Output</div>
                    <h3 class="section-title">Emotion Result</h3>
                    <p class="section-copy">
                        Your captured selfie preview and AI emotion result will appear here.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if photo is not None:
    file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.warning("Failed to decode the captured image.")
    else:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_col, preview_col = right_col.columns([0.76, 1.24], gap="medium")

        try:
            result = DeepFace.analyze(
                rgb_frame,
                actions=["emotion"],
                enforce_detection=False,
            )
            emotion = result[0]["dominant_emotion"]
            emotion_scores = result[0]["emotion"]
            theme = emotion_theme(emotion)
            confidence = float(emotion_scores.get(emotion, 0.0))

            st.markdown('<div class="preview-grid">', unsafe_allow_html=True)
            preview_col, result_col = st.columns([1.05, 0.95], gap="medium")

            with preview_col:
                st.markdown('<div class="preview-card"><h4 class="subcard-title">Captured Selfie</h4><div class="image-shell">', unsafe_allow_html=True)
                st.image(rgb_frame, use_container_width=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

            with result_col:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-header">Detected Emotion</div>
                        <div class="result-emotion">
                            <div class="result-emoji">{theme["emoji"]}</div>
                            <h3 class="result-value">{escape(emotion.title())}</h3>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-label">🛡️ Confidence Score</div>
                            <div class="confidence-score">{confidence:.2f}%</div>
                        </div>
                        <p class="result-copy">
                            This is the dominant emotion detected from your selfie using the DeepFace emotion analysis model.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="probability-card">
                    <h4 class="subcard-title">Emotion Probability Breakdown</h4>
                    {render_probability_breakdown(emotion_scores)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            with right_col:
                st.markdown(
                    """
                    <div class="preview-card">
                        <h4 class="subcard-title">Emotion Status</h4>
                        <div class="placeholder-box">
                            Face not detected. Retake the selfie with your face larger in frame and more even lighting.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

else:
    with right_col:
        st.markdown(
            """
            <div class="preview-card">
                <h4 class="subcard-title">Captured Selfie</h4>
                <div class="placeholder-box">
                    Your captured selfie preview will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="result-card">
                <div class="result-header">Detected Emotion</div>
                <div class="result-emotion">
                    <div class="result-emoji">🙂</div>
                    <h3 class="result-value">Waiting</h3>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-label">🛡️ Confidence Score</div>
                    <div class="confidence-score">0.00%</div>
                </div>
                <p class="result-copy">Your AI emotion result will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="probability-card">
                <h4 class="subcard-title">Emotion Probability Breakdown</h4>
                {render_probability_breakdown({})}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="app-footer">© 2024 EmotiAvatar &nbsp;•&nbsp; Powered by DeepFace AI ❤️</div>
    """,
    unsafe_allow_html=True,
)
