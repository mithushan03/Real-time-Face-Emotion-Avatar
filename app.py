import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import cv2
import numpy as np

try:
    from deepface import DeepFace
except ValueError as exc:
    if "requires tf-keras package" in str(exc):
        st.set_page_config(page_title="Emotion Detector", layout="centered")
        st.title("😊 Real-Time Face Emotion Detector")
        st.error(
            "DeepFace could not start because `tf-keras` is missing for the installed "
            "TensorFlow version. Install dependencies again with "
            "`pip install -r requirements.txt`."
        )
        st.stop()
    raise

st.set_page_config(page_title="Emotion Detector", layout="wide")

EMOTION_STYLES = {
    "happy": {"color": "#f28c28", "accent": "Radiant and upbeat"},
    "sad": {"color": "#4f7cff", "accent": "Quiet and reflective"},
    "angry": {"color": "#e63946", "accent": "High energy and intense"},
    "surprise": {"color": "#ffb703", "accent": "Alert and reactive"},
    "fear": {"color": "#6a4c93", "accent": "Tense and cautious"},
    "disgust": {"color": "#2a9d8f", "accent": "Strong aversion detected"},
    "neutral": {"color": "#8d99ae", "accent": "Steady and composed"},
}


def render_shell() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

        :root {
            --bg: #f3ede2;
            --ink: #1f2933;
            --muted: #58626c;
            --panel: rgba(255, 251, 245, 0.88);
            --panel-strong: rgba(255, 255, 255, 0.92);
            --line: rgba(31, 41, 51, 0.09);
            --shadow: 0 20px 60px rgba(61, 47, 25, 0.10);
            --shadow-soft: 0 12px 30px rgba(31, 41, 51, 0.05);
            --accent: #cb7a2f;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(223, 171, 73, 0.22), transparent 30%),
                radial-gradient(circle at top right, rgba(92, 134, 216, 0.14), transparent 24%),
                linear-gradient(180deg, #faf6ef 0%, #efe5d6 100%);
            color: var(--ink);
            font-family: "Manrope", sans-serif;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1240px;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
        }

        .app-shell {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .topbar {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 220px;
            gap: 0.75rem;
            align-items: stretch;
        }

        .hero {
            padding: 1.15rem 1.35rem;
            border: 1px solid var(--line);
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,243,216,0.82)),
                linear-gradient(135deg, rgba(232,177,88,0.14), rgba(84,126,206,0.08));
            box-shadow: var(--shadow);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .hero-kicker {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(31, 41, 51, 0.06);
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-title {
            margin: 0.55rem 0 0.2rem 0;
            font-size: clamp(1.7rem, 3vw, 2.5rem);
            line-height: 0.98;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0;
        }

        .summary-grid {
            display: flex;
        }

        .hero-stat {
            padding: 1rem 1.05rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(31, 41, 51, 0.08);
            box-shadow: var(--shadow-soft);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .hero-stat-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            font-weight: 700;
        }

        .hero-stat-value {
            margin-top: 0.25rem;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .panel {
            height: 100%;
            padding: 1rem;
            border-radius: 22px;
            border: 1px solid var(--line);
            background: var(--panel);
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-soft);
        }

        .panel-strong {
            background: var(--panel-strong);
        }

        .panel-title {
            margin: 0 0 0.35rem 0;
            font-size: 1rem;
        }

        .panel-copy {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
            font-size: 0.92rem;
        }

        .eyebrow {
            margin-bottom: 0.45rem;
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .stack {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .emotion-card {
            padding: 1.2rem;
            border-radius: 20px;
            color: white;
            box-shadow: 0 20px 50px rgba(31, 41, 51, 0.14);
        }

        .emotion-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.84;
        }

        .emotion-value {
            margin: 0.4rem 0 0.35rem 0;
            font-family: "Space Grotesk", sans-serif;
            font-size: clamp(1.8rem, 3vw, 2.5rem);
            line-height: 0.95;
        }

        .emotion-note {
            margin: 0;
            font-size: 0.95rem;
            opacity: 0.94;
        }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .metric-card {
            padding: 0.9rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(31, 41, 51, 0.07);
        }

        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            font-weight: 800;
        }

        .metric-value {
            margin-top: 0.3rem;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--ink);
        }

        .status-empty {
            min-height: 260px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        [data-testid="stImage"] img {
            border-radius: 18px;
            border: 1px solid rgba(31, 41, 51, 0.08);
            max-height: 52vh;
            object-fit: cover;
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
            background: linear-gradient(135deg, #cb7a2f, #a9572a);
            color: white;
            border: none;
            font-weight: 700;
            box-shadow: 0 12px 24px rgba(169, 87, 42, 0.24);
            min-height: 48px;
            padding: 0.7rem 1.2rem;
        }

        [data-testid="stCameraInput"] video,
        [data-testid="stCameraInput"] img {
            transform: scaleX(-1);
            border-radius: 18px;
        }

        [data-testid="stCaptionContainer"] {
            margin-top: 0.35rem;
        }

        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }

        footer, header[data-testid="stHeader"] {
            display: none;
        }

        @media (max-width: 900px) {
            .topbar,
            .summary-grid,
            .metrics-row {
                grid-template-columns: 1fr;
            }

            .hero {
                padding: 1.2rem;
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
    return EMOTION_STYLES.get(emotion.lower(), {"color": "#355070", "accent": "Expression captured"})


render_shell()

st.markdown(
    """
    <div class="app-shell">
        <div class="topbar">
            <section class="hero">
                <div>
                    <span class="hero-kicker">Emotion Inference Demo</span>
                    <h1 class="hero-title">Take Selfie. Get Emotion.</h1>
                    <p class="hero-copy">
                        A minimal browser-based emotion detector with single-shot analysis.
                    </p>
                </div>
            </section>
            <div class="summary-grid">
                <div class="hero-stat">
                    <div class="hero-stat-label">Model</div>
                    <div class="hero-stat-value">DeepFace Emotion</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.92, 1.08], gap="large")

with left_col:
    st.markdown(
        """
        <div class="panel panel-strong">
            <div class="eyebrow">Camera</div>
            <h3 class="panel-title">Take Selfie</h3>
            <p class="panel-copy">
                Capture one clear face photo to analyze the dominant emotion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    photo = st.camera_input("Take Selfie")

with right_col:
    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">Analysis</div>
            <h3 class="panel-title">Result</h3>
            <p class="panel-copy">
                The captured selfie is mirrored, analyzed, and shown here.
            </p>
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
            theme = emotion_theme(emotion)
            cv2.putText(
                rgb_frame,
                f"Emotion: {emotion}",
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            with result_col:
                st.markdown(
                    f"""
                    <div class="stack">
                        <div class="emotion-card" style="background:
                            linear-gradient(135deg, {theme["color"]}, #1f2933);">
                            <div class="emotion-label">Detected emotion</div>
                            <div class="emotion-value">{emotion.title()}</div>
                            <p class="emotion-note">{theme["accent"]}</p>
                        </div>
                        <div class="metrics-row">
                            <div class="metric-card">
                                <div class="metric-label">Input type</div>
                                <div class="metric-value">Selfie frame</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Mode</div>
                                <div class="metric-value">Single-shot</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Prediction is based on the dominant facial expression in the current capture.")
        except Exception:
            cv2.putText(
                rgb_frame,
                "No face detected",
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )
            with result_col:
                st.markdown(
                    """
                    <div class="panel status-empty">
                        <div class="eyebrow">Status</div>
                        <h3 class="panel-title">Face not detected</h3>
                        <p class="panel-copy">
                            Retake the selfie with your face larger in frame and more even lighting.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with preview_col:
            st.image(rgb_frame, caption="Selfie preview", use_container_width=True)
else:
    with right_col:
        st.markdown(
            """
            <div class="panel status-empty">
                <div class="eyebrow">Waiting</div>
                <h3 class="panel-title">Take Selfie to begin</h3>
                <p class="panel-copy">
                    The result card and preview will appear here after you capture an image.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
