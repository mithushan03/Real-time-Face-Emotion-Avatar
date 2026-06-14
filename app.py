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
            --bg: #f4efe6;
            --ink: #16212f;
            --muted: #5d6773;
            --panel: rgba(255, 252, 247, 0.90);
            --panel-strong: rgba(255, 255, 255, 0.95);
            --line: rgba(22, 33, 47, 0.08);
            --shadow: 0 24px 70px rgba(55, 43, 24, 0.10);
            --shadow-soft: 0 14px 30px rgba(22, 33, 47, 0.05);
            --accent: #bf6c2c;
            --accent-deep: #7b3f1d;
            --navy: #1c3556;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(235, 191, 96, 0.22), transparent 24%),
                radial-gradient(circle at top right, rgba(71, 117, 201, 0.12), transparent 20%),
                linear-gradient(180deg, #fbf7f0 0%, #eee3d3 100%);
            color: var(--ink);
            font-family: "Manrope", sans-serif;
        }

        .block-container {
            padding-top: 0.9rem;
            padding-bottom: 0.9rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100vw;
            width: 100%;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
        }

        .app-shell {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            width: 100%;
        }

        .topbar {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 0.85rem;
            align-items: stretch;
            width: 100%;
        }

        .hero {
            padding: 1.35rem 1.5rem;
            border: 1px solid var(--line);
            border-radius: 26px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,244,220,0.86)),
                linear-gradient(135deg, rgba(226,170,82,0.14), rgba(76,118,193,0.08));
            box-shadow: var(--shadow);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            gap: 0.7rem;
            align-items: flex-start;
        }

        .hero-kicker {
            display: inline-block;
            padding: 0.38rem 0.78rem;
            border-radius: 999px;
            background: rgba(28, 53, 86, 0.07);
            color: var(--navy);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }

        .hero-title {
            margin: 0.4rem 0 0.25rem 0;
            font-size: clamp(2.2rem, 3vw, 3.4rem);
            line-height: 0.98;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.5;
            margin: 0;
        }

        .hero-stat {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(22, 33, 47, 0.07);
            box-shadow: var(--shadow-soft);
            display: flex;
            flex-direction: column;
            justify-content: center;
            width: 100%;
        }

        .hero-stat-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            font-weight: 700;
        }

        .hero-stat-value {
            margin-top: 0.2rem;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1rem;
            font-weight: 700;
        }

        .workspace-grid {
            display: grid;
            grid-template-columns: minmax(380px, 0.88fr) minmax(0, 1.12fr);
            gap: 1rem;
            align-items: start;
        }

        .panel {
            height: 100%;
            padding: 1.1rem;
            border-radius: 24px;
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
            font-size: 1.05rem;
        }

        .panel-copy {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
            font-size: 0.93rem;
        }

        .section-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
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
            padding: 1.25rem;
            border-radius: 22px;
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

        .camera-frame {
            padding: 0.35rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(28,53,86,0.08), rgba(191,108,44,0.08));
        }

        .camera-note {
            font-size: 0.82rem;
            color: var(--muted);
            margin: 0.2rem 0 0 0;
        }

        [data-testid="stImage"] img {
            border-radius: 18px;
            border: 1px solid rgba(31, 41, 51, 0.08);
            max-height: 54vh;
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
            background: linear-gradient(135deg, var(--accent), var(--accent-deep));
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
            width: 100%;
        }

        [data-testid="column"] {
            width: 100%;
        }

        [data-testid="column"] > div {
            height: 100%;
        }

        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] {
            gap: 0.85rem;
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
            .topbar,
            .hero,
            .workspace-grid,
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
                    <span class="hero-kicker">EmotiAvatar</span>
                    <h1 class="hero-title">EmotiAvatar</h1>
                    <p class="hero-copy">
                        Capture a selfie, run facial emotion inference, and review the dominant expression.
                    </p>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Inference Engine</div>
                    <div class="hero-stat-value">DeepFace emotion analysis</div>
                </div>
            </section>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="workspace-grid">', unsafe_allow_html=True)
left_col, right_col = st.columns([0.88, 1.12], gap="large")

with left_col:
    st.markdown(
        """
        <div class="panel panel-strong">
            <div class="section-head">
                <div>
                    <div class="eyebrow">Camera Input</div>
                    <h3 class="panel-title">Take Selfie</h3>
                </div>
            </div>
            <p class="camera-note">Mirrored live view for natural selfie capture.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="camera-frame">', unsafe_allow_html=True)
    photo = st.camera_input("Take Selfie")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">Output</div>
            <h3 class="panel-title">Emotion Result</h3>
            <p class="panel-copy">
                The captured selfie is analyzed and displayed with the dominant detected emotion.
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
                                <div class="metric-label">Assessment</div>
                                <div class="metric-value">Dominant expression</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("EmotiAvatar reports the strongest facial emotion inferred from the captured frame.")
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
            st.image(rgb_frame, caption="EmotiAvatar preview", use_container_width=True)
else:
    with right_col:
        st.markdown(
            """
            <div class="panel status-empty">
                <div class="eyebrow">Waiting</div>
                <h3 class="panel-title">Capture a selfie to begin analysis</h3>
                <p class="panel-copy">
                    The result card and processed preview will appear here after a capture is taken.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)
