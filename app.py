import os

# Keep TensorFlow quieter and CPU-friendly on deployment targets.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
import streamlit as st
from streamlit.runtime import exists as streamlit_runtime_exists

if not streamlit_runtime_exists():
    raise SystemExit(
        "This app must be launched with Streamlit.\n"
        "Run: .\\.venv\\Scripts\\python.exe -m streamlit run app.py"
    )

st.set_page_config(
    page_title="EmotiAvatar",
    page_icon="🙂",
    layout="centered",
)

st.markdown(
    """
    <style>
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img {
        transform: scaleX(-1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


EMOTION_META = {
    "happy": {"emoji": "😊", "color": "#ff8a3d", "tone": "Open, warm, and positive."},
    "neutral": {"emoji": "🙂", "color": "#4f7cff", "tone": "Balanced and steady."},
    "surprise": {"emoji": "😮", "color": "#9a6bff", "tone": "Alert and reactive."},
    "sad": {"emoji": "😔", "color": "#f0a23b", "tone": "Low-energy and reflective."},
    "angry": {"emoji": "😠", "color": "#f14f5d", "tone": "Tense, forceful, and intense."},
    "fear": {"emoji": "😨", "color": "#23b9b0", "tone": "Guarded and uneasy."},
    "disgust": {"emoji": "🤢", "color": "#d864b2", "tone": "Strong aversion detected."},
}

EMOTION_ORDER = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]


def meta_for_emotion(emotion: str) -> dict:
    return EMOTION_META.get(
        str(emotion).lower(),
        {"emoji": "🙂", "color": "#7a8699", "tone": "Expression captured."},
    )


def read_camera_image(uploaded_photo) -> np.ndarray | None:
    raw = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    frame = cv2.flip(frame, 1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner=False)
def get_deepface():
    try:
        from deepface import DeepFace
    except ValueError as exc:
        if "requires tf-keras package" in str(exc):
            st.error(
                "DeepFace could not start because `tf-keras` is missing for the installed "
                "TensorFlow version. Install dependencies again with `pip install -r requirements.txt`."
            )
            st.stop()
        raise
    except ModuleNotFoundError:
        st.error(
            "DeepFace is not installed. Install your dependencies with `pip install -r requirements.txt`."
        )
        st.stop()
    return DeepFace


def analyze_emotion(rgb_frame: np.ndarray) -> tuple[str, dict]:
    deepface = get_deepface()
    analysis = deepface.analyze(
        img_path=rgb_frame,
        actions=["emotion"],
        enforce_detection=False,
    )
    if isinstance(analysis, list):
        analysis = analysis[0]
    emotion = str(analysis.get("dominant_emotion", "neutral")).lower()
    scores = analysis.get("emotion", {}) or {}
    return emotion, scores


def render_score_breakdown(scores: dict) -> None:
    st.subheader("Emotion scores")
    for name in EMOTION_ORDER:
        meta = meta_for_emotion(name)
        score = float(scores.get(name, 0.0) or 0.0)
        st.markdown(f"**{meta['emoji']} {name.title()}** - {score:.2f}%")
        st.progress(min(max(score / 100, 0.0), 1.0))


def render_result(rgb_frame: np.ndarray, emotion: str, scores: dict) -> None:
    meta = meta_for_emotion(emotion)
    confidence = float(scores.get(emotion, 0.0) or 0.0)

    left_col, right_col = st.columns(2)

    with left_col:
        st.image(rgb_frame, caption="Captured photo", use_container_width=True)

    with right_col:
        st.subheader("Detected emotion")
        st.markdown(f"## {meta['emoji']} {emotion.title()}")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.write(meta["tone"])

    render_score_breakdown(scores)


st.title("EmotiAvatar")
st.write("Take a selfie and check the detected facial emotion.")

photo = st.camera_input("Take photo")

if photo is None:
    st.info("Capture a clear front-facing photo to see the result.")
else:
    rgb_frame = read_camera_image(photo)
    if rgb_frame is None:
        st.warning("Failed to decode the captured image. Please retake the photo.")
    else:
        with st.spinner("Analyzing emotion..."):
            try:
                emotion, scores = analyze_emotion(rgb_frame)
                render_result(rgb_frame, emotion, scores)
            except Exception as exc:
                st.error("Could not analyze the photo. Retake it with your face centered and clearly visible.")
                st.caption(f"Technical detail: {exc}")
