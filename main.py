import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import joblib

st.set_page_config(page_title="ASL Hand Pose Recognition", page_icon="🤟", layout="wide")

CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


@st.cache_resource
def load_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="mediapipe/hand_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_presence_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_rgb, landmarker):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None, None
    hand = result.hand_landmarks[0]
    x_coords = [lm.x for lm in hand]
    y_coords = [lm.y for lm in hand]
    features = np.concatenate((x_coords, y_coords), axis=0).reshape(1, -1)
    return features, result


def draw_landmarks(image, result, prediction):
    annotated = image.copy()
    mp.tasks.vision.drawing_utils.draw_landmarks(
        annotated,
        result.hand_landmarks[0],
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
        mp.tasks.vision.drawing_styles.get_default_hand_landmarks_style(),
        mp.tasks.vision.drawing_styles.get_default_hand_connections_style(),
    )
    h, w, _ = annotated.shape
    hand = result.hand_landmarks[0]
    x_min = int(min(lm.x for lm in hand) * w)
    y_min = int(min(lm.y for lm in hand) * h) - 15
    cv2.putText(
        annotated,
        prediction,
        (x_min, y_min),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 200, 0),
        3,
        cv2.LINE_AA,
    )
    return annotated


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("ASL Hand Pose Recognition")
    st.markdown(
        "This app recognises **American Sign Language** letters **A – J** "
        "from a single photo of your hand."
    )
    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. Show an ASL letter (A–J) to your camera\n"
        "2. Take a snapshot\n"
        "3. The model detects hand landmarks and predicts the letter"
    )
    st.divider()
    st.subheader("Model details")
    st.markdown(
        "- **Landmarks:** MediaPipe Hand Landmarker\n"
        "- **Classifier:** SVM (RBF kernel, C=10)\n"
        "- **Dataset:** 3,648 samples across 10 classes\n"
        "- **Tuning:** GridSearchCV (5-fold)"
    )
    st.divider()
    st.subheader("Realtime mode")
    st.markdown(
        "Want **live, real-time** predictions from your webcam? "
        "Run the following command in your terminal:"
    )
    st.code("python landmark.py", language="bash")

# ── Main area ────────────────────────────────────────────────────────────────

st.header("ASL Letter Recognition")

col_ref, col_cam = st.columns(2)

with col_ref:
    st.subheader("ASL Reference (A – J)")
    st.image("asl.jpg", use_container_width=True)

with col_cam:
    st.subheader("Take a photo")
    picture = st.camera_input("Show an ASL letter (A–J) to the camera")

if picture is not None:
    file_bytes = np.frombuffer(picture.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    landmarker = load_landmarker()
    features, result = extract_landmarks(image_rgb, landmarker)

    if features is not None:
        model = load_model()
        prediction = model.predict(features)[0]
        annotated = draw_landmarks(image_rgb, result, prediction)

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Predicted Letter", value=prediction)
        with res_col2:
            st.image(annotated, caption="Detected hand with landmarks", use_container_width=True)
    else:
        st.warning(
            "No hand detected in the photo. "
            "Make sure your hand is clearly visible and well-lit, then try again."
        )
