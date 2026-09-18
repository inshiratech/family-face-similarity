from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


ROOT = Path(__file__).parent
DETECTOR_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = ROOT / "models" / "face_recognition_sface_2021dec_int8.onnx"


st.set_page_config(
    page_title="Family Face Similarity",
    page_icon="👨‍👩‍👧",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem;}
    [data-testid="stFileUploader"] {background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: .65rem .85rem;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 20px; color: white;
        background: linear-gradient(120deg, #4338ca, #7c3aed 55%, #db2777); margin-bottom: 1.4rem;}
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.25rem;}
    .hero p {margin: 0; opacity: .92; font-size: 1.05rem;}
    .score-card {text-align:center; background:#faf5ff; border:1px solid #e9d5ff;
        border-radius:18px; padding:1.2rem; margin:.4rem 0 1rem;}
    .score {font-size:3.2rem; line-height:1; font-weight:800; color:#6d28d9;}
    .score-label {color:#475569; margin-top:.45rem;}
    .fineprint {color:#64748b; font-size:.87rem;}
    </style>
    <div class="hero">
      <h1>Family Face Similarity</h1>
      <p>Compare a child with a parent now — and optionally with the parent at a similar age.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@dataclass
class Face:
    row: np.ndarray
    crop_rgb: np.ndarray
    quality: float


@st.cache_resource
def load_models():
    if not DETECTOR_MODEL.exists() or not RECOGNIZER_MODEL.exists():
        raise FileNotFoundError("Face-model files are missing from the app.")
    detector = cv2.FaceDetectorYN.create(str(DETECTOR_MODEL), "", (320, 320), 0.55, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_MODEL), "")
    return detector, recognizer


def uploaded_to_bgr(uploaded) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def face_quality(image: np.ndarray, row: np.ndarray) -> float:
    x, y, w, h = [int(v) for v in row[:4]]
    x, y = max(x, 0), max(y, 0)
    roi = image[y : min(y + h, image.shape[0]), x : min(x + w, image.shape[1])]
    if roi.size == 0:
        return 0.0
    sharpness = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    size_ratio = (w * h) / float(image.shape[0] * image.shape[1])
    sharp_score = np.clip(np.log1p(sharpness) / 6.0, 0, 1)
    size_score = np.clip(size_ratio / 0.05, 0, 1)
    detection_score = float(row[-1])
    return float(0.45 * sharp_score + 0.30 * size_score + 0.25 * detection_score)


def detect_faces(image: np.ndarray) -> list[Face]:
    detector, _ = load_models()
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, rows = detector.detect(image)
    if rows is None:
        return []
    found: list[Face] = []
    for row in rows:
        x, y, fw, fh = [int(v) for v in row[:4]]
        pad = int(max(fw, fh) * 0.20)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + fw + pad), min(h, y + fh + pad)
        crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        found.append(Face(row=row, crop_rgb=crop, quality=face_quality(image, row)))
    return sorted(found, key=lambda f: float(f.row[0]))


def annotate(image: np.ndarray, faces: list[Face]) -> np.ndarray:
    canvas = image.copy()
    for number, face in enumerate(faces, 1):
        x, y, w, h = [int(v) for v in face.row[:4]]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (124, 58, 237), 5)
        cv2.putText(canvas, str(number), (x, max(35, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (124, 58, 237), 4, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def embedding(image: np.ndarray, face: Face) -> np.ndarray:
    _, recognizer = load_models()
    aligned = recognizer.alignCrop(image, face.row)
    feature = recognizer.feature(aligned).flatten().astype(np.float32)
    return feature / (np.linalg.norm(feature) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def display_score(raw_cosine: float) -> int:
    # A bounded, entertainment-only presentation scale. It is intentionally
    # labelled an index rather than a probability of kinship.
    value = 100.0 / (1.0 + np.exp(-7.0 * (raw_cosine - 0.18)))
    return int(round(np.clip(value, 0, 100)))


def quality_label(q: float) -> str:
    if q >= 0.72:
        return "High"
    if q >= 0.52:
        return "Moderate"
    return "Low"


def upload_panel(label: str, help_text: str, key: str, optional: bool = False):
    suffix = " (optional)" if optional else ""
    st.subheader(label + suffix)
    st.caption(help_text)
    return st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"], key=key,
                            label_visibility="collapsed")


st.markdown("### 1. Add the photos")
c1, c2, c3 = st.columns(3)
with c1:
    child_file = upload_panel("Child", "Use a clear, front-facing photo.", "child")
with c2:
    parent_file = upload_panel("Parent now", "Neutral expression works best.", "parent")
with c3:
    childhood_file = upload_panel("Parent as a child", "Adds an age-comparable result.", "past", True)


def prepare(uploaded, title: str, allow_choice: bool = False):
    image = uploaded_to_bgr(uploaded)
    faces = detect_faces(image)
    if not faces:
        st.error(f"No face was detected in **{title}**. Try a clearer or more front-facing photo.")
        return image, None
    selected = 0
    if allow_choice and len(faces) > 1:
        st.markdown(f"#### Select the correct face in {title}")
        st.image(annotate(image, faces), caption="Detected faces are numbered from left to right.")
        selected = st.selectbox("Which face should be compared?", range(len(faces)),
                                format_func=lambda i: f"Face {i + 1}", key=f"pick-{title}")
    return image, faces[selected]


if child_file and parent_file:
    try:
        child_img, child_face = prepare(child_file, "the child photo", True)
        parent_img, parent_face = prepare(parent_file, "the current parent photo", True)
        past_img, past_face = (None, None)
        if childhood_file:
            past_img, past_face = prepare(childhood_file, "the childhood photo", True)

        if child_face and parent_face:
            st.divider()
            st.markdown("### 2. Instant comparison")
            child_vec = embedding(child_img, child_face)
            parent_vec = embedding(parent_img, parent_face)
            current_raw = cosine(child_vec, parent_vec)
            current_score = display_score(current_raw)

            past_raw = past_score = None
            if past_face is not None:
                past_vec = embedding(past_img, past_face)
                past_raw = cosine(child_vec, past_vec)
                past_score = display_score(past_raw)

            if past_score is None:
                overall = current_score
            else:
                # Weight each result by image quality, while keeping the clear
                # current-parent image as the anchor comparison.
                w_current = max(0.35, parent_face.quality)
                w_past = max(0.15, past_face.quality) * 0.8
                overall = int(round((current_score * w_current + past_score * w_past) /
                                    (w_current + w_past)))

            st.markdown(
                f'<div class="score-card"><div class="score">{overall}/100</div>'
                '<div class="score-label">Overall facial resemblance index</div></div>',
                unsafe_allow_html=True,
            )

            r1, r2 = st.columns(2)
            with r1:
                st.metric("Child ↔ parent now", f"{current_score}/100")
                st.caption(f"Image confidence: {quality_label(min(child_face.quality, parent_face.quality))}")
            with r2:
                if past_score is not None:
                    st.metric("Child ↔ parent at a similar age", f"{past_score}/100")
                    st.caption(f"Image confidence: {quality_label(min(child_face.quality, past_face.quality))}")
                else:
                    st.metric("Similar-age comparison", "Not added")
                    st.caption("Upload a childhood photo of the parent to add this result.")

            st.markdown("#### Faces used")
            previews = st.columns(3 if past_face is not None else 2)
            previews[0].image(child_face.crop_rgb, caption="Child", use_container_width=True)
            previews[1].image(parent_face.crop_rgb, caption="Parent now", use_container_width=True)
            if past_face is not None:
                previews[2].image(past_face.crop_rgb, caption="Parent as a child", use_container_width=True)

            with st.expander("How this score works"):
                st.write(
                    "The app detects and aligns each face, converts it into a numerical facial embedding, "
                    "and compares the embeddings using cosine similarity. The 0–100 result is a friendly "
                    "presentation of that model similarity, adjusted for image quality."
                )
                st.caption(
                    f"Technical similarity — current: {current_raw:.3f}"
                    + (f" · similar-age: {past_raw:.3f}" if past_raw is not None else "")
                )
    except Exception as exc:
        st.error("The comparison could not be completed. Please try different images.")
        with st.expander("Technical details"):
            st.code(str(exc))
else:
    st.info("Upload the child and current parent photos to generate the score. The childhood photo is optional.")

st.divider()
st.markdown(
    "<p class='fineprint'><b>Just for fun.</b> This is a computational resemblance index, not a DNA, "
    "paternity, maternity, identity, or biological-relationship test. Uploaded photos are processed "
    "in memory and are not intentionally retained by this app.</p>",
    unsafe_allow_html=True,
)
