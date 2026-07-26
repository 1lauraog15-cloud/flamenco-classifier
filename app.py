import sys
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from flamenco import extract_features
from palo_templates import PALO_TEMPLATES

MODEL_PATH = os.path.join(_HERE, "flamenco_classifier_v3.joblib")
FALLBACK_TEMPLATE = PALO_TEMPLATES["soleá"]
CONFIDENCE_THRESHOLD = 0.4

PALO_LABELS = {
    "soleares":  "Soleares",
    "seguiriyas": "Seguiriyas",
    "bulerias":  "Bulerías",
    "alegrias":  "Alegrías",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ── Layout ──────────────────────────────────────────────────────────────────
st.title("Flamenco Palo Classifier")
st.markdown(
    "Classifies a flamenco audio recording into one of four palos: "
    "**soleares**, **seguiriyas**, **bulerías**, or **alegrías**. "
    "The model uses 47 musicologically informed audio features "
    "(chroma, F0, MFCCs, Phrygian mode indicators, positional beat-cycle features) "
    "trained on the [cante100](https://mtg.upf.edu/research/datasets/cante100) dataset."
)

st.divider()

uploaded = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

if uploaded is not None:
    suffix = ".mp3" if uploaded.name.lower().endswith(".mp3") else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Extracting audio features… this takes 20–40 seconds."):
        try:
            feats = extract_features(tmp_path, palo_template=FALLBACK_TEMPLATE)
        except Exception as exc:
            st.error(f"Feature extraction failed: {exc}")
            os.unlink(tmp_path)
            st.stop()

    os.unlink(tmp_path)

    try:
        model_data = load_model()
        pipeline = model_data["pipeline"]
        feature_cols = model_data["feature_cols"]
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()

    X = pd.DataFrame([feats])[feature_cols]
    proba = pipeline.predict_proba(X)[0]
    classes = pipeline.classes_

    max_idx = int(np.argmax(proba))
    predicted = classes[max_idx]
    confidence = float(proba[max_idx])

    st.divider()

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠️ **Uncertain prediction** (max confidence: {confidence:.0%}). "
            "The recording may not belong to any of the four target palos, "
            "or the audio quality may be too low for reliable classification."
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Predicted palo: **{PALO_LABELS.get(predicted, predicted)}**")
    with col2:
        st.metric("Confidence", f"{confidence:.0%}")

    st.markdown("#### Probability by palo")

    sorted_pairs = sorted(zip(classes, proba), key=lambda x: -x[1])
    for cls, prob in sorted_pairs:
        label = PALO_LABELS.get(cls, cls)
        st.markdown(f"**{label}**")
        st.progress(float(prob), text=f"{prob:.0%}")
