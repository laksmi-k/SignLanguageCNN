import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="SignSense | Sign Language AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0b1020;
    color: #edf0ff;
    font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"] {
    background: rgba(11,16,32,0.85);
}
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #10172a;
    border-right: 1px solid #27304a;
}
[data-testid="stSidebar"] * {
    color: #dce3fa;
}
.hero {
    background: linear-gradient(120deg, #171f3c, #241747 70%, #182b4b);
    border: 1px solid #38416b;
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 24px;
}
.eyebrow {
    color: #b6a6ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero h1 {
    color: #ffffff;
    font-size: clamp(28px, 4vw, 42px);
    line-height: 1.15;
    margin: 12px 0;
}
.hero p {
    color: #c3cbe5;
    font-size: 15px;
    line-height: 1.8;
    max-width: 760px;
}
.panel {
    background: #121a2e;
    border: 1px solid #283450;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
}
.panel-title {
    font-size: 17px;
    font-weight: 700;
    color: #f2f4ff;
    margin-bottom: 8px;
}
.muted {
    color: #a5b0cb;
    font-size: 13px;
    line-height: 1.7;
}
.prediction {
    background: linear-gradient(145deg, #1b2847, #20183c);
    border: 1px solid #6252a4;
    border-radius: 22px;
    padding: 25px;
    text-align: center;
    margin: 12px 0 20px;
}
.prediction-label {
    color: #b7a9ff;
    text-transform: uppercase;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}
.prediction-letter {
    font-size: 76px;
    line-height: 1.2;
    font-weight: 800;
    color: #ffffff;
    margin: 8px 0;
}
div[data-testid="stMetric"] {
    background: #151e34;
    border: 1px solid #2d3a58;
    padding: 16px;
    border-radius: 15px;
}
div[data-testid="stMetricLabel"] {
    color: #b8c2dd;
}
.stButton button, .stDownloadButton button {
    border-radius: 10px;
}
.stProgress > div > div > div > div {
    background-color: #a78bfa;
}
hr {
    border-color: #29334c;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("sign_language_cnn.keras")
    with open("class_names.json", "r") as file:
        classes = json.load(file)
    return model, classes


with st.sidebar:
    st.markdown("## 🤟 SignSense")
    st.caption("AI-powered gesture recognition")
    st.divider()
    st.markdown("### Project")
    st.write("**Sign Language Alphabet Recognition Using CNN**")
    st.markdown(
        '<p class="muted">A deep-learning prototype that classifies '
        'individual hand-sign images into 29 classes.</p>',
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown("### Technology")
    st.write("• Python")
    st.write("• TensorFlow / Keras")
    st.write("• Convolutional Neural Network")
    st.write("• Streamlit")
    st.divider()
    st.caption("Academic project • Image classification demo")


st.markdown("""
<div class="hero">
    <div class="eyebrow">Computer Vision · Deep Learning</div>
    <h1>Sign Language<br>Recognition Studio</h1>
    <p>
        Explore hand-sign recognition using a trained Convolutional Neural
        Network. Upload an image or capture a photo, run a prediction, and
        inspect the model's leading class scores.
    </p>
</div>
""", unsafe_allow_html=True)

try:
    model, class_names = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

st.write("")
left, right = st.columns([1, 1.15], gap="large")

with left:
    upload_tab, camera_tab = st.tabs(["📁 Upload image", "📷 Camera"])


    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose a hand-sign image",
            type=["jpg", "jpeg", "png"],
            key="sign_upload"
        )

    with camera_tab:
        camera_file = st.camera_input(
            "Capture a hand sign",
            key="sign_camera"
        )

    image_file = uploaded_file if uploaded_file is not None else camera_file

    if image_file is not None:
        try:
            image = Image.open(image_file).convert("RGB")
            st.image(image, caption="Selected input image", use_container_width=True)
        except Exception as error:
            st.error(f"Unable to open the image: {error}")
            image_file = None

with right:

    if image_file is None:
        st.markdown("""
        <div class="prediction">
            <div class="prediction-label">Awaiting input</div>
            <div style="font-size:54px;margin:12px;">✦</div>
            <div style="font-size:20px;font-weight:700;color:#fff;">
                Your results will appear here
            </div>
            <p class="muted">
                Upload an image or capture a photo to begin.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            image = Image.open(image_file).convert("RGB")
            # Keep the same preprocessing as the original working demo.
            resized_image = image.resize(
                (64, 64), Image.Resampling.LANCZOS
            )
            array = np.asarray(resized_image, dtype=np.float32) / 255.0
            batch = np.expand_dims(array, axis=0)

            with st.spinner("Analyzing hand-sign image..."):
                probabilities = model.predict(batch, verbose=0)[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

            st.markdown(f"""
            <div class="prediction">
                <div class="prediction-label">Predicted class</div>
                <div class="prediction-letter">{predicted_class}</div>
                <div style="color:#c5ccef;font-size:14px;">
                    Model score: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            metric1, metric2 = st.columns(2)
            metric1.metric("Top prediction", predicted_class)
            metric2.metric("Model score", f"{confidence:.2f}%")

            if confidence < 60:
                st.warning(
                    "The model is uncertain. Try a clearer image with "
                    "the hand centered and fully visible."
                )

            st.markdown("#### Top 5 candidate classes")
            top_indices = np.argsort(probabilities)[-5:][::-1]

            for rank, index in enumerate(top_indices, start=1):
                label = class_names[int(index)]
                score = float(probabilities[index])
                st.write(f"**{rank}. {label}**")
                st.progress(min(max(score, 0.0), 1.0))
                st.caption(f"{score * 100:.2f}% model score")

            st.caption(
                "Scores are model outputs, not guaranteed probabilities "
                "of real-world correctness."
            )

        except Exception as error:
            st.error(f"Could not process this image: {error}")

st.divider()

st.markdown("### About this prototype")
a, b, c = st.columns(3)

with a:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Image input</div>
        <div class="muted">
            Upload an image or capture a frame using your device camera.
        </div>
    </div>
    """, unsafe_allow_html=True)

with b:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">CNN inference</div>
        <div class="muted">
            The trained network processes the image and scores the
            available gesture classes.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Current scope</div>
        <div class="muted">
            This prototype classifies individual images. Continuous
            sign-language sentence translation is not implemented.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.caption("SignSense · Sign Language Alphabet Recognition Using CNN")
