import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Emotion Labels
# -------------------------------
class_names = [
    "😲 Surprise",
    "😨 Fear",
    "🤢 Disgust",
    "😄 Happy",
    "😢 Sad",
    "😠 Angry",
    "😐 Neutral"
]

# -------------------------------
# Load Model
# -------------------------------
model = load_model("best_CNNModel.keras")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("😊 Emotion Detection App")
st.write("Upload a face image and AI will predict the emotion.")

# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # Preprocess (match training)
    # -------------------------------
    image = image.resize((100, 100))
    image = np.array(image) / 255.0

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = np.expand_dims(image, axis=0)

    # -------------------------------
    # Predict
    # -------------------------------
    prediction = model.predict(image)[0]

    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100

    # -------------------------------
    # Main Result
    # -------------------------------
    st.success(f"### Prediction: {class_names[predicted_class]}")
    st.info(f"Confidence: {confidence:.2f}%")

    # -------------------------------
    # Show Top Probabilities
    # -------------------------------
    st.subheader("All Emotion Probabilities")

    for i, prob in enumerate(prediction):
        st.write(f"{class_names[i]} — {prob*100:.2f}%")
        st.progress(float(prob))
