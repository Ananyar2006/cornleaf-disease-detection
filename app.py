import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import base64
import os

# Background Image
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
        encoded_string = base64.b64encode(data).decode()

        st.markdown(f"""
            <style>
            .stApp {{
                background:
                    linear-gradient(rgba(0, 0, 0, 0.48), rgba(0, 0, 0, 0.48)),
                    url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
        """, unsafe_allow_html=True)

# Model Loading
@st.cache_resource
def load_model():
    try:
        model_dict = joblib.load("corn_leaf.pkl")
        rf_model = model_dict["random_forest"]
        scaler = model_dict["scaler"]
        svm_model = model_dict["svm"]
        label_encoder = model_dict["label_encoder"]
        return rf_model, scaler, svm_model, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

rf_model, scaler, svm_model, label_encoder = load_model()

# Disease Information
disease_info = {
    "Healthy":
        "The corn leaf appears healthy with no visible disease symptoms.",

    "Common_Rust":
        "Common Rust is a fungal disease that appears as reddish-brown pustules on the leaf surface. Severe infection can reduce photosynthesis and affect crop productivity.",

    "Blight":
        "Blight causes elongated tan or brown lesions on leaves. Under favorable environmental conditions, the disease can spread rapidly and reduce yield.",

    "Gray_Leaf_Spot":
        "Gray Leaf Spot produces rectangular gray-colored lesions on leaves and may significantly reduce photosynthetic activity and crop performance."
}

# Background
add_bg_from_local("background.png")

# Title
st.title("🌽 AI-Powered Corn Leaf Disease Detection System")
st.write("Upload a corn leaf image to identify whether the leaf is healthy or affected by disease.")

# Upload Image
uploaded_file = st.file_uploader("Choose a corn leaf image...", type=["jpg", "jpeg", "png"])

# Feature Extraction
def extract_hsv_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Prediction
def predict_leaf_disease(image_pil):
    image = np.array(image_pil.convert("RGB"))
    image_resized = cv2.resize(image, (128, 128))
    features = extract_hsv_histogram(image_resized).reshape(1, -1)
    prf_features = rf_model.predict_proba(features)
    prf_scaled = scaler.transform(prf_features)
    probabilities = svm_model.predict_proba(prf_scaled)
    prediction = np.argmax(probabilities)
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    confidence = np.max(probabilities)
    return predicted_class, confidence, probabilities

# Confidence Graph
def show_probabilities(probabilities, class_names):
    st.subheader("🔬 Class Confidence")
    fig, ax = plt.subplots(figsize=(8, 4))
    max_prob = np.max(probabilities[0])
    colors = ["green" if p == max_prob else "skyblue" for p in probabilities[0]]
    ax.barh(class_names, probabilities[0], color=colors)
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    for i, value in enumerate(probabilities[0]):
        ax.text(value + 0.01, i, f"{value*100:.1f}%")
    st.pyplot(fig)

# Prediction Section
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", width="stretch")
    st.write("⏳ Analyzing...")
    try:
        label, confidence, probabilities = predict_leaf_disease(image)
        st.success(f"Predicted Disease: **{label}**")
        st.info(f"Confidence: **{confidence * 100:.2f}%**")
        show_probabilities(probabilities, label_encoder.classes_)
        st.subheader("🩺 Disease Information")
        st.info(disease_info.get(label, "Information unavailable."))
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
