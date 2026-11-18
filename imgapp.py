import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import base64

# Set background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded_string = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Safe model loading block
@st.cache_resource
def load_model():
    try:
        model_path = "corn_leaf.pkl"
        model_dict = joblib.load(model_path)

        rf_model = model_dict["random_forest"]
        scaler = model_dict["scaler"]
        svm_model = model_dict["svm"]
        label_encoder = model_dict["label_encoder"]

        return rf_model, scaler, svm_model, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()  # Stop the app if model fails to load

# Load models
rf_model, scaler, svm_model, label_encoder = load_model()

add_bg_from_local("back.jpg")
# App title and instruction
st.title("🌽 Corn Leaf Disease Detection")
st.write("Upload a corn leaf image to detect if it is **healthy** or **diseased**.")

# Upload image
uploaded_file = st.file_uploader("Choose a corn leaf image...", type=["jpg", "jpeg", "png"])

# Feature extraction (HSV histogram)
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

# Plot probabilities
def show_probabilities(probabilities, class_names):
    st.subheader("🔬 Class Confidence")
    fig, ax = plt.subplots()
    ax.barh(class_names, probabilities[0], color='teal')
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

# Run prediction if image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
    st.write("⏳ Analyzing...")

    try:
        label, confidence, probabilities = predict_leaf_disease(image)
        st.success(f"Predicted Class: **{label}**")
        st.info(f"Confidence: **{confidence * 100:.2f}%**")

        show_probabilities(probabilities, label_encoder.classes_)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
