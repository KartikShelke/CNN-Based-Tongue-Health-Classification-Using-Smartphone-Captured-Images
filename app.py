import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === SETTINGS ===
IMG_SIZE = (128, 128)
MODEL_PATH = "tongue_health_model.h5"
GDRIVE_MODEL_URL = "https://drive.google.com/uc?id=1cFc1hSHY02PGTdgWIsX3y3ubyOCbpkU4"

# === MODEL LOADER WITH DOWNLOAD ===
@st.cache_resource
def load_tongue_model():
    if not os.path.exists(MODEL_PATH):
        st.info("🔄 Downloading model...")
        with open(MODEL_PATH, "wb") as f:
            response = requests.get(GDRIVE_MODEL_URL)
            f.write(response.content)
        st.success("✅ Model downloaded.")
    return load_model(MODEL_PATH)

# === TONGUE SEGMENTATION FUNCTION ===
def extract_tongue_region(image):
    img = np.array(image)
    img = cv2.resize(img, (512, 512))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Red-pink tongue ranges
    lower1 = np.array([0, 30, 50])
    upper1 = np.array([20, 255, 255])
    lower2 = np.array([160, 30, 50])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    tongue_mask = np.zeros_like(mask)
    cv2.drawContours(tongue_mask, [largest_contour], -1, 255, -1)

    result = cv2.bitwise_and(img, img, mask=tongue_mask)
    result[tongue_mask == 0] = (255, 255, 255)
    return Image.fromarray(result)

# === PREDICT FUNCTION ===
def predict(image):
    model = load_tongue_model()
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index] * 100
    label = "Healthy (Non-staining moss)" if class_index == 0 else "Unhealthy (Stained moss)"
    return label, confidence

# === STREAMLIT UI ===
st.set_page_config(page_title="Tongue Health Detection", page_icon="🧠")

# Custom CSS for background
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #fff1eb, #ace0f9);
    }
    .stContainer {
        padding: 20px;
    }
    .stImage {
        border: 2px solid #1e3c72;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Tongue Health Detection")
st.markdown("Upload a tongue image, and the model will predict its health status based on color features.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🧪 Processing..."):
        tongue_img = extract_tongue_region(image)

    if tongue_img is not None:
        st.image(tongue_img, caption="👅 Extracted Tongue Region", use_column_width=True)
        label, confidence = predict(tongue_img)
        st.success(f"✅ Prediction: **{label}**")
        st.info(f"🔍 Confidence: **{confidence:.2f}%**")
    else:
        st.error("❌ Could not detect tongue region. Try a clearer or more zoomed-in image.")
