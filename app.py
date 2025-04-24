import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

IMG_SIZE = (128, 128)
MODEL_PATH = "tongue_health_model.h5"

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

def extract_tongue_region(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512, 512))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 50])
    upper = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)

    lower2 = np.array([160, 30, 50])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)
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
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

def predict(image):
    model = load_trained_model()
    image = image.resize(IMG_SIZE)
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index] * 100
    class_name = "Healthy (Non-staining moss)" if class_index == 0 else "Unhealthy (Stained moss)"
    return class_name, confidence

# === STREAMLIT UI ===
st.title("🧠 Tongue Health Detection from Image")
st.markdown("Upload a tongue image to predict its health status using a CNN model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    tongue_img = extract_tongue_region(image)
    if tongue_img is not None:
        st.image(tongue_img, caption="Extracted Tongue Region", use_column_width=True)
        label, confidence = predict(tongue_img)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
    else:
        st.error("Could not detect tongue region. Try a clearer image.")
