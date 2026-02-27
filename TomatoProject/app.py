import streamlit as st
import tensorflow as tf
import keras
from PIL import Image
import numpy as np
import cv2 

# Page Configuration
st.set_page_config(page_title="Tomato Grader", page_icon="🍅", layout="wide")

# Configuration
MODEL_PATH = 'C:/Trained_Models/tomato_grader_model.keras'
# Fixed alphabetical order to match Keras training directory inference
CLASS_NAMES = ['Defective', 'Ripe', 'Unripe']

# Load the Trained Model
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

def segment_tomato(image):
    """Isolates the tomato using wider color masking to prevent misclassification."""
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Broader Masks for Red (catches more shades of ripe tomatoes)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([15, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Mask for Green (Unripe)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask3 = cv2.inRange(hsv, lower_green, upper_green)

    mask = mask1 + mask2 + mask3
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None 

    main_contour = max(contours, key=cv2.contourArea)
    contour_mask = np.zeros(img_bgr.shape[:2], dtype="uint8")
    cv2.drawContours(contour_mask, [main_contour], -1, (255), thickness=cv2.FILLED)
    
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=contour_mask)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_rgb)

# UI Header
st.title("🍅 Tomato Grading System")
st.write("Upload an image of a tomato to isolate it from the background and classify its grade.")
st.markdown("---")

# Image Uploader
uploaded_file = st.file_uploader("Choose a tomato image...", type=["jpg", "jpeg", "png"])

# Prediction Logic
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.image(original_image, caption='Uploaded Image', use_column_width=True)

    with col2:
        with st.spinner("Isolating tomato..."):
            segmented_image = segment_tomato(original_image)
            
            if segmented_image is None:
                st.warning("Masking failed. Falling back to the original image.")
                processing_image = original_image
            else:
                st.image(segmented_image, caption='Segmented Input', use_column_width=True)
                processing_image = segmented_image

    with col3:
        with st.spinner("Classifying..."):
            image_np = np.array(processing_image.resize((224, 224)))
            image_np = np.expand_dims(image_np, axis=0)

            prediction = model.predict(image_np)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index])

            st.write("<br><br>", unsafe_allow_html=True) 
            st.metric(
                label="Predicted Grade", 
                value=predicted_class_name, 
                delta=f"Confidence: {confidence:.1%}",
                delta_color="normal" if predicted_class_name != "Defective" else "inverse"
            )