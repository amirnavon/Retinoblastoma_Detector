import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import numpy as np
import mediapipe as mp
from models.detector import RetinoblastomaDetector

# Set page configuration
st.set_page_config(page_title="Retinoblastoma Detector", layout="wide", page_icon="🔍")

# Load the model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinoblastomaDetector().to(device)
    model.load_state_dict(torch.load("models/retinoblastoma_detector.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# Mediapipe helper
mp_face_detection = mp.solutions.face_detection

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Helper function to extract eyes from face
def extract_eyes_from_face(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(np.array(image))
        if results.detections:
            cropped_eyes = []
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = np.array(image).shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                # Split the face into left and right halves for eyes
                left_eye = image.crop((x, y, x + w // 2, y + h))
                right_eye = image.crop((x + w // 2, y, x + w, y + h))
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                draw.text((x + 5, y - 10), "Face", fill="red")
                cropped_eyes.extend([("Left Eye", left_eye), ("Right Eye", right_eye)])
            return cropped_eyes, draw_image
    return None, image

# Helper function to determine if the image is a single eye
def is_single_eye_image(image):
    width, height = image.size
    return width / height < 1.2

# Helper function to detect eyes directly
def detect_eyes_directly(image):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    width, height = image.size
    mid_x = width // 2

    # Assume the left and right halves correspond to left and right eyes
    left_eye_region = (0, 0, mid_x, height)
    right_eye_region = (mid_x, 0, width, height)

    left_eye = image.crop(left_eye_region)
    right_eye = image.crop(right_eye_region)

    # Draw rectangles and label regions
    draw.rectangle(left_eye_region, outline="blue", width=3)
    draw.text((5, 5), "Left Eye", fill="blue")
    draw.rectangle(right_eye_region, outline="blue", width=3)
    draw.text((mid_x + 5, 5), "Right Eye", fill="blue")

    return [("Left Eye", left_eye), ("Right Eye", right_eye)], draw_image

# Helper function for prediction
def predict(image, model, device, threshold=0.45):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        confidence = probabilities.max()
        predicted_class = 1 if probabilities[1] > threshold else 0
    return predicted_class, confidence, probabilities

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# App title
st.title("🔍 **Retinoblastoma Detector**")
st.markdown("""
Upload a photo of an eye or face to detect retinoblastoma and assess the risk level. 
This AI-powered tool uses deep learning to provide reliable predictions.
""")

# Sidebar for settings
st.sidebar.header("🔧 **Settings**")
threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.05)
st.sidebar.write(f"Current Threshold: **{threshold:.2f}**")

# User-assisted mode selection
mode = st.sidebar.selectbox(
    "Select Input Type",
    options=["Auto-Detect", "Single Eye", "Two Eyes", "Face"],
    index=0
)

# File uploader
uploaded_file = st.file_uploader("📤 **Upload a Photo (JPEG or PNG)**", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Eye detection and diagnosis
    st.subheader("📋 **Detection and Diagnosis**")

    if mode == "Auto-Detect":
        # Attempt to extract eyes from face
        cropped_eyes, annotated_image = extract_eyes_from_face(image)
        if not cropped_eyes:
            st.warning("⚠️ No faces detected. Attempting to detect eye regions directly.")
            cropped_eyes, annotated_image = detect_eyes_directly(image)
    elif mode == "Face":
        cropped_eyes, annotated_image = extract_eyes_from_face(image)
    elif mode == "Single Eye":
        cropped_eyes, annotated_image = [("Single Eye", image)], image
    elif mode == "Two Eyes":
        cropped_eyes, annotated_image = detect_eyes_directly(image)

    st.image(annotated_image, caption="Detected Regions (Highlighted)", use_container_width=True)

    # Diagnose each eye
    for idx, (label, eye) in enumerate(cropped_eyes):
        predicted_class, confidence, probabilities = predict(eye, model, device, threshold)
        class_names = ["Healthy Eye", "Retinoblastoma"]
        prediction = class_names[predicted_class]

        st.write(f"### Prediction for {label}: **{prediction}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        st.progress(int(confidence * 100))

        # Add to history
        st.session_state.history.append({
            "filename": uploaded_file.name,
            "eye_label": label,
            "prediction": prediction,
            "confidence": f"{confidence:.2%}"
        })

# Display history as a table
if st.session_state.history:
    st.subheader("📜 **Diagnosis History**")
    st.write("Below is the summary of your uploaded images:")
    history_table = [
        {
            "File Name": record["filename"],
            "Eye Label": record["eye_label"],
            "Prediction": record["prediction"],
            "Confidence": record["confidence"]
        }
        for record in st.session_state.history
    ]
    st.table(history_table)

# Footer
st.markdown("---")
st.markdown("💻 **Source code available on [GitHub](https://github.com/amirnavon/Retinoblastoma_Detector.git)**")
