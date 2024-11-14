import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from models.detector import RetinoblastomaDetector
from models.eye_detector import extract_eyes_from_face
import cv2
import numpy as np

# Load the model
model = RetinoblastomaDetector()
model.load_state_dict(torch.load("models/retinoblastoma_detector.pth"))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def classify_eye(eye_image):
    eye_tensor = transform(eye_image).unsqueeze(0)
    outputs = model(eye_tensor)
    _, predicted = torch.max(outputs, 1)
    confidence = torch.softmax(outputs, 1).max().item()
    return predicted.item(), confidence

# Streamlit UI
st.title("Retinoblastoma Detector")
uploaded_file = st.file_uploader("Upload a face or eye photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    eyes = extract_eyes_from_face(image)
    if eyes:
        for i, eye in enumerate(eyes):
            eye_pil = Image.fromarray(cv2.cvtColor(eye, cv2.COLOR_BGR2RGB))
            label, confidence = classify_eye(eye_pil)
            result = "Healthy" if label == 0 else "Retinoblastoma"
            st.image(eye, caption=f"Eye {i+1}: {result} ({confidence:.2%})")
    else:
        st.write("No eyes detected.")

