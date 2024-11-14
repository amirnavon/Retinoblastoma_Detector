import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from models.detector import RetinoblastomaDetector
from models.eye_detector import extract_eyes_from_face

# Initialize the model
device = torch.device("cpu")  # Ensure it runs on CPU
model = RetinoblastomaDetector()
model.load_state_dict(torch.load("models/retinoblastoma_detector.pth", map_location=device))
model.eval()

# Define classes
class_names = ["Healthy", "Retinoblastoma"]

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# Define function for classification
def classify_eye(image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, 1).max().item()
    return class_names[predicted.item()], confidence


# Streamlit App Layout
st.set_page_config(page_title="Retinoblastoma Detector", layout="wide", initial_sidebar_state="expanded")

# Sidebar Section
st.sidebar.title("🩺 Retinoblastoma Detector")
st.sidebar.write("""
This app detects **retinoblastoma** from uploaded images of eyes or faces.
- Upload an image (eye or face).
- Adjust the confidence threshold if needed.
- View results and confidence levels.
""")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Main Section
st.title("🎯 Retinoblastoma Detection App")
st.markdown("""
Detect **retinoblastoma** from eye or face images using deep learning. 
Upload your images and let the app analyze them for signs of the disease.
""")

uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load the uploaded image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect eyes if the photo is of a face
    eyes = extract_eyes_from_face(image)

    if eyes:
        st.subheader(f"👀 Detected {len(eyes)} Eye(s) in the Image")
        results = []
        for idx, eye in enumerate(eyes):
            eye_pil = Image.fromarray(cv2.cvtColor(eye, cv2.COLOR_BGR2RGB))
            prediction, confidence = classify_eye(eye_pil)

            if confidence >= confidence_threshold:
                st.write(f"**Eye {idx + 1}: {prediction} ({confidence:.2%} confidence)**")
                st.progress(confidence)
                st.image(eye, caption=f"Eye {idx + 1}: {prediction}", use_column_width=True)
            else:
                st.write(f"**Eye {idx + 1}: Low Confidence ({confidence:.2%})**")
                st.progress(confidence)

            results.append((prediction, confidence))

        # Results Summary
        st.subheader("📊 Summary of Results")
        healthy_count = sum(1 for r, c in results if r == "Healthy")
        retinoblastoma_count = sum(1 for r, c in results if r == "Retinoblastoma")
        st.write(f"**Healthy Eyes:** {healthy_count}")
        st.write(f"**Detected Retinoblastoma Cases:** {retinoblastoma_count}")

    else:
        # If no eyes are detected, treat the uploaded image as an eye photo
        st.subheader("🖼️ Single Eye Photo Detected")
        eye_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        prediction, confidence = classify_eye(eye_pil)

        if confidence >= confidence_threshold:
            st.write(f"**Prediction: {prediction} ({confidence:.2%} confidence)**")
            st.progress(confidence)
        else:
            st.write(f"**Low Confidence Prediction: {prediction} ({confidence:.2%})**")
            st.progress(confidence)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("""
**Developed by:**  
[Your Name] | AI-Powered Retinoblastoma Detection  
[GitHub](https://github.com/YourGitHubUsername) | [LinkedIn](https://linkedin.com/in/YourProfile)
""")


