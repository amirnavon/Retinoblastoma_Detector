import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import numpy as np
from models.detector import RetinoblastomaDetector

# Set page configuration - must be the first Streamlit command
st.set_page_config(page_title="Retinoblastoma Detector", layout="wide")

# Initialize the model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinoblastomaDetector().to(device)
    model.load_state_dict(torch.load("models/retinoblastoma_detector.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Helper function for prediction
def predict(image, model, device, threshold=0.5):  # Match training threshold
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        confidence = probabilities.max()
        predicted_class = 1 if probabilities[1] > threshold else 0
    return predicted_class, confidence, probabilities

# App title
st.title("🔍 Retinoblastoma Detector")
st.markdown("""
Upload a photo of an eye or face to detect retinoblastoma and assess the risk level. 
This AI-powered tool uses deep learning to provide reliable predictions.
""")

# Sidebar for settings
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)  # Default threshold = training threshold
st.sidebar.write(f"Current Threshold: {threshold:.2f}")

# File uploader
uploaded_file = st.file_uploader("Upload a Photo (JPEG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Make prediction
    predicted_class, confidence, probabilities = predict(image, model, device, threshold)

    # Class mapping
    class_names = ["Healthy Eye", "Retinoblastoma"]
    prediction = class_names[predicted_class]

    # Display prediction
    st.subheader("Prediction Results")
    if confidence >= threshold:
        st.success(f"Prediction: **{prediction}** (Confidence: {confidence:.2%})")
    else:
        st.warning(f"Prediction: Uncertain (Confidence: {confidence:.2%})")

    # Improved Confidence Display
    st.subheader("Confidence Scores")
    for i, score in enumerate(probabilities):
        st.write(f"{class_names[i]}: {score:.2%}")
        st.progress(int(score * 100))

    # Textual Interpretation based on confidence levels
    if confidence >= 0.7:
        st.info(f"The model has **high confidence** in the prediction: {prediction}.")
    elif 0.5 <= confidence < 0.7:
        st.warning(f"The model has **moderate confidence** in the prediction: {prediction}. Further testing is recommended.")
    else:
        st.error(f"The model is **uncertain** about the prediction. Confidence: {confidence:.2%}")

    # Visual feedback (Draw rectangle or highlight region)
    draw = ImageDraw.Draw(image)
    if prediction == "Retinoblastoma":
        draw.rectangle([(10, 10), (214, 214)], outline="red", width=5)
    else:
        draw.rectangle([(10, 10), (214, 214)], outline="green", width=5)
    st.image(image, caption="Processed Image", use_container_width=True)

# Add prediction history
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file and confidence >= threshold:
    st.session_state.history.append({"Image": uploaded_file.name, "Prediction": prediction, "Confidence": confidence})

if st.session_state.history:
    st.subheader("Prediction History")
    st.table(st.session_state.history)

# Footer
st.markdown("---")
# st.markdown("📧 **For questions or feedback, contact us at**: support@example.com")
st.markdown("💻 **Source code available on [GitHub](https://github.com/amirnavon/Retinoblastoma_Detector.git)**")
