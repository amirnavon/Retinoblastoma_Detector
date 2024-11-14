# Retinoblastoma_Detector
 BIU DS17

 # Retinoblastoma Detector 🎯

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

This is a deep learning-based project for detecting **Retinoblastoma**, a rare form of eye cancer, from images of eyes or faces. The tool provides an intuitive **web-based interface** for uploading images, performing predictions, and visualizing results. It uses **PyTorch** for model training, **Mediapipe** for eye detection, and **Streamlit** for the user interface.

---

## 🚀 **Features**

1. **Eye Detection**:
   - Automatically detects eyes in face images using **Mediapipe**.
   - Supports direct analysis of eye photos.

2. **Disease Prediction**:
   - Classifies eyes as either:
     - **Healthy**
     - **Retinoblastoma**

3. **Interactive Web Interface**:
   - Built with **Streamlit** for ease of use.
   - Upload images, adjust thresholds, and view predictions dynamically.

4. **Customizable Model**:
   - Train a Convolutional Neural Network (CNN) with your own data.
   - Includes data augmentation for better generalization.

---

## 🛠️ **Technologies Used**

| Technology          | Purpose                                   |
|----------------------|-------------------------------------------|
| **Python**           | Programming language                     |
| **PyTorch**          | Deep learning model                      |
| **Streamlit**        | Web app for user interaction             |
| **OpenCV (Headless)**| Image preprocessing                      |
| **Mediapipe**        | Eye detection in face images             |
| **Torchvision**      | Data augmentation and preprocessing      |

---

## 📂 **Project Structure**

```plaintext
retinoblastoma-detector/
├── Training/                     # Folder for training images
│   ├── eye/                      # Healthy eye images
│   ├── retinoblastoma/           # Retinoblastoma-affected eye images
├── models/                       # Contains the model definitions
│   ├── detector.py               # CNN model for classification
│   ├── eye_detector.py           # Eye detection logic (Mediapipe)
├── utils/                        # Utility scripts
│   ├── dataset.py                # Dataset loading and preprocessing
│   ├── evaluation.py             # Model evaluation and visualization
├── train.py                      # Script for training the CNN
├── app.py                        # Streamlit app file
├── requirements.txt              # Dependency file for deployment
└── README.md                     # Project documentation (this file)



