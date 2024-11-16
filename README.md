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
```
---

## ⚙️ **Setup and Installation**

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/retinoblastoma-detector.git
cd retinoblastoma-detector
```

### 2. Install Dependencies
This project uses Poetry for dependency management. If you don’t have Poetry installed, you can install it with pip:
```bash
pip install poetry
```

Then, install the dependencies:
```bash
poetry install
```

### 3. Prepare Training Data
Organize the `Training/` folder with the following structure:
```
Training/
├── eye/                      # Healthy eye images
│   ├── image1.jpg
│   ├── image2.jpg
├── retinoblastoma/           # Retinoblastoma-affected eye images
│   ├── image1.jpg
│   ├── image2.jpg
```

You can use your own images or publicly available datasets. Ensure there are sufficient samples in each category for effective training.

### 4. Train the Model
Run the training script:
```bash
poetry run python train.py
```

This script:
- Loads the training data.
- Trains a CNN to classify healthy and diseased eyes.
- Saves the trained model as `models/retinoblastoma_detector.pth`.

---

## 🌐 Run the Streamlit App
Launch the Streamlit app with the following command:
```bash
poetry run streamlit run app.py
```

Then open your browser and navigate to:
```
http://localhost:8501
```

---

## 📊 How It Works

1. **Eye Detection**:
   - Face images are analyzed using Mediapipe to locate and crop eye regions.

2. **Image Preprocessing**:
   - Images are resized to `224x224`.
   - Pixel values are normalized to `[0.5, 0.5]` mean and standard deviation.

3. **Model Architecture**:
   - The CNN is designed to classify eyes into two categories:
     - Healthy
     - Retinoblastoma

4. **Inference**:
   - Each cropped eye is passed through the trained model.
   - The app outputs the classification along with a confidence score.

---

## 📈 Future Improvements

- **Pretrained Models**:
  - Integrate pretrained models like ResNet for higher accuracy.
  
- **Dataset Expansion**:
  - Incorporate larger and more diverse datasets for better generalization.

- **Batch Uploads**:
  - Add support for analyzing multiple images in a single session.

- **Mobile Optimization**:
  - Improve the Streamlit interface for better usability on mobile devices.

---

## 🤝 Contributing

We welcome contributions to improve this project! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request on GitHub.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 📬 Contact

For questions or feedback, feel free to reach out:

- **Your Name**
- [Your GitHub Profile](https://github.com/your-username)
- [Your Email Address](mailto:youremail@example.com)

---

