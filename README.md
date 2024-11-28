
# **🎯 Retinoblastoma Detector**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.1-orange)

## **Introduction**
Retinoblastoma is a rare but aggressive eye cancer, predominantly affecting children under the age of five. Early diagnosis is critical for preserving vision and improving survival rates. This project leverages **Convolutional Neural Networks (CNNs)** to classify eye images as either healthy or affected by retinoblastoma. The project integrates **PyTorch** for training, **data augmentation** for improved generalization, and a **Streamlit** interface for user-friendly, real-time predictions.

---

## **✨ Features**
- **Deep Learning Model**: Custom CNN for binary classification of eye images.
- **Eye Detection with Mediapipe**: Automatically detects and crops eye regions from face images for diagnosis.
- **Data Augmentation**: Techniques like flipping, rotation, and cropping to enhance dataset variability.
- **Interactive Web App**: Upload an image and get predictions with confidence scores.
- **Balanced Loss Function**: Weighted loss to address class imbalance.

---

## **📂 Project Structure**
| File/Directory       | Purpose                                                      |
|-----------------------|-------------------------------------------------------------|
| `train.py`           | Contains the code to train the CNN model, handle data, and apply early stopping. |
| `evaluation.py`      | Evaluates the trained model, computes metrics, and visualizes results. |
| `app.py`             | Implements the Streamlit application for real-time predictions, integrating Mediapipe for eye detection. |
| `utils/dataset.py`   | Handles data loading, preprocessing, and augmentations.       |
| `utils/evaluation.py`| Contains utility functions for metric computation and visualization. |
| `models/`            | Stores the custom model architecture (RetinoblastomaDetector). |
| `Training/`          | Directory to hold the dataset with subfolders for healthy and diseased images. |
| `models/losses.json` | Stores training and validation losses for analysis.          |
| `models/retinoblastoma_detector.pth` | Saved weights of the trained model.           |

----

## **💻 Technologies Used**
| Technology           | Purpose                                                      |
|-----------------------|-------------------------------------------------------------|
| **Python**           | Main programming language used for the project.              |
| **PyTorch**          | Deep learning framework used for model development.          |
| **Torchvision**      | Provides utilities for image transformations and datasets.   |
| **Mediapipe**        | Detects and crops eye regions from face images.              |
| **Streamlit**        | Web-based interface for real-time interaction with the model.|
| **Pillow (PIL)**     | Image handling library for preprocessing and manipulation.   |
| **Scikit-learn**     | Utility functions for metrics such as classification report. |
| **JSON**             | Stores training and validation losses for plotting.          |

---

## **📂 Dataset**
The dataset is organized as follows:
```
Training/
├── eye/                      # Healthy eye images
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
├── retinoblastoma/           # Retinoblastoma-affected eye images
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
```

**Data Augmentation**:
- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Random Resized Crop

---

## **🚀 Pipeline Overview**

### **1. Data Preparation**
- Augmentations are applied to increase the dataset's robustness.
- Weighted cross-entropy loss balances class distribution.

### **2. Model Training**
- **Architecture**:
  - Convolutional layers extract spatial features.
  - Pooling layers reduce dimensionality.
  - Fully connected layers classify features.
- **Optimizer**: Adam optimizer with a learning rate of `2e-4`.
- **Early Stopping**: Stops training when validation loss does not improve.

### **3. Evaluation**
- **Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC Score
- **Sample Results**:
```
Classification Report:
                precision    recall  f1-score   support

           eye       0.94      0.94      0.94        16
retinoblastoma       0.93      0.93      0.93        15

      accuracy                           0.94        31
     macro avg       0.94      0.94      0.94        31
  weighted avg       0.94      0.94      0.94        31

ROC-AUC Score: 0.97
```
- **Confusion Matrix**:
```
[[15, 1],
 [ 1, 14]]
```

### **4. Interactive Analysis**
- The **Streamlit App** allows:
  - Image uploads.
  - Class predictions and confidence scores.
  - Threshold adjustments for precision-recall trade-offs.

---

## **⚙️ Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/retinoblastoma-detector.git
cd retinoblastoma-detector
```

### **2. Install Dependencies**
Install Poetry if not already installed:
```bash
pip install poetry
```

Then, install the project dependencies:
```bash
poetry install
```

### **3. Prepare the Dataset**
Organize the dataset into the following structure:
```
Training/
├── eye/
├── retinoblastoma/
```
Augmentations will be applied automatically during training.

### **4. Train the Model**
Run the training script:
```bash
poetry run python train.py
```

### **5. Evaluate the Model**
Run the evaluation script:
```bash
poetry run python evaluation.py
```

### **6. Launch the Web App**
Run the Streamlit app:
```bash
poetry run streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

---

## **🌍 Real-World Applications**
1. **Telemedicine**:
   - Enables remote diagnosis using standard smartphone images.
2. **Pre-Screening Tool**:
   - Flags high-risk cases for further examination.
3. **Low-Resource Settings**:
   - Compatible with non-specialized equipment for wider accessibility.

---

## **🚧 Future Improvements**
1. **Dataset Expansion**:
   - Acquire larger, diverse datasets from various demographics and real-world scenarios to improve model accuracy and generalizability.

2. **Advanced Augmentation**:
   - Implement techniques such as low-light adjustments and varying angles for better real-world robustness.

3. **Pretrained Models**:
   - Use pretrained architectures like ResNet or EfficientNet to enhance feature extraction and accuracy.

4. **Explainable AI**:
   - Add Grad-CAM visualizations to interpret model predictions and build user trust.

5. **Enhanced Usability**:
   - Support batch image uploads and develop a lightweight mobile app for remote access.

---

## **📜 License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **📬 Contact**
For feedback or contributions:
- **Name**: Taina Trahtenberg , Adi Albeg, Amir Navon
- **GitHub**: [Your GitHub Profile](https://github.com/amirnavon)

