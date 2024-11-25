import torch
from torchvision import datasets, transforms
from PIL import Image
import mediapipe as mp
import numpy as np
import os

# Mediapipe helper for eye extraction
def extract_eyes_from_faces(image):
    """
    Extract eye regions from face images using Mediapipe.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        list: A list of cropped eye regions as PIL images.
    """
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(np.array(image))
        if results.detections:
            cropped_eyes = []
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = np.array(image).shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                eye_region = image.crop((x, y, x + w, y + h))
                cropped_eyes.append(eye_region)
            return cropped_eyes
    return []

# Dataset loader with preprocessing
def load_dataset(data_dir, validation_split=0.2, batch_size=32, preprocess_faces=True):
    """
    Load the dataset, preprocess face images (if required), and apply transformations.

    Args:
        data_dir (str): Directory containing the dataset.
        validation_split (float): Proportion of data to use for validation.
        batch_size (int): Number of samples per batch.
        preprocess_faces (bool): Whether to preprocess face images to extract eyes.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for validation data.
        class_names (list): List of class names.
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Preprocess face images if needed
    if preprocess_faces:
        face_dir = os.path.join(data_dir, "faces")
        eye_dir = os.path.join(data_dir, "eyes")
        os.makedirs(eye_dir, exist_ok=True)
        for face_file in os.listdir(face_dir):
            face_path = os.path.join(face_dir, face_file)
            if os.path.isfile(face_path):
                image = Image.open(face_path).convert("RGB")
                cropped_eyes = extract_eyes_from_faces(image)
                for idx, eye in enumerate(cropped_eyes):
                    eye.save(os.path.join(eye_dir, f"{os.path.splitext(face_file)[0]}_eye{idx + 1}.jpg"))

    # Load full dataset with transformations
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    # Split dataset into training and validation
    train_size = int((1 - validation_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )

    # Apply validation transformations to the validation dataset
    validation_dataset.dataset.transform = validation_transform

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Get class names
    class_names = full_dataset.classes

    return train_loader, validation_loader, class_names
