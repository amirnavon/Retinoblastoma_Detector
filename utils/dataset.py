import torch
from torchvision import datasets, transforms

def load_dataset(data_dir, validation_split=0.2, batch_size=32):
    """
    Load the dataset, split into training and validation, and apply appropriate transformations.

    Args:
        data_dir (str): Directory containing the dataset.
        validation_split (float): Proportion of data to use for validation.
        batch_size (int): Number of samples per batch.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for validation data.
        class_names (list): List of class names.
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),               # Randomly flip images horizontally
        transforms.RandomRotation(degrees=15),                # Random rotation within ±15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random changes to brightness and contrast
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # Random crop and resize
        transforms.ToTensor(),                                # Convert to PyTorch tensor
        transforms.Normalize([0.5], [0.5])                    # Normalize pixel values
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),                        # Resize to match the training pipeline
        transforms.ToTensor(),                                # Convert to PyTorch tensor
        transforms.Normalize([0.5], [0.5])                    # Normalize using same stats as training
    ])

    # Load full dataset with training transformations
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
