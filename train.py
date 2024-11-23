import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utils.dataset import load_dataset
from models.detector import RetinoblastomaDetector
from utils.evaluation import plot_loss
import json
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data_dir = "Training"
train_loader, validation_loader, class_names = load_dataset(data_dir, validation_split=0.2, batch_size=32)

# Initialize the model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RetinoblastomaDetector().to(device)

# Class weights for imbalanced dataset
class_weights = torch.tensor([1.0, 1.05])  # Prioritize 'retinoblastoma'
criterion = CrossEntropyLoss(weight=class_weights.to(device))

optimizer = Adam(model.parameters(), lr=2e-4)

# Training
num_epochs = 20
train_losses, validation_losses = [], []

best_val_loss = float('inf')  # Initialize best validation loss
patience = 3  # Early stopping patience
wait = 0  # Counter for early stopping
threshold = 0.45  # Custom threshold for predictions

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop
    model.eval()
    validation_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Compute validation loss
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            # Apply threshold to predictions
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predictions = (probabilities > threshold).astype(int)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions)
            all_probs.extend(probabilities)

    # Log metrics
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {validation_loss / len(validation_loader):.4f}")
    print("Classification Report (Validation):")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=1))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    train_losses.append(train_loss / len(train_loader))
    validation_losses.append(validation_loss / len(validation_loader))

    # Early stopping logic
    if validation_losses[-1] < best_val_loss:
        best_val_loss = validation_losses[-1]
        torch.save(model.state_dict(), "models/retinoblastoma_detector.pth")
        wait = 0  # Reset wait counter
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# Save loss data
loss_data = {
    "train_losses": train_losses,
    "validation_losses": validation_losses
}
with open("models/losses.json", "w") as f:
    json.dump(loss_data, f)

# Plot the loss curves
plot_loss(train_losses, validation_losses)

print("Training complete. Model and loss data saved.")
