import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
from models.detector import RetinoblastomaDetector
from utils.dataset import load_dataset
from utils.evaluation import evaluate_model, plot_loss
import json

# Load data
data_dir = "Training"
train_loader, class_names = load_dataset(data_dir)

# Split the dataset into training and testing
train_size = int(0.7 * len(train_loader.dataset))
validation_size = len(train_loader.dataset) - train_size
train_dataset, validation_dataset = random_split(train_loader.dataset, [train_size, validation_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=False)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RetinoblastomaDetector().to(device)

# criterion = CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=0.001)

# Class weights based on the dataset distribution
class_weights = torch.tensor([1.0, 1.0])  # Slightly prioritize 'retinoblastoma'
criterion = CrossEntropyLoss(weight=class_weights.to(device))

optimizer = Adam(model.parameters(), lr=2e-4)

# Training
num_epochs = 25
train_losses, validation_losses = [], []

best_val_loss = float('inf')  # Initialize the best validation loss
patience = 100  # Number of epochs to wait for improvement
wait = 0  # Counter for early stopping

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

    # Evaluate on validation set
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        all_preds = []  # To store all predictions
        all_labels = []  # To store all ground truth labels

        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Calculate validation loss
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            # Apply custom threshold
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability for 'retinoblastoma'
            preds = (probabilities > 0.4).int()  # Predictions with threshold 0.4

            # Store predictions and ground truth
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Append average validation loss for the epoch
    train_losses.append(train_loss / len(train_loader))
    validation_losses.append(validation_loss / len(validation_loader))

    # Print epoch summary
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {validation_losses[-1]:.4f}")

    # Early stopping logic
    if validation_losses[-1] < best_val_loss:
        best_val_loss = validation_losses[-1]
        torch.save(model.state_dict(), "models/retinoblastoma_detector.pth")  # Save the best model
        wait = 0  # Reset the wait counter
    else:
        wait += 1  # Increment the wait counter
        if wait >= patience:
            print("Early stopping triggered.")
            break


# Save the model
# torch.save(model.state_dict(), "models/retinoblastoma_detector.pth")

# Evaluate and visualize
evaluate_model(model, validation_loader, class_names, device)
plot_loss(train_losses, validation_losses)


loss_data = {
    "train_losses": train_losses,
    "validation_losses": validation_losses
}
with open("models/losses.json", "w") as f:
    json.dump(loss_data, f)

print("Training complete. Model and loss data saved.")
