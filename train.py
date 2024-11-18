import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from models.detector import RetinoblastomaDetector
from utils.dataset import load_dataset
from utils.evaluation import evaluate_model, plot_loss
import json

# Load data
data_dir = "Training"
train_loader, class_names = load_dataset(data_dir)

# Split the dataset into training and testing
from torch.utils.data import random_split
train_size = int(0.8 * len(train_loader.dataset))
test_size = len(train_loader.dataset) - train_size
train_dataset, test_dataset = random_split(train_loader.dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RetinoblastomaDetector().to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 13
train_losses, test_losses = [], []

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

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    test_losses.append(test_loss / len(test_loader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

# Save the model
torch.save(model.state_dict(), "models/retinoblastoma_detector.pth")

# Evaluate and visualize
evaluate_model(model, test_loader, class_names, device)
plot_loss(train_losses, test_losses)


loss_data = {
    "train_losses": train_losses,
    "test_losses": test_losses
}
with open("models/losses.json", "w") as f:
    json.dump(loss_data, f)

print("Training complete. Model and loss data saved.")
