from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    return train_loader, dataset.classes



