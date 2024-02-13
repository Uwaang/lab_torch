import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import yaml

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a simple CNN classifier.')
parser.add_argument('--data_config', type=str, required=True, help='Path to data config file')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained model')
parser.add_argument('--batch', type=int, default=16, help='Batch size')
parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
args = parser.parse_args()

# Load data config
with open(args.data_config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
train_dir = config['train']
val_dir = config['val']
num_classes = config['nc']
class_names = config['names']

# Define dataset
class SimpleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.labels = [open(os.path.join(data_dir.replace('images', 'labels'), f.replace('.jpg', '.txt'))).read().strip() for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = transforms.ToPILImage()(torchvision.io.read_image(image_path))
        label = self.class_to_idx[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets and DataLoaders
train_dataset = SimpleDataset(train_dir, transform=transform)
val_dataset = SimpleDataset(val_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model, criterion, optimizer, and device
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# TensorBoard writer
writer = SummaryWriter()

# Training and validation function
def train_and_validate(model, criterion, optimizer, train_loader, val_loader, epochs, device, save_path):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('training loss', avg_loss, epoch)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        writer.add_scalar('validation accuracy', accuracy, epoch)

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_path, f'checkpoint_epoch_{epoch}.pt'))
        print(f'Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {accuracy}%')

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))

if __name__ == '__main__':
    train_and_validate(model, criterion, optimizer, train_loader, val_loader, args.epoch, device, args.save_path)
