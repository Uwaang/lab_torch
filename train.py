# docker run -it --rm --gpus all -p 6606:6606 -w /workspace -v $PWD:/workspace dl_image:1.1
# python train.py --data_config ./config.yaml --save_path ./data/set1/pt/ --batch 4 --epoch 100
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
    # transforms.Resize((224, 224)),
    transforms.Resize((40 , 40)),
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

class Simage_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Simage_CNN, self).__init__()
        self.features = nn.Sequential(
            # 첫 번째 계층: 작은 커널 크기와 적은 필터 수로 시작
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 두 번째 계층: 조금 더 많은 필터를 사용하지만 여전히 작은 커널
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 세 번째 계층: 필터 수를 늘리고, 작은 이미지에 맞게 조정
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # 평탄화된 특성 맵을 기반으로 분류기를 구성
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model, criterion, optimizer, and device
# model = SimpleCNN(num_classes=num_classes)
model = Simage_CNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# TensorBoard writer
writer = SummaryWriter()

def train_and_validate(model, criterion, optimizer, train_loader, val_loader, epochs, device, save_path):
    best_accuracy = 0.0
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
            # Optionally log parameter histograms
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}/gradients', param.grad, epoch)
                writer.add_histogram(f'{name}/weights', param, epoch)
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('training loss', avg_loss, epoch)

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = running_val_loss / len(val_loader)
        accuracy = 100 * correct / total
        writer.add_scalar('validation loss', avg_val_loss, epoch)
        writer.add_scalar('validation accuracy', accuracy, epoch)

        # Log images from the last batch of the epoch
        img_grid = torchvision.utils.make_grid(inputs)
        writer.add_image('validation images', img_grid, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(save_path, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with accuracy: {accuracy}%')

        print(f'Epoch {epoch+1}, Loss: {avg_loss}, Val Loss: {avg_val_loss}, Accuracy: {accuracy}%')


    # Optionally, you can also save the final model
    # torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pt'))


if __name__ == '__main__':
    train_and_validate(model, criterion, optimizer, train_loader, val_loader, args.epoch, device, args.save_path)
