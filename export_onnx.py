import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

# Define the same model architecture as used for training
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

def export_model_to_onnx(model_path, onnx_path, num_classes):
    # Ensure that the model architecture matches what was used in training
    model = SimpleCNN(num_classes=num_classes)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Specify the input to the model as a dummy variable
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    # Export the model
    torch.onnx.export(model,               # model being run
                      x,                   # model input (or a tuple for multiple inputs)
                      onnx_path,           # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'output': {0: 'batch_size'}})

if __name__ == '__main__':
    # Specify the path to the trained model and the output ONNX file
    model_path = './model.pt'  # Adjust this to the correct path
    onnx_path = './model.onnx'  # Adjust this to the desired output path
    num_classes = 2  # Adjust this to the correct number of classes used in your model

    export_model_to_onnx(model_path, onnx_path, num_classes)
