import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomSignCNN(nn.Module):
    """
    Scratch-built Custom Convolutional Neural Network for Static Sign Recognition.
    Zero pretrained weights. Built entirely from scratch.
    Dynamically scales to num_classes output dimensions.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        
        # Conv Block 1: 3 -> 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112
        
        # Conv Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56
        
        # Conv Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28
        
        # Conv Block 4: 128 -> 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 28 -> 14
        
        # Adaptive Average Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully Connected Head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Pooling & Flatten
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def build_model(num_classes: int) -> CustomSignCNN:
    """Factory function to instantiate the scratch CNN model."""
    if num_classes < 2:
        num_classes = 2  # Minimum binary classification headroom
    return CustomSignCNN(num_classes=num_classes)
