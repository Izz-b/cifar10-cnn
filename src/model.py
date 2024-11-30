import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # nn.conv2d(input channels(RGB), nb of Kernels, Kernel size(3x3), padding to resize the output image)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Convolution 1
        # nn.MaxPool2d to reduce the spatial dimensions of the input
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)        # MaxPooling
        # the second convolution takes 32 as input since it is the output of the first layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Convolution 2
        # nn.Linear(input size, output size)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Fully connected 1
        self.fc2 = nn.Linear(128, 10)  # Fully connected 2 (10 classes)

    def forward(self, x):
        # ReLU is an activation function that introduces non-linearity into the network
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x