import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import CNN

transform = transforms.Compose([ # compose allows to chain multiple transformation operations together
    transforms.RandomHorizontalFlip(),  # Flip horizontal aléatoire
    transforms.RandomRotation(10),      # Rotation aléatoire de 10 degrés
    #the previous two are optional: it is Data augmentation used to generate more training data by applying transformations such as rotation, cropping, etc to prevent overfitting and improve the model's performance.
    transforms.ToTensor(),              # convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize each channel (R,G,B)
])


# downloading Dataset CIFAR10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# DataLoader handles batching, shuffling ... and overall makes handling datasets for training and evaluation simpler
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initializing our Model
model = CNN().to(device)

# CrossEntropyLoss() is a loss function for classification tasks
criterion = torch.nn.CrossEntropyLoss()
# Adam optimizer with a learning rate of 0.001
# the lr can't be too high as to not overshoot the minimum
# the lr can't be too low either as to not slow down the training process
optimizer = optim.Adam(model.parameters(), lr=0.001)

# epochs are the number of iterations that go over the full training data
for epoch in range(10):
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Move inputs and labels to the device (GPU/CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradients to zero for each new batch
        optimizer.zero_grad()
        # Forward pass to get the predictions
        outputs = model(inputs)
        # Compute the loss between the model's predictions and the actual labels
        loss = criterion(outputs, labels)
        # Backward pass: compute gradients of the loss with respect to the model's parameters
        loss.backward()
        # Update the model parameters using the optimizer 
        optimizer.step()
        # Add the loss for this batch to the running loss for this epoch
        running_loss += loss.item()
    # `running_loss / len(train_loader)` gives the average loss per batch for this epoch
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
