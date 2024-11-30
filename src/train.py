import torch
import torch.optim as optim
from model import CNN
from utils import imshow
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


def train_and_evaluate_model(model, device):
             #****LOAD AND PREPROCESS THE CIFAR-10 DATASET****

    transform = transforms.Compose([#compose allows to chain multiple transformation operations together
    
     transforms.RandomHorizontalFlip(),  # Flip horizontal aléatoire
     transforms.RandomRotation(10),      # Rotation aléatoire de 10 degrés
     #the previous two are optional:it is Data augmentation used to generate more training data by applying transformations such as rotation, cropping, etc to prevent overfitting and improve the model's performance.
     transforms.ToTensor(),              #convert the image to a tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #Normalize each channel (R,G,B)
 ])
 #downloading Dataset CIFAR10
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #DataLoader handels batching,shuffling ...  and overall makes handeling datasets for training and evaluation simpler
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    #Explore the classes
    classes = train_dataset.classes
    print("Classes :", classes)
    print("Number of Classes:",len(classes))
    #Initilazing our Model
    
    #CrossEntropyLoss() is a loss function for classification tasks
    criterion = nn.CrossEntropyLoss()
    #Adam optimizer with a learning rate of 0.001
    #the lr can't be  too high  as to not overshoot the minimum
    #the lr can't be too low either as to not slow down the training procees
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #epochs are the number of iterations that goes over the full training data

             #****TRAINING THE MODEL****

    for epoch in range(10): 
        running_loss = 0.0
        for inputs, labels in train_loader:
                #Move inputs and labels to thr device (GPU/CPU)
            inputs, labels = inputs.to(device), labels.to(device)


        # Reset gradients to zero for each new batch
            optimizer.zero_grad()

        # Forward pass to get the predictions
            outputs = model(inputs)

        # Compute the loss between the model's predictions and the actual labels
            loss = criterion(outputs, labels)
        # Backward pass:compute gradients of the loss with respect to the model's parameters
            loss.backward()
        # Update the model parameters using the optimizer 
            optimizer.step()
        # Add the loss for this batch to the running loss for this epoch
            running_loss += loss.item()
    # `running_loss / len(train_loader)` gives the average loss per batch for this epoch
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
                    #****VISUALISATION OF PREDICTIONS****(OPTIONAL)

    dataiter = iter(test_loader)
    images, labels = next(dataiter)

# Display the images

    imshow(torchvision.utils.make_grid(images))


# Make predictions with the model
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)
    
             #****EVALUATE THE MODEL****

    correct = 0 # Variable to store the number of correct predictions
    total = 0   #Variable to store the total number of samples

    # Disable gradient calculation, since we're only evaluating the model and not training it
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

        # Get the predicted class for each sample
            _, predicted = torch.max(outputs, 1)

        # Update the total count 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
# Calculate the test accuracy as the ratio of correct predictions to total samples, and print it
    print(f"Test Accuracy after training the model: {100 * correct / total:.2f}%")

                 #****TESTING****

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f' Test Accuracy of the model on the 10000 test images: {100 * correct / total}%')

             #****SAVING AND LOADING THE MODEL****
    #saving the model's state dictionary as opposed to the whole model is better since it's more lightweight 
    torch.save(model.state_dict(), 'cnn_cifar10.pth')
    model = CNN()  # Initialize the model again
    model.load_state_dict(torch.load('cnn_cifar10.pth',weights_only=True))  # Load the saved model state into the model
    model.eval()  # Set the model to evaluation mode


