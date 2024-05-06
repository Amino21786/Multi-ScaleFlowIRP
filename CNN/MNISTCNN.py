import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


#2 convolutional layers and 2 max pooling layers
class CNN(nn.Module):
    def __init__(self, n_channels=32):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, n_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, trainloader, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

def plotting(train_loss, test_loss, train_accuracy, test_accuracy):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(test_loss, label='Test Loss', linestyle='--')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy', color='orange')
    plt.plot(test_accuracy, label='Test Accuracy', color='red', linestyle='--')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def ApplyingCNN(train_loader, test_loader, model, criterion, optimizer):
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    #Training the model
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        #Loading the images and applying the model (CNN), computing the loss functions
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total

        print(f'Train Epoch {epoch+1}/{5}, Loss: {epoch_loss:.4f}, Accuracy: {100 * epoch_accuracy:.2f}%')

        # Store training loss and accuracy for plotting
        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_accuracy)

        # Evaluation on the test set
        model.eval()  # Set the model to evaluation mode
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for test_images, test_labels in test_loader:
                test_outputs = model(test_images)
                test_loss = criterion(test_outputs, test_labels)
                test_running_loss += test_loss.item()

                _, test_predicted = torch.max(test_outputs.data, 1)
                test_total += test_labels.size(0)
                test_correct += (test_predicted == test_labels).sum().item()

        test_epoch_loss = test_running_loss / len(test_loader)
        test_epoch_accuracy = test_correct / test_total

        print(f'Test Epoch {epoch+1}/{5}, Loss: {test_epoch_loss:.4f}, Accuracy: {100 * test_epoch_accuracy:.2f}%')

        # Store test loss and accuracy for plotting
        test_loss_history.append(test_epoch_loss)
        test_accuracy_history.append(test_epoch_accuracy)

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history


# For MNIST Fashion Example (92% accuracy over 5 epochs)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
train_loss, test_loss, train_accuracy, test_accuracy = ApplyingCNN(train_loader, test_loader, model, criterion, optimizer)
plotting(train_loss, test_loss, train_accuracy, test_accuracy)


