import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

#Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_digits_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_digits_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#Base CNN
#Without using Darcy's law (having the velocity map and their peremabilities) 
class kCNN(nn.Module):
    def __init__(self):
        super(kCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 1)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_and_evaluate(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for velocity, permeability in train_loader:
            optimizer.zero_grad()
            output = model(velocity)
            loss = criterion(output, permeability)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.numpy())
            y_pred.extend(predicted.numpy())
    
    return y_true, y_pred



#Synthetic velocity maps and corresponding permeability values
def generate_data(num_samples):
    velocity_maps = np.random.rand(num_samples, 1, 28, 28)  # Example: 28x28 velocity maps
    permeabilities = np.random.rand(num_samples, 1)  # Example: single scalar permeability values
    return velocity_maps.astype(np.float32), permeabilities.astype(np.float32)

#Training dataset
velocity_maps, permeabilities = generate_data(num_samples=1000)
dataset = TensorDataset(torch.from_numpy(velocity_maps), torch.from_numpy(permeabilities))
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

#Test dataset
Tvelocity_maps, Tpermeabilities = generate_data(num_samples=500)
Tdataset = TensorDataset(torch.from_numpy(Tvelocity_maps), torch.from_numpy(Tpermeabilities))
test_loader = DataLoader(Tdataset, batch_size=64, shuffle=True)


"""
model = kCNN()
criterion = nn.MSELoss()  #Mean Squared Error loss 
optimizer = optim.Adam(model.parameters(), lr=0.001)
"""

# Hyperparameters
learning_rate = [0.001, 0.01, 0.1]
batch_size = [16, 32, 64]
num_epochs = [50]



# Function to train and evaluate a given model


predicted_permeability_values = []
ground_truth_permeability_values = []
model = kCNN()
criterion = nn.MSELoss()  #Mean Squared Error loss 
optimizer = optim.Adam(model.parameters(), lr=0.1)
num_epochs = 50
for epoch in range(num_epochs):
    for velocity, permeability in train_loader:
        optimizer.zero_grad()
        output = model(velocity)
        loss = criterion(output, permeability)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')




#Evaluating mode on test dataset
with torch.no_grad():
    total_loss = 0
    total_samples = 0
    for velocity_map, permeability_value in test_loader:
        predicted_permeability = model(velocity_map)
        
        loss = criterion(predicted_permeability, permeability_value)
    
        total_loss += loss.item() * velocity_map.size(0)
        total_samples += velocity_map.size(0)
        
        predicted_permeability_values.extend(predicted_permeability.numpy().tolist())
        ground_truth_permeability_values.extend(permeability_value.numpy().tolist())

# Calculate the average loss over all samples in the test dataset
average_loss = total_loss / total_samples
print("Average Loss:", average_loss)

predicted_permeability_values = np.array(predicted_permeability_values)
ground_truth_permeability_values = np.array(ground_truth_permeability_values)


print(ground_truth_permeability_values[-10])
print(predicted_permeability_values[-10])
plt.scatter(ground_truth_permeability_values, predicted_permeability_values, color='blue', alpha=0.5)
plt.plot(ground_truth_permeability_values, ground_truth_permeability_values, color='red', linestyle='--')
plt.xlabel('Ground Truth Permeability')
plt.ylabel('Predicted Permeability')
plt.title('Predicted vs. Ground Truth Permeability')
plt.grid()
plt.show()



"""
best_score = 0
best_params = None
for lr in learning_rate:
    for bs in batch_size:
        for ne in num_epochs:
            # Create DataLoaders with current batch_size
            train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(Tdataset, batch_size=64, shuffle=True)

            # Initialize the model, loss function, and optimizer with current learning_rate
            model = kCNN()
            criterion = nn.MSELoss()  #Mean Squared Error loss 
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Train and evaluate the model
            y_true, y_pred = train_and_evaluate(model, criterion, optimizer, ne)
            current_score = np.mean(np.array(y_true) == np.array(y_pred))

            # Update best_score and best_params if current_score is better
            if current_score > best_score:
                best_score = current_score
                best_params = {'learning_rate': lr, 'batch_size': bs, 'num_epochs': ne}

print("Best hyperparameters:", best_params)
print("Best score:", best_score) 
"""




"""



example_velocity = torch.rand(1, 1, 28, 28)
predicted_permeability = model(example_velocity)
#print(f'Predicted permeability: {predicted_permeability.item()}')

"""
