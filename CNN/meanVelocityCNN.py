import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms


class VelocityPermeabilityDataset(Dataset):
    def __init__(self, num_samples, map_shape=(28, 28), permeability_range=(0.1, 21.0)):
        self.num_samples = num_samples
        self.map_shape = map_shape
        self.permeability_range = permeability_range

        # Generate velocity maps and corresponding permeability values
        self.velocity_maps = [self.generate_velocity_map() for _ in range(num_samples)]
        self.permeability_values = [self.calculate_permeability(map) for map in self.velocity_maps]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        velocity_map = self.velocity_maps[idx]
        permeability_value = self.permeability_values[idx]
        return torch.Tensor(velocity_map), torch.Tensor([permeability_value])

    def generate_velocity_map(self):
        return np.random.rand(*self.map_shape)  # Generate random velocity map (example)

    def calculate_permeability(self, velocity_map):
        """
        Very rudimentary way of calculating permeability (assigning a relationship between velocity maps and K)
        """
        mean_velocity = np.mean(velocity_map)
        min_permeability, max_permeability = self.permeability_range
        return (mean_velocity) # * (max_permeability - min_permeability) + 0.25*min_permeability



#dataset = VelocityPermeabilityDataset(num_samples=3)
#data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

#for velocity_map, permeability_value in data_loader:
#    print("Velocity Map:", velocity_map)  # Shape: (batch_size, height, width)
#    print("Permeability Value:", permeability_value)  # Shape: (batch_size, 1)
    




#Base CNN
#Without using Darcy's law (having the velocity map and their peremabilities) 
class KCNN(nn.Module):
    def __init__(self):
        super(KCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)  

    def forward(self, x):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print("C and MP:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print("2, C and MP:",x.shape)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        print("fc1:",x.shape)
        x = self.fc2(x)
        print("fc2:",x.shape)
        return x


dataset = VelocityPermeabilityDataset(num_samples=1000)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

Tdataset = VelocityPermeabilityDataset(num_samples=200)
test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = KCNN()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 3

#Training loop
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for velocity_map, permeability_value in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(velocity_map)
        loss = criterion(outputs, permeability_value)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item() * velocity_map.size(0)

    # Calculate average loss per epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")





#example_velocity = torch.rand(1, 1, 28, 28)
#predicted_permeability = model(example_velocity)
#print(f'Predicted permeability: {predicted_permeability.item()}')

predicted_permeability_values = []
ground_truth_permeability_values = []

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

#predicted_permeability_values = np.array(predicted_permeability_values)
#ground_truth_permeability_values = np.array(ground_truth_permeability_values)

print(ground_truth_permeability_values[-1])
print(predicted_permeability_values[-1])
plt.scatter(ground_truth_permeability_values, predicted_permeability_values, color='blue', alpha=0.5)
plt.plot(ground_truth_permeability_values, ground_truth_permeability_values, color='red', linestyle='--')
plt.xlabel('Ground Truth Permeability')
plt.ylabel('Predicted Permeability')
plt.title('Predicted vs. Ground Truth Permeability')
plt.grid()
plt.show()

