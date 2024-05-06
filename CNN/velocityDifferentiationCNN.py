import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling with kernel size 2 and stride 2
        self.fc1 = nn.Linear(32 * 1 * 1, 64)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(64, 4)  # Output is a 2x2 permeability tensor

    def forward(self, x):
        #print("Input shape:", x.shape)
        x = F.relu(self.conv1(x))
        #print("After conv1 shape:", x.shape)
        x = self.pool(x)
        #print("After pool1 shape:", x.shape)
        x = F.relu(self.conv2(x))
        #print("After conv2 shape:", x.shape)
        #x = self.pool(x)
        #print("After pool2 shape:", x.shape)
        x = x.view(-1, 32 * 1 * 1)
        #print("Flattened shape:", x.shape)
        x = F.relu(self.fc1(x))
        #print("After fc1 shape:", x.shape)
        x = self.fc2(x)
        #print("After fc2 shape:", x.shape)
        x = x.view(-1, 4)
        #print("Final output shape:", x.shape)
        return x

class VelocityPermeabilityDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random velocity map (2x2 tensor matrix)
        velocity_map = np.random.rand(2, 2).astype(np.float32)

        # Calculate gradient of velocity map
        gradient_x = np.gradient(velocity_map, axis=0)
        gradient_y = np.gradient(velocity_map, axis=1)

        # Construct permeability tensor
        permeability_tensor = np.array([
            [np.sum(gradient_x**2), np.sum(gradient_x * gradient_y)],
            [np.sum(gradient_y * gradient_x), np.sum(gradient_y**2)]
        ])

        #permeability_tensor += 0.1*np.random.rand(2, 2)
        # Convert to PyTorch tensors
        velocity_map_tensor = torch.tensor(velocity_map).unsqueeze(0)  # Add channel dimension
        permeability_tensor = torch.tensor(permeability_tensor).float()

        return velocity_map_tensor, permeability_tensor

dataset = VelocityPermeabilityDataset(num_samples=1000)

# Access a sample
sample_velocity, sample_permeability = dataset[0]
#print("Sample Velocity Map:")
#print(sample_velocity)
#print("Sample Permeability Tensor:")
#print(sample_permeability)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader instances for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize the CNN model
model = CNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40
# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for velocity_maps, permeability_tensors in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        optimizer.zero_grad()
        outputs = model(velocity_maps)
        loss = criterion(outputs, permeability_tensors.view(-1, 4))  # Reshape targets to match output shape
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * velocity_maps.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Evaluation loop
predicted_values = []
ground_truth_values = []
model.eval()
test_loss = 0.0
with torch.no_grad():
    for velocity_maps, permeability_tensors in tqdm(test_loader):
        outputs = model(velocity_maps)
        #print('output:', outputs)
        #print('p tensor:', permeability_tensors)
        permeability_tensors = permeability_tensors.view(-1,4)
        test_loss += criterion(outputs, permeability_tensors).item() * velocity_maps.size(0)
        # Append predicted and ground truth values
        outputs_np = outputs.numpy()
        targets_np = permeability_tensors.numpy()
        predicted_values.append(outputs_np)
        ground_truth_values.append(targets_np)


test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")



predicted_values = np.concatenate(predicted_values, axis = 0)
ground_truth_values = np.concatenate(ground_truth_values, axis = 0)

print(ground_truth_values.shape)
print(predicted_values.shape)

# For the comparison

min_val = min(np.min(ground_truth_values), np.min(predicted_values))
max_val = max(np.max(ground_truth_values), np.max(predicted_values))
components = ['$v_x^2$', '$v_xv_y$', '$v_xv_y$', '$v_y^2$' ]
# Plotting
plt.figure(figsize=(10, 5))
for j in range(4):
    plt.subplot(2, 2, j+1)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', label='')  
    plt.scatter(ground_truth_values[:,j], predicted_values[:,j], alpha = 0.5, label=f'{components[j]}')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.title(f'Simple permability tensor component {j+1}')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()