import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the dataset class to generate 64x64 velocity maps and 2x2 permeability tensors
class VelocityPermeabilityDataset(Dataset):
    def __init__(self, num_samples, map_size=64):
        self.num_samples = num_samples
        self.map_size = map_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random velocity map (64x64 tensor matrix)
        velocity_map = np.random.rand(self.map_size, self.map_size).astype(np.float32)

        # Calculate gradient of velocity map
        gradient_x = np.gradient(velocity_map, axis=0)
        gradient_y = np.gradient(velocity_map, axis=1)

        # Construct permeability tensor (2x2 tensor)
        permeability_tensor = np.array([
            [np.sum(gradient_x**2), np.sum(gradient_x * gradient_y)],
            [np.sum(gradient_y * gradient_x), np.sum(gradient_y**2)]
        ])

        # Add noise to permeability tensor
        permeability_tensor += 0.1 * np.random.rand(2, 2)

        # Convert to PyTorch tensors
        velocity_map_tensor = torch.tensor(velocity_map).unsqueeze(0)  # Add channel dimension
        permeability_tensor = torch.tensor(permeability_tensor).float()

        return velocity_map_tensor, permeability_tensor

# Define the CNN model
class KCNN(nn.Module):
    def __init__(self):
        super(KCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define hyperparameters and create data loaders
num_samples_train = 800
num_samples_test = 100
batch_size = 1
lr = 0.001
num_epochs = 3

train_dataset = VelocityPermeabilityDataset(num_samples=num_samples_train)
test_dataset = VelocityPermeabilityDataset(num_samples=num_samples_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = KCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for velocity_map, permeability_value in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        outputs = model(velocity_map)
        loss = criterion(outputs, permeability_value)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Evaluation loop
predicted_values = []
ground_truth_values = []
model.eval()
total_loss = 0.0
total_samples = 0
with torch.no_grad():
    for velocity_map, permeability_value in tqdm(test_loader):
        predicted_permeability = model(velocity_map)
        loss = criterion(predicted_permeability, permeability_value)
        total_loss += loss.item()
        total_samples += velocity_map.size(0)

        outputs_np = outputs.numpy()
        targets_np = permeability_value.numpy()
        predicted_values.append(outputs_np)
        ground_truth_values.append(targets_np)

average_loss = total_loss / total_samples
print("Average Test Loss:", average_loss)


predicted_values = np.concatenate(predicted_values, axis = 0)
ground_truth_values = np.concatenate(ground_truth_values, axis = 0)

print(ground_truth_values.shape)
print(predicted_values.shape)

# For the comparison

min_val = min(np.min(ground_truth_values), np.min(predicted_values))
max_val = max(np.max(ground_truth_values), np.max(predicted_values))
# Plotting
plt.figure(figsize=(10, 5))
for j in range(4):
    plt.subplot(2, 2, j+1)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', label='')  
    plt.scatter(ground_truth_values[:,j], predicted_values[:,j], alpha = 0.5, label=f'Tensor component {j+1}')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.title(f''"Permeability-like"'  tensor component {j+1} comparison')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()