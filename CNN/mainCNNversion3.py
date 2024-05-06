import sys
sys.path.append('../')
import torch
import numpy as np
import matplotlib.pyplot as plt
import vtk
import pandas as pd
import os
from torch.utils.data import Dataset, random_split, dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from tqdm import tqdm


# Load the numpy arrays from dataCNN
vmap = np.load('dataCNN/vmap.npy', allow_pickle=True)
Ks = np.load('dataCNN/Ksnorms.npy', allow_pickle=True)
#vmap = vmap.squeeze()
vmaps = torch.from_numpy(vmap[:,:,:,:]).type(torch.float32).clone()
Ks = torch.from_numpy(Ks).float()

class MyDataset(Dataset):
    def __init__(self, vmaps, Ks):
        self.vmaps = vmaps
        self.Ks = Ks

    def __len__(self):
        return len(self.vmaps)

    def __getitem__(self, idx):
        #vmap = self.vmaps[idx].squeeze(0)
        xvmap, yvmap = self.vmaps[idx]
        K = self.Ks[idx]
        return xvmap, yvmap, K

# Instantiate the dataset
dataset = MyDataset(vmaps, Ks)
#batch_size = 16
#shuffle = True
#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
print(dataset[0])
#Generate the dataset from the results of FNO
train_size = int(0.8 * len(dataset))  
test_size = len(dataset) - train_size  

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#batch size defined
bs = 8
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

# CNN layout and structure with 2 convolutional layers, 1 max pooling layer applied immediately after each one and two fully connected layers
class KCNN(nn.Module):
    def __init__(self):
        super(KCNN, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16 * 2, 128) #Not sure about the construction setup
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # Output 4 quantities
        self.dropout = nn.Dropout(0.3)


    def forward(self, x1 , x2):
        #forward pass for x velocity map
        #print('Input:', x1.shape)
        x1 = F.relu(self.conv1(x1))
        #print("C1:", x1.shape)
        x1 = self.pool(x1)
        #print("MP1:", x1.shape)
        x1 = F.relu(self.conv2(x1))
        #print("C2:", x1.shape)
        x1 = self.pool(x1)
        #print("MP2:", x1.shape)
        x1 = x1.view(-1, 32 * 16 * 16) 
        #print("view:", x1.shape)

        #forward pass for y velocity map
        #print('Input:', x2.shape)
        x2 = F.relu(self.conv1(x2))
        #print("C1:", x2.shape)
        x2 = self.pool(x2)
        #print("MP1:", x2.shape)
        x2 = F.relu(self.conv2(x2))
        #print("C2:", x2.shape)
        x2 = self.pool(x2)
        #print("MP2:", x2.shape)
        x2 = x2.view(-1, 32 * 16 * 16) 
        #print("view:", x2.shape)

        # Concatenate the features from both images
        x = torch.cat((x1, x2), dim=1)
        #print("full features:", x.shape)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        #print("fc1:", x.shape)
        x = F.relu(self.fc2(x))
        #print("fc2:", x.shape)
        x = self.fc3(x)
        #print("fc3:", x.shape)
        return x

      


#Initialise the model
model = KCNN()




#loss and optimiser defined
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)


#training loop constructed
num_epochs = 3
for epoch in range(num_epochs):
    #model.train() 
    running_loss = 0.0
    for xvmap_batch, yvmap_batch, Ks_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        #print("x:", xvmap_batch)
        #print("y:", yvmap_batch)
        
        #extract the inputs and outputs for a sample (2 images input to 4 tensor components output)
    
        optimiser.zero_grad()

        #forward
        outputs = model(xvmap_batch, yvmap_batch)
        #print("Prediction:", outputs)
        #loss
        loss = criterion(outputs, Ks_batch)

        loss.backward()
        optimiser.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")



model.eval() 
test_loss = 0.0
predicted_values = []
ground_truth_values = []
Kpred =[]
Ktrue = []

with torch.no_grad():
    for xvmap_batch, yvmap_batch, Ks_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        #forward pass of the x and y velocity maps
        outputs = model(xvmap_batch, yvmap_batch)

        loss = criterion(outputs, Ks_batch)

        
        test_loss += loss.item()

        Kpred.append(outputs)
        Ktrue.append(Ks_batch)

print(f"Test Loss: {test_loss / len(train_loader)}")



# For the comparison
quantity_names = ['$K_{11}$', '$K_{12}$', '$K_{21}$', '$K_{22}$']


Ktrue = np.array(Ktrue)
Kpred = np.array(Kpred)


print(Ktrue.shape)
print(Kpred.shape)

min_val = min(np.min(Ktrue), np.min(Kpred))
max_val = max(np.max(Ktrue), np.max(Kpred))

for j in range(4):
    plt.subplot(2, 2, j+1)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-')  
    plt.scatter(Ktrue[:,:, j], Kpred[:,:, j], color='blue', alpha=0.5, label='Predicted vs Ground Truth')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.title(f'Permeability tensor component {quantity_names[j]}')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()





