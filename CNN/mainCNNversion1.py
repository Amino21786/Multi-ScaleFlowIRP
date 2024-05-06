import torch
import numpy as np
import matplotlib.pyplot as plt
import vtk
import pandas as pd
import os
from torch.utils.data import Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split

#Class to convert the vti velocity maps (x, y, z) images to numpy arrays for use in the CNN
#Preprocessing dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): filenames with respective K tensors (2 x 2) values
            root_dir (string): Directory with all the vti images (here 64 x 64)
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder_name = self.data.iloc[idx, 0]
        folder_path = os.path.join(self.root_dir, folder_name)
        
        vti_files = [f for f in os.listdir(folder_path) if f.endswith('.vti')]
        
        if not vti_files:
            print(f"No VTI files found in folder: {folder_path}")
            return None
        
        vti_file_path = os.path.join(folder_path, vti_files[0])  # Assuming only one VTI file per folder
        
        # Load VTI file
        vti_data = load_vti(vti_file_path)
        if vti_data is None:
            print(f"Failed to load VTI file: {vti_file_path}")
            return None

        #print(vti_data)

        
        
        #Extract the four quantities
        K11 = self.data.iloc[idx, 1]
        K12 = self.data.iloc[idx, 2]
        K21 = self.data.iloc[idx, 3]
        K22 = self.data.iloc[idx, 4]
        #print(vti_data.size)
        sample = {'vti': vti_data, 'K11': K11, 'K12': K12, 'K21': K21, 'K22': K22}

        #for k, v in sample.items():
        #    print(k, v)
        
        return sample

def load_vti(vti_path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_path)
    reader.Update()
    vti_data = reader.GetOutput()
    if vti_data:
        #vti_array = vtk_to_numpy(vti_data.GetPointData().GetScalars())
        #print(vti_data.GetDimensions())
        numpy_array = np.array(vti_data.GetPointData().GetScalars())
        numpy_array.reshape(64, 64, 3) 
        return numpy_array
    else:
        print(f"Failed to load VTI file: {vti_path}")
        return None

dir0 = r"R64/resu/"
dir1 = r"R64/resu/resu_kM1.000E-04_kS1.000E+00_scx1.625E-01_scy1.625E-01_ang0.000E+00_UC0/"
dir2 = r"R64/resu/resu_kM1.000E-11_kS1.000E+00_scx5.000E-02_scy3.875E-01_ang9.000E+01_UC0/"

"""
t1 = load_vti(dir1 + "vmacro_dir1_velocity.vti")
t2 = load_vti(dir1 + "vmacro_dir2_velocity.vti")
print(t1)
print(t2)
"""


dataset = CustomDataset(csv_file= 'KTensors.csv', root_dir=dir0)

#Access an individual sample using its index
#idx = 1
#print(dataset[idx])
#t1 = load_vti(dir1 + "vmacro_dir1_velocity.vti")


# CNN layout and structure with 2 convolutional layers, 1 max pooling layer applied immediately after each one and two fully connected layers
class KCNN(nn.Module):
    def __init__(self):
        super(KCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(4094, 1) 
        self.fc2 = nn.Linear(1, 4)  # Output 4 quantities
        self.dropout = nn.Dropout(0.9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #print("C and MP:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print("2, C and MP:",x.shape)
        x = x.view(x.size(0), -1)
        #print("view",x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #print("fc1:",x.shape)
        x = self.fc2(x)
        #print("fc2:",x.shape)
        return x
    

model = KCNN()

train_size = int(0.8 * len(dataset))  
test_size = len(dataset) - train_size  

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

bs = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)



num_epochs = 7
for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    for batch_idx, batch_data in enumerate(train_loader):
        inputs = batch_data['vti']
        #print(inputs.shape)
        #print(batch_data['K11'].shape)
        K11 = batch_data['K11'].double()
        K12 = batch_data['K12'].double()
        K21 = batch_data['K21'].double()
        K22 = batch_data['K22'].double()
        
        targets = torch.stack([K11, K12, K21, K22], dim=1)

        optimiser.zero_grad()
        outputs = model(inputs).double()
        

        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")



model.eval() 
test_loss = 0.0
predicted_values = []
ground_truth_values = []

with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_loader):
        inputs = batch_data['vti']
        K11 = batch_data['K11'].double()
        K12 = batch_data['K12'].double()
        K21 = batch_data['K21'].double()
        K22 = batch_data['K22'].double()
        targets = torch.stack([K11, K12, K21, K22], dim=1)
        #print("Ground truth:",targets)
        outputs = model(inputs).double()
        print("Prediction:", outputs)
        #outputs = torch.mean(outputs, dim = 0)
        #outputs =outputs.view(1, 4)
        #print(outputs)

        # Convert tensors to numpy arrays
        outputs_np = outputs.numpy()
        targets_np = targets.numpy()
        
        # Append predicted and ground truth values
        predicted_values.append(outputs_np)
        ground_truth_values.append(targets_np)

        loss = criterion(outputs, targets) 
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader)}")

predicted_values = np.concatenate(predicted_values, axis = 0)
ground_truth_values = np.concatenate(ground_truth_values, axis = 0)
print(predicted_values.shape)
print(ground_truth_values.shape)

# For the comparison

def plot_K(predicted_values, ground_truth_values, quantity_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(ground_truth_values, predicted_values, color='blue', label='Predicted vs Ground Truth')
    plt.plot(ground_truth_values, ground_truth_values, color='red', linestyle='-', label='')
    plt.xlabel('Ground Truth ' + quantity_name)
    plt.ylabel('Predicted ' + quantity_name)
    plt.title('Predicted vs Ground Truth ' + quantity_name)
    plt.legend()
    plt.grid(True)
    plt.show()



quantity_names = ['$K_{11}$', '$K_{12}$', '$K_{21}$', '$K_{22}$']
for i in range(4):
    plot_K(predicted_values[:, i], ground_truth_values[:, i], quantity_names[i])

