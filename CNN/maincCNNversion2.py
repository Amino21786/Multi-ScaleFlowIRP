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
from tqdm import tqdm

#Class to convert the vti velocity maps (x, y, z) images to numpy arrays for use in the CNN
#Preprocessing dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        Constructor
        Args:
            csv_file (string): filenames with respective K tensors (2 x 2) values
            root_dir (string): Directory with all the vti images (here 64 x 64)
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Extracts the vti velocity map files to obtain numpy arrays equivalent for each fibre image
        for (x, y) vti images (2 of them) --> 3 numpy arrays and puts the files in a dictionary with their corresponding 4 tensor values (sample has 3 velocity maps)
        Returns the sample
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder_name = self.data.iloc[idx, 0]
        folder_path = os.path.join(self.root_dir, folder_name)
        
        vti_files = [f for f in os.listdir(folder_path) if f.endswith('.vti')]
    
        if not vti_files or len(vti_files) != 3:
            print(f"Expected exactly 2 VTI files in folder: {folder_path}")
            return None
        
        vti_data_list = []
        for vti_file in vti_files:
            vti_file_path = os.path.join(folder_path, vti_file)
            vti_data = load_vti(vti_file_path)
            if vti_data is None:
                print(f"Failed to load VTI file: {vti_file_path}")
                return None
            vti_data_list.append(vti_data)
        
        #print(len(vti_data_list))

        # Extract the four quantities
        K11 = self.data.iloc[idx, 1]
        K12 = self.data.iloc[idx, 2]
        K21 = self.data.iloc[idx, 3]
        K22 = self.data.iloc[idx, 4]
        
        # Create the dictionary for each vti to numpy array image (x, y) to their respective 4 tensor values
        sample = {
            'vti1': vti_data_list[0][:, :2],
            'vti2': vti_data_list[1][:, :2],
            'K11': K11,
            'K12': K12,
            'K21': K21,
            'K22': K22
        }

         #for k, v in sample.items():
        #    print(k, v)

        return sample
        

def load_vti(vti_path):
    """
    vti_path: Takes in folder directory containing the vti files

    Takes in a sample folder with 3 vti files and converts them to their respective numpy arrays for use in the neural network
    """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_path)
    reader.Update()
    vti_data = reader.GetOutput()
    if vti_data:
        #print(vti_data.GetDimensions())
        numpy_array = np.array(vti_data.GetPointData().GetScalars())
        numpy_array.reshape(64, 64, 3) #to represent the three different colours in each vti image file
        return numpy_array
    else:
        print(f"Failed to load VTI file: {vti_path}")
        return None

dir0 = r"R64/resu/"
dir1 = r"R64/resu/resu_kM1.000E-04_kS1.000E+00_scx1.625E-01_scy1.625E-01_ang0.000E+00_UC0/"


#t1 = load_vti(dir1 + "vmacro_dir1_velocity.vti")
#print(t1)


dataset = CustomDataset(csv_file= 'KTensors.csv', root_dir=dir0)

#Access an individual sample using its index
idx = 0
sample = dataset[idx]
for item in sample:
    print(sample[item])

"""

# CNN layout and structure with 2 convolutional layers, 1 max pooling layer applied immediately after each one and two fully connected layers
class KCNN(nn.Module):
    def __init__(self):
        super(KCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(12286, 64) #Not sure about the construction setup
        self.fc2 = nn.Linear(64, 4)  # Output 4 quantities
        #self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        print('Input:', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print("C1 and MP1:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print("C2 and MP2:",x.shape)
        x = x.view(x.size(0), -1)
        print("view:",x.shape)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        print("fc1:",x.shape)
        x = self.fc2(x)
        print("fc2:",x.shape)
        x = x.view(-1, 4) # important change
        print("final:",x.shape)
        return x

    
#Initialise the model
model = KCNN()

#Generate the dataset from the results of FNO
train_size = int(0.875 * len(dataset))  
test_size = len(dataset) - train_size  

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#batch size defined
bs = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

#loss and optimiser defined
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)


#training loop constructed
num_epochs = 6
for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    for batch_data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        
        #How to ensure that the prediction is different for each sample?
        #extract the inputs and outputs for a sample (3 images input to 4 tensor components output)
        vti1 = batch_data['vti1']
        vti2 = batch_data['vti2']
        vti3 = batch_data['vti3']
        K11 = batch_data['K11'].double()
        K12 = batch_data['K12'].double()
        K21 = batch_data['K21'].double()
        K22 = batch_data['K22'].double()
        
        #print(K11)

        #combine them in their relevant forms
        inputs = torch.cat([vti1, vti2, vti3], dim=1)
        targets = torch.stack([K11, K12, K21, K22], dim=1)
        #print("input:", inputs.shape)
        
        

        optimiser.zero_grad()
        outputs = model(inputs).double()
        #print("targets:", targets)
        #print("outputs:", outputs)

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
    for batch_idx, batch_data in enumerate(train_loader):
        vti1 = batch_data['vti1']
        vti2 = batch_data['vti2']
        vti3 = batch_data['vti3']
        K11 = batch_data['K11'].double()
        K12 = batch_data['K12'].double()
        K21 = batch_data['K21'].double()
        K22 = batch_data['K22'].double()


        inputs = torch.cat([vti1, vti2, vti3], dim=1)
        targets = torch.stack([K11, K12, K21, K22], dim=1)
        print("Ground truth:",targets)
        outputs = model(inputs).double()
        print("Prediction:", outputs)
        #outputs = outputs.view(1, 4)
        #print(outputs)

        # Convert tensors to numpy arrays
        outputs_np = outputs.numpy()
        targets_np = targets.numpy()
        
        # Append predicted and ground truth values
        predicted_values.append(outputs_np)
        ground_truth_values.append(targets_np)

        loss = criterion(outputs, targets) 
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(train_loader)}")

predicted_values = np.concatenate(predicted_values, axis = 0)
ground_truth_values = np.concatenate(ground_truth_values, axis = 0)
#print(predicted_values.shape)
#print(ground_truth_values.shape)
"""
# For the comparison
quantity_names = ['$K_{11}$', '$K_{12}$', '$K_{21}$', '$K_{22}$']
"""
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

for i in range(4):
    plot_K(predicted_values[:, i], ground_truth_values[:, i], quantity_names[i])




min_val = min(np.min(ground_truth_values), np.min(predicted_values))
max_val = max(np.max(ground_truth_values), np.max(predicted_values))
# Plotting
plt.figure(figsize=(10, 5))
for j in range(4):
    plt.subplot(2, 2, j+1)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-')  
    plt.scatter(ground_truth_values[:,j], predicted_values[:,j], color='blue', alpha=0.5, label='Predicted vs Ground Truth')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.title(f'Permeability tensor component {quantity_names[j]}')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()
"""