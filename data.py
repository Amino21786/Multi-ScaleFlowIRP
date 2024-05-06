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
from utils.brinkman_amitex import *
from utils.IOfcts import *

# Class to convert the vti velocity maps (x, y, z) images to numpy arrays for use in the CNN
# Preprocessing dataset
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
        for (x, y, z) vti images (3 of them) --> 3 numpy arrays and puts the files in a dictionary with their corresponding 4 tensor values (sample has 3 velocity maps)
        Returns the sample
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder_name = self.data.iloc[idx, 0]
        folder_path = os.path.join(self.root_dir, folder_name)
        
        vti_files = [f for f in os.listdir(folder_path) if f.endswith('.vti')]
    
        if not vti_files or len(vti_files) != 3:
            print(f"Expected exactly 3 VTI files in folder: {folder_path}")
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
        
        # Create the dictionary for each vti to numpy array image (x, y, z) to their respective 4 tensor values
        sample = {
            'vti1': vti_data_list[0],
            'vti2': vti_data_list[1],
            'vti3': vti_data_list[2],
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

# Accessing the right directory for use (64 x 64 results, 128 x 128 is possible also)
root_dir = "R64/"
# Identifying the folders of the results and matching with the Ktensors.csv results
idnames = datasets.utils.list_files(root_dir+'mesh/', ".vtk")
idnames = [idname[4:-4] for idname in idnames]
row_names = ["resu_" + str(idname) for idname in idnames]
f = pd.read_csv('Ktensors.csv', index_col = 0)


# Function to extract the folder with the vti file velocity map and its K tensor
def extract(df, idnames, row_names, root_dir):
    vmaps = []
    Ks = []
    for idx, idname in enumerate(idnames):
        vmaps.append(load_velomap(root_dir, idname, velo_dir = 1))
        row_name = row_names[idx]
        K11 = df.loc[row_name, 'K11']
        K12 = df.loc[row_name, 'K12']
        K21 = df.loc[row_name, 'K21']
        K22 = df.loc[row_name, 'K22']
        Ksample = [K11, K12, K21, K22]
        Ks.append(Ksample)

    vmaps = np.array(vmaps) 
    Ks = np.array(Ks)
    return vmaps, Ks

# Running this function to get the pairing of the velocity map and K tensor
vmaps, Ks = extract(f, idnames, row_names, root_dir)
vmaps = vmaps.squeeze()
print(vmaps.shape)
print(Ks.shape)

# Normalising the K tensors for later use in the MultiScaleFlowCNN.py file
mean = np.mean(Ks, axis=0, keepdims=True)
std = np.std(Ks, axis=0, keepdims=True)
Ksnorms = (Ks - mean)/std

# Saving of the velocity maps, K and normalised K tensors
np.save('dataCNN/vmap.npy', vmaps)
np.save('dataCNN/Ks.npy', Ks)
np.save('dataCNN/Ksnorms.npy', Ksnorms)
























# Old code to access the CustomDataset and use as a dictionary
"""
dataset = CustomDataset(csv_file= 'KTensors.csv', root_dir=dir0)

idx = 0
print(dataset[idx].values())
print(type(dataset))

dataset_dict = {}
for idx in range(len(dataset)):
    vti1, vti2, vti3, K11, K12, K21, K22 = dataset[idx].values()
    dataset_dict[idx] = {'vti1': vti1, 'vti2': vti2, 'vti3': vti3, 'K11': K11, 'K12': K12, 'K21': K21, 'K22': K22}

print(dataset_dict[0])
print(type(dataset_dict))
"""

