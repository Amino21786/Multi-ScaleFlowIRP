import sys
sys.path.append("../")
from utils.IOfcts import vtkFieldReader, vtiFieldReader, extract_mat

import torch
from torch.utils.data import Dataset
from torchvision import datasets

import numpy as np
import os
import vtk

from neuralop.datasets.tensor_dataset import TensorDataset
#from ..layers.embeddings import PositionalEmbedding
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.data_transforms import DefaultDataProcessor


def load_betamap(dir0, idname):
    # matID & zone ID
    matID = os.path.join(dir0, 'mesh', 'mID_' + idname + '.vtk')
    matID, orig, spac = vtkFieldReader(matID, 'matID')
    #zoneID = dir0 + 'zoneID/zID_' + idname + '.vtk'
    #zoneID, orig, spac = vtkFieldReader(zoneID, 'zoneID')
    
    # local field beta (mu/ks)
    matxml = os.path.join(dir0, 'mate', 'mate_' + idname + '.xml')
    beta = extract_mat(matxml, mID=1, idx=[1, 2, 3], relativePath=True)
    
    # map of local field beta
    betamap = np.zeros(matID.shape, dtype=beta.dtype)
    betamap[matID==1] = beta[:,0]  #isotropic
    
    return betamap

def load_velomap(dir0, idname, prefix='vmacro', velo_dir=1):
    vti_name = dir0+'resu/resu_'+idname+'/'+prefix+'_dir'+str(velo_dir)+'_velocity.vti'
    return vtiFieldReader(vti_name, components=[0,1])
    


#################
#################
#################
def load_stokesbrinkman( 
     root_dir,
     n_train, 
     n_tests,
     batch_size, 
     test_batch_sizes,
     test_resolutions=[256],
     train_resolution=256,
     grid_boundaries=[[0,1],[0,1]],
     positional_encoding=True,
     encode_input=False,
     encode_output=True,
     encoding='channel-wise',
     ):
                           
    ## load the whole dataset
    print(root_dir)
    idnames = datasets.utils.list_files(root_dir+'mesh/', ".vtk")
    idnames = [idname[4:-4] for idname in idnames]
    
    x = [ load_betamap(root_dir, idname) for idname in idnames]
    x = np.expand_dims(np.array(x), 1)  #BxCxWxHxD
    x[x>0] = np.log10(x[x>0])           #log scale
    y = [ load_velomap(root_dir, idname, velo_dir=1) for idname in idnames ]
    
    y = np.array(y)                     #BxCxWxHxD
    print("Shape of all vvmaps:", y.shape)
    idx_train = np.random.choice(len(x), n_train)
    print("train:", idx_train)
    x_train = torch.from_numpy(x[idx_train,:,:,:,0]).type(torch.float32).clone()
    y_train = torch.from_numpy(y[idx_train,:,:,:,0]).type(torch.float32).clone()
    print('')
    #
    test_batch_size = test_batch_sizes[0]
    test_resolution = test_resolutions[0]
    
    n_test = n_tests[0]  #currently, only 1 resolution possible
    idx_test = np.random.choice(np.setdiff1d(np.arange(len(x)),idx_train), n_test)
    x_test = torch.from_numpy(x[idx_test,:,:,:,0]).type(torch.float32).clone()
    y_test = torch.from_numpy(y[idx_test,:,:,:,0]).type(torch.float32).clone()
    
    #x_valid = torch.from_numpy(x[n_train+n_test,:,:,:,0]).clone()
    #y_valid = torch.from_numpy(y[n_train+n_test,:,:,:,0]).clone()
    

    ## input encoding
    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]
        
        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
        #x_train = input_encoder.encode(x_train)
        #x_test = input_encoder.encode(x_test.contiguous())
    else:
        input_encoder = None

    ## output encoding
    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        #y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None

    ## training dataset 
    train_db = TensorDataset(
            x_train, 
            y_train)

    train_loader = torch.utils.data.DataLoader(
            train_db,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0, 
            pin_memory=True, 
            persistent_workers=False)
    
    ## test dataset
    test_db = TensorDataset(
            x_test, 
            y_test)
    test_loader = torch.utils.data.DataLoader(
            test_db,
            batch_size=test_batch_size, 
            shuffle=False,
            num_workers=0, 
            pin_memory=True, 
            persistent_workers=False)
    
    test_loaders =  {train_resolution: test_loader}
    test_loaders[0] = test_loader #currently, only 1 resolution is possible


    ## positional encoding
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
    else:
        pos_encoding = None
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )
    
    return train_loader, test_loaders, data_processor
    
    
        
    