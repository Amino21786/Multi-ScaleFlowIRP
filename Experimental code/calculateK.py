import torch
import numpy as np
import sys
import neuralop
sys.path.append("../")
from neuralop.models import TFNO
from neuralop.training.trainer import Trainer
from neuralop.training.callbacks import CheckpointCallback, BasicLoggerCallback
from neuralop.utils import count_model_params
from neuralop.losses.data_losses import LpLoss, H1Loss
from utils.brinkman_amitex import load_stokesbrinkman
from utils.helpers import calc_pressure_grad, preparation
import matplotlib.pyplot as plt

device = 'cpu'

for index in range(3):
    print(index)

def unpack_data(data, idir, encode_output=None):
    # Model prediction
    if idir==0:
        out = model(data['x'].unsqueeze(0))
    elif idir==1:
        out = model(torch.transpose(data['x'].unsqueeze(0), -2, -1))
        out = torch.transpose(out, -2, -1)
        print(encode_output)
        #if encode_output.options['pvout']==True:
        #    out = out[:,[1,0,3,2],:,:]
        #else: 
        out = out[:,[1,0],:,:]
    out, data = data_processor.postprocess(out, data) 
    out = out.detach().cpu()
    
    #Input
    x = data['x'].cpu()
    #Ground trueth
    y = data['y'].cpu()
    
    #decoding
    if encode_output is not None:
        out[0] = torch.as_tensor(encode_output.decode(out.numpy()[0]))
        y = torch.as_tensor(encode_output.decode(y))
    
    return x, y, out


def compare_vfield(data, idir, encode_output=None, fig=None):
    
    #unpack
    x, y, out = unpack_data(data, idir, encode_output)
    print(y.shape)
    print(out.shape)
    
    #
    if encode_output.options['pvout']==True:
        ncols = 9
    else:
        ncols = 5
    #input - log beta
    ax = fig.add_subplot(3, ncols, index*ncols + 1)
    im = ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('log beta')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation='horizontal')
    #output - ux
    vmin, vmax = y[0].min(), y[0].max()
    ax = fig.add_subplot(3, ncols, index*ncols + 2)
    im = ax.imshow(y[0].squeeze(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('ux (FFT)')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation='horizontal')
    #
    ax = fig.add_subplot(3, ncols, index*ncols + 3)
    im = ax.imshow(out[0,0].squeeze().detach().numpy(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('ux (FNO)')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation='horizontal')
    #output - uy
    vmin, vmax = y[1].min(), y[1].max()
    ax = fig.add_subplot(3, ncols, index*ncols + 4)
    im = ax.imshow(y[1].squeeze(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('uy (FFT)')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation='horizontal')
    #
    ax = fig.add_subplot(3, ncols, index*ncols + 5)
    im = ax.imshow(out[0,1].squeeze().detach().numpy(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('uy (FNO)')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation='horizontal')
    
    if encode_output.options['pvout']==True:
        #output - px
        vmin, vmax = y[2].min(), y[2].max()
        ax = fig.add_subplot(3, ncols, index*ncols + 6)
        im = ax.imshow(y[2].squeeze(), vmin=vmin, vmax=vmax)
        if index == 0: 
            ax.set_title('px (FFT)')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar(im, ax=ax, orientation='horizontal')
        #
        ax = fig.add_subplot(3, ncols, index*ncols + 7)
        im = ax.imshow(out[0,2].squeeze().detach().numpy(), vmin=vmin, vmax=vmax)
        if index == 0: 
            ax.set_title('px (FNO)')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar(im, ax=ax, orientation='horizontal')
        #output - py
        vmin, vmax = y[3].min(), y[2].max()
        ax = fig.add_subplot(3, ncols, index*ncols + 8)
        im = ax.imshow(y[3].squeeze(), vmin=vmin, vmax=vmax)
        if index == 0: 
            ax.set_title('py (FFT)')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar(im, ax=ax, orientation='horizontal')
        #
        ax = fig.add_subplot(3, ncols, index*ncols + 9)
        im = ax.imshow(out[0,3].squeeze().detach().numpy(), vmin=vmin, vmax=vmax)
        if index == 0: 
            ax.set_title('py (FNO)')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar(im, ax=ax, orientation='horizontal')
    plt.tight_layout()
    
    return fig


def plot_Kcompar(K11_true, K11_pred, K12_true, K12_pred, 
                 K21_true, K21_pred, K22_true, K22_pred,
                 indices=None):
    plt.rcParams['font.size'] = '12'
    fig, ax = plt.subplots(2,2)
    mksiz = 2
    
    ax[0,0].loglog(K11_true, K11_pred, '.', markersize=mksiz)
    #ax[0,0].plot(K11_true, K11_pred, '.')
    lims = [min(K11_true+K11_pred), max(K11_true+K11_pred)]
    ax[0,0].set_xlim(lims)
    ax[0,0].set_ylim(lims)
    ax[0,0].loglog(lims, lims, '-k', linewidth=1)
    xticks = np.array(ax[0,0].get_xticks())
    xticks = xticks[(xticks >= lims[0]) & (xticks <= lims[1])]
    ax[0,0].set_xticks(xticks)
    ax[0,0].set_yticks(xticks)
    ax[0,0].set_aspect('equal', 'box')
    
    ax[0,1].loglog(K12_true, K12_pred, '.', markersize=mksiz)
    #ax[0,1].plot(K12_true, K12_pred, '.')
    lims = [min(K12_true+K12_pred), max(K12_true+K12_pred)]
    ax[0,1].set_xlim(lims)
    ax[0,1].set_ylim(lims)
    ax[0,1].loglog(lims, lims, '-k', linewidth=1)
    xticks = np.array(ax[0,1].get_xticks())
    xticks = xticks[(xticks >= lims[0]) & (xticks <= lims[1])]
    ax[0,1].set_xticks(xticks)
    ax[0,1].set_yticks(xticks)
    ax[0,1].set_aspect('equal', 'box')
    
    ax[1,0].loglog(K21_true, K21_pred, '.', markersize=mksiz)
    #ax[1,0].plot(K21_true, K21_pred, '.')
    lims = [min(K21_true+K21_pred), max(K21_true+K21_pred)]
    ax[1,0].set_xlim(lims)
    ax[1,0].set_ylim(lims)
    ax[1,0].loglog(lims, lims, '-k', linewidth=1)
    xticks = np.array(ax[1,0].get_xticks())
    xticks = xticks[(xticks >= lims[0]) & (xticks <= lims[1])]
    ax[1,0].set_xticks(xticks)
    ax[1,0].set_yticks(xticks)
    ax[1,0].set_aspect('equal', 'box')
    
    ax[1,1].loglog(K22_true, K22_pred, '.', markersize=mksiz)
    #ax[1,1].plot(K22_true, K22_pred, '.')
    lims = [min(K22_true+K22_pred), max(K22_true+K22_pred)]
    ax[1,1].set_xlim(lims)
    ax[1,1].set_ylim(lims)
    ax[1,1].loglog(lims, lims, '-k', linewidth=1)
    xticks = np.array(ax[1,1].get_xticks())
    xticks = xticks[(xticks >= lims[0]) & (xticks <= lims[1])]
    ax[1,1].set_xticks(xticks)
    ax[1,1].set_yticks(xticks)
    ax[1,1].set_aspect('equal', 'box')
    
    if indices is not None:
        indices = np.array(indices)
        ax[0,0].loglog(K11_true[indices], K11_pred[indices], 'd')
        ax[0,1].loglog(K12_true[indices], K12_pred[indices], 'd')
        ax[1,0].loglog(K21_true[indices], K21_pred[indices], 'd')
        ax[1,1].loglog(K22_true[indices], K22_pred[indices], 'd')
        
    fig.tight_layout()


#model setup
model = TFNO( n_modes=(64, 64), 
              in_channels=3, out_channels=2,
              hidden_channels=32, projection_channels=64, 
              factorization='tucker', rank=0.42)
model = model.to(device)
save_dir = r'model/bkm/'
model.load_state_dict(torch.load(save_dir+"model_state_dict.pt"))

###############################
###############################
# Calulate Macro-permeability #
###############################
###############################
# load the data (two simulations per sample)
RESs = [64] #64 resolution for now (original was 64, 128)
test_loaders = ({}, {})
test_names = ({}, {})
for velo_dir in range(2):
    for RES in RESs:
        #root_dir = f'/data/yc2634/ML/StokesBrinkman/R{RES}/'
        root_dir = r'R'+ str(RES) +'/' #from my directory
        #CHANGED ORDERING HERE
        test_names[velo_dir][RES], test_loaders[velo_dir][RES],  data_processor = load_stokesbrinkman(
                                 root_dir=root_dir,
                                 n_train=1, 
                                 n_tests=[3],
                                 batch_size=1, 
                                 test_batch_sizes=[1],
                                 test_resolutions=[RES],
                                 train_resolution=RES,
                                 grid_boundaries=[[0,1],[0,1]],
                                 positional_encoding=True,
                                 encode_input=False,
                                 encode_output=False,
                                 encoding='channel-wise')
        data_processor = data_processor.to(device)


# Plot the prediction, and compare with the ground-truth 
#COMMENTED OUT HERE ISSUE WITH COMPARE_VFIELD - ENCODE_OUTPUT.OPTIONS - NoneType
"""
idir = 0
fig = plt.figure(figsize=(7, 7))
for index in range(3):
    test_samples = test_loaders[idir][RESs[index]][RESs[index]].dataset
    
    idx = torch.randint(0, 143, (1,)).item()
    #idx = 40
    print(f'Printing the {idx}th sample of RES {RESs[index]}')
    data = test_samples[idx]
    data = data_processor.preprocess(data, batched=False)
    
    compare_vfield(data, idir, fig=fig)

plt.show()
"""

# Macro-permeability
pressureProvided = False

l2 = {}
#different viscosities
mue = 1
mu = 1

l2 = ({}, {})
K11_true, K12_true, K21_true, K22_true = [], [], [], []

K11_pred, K12_pred, K21_pred, K22_pred = [], [], [], []


errPI1 = []
errPI2 = []

idx_RES = []
idx_SAM = []

for RES in RESs:
    l2[0][RES] = []
    l2[1][RES] = []
    i = 0
    for data1, data2 in zip(test_loaders[0][RES][RES].dataset, 
                            test_loaders[1][RES][RES].dataset):
        idx_RES.append( RES )
        idx_SAM.append( i )
        i = i+1
        # LOAD 1 and 2
        data1 = data_processor.preprocess(data1, batched=False)
        data2 = data_processor.preprocess(data2, batched=False)
        print(data1)
        print(data2)
        #ISSUE HERE
        if torch.count_nonzero(data1['x'][0]-data2['x'][0])>0: #sanity check
            raise AssertionError(f'Load 1 and Load 2 are not sharing the same input: RES {RES}, isample {i}')
        
        # unpack data
        x1, y1, out1 = unpack_data(data1, 0, encode_output=None)
        x2, y2, out2 = unpack_data(data2, 1, encode_output=None)
        data1['y'], data2['y'] = y1, y2
        
        #L2 difference norm
        l2[0][RES].append( (torch.sqrt(torch.sum((y1-out1[0])**2)) / RES).item() )
        l2[1][RES].append( (torch.sqrt(torch.sum((y2-out2[0])**2)) / RES).item() )
        
        #local pressure gradient field
        pgrad1_true, pgrad1_pred = calc_pressure_grad(data1, out1, mu=1, mue=1, L=[1,1,1], device=device, inputEncoded=True, pressureProvided=pressureProvided)
        pgrad2_true, pgrad2_pred = calc_pressure_grad(data2, out2, mu=1, mue=1, L=[1,1,1], device=device, inputEncoded=True, pressureProvided=pressureProvided)
        
        #darcys law
        H_true = torch.zeros((2,2))
        H_true[:,0] = -torch.as_tensor([pgrad1_true[...,i].mean().item() for i in range(2)]) / mu
        H_true[:,1] = -torch.as_tensor([pgrad2_true[...,i].mean().item() for i in range(2)]) / mu
        
        H_pred = torch.zeros((2,2))
        H_pred[:,0] = -torch.as_tensor([pgrad1_pred[...,i].mean().item() for i in range(2)]) / mu
        H_pred[:,1] = -torch.as_tensor([pgrad2_pred[...,i].mean().item() for i in range(2)]) / mu
        
        #macro permeability
        K_true = torch.linalg.inv(H_true).numpy()
        K_pred = torch.linalg.inv(H_pred).numpy()
        
        #register
        K11_true.append( abs(K_true[0,0]) )
        K22_true.append( abs(K_true[1,1]) )
        K12_true.append( abs(K_true[0,1]) )
        K21_true.append( abs(K_true[1,0]) )
        K11_pred.append( abs(K_pred[0,0]) )
        K22_pred.append( abs(K_pred[1,1]) )
        K12_pred.append( abs(K_pred[0,1]) )
        K21_pred.append( abs(K_pred[1,0]) )


## compare K
plot_Kcompar(K11_true, K11_pred, K12_true, K12_pred, 
             K21_true, K21_pred, K22_true, K22_pred)
plt.show()
