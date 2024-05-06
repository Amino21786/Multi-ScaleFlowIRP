
############
# FNO
############
import torch
import sys
import neuralop
sys.path.append("../")
from neuralop.models import TFNO
from neuralop.training.trainer import Trainer
from neuralop.training.callbacks import CheckpointCallback, BasicLoggerCallback
from neuralop.utils import count_model_params
from neuralop.losses.data_losses import LpLoss, H1Loss
from utils.brinkman_amitex import load_stokesbrinkman
import matplotlib.pyplot as plt

device = 'cpu'

# load the data
root_dir = r"R64/"
train_loader, test_loaders, data_processor = load_stokesbrinkman(
                         root_dir=root_dir, 
                         n_train=35, 
                         n_tests=[7],
                         batch_size=16, 
                         test_batch_sizes=[16],
                         test_resolutions=[64],
                         train_resolution=64,
                         grid_boundaries=[[0,1],[0,1]],
                         positional_encoding=True,
                         encode_input=False,
                         encode_output=False,
                         encoding='channel-wise')
data_processor = data_processor.to(device)


# create a tensorised FNO model

model = TFNO( n_modes=(64, 64), 
              in_channels=3, out_channels=2,
              hidden_channels=32, projection_channels=64, 
              factorization='tucker', rank=0.42)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

# create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)

# creating the losses
#l2loss = LpLoss(d=2, p=2)
#h1loss = H1Loss(d=2)
l2loss = LpLoss(d=2, p=2, reduce_dims=[0,1])
h1loss = H1Loss(d=2, reduce_dims=[0,1])

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%
print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# Create the trainer
save_dir = r'model/bkm/'
trainer = Trainer(model=model, n_epochs=200,
                  device=device,
                  callbacks=[
                      CheckpointCallback(save_dir=save_dir,
                                         save_best=False,
                                         save_interval=10,
                                         save_optimizer=True,
                                         save_scheduler=True),
                      BasicLoggerCallback(),
                        ],
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# Train the model
train = True
if train == True:
    trainer.train(train_loader=train_loader, 
                  test_loaders=test_loaders,
                  optimizer=optimizer,
                  scheduler=scheduler, 
                  regularizer=None, 
                  training_loss=train_loss,
                  eval_losses=eval_losses)
 
## save the model
"""
checkpoint = {
   'epoch': float('inf'),
   'valid_loss_min': 0,
   'state_dict': model.state_dict(),
   'optimizer': optimizer.state_dict(),
}
torch.save(checkpoint, 'model/bkm/last.pt')
"""
# load the trained model
model.load_state_dict(torch.load(save_dir+"model_state_dict.pt"))
print("It is working")


# %%
# Plot the prediction, and compare with the ground-truth 

test_samples = test_loaders[64].dataset
model.to(device)
#print(len(test_samples))
#print(test_samples[1])


fig = plt.figure(figsize=(7, 7))
for index in range(3):
    #something up with the indexing given there is only one sample
    data = test_samples[index+0]
    data = data_processor.preprocess(data, batched=False)
    
    # Model prediction
    out = model(data['x'].unsqueeze(0))
    out, data = data_processor.postprocess(out, data)
    out = out.cpu()
    
    #Input
    x = data['x'].cpu()
    #Ground trueth
    y = data['y'].cpu()
    
    #
    ax = fig.add_subplot(3, 5, index*5 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])
    #
    ax = fig.add_subplot(3, 5, index*5 + 2)
    ax.imshow(y[0].squeeze())
    if index == 0: 
        ax.set_title('Ground-truth y1')
    plt.xticks([], [])
    plt.yticks([], [])
    #
    ax = fig.add_subplot(3, 5, index*5 + 3)
    ax.imshow(out[0,0].squeeze().detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])
    #
    ax = fig.add_subplot(3, 5, index*5 + 4)
    ax.imshow(y[1].squeeze())
    if index == 0: 
        ax.set_title('Ground-truth y2')
    plt.xticks([], [])
    plt.yticks([], [])
    #
    ax = fig.add_subplot(3, 5, index*5 + 5)
    ax.imshow(out[0,1].squeeze().detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()


############################
############################
############################
############################
### test on unseen data  ###
############################
############################
############################
############################
root_dir = 'R128/'
train_loader, test_loaders, data_processor = load_stokesbrinkman(
                         root_dir=root_dir,
                         n_train=1, 
                         n_tests=[50],
                         batch_size=16, 
                         test_batch_sizes=[16],
                         test_resolutions=[128],
                         train_resolution=128,
                         grid_boundaries=[[0,1],[0,1]],
                         positional_encoding=True,
                         encode_input=False,
                         encode_output=False,
                         encoding='channel-wise')
data_processor = data_processor.to(device)




test_samples = test_loaders[128].dataset
model.to(device)

import matplotlib.pyplot as plt
from matplotlib import cm
cm_colo = cm.ScalarMappable(norm=None, cmap='viridis')

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index+0]
    data = data_processor.preprocess(data, batched=False)
    
    # Model prediction
    out = model(data['x'].unsqueeze(0))
    out, data = data_processor.postprocess(out, data)
    out = out.cpu()
    
    #Input
    x = data['x'].cpu()
    #Ground truth
    y = data['y'].cpu()
    


    
    #
    ax = fig.add_subplot(3, 5, index*5 + 1)
    im = ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax)
    #
    ax = fig.add_subplot(3, 5, index*5 + 2)
    vmin = y[0].min()
    vmax = y[0].max()
    im = ax.imshow(y[0].squeeze(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Ground-truth y1')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax)
    #
    ax = fig.add_subplot(3, 5, index*5 + 3)
    im = ax.imshow(out[0,0].squeeze().detach().numpy(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax)
    #
    ax = fig.add_subplot(3, 5, index*5 + 4)
    vmin = y[1].min()
    vmax = y[1].max()
    im = ax.imshow(y[1].squeeze(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Ground-truth y2')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax)
    #
    ax = fig.add_subplot(3, 5, index*5 + 5)
    im = ax.imshow(out[0,1].squeeze().detach().numpy(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax)

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.show()




###############################
###############################
# Calulate Macro-permeability #
###############################
###############################
from utils.helpers import calc_pressure_grad, preparation

def unpack_data(data, idir, encode_output=None):
    # Model prediction
    if idir==0:
        out = model(data['x'].unsqueeze(0))
    elif idir==1:
        out = model(torch.transpose(data['x'].unsqueeze(0), -2, -1))
        out = torch.transpose(out, -2, -1)
        if encode_output.options['pvout']==True:
            out = out[:,[1,0,3,2],:,:]
        else:
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



# load the data (two simulations per sample)
RESs = [64, 128]
test_loaders = ({}, {})
test_names = ({}, {})
for velo_dir in range(2):
    for RES in RESs:
        #root_dir = f'/data/yc2634/ML/StokesBrinkman/R{RES}/'
        root_dir = r'R'+ str(RES) +'/' #from my directory
        _, test_loaders[velo_dir][RES], data_processor, \
        _, test_names[velo_dir][RES] = load_stokesbrinkman(
                                 root_dir=root_dir,
                                 n_train=1, 
                                 n_tests=[143],
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


# Macro-permeability
pressureProvided = False

l2 = {}
mue = 1
mu = 1

l2 = ({}, {})
K11_true = []
K22_true = []
K12_true = []
K21_true = []

K11_pred = []
K22_pred = []
K12_pred = []
K21_pred = []

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












# unused code below
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################
######################################

import numpy as np
import time
import tracemalloc

from torchvision import datasets
from utils.brinkman_amitex import load_betamap, load_velomap


from utils.classes import microstructure, load_fluid_condition, param_algo
from utils.brinkman_mod import *
from utils.math_fcts import *


###
### load a pair of input/output
###
root_dir = 'R128/'
idnames = datasets.utils.list_files(root_dir+'mesh/', ".vtk")
idnames = [idname[4:-4] for idname in idnames]

idx = 10
x = load_betamap(root_dir, idnames[idx])
y = load_velomap(root_dir, idnames[idx], velo_dir=1)

print(x.shape)
print(y[0].shape)



'''
#============ 2D unit cell, ellipsoid inclusion, reguler packing ==============
vxsiz = 10e-3
L1, L2, L3 = 1, 6.6, 1*vxsiz
L1, L2, L3 = 1, 1, 1*vxsiz
nx, ny, nz = int(L1/vxsiz), int(L2/vxsiz), 1

x0 = [L1/2.]
y0 = [L2/2.]
b  = [0.2]
a  = [0.25]

Ifn = np.zeros([nx, ny, nz], dtype=np.uint8) + 0
x, y, z = np.meshgrid(np.linspace(0, L1, nx),
                      np.linspace(0, L2, ny),
                      np.linspace(0, L3, nz))
x = np.moveaxis(x, 0, 1)
y = np.moveaxis(y, 0, 1)
z = np.moveaxis(z, 0, 1)
zID = list()
d = np.sqrt( (x-x0[0])**2/a[0]**2 + (y-y0[0])**2/b[0]**2 )
id = d <= 1.
Ifn[id] = 1
zID.append( 1 )
    
v_tow = np.pi * a[0]*b[0]/L1/L2
print('tow fraction: ', str(v_tow))


# micro permeability tensor
Rf = 0.023
vp = 0.53
Kp = 16/(9*np.pi*np.sqrt(6)) * (np.sqrt(0.91/(1-vp))-1)**2.5 * Rf**2
ks = np.array([[Kp, Kp, Kp, 0, 0, 0]])

# fibre orientation vectors
ex = np.array([[0,0,1]]) #fibre orientation
ey = np.array([[0,1,0]]) #transverse direction


# fluid viscosity
mu = 1.
mue = 1.

# reference parameters
phi0  = ( mu + mue ) / 2.
beta0 = ( 0 + mu/ks[0][0] ) / 2.


#calculate the coefficient beta from ks
ks_inv = inv_matrix3x3sym_vec(ks)
        
lst_beta = mu * np.array(ks_inv)        #in the local coord syst.
lst_beta = rot_tensorSym_loc2glob(lst_beta, ex, ey) #in global coord syst.

beta = np.zeros((nx,ny,nz, 6))
beta[Ifn!=0,:] = lst_beta         #beta in solid region

x = beta[:,:,:,0]
y = [np.zeros((nx,ny,nz)), np.zeros((nx,ny,nz))]


# varying beta field
c0 = 1
x = lst_beta[0][0] * 10**(-c0*d**2)
x[Ifn!=1] = 0

from matplotlib.colors import LogNorm
plt.figure();
plt.imshow(x, norm=LogNorm(vmin=x.min(), vmax=x.max())); plt.show()


print(x.shape)
print(y[1].shape)
#=============================================================================
'''

#============ 2D unit cell, ellipsoid inclusion, hexagonal packing ==============
vxsiz = 2e-2
L1, L2, L3 = 4, 4, 1*vxsiz
nx, ny, nz = int(L1/vxsiz), int(L2/vxsiz), 1

x0 = [L1/2., 0, 0, L1, L1]
y0 = [L2/2., 0, L2, 0, L2]
b  = [1.5, 1.5, 1.5, 1.5, 1.5]
a  = [0.8, 0.8, 0.8, 0.8, 0.8]

Ifn = np.zeros([nx, ny, nz], dtype=np.uint8) + 0
x, y, z = np.meshgrid(np.linspace(0, L1, nx),
                      np.linspace(0, L2, ny),
                      np.linspace(0, L3, nz))
x = np.moveaxis(x, 0, 1)
y = np.moveaxis(y, 0, 1)
z = np.moveaxis(z, 0, 1)
zID = list()
dmap = list()
for i in range(5):
    dmap.append( np.sqrt((x-x0[i])**2/a[i]**2 + (y-y0[i])**2/b[i]**2) )
    id =  dmap[i] <= 1.
    Ifn[id] = i+1
    zID.append( i+1 )

v_tow = np.pi * a[0]*b[0]/L1/L2 *2
print('tow fraction: ', str(v_tow))


# micro permeability tensor
Rf = 0.023
vp = 0.3
Kp = 16/(9*np.pi*np.sqrt(6)) * (np.sqrt(0.91/(1-vp))-1)**2.5 * Rf**2
ks = np.array([[Kp, Kp, Kp, 0, 0, 0]])

# fibre orientation vectors
ex = np.array([[0,0,1]]) #fibre orientation
ey = np.array([[0,1,0]]) #transverse direction


# fluid viscosity
mu = 1.
mue = 1.

# reference parameters
phi0  = ( mu + mue ) / 2.
beta0 = ( 0 + mu/ks[0][0] ) / 2.


#calculate the coefficient beta from ks
ks_inv = inv_matrix3x3sym_vec(ks)
        
lst_beta = mu * np.array(ks_inv)        #in the local coord syst.
lst_beta = rot_tensorSym_loc2glob(lst_beta, ex, ey) #in global coord syst.

beta = np.zeros((nx,ny,nz, 6))
beta[Ifn!=0,:] = lst_beta         #beta in solid region

x = beta[:,:,:,0]
y = [np.zeros((nx,ny,nz)), np.zeros((nx,ny,nz))]


# varying beta field
c0 = 10
x = x*0
for i in range(5):
    #x[Ifn==i+1] = lst_beta[0][0] * 10**(-c0*dmap[i][Ifn==i+1]**3)
    x[Ifn==i+1] = lst_beta[0][0] * np.exp(-c0*dmap[i][Ifn==i+1])


#from matplotlib.colors import LogNorm
#plt.figure();
#plt.imshow(x, norm=LogNorm(vmin=x.min(), vmax=x.max())); plt.show()
plt.figure(); plt.imshow(x); plt.show()

print(x.shape)
print(y[1].shape)
#=============================================================================









###
### FNO model prediction
###
xlog = np.copy(x)
xlog[x>0] = np.log10(x[x>0])
xlog = np.moveaxis(xlog, -1, 0)
data = {'x': torch.from_numpy(xlog).to(device).type(torch.float32), 
        'y': torch.from_numpy(np.array(y)[:,:,:,0]).to(device).type(torch.float32)}
data = data_processor.preprocess(data, batched=False)

out = model(data['x'].unsqueeze(0))
out, data = data_processor.postprocess(out, data)
out = out.cpu()


##
## preparation
##
Ifn = (x>0).astype(np.uint8)
beta = np.array([x, x, x, x*0, x*0, x*0])
beta = np.moveaxis(beta, 0, -1)

mu, mue = 1.,  1.

beta0 = beta.max() / 2
phi0 = (mu+mue) / 2

#initial velocity field
vfield0 = np.stack( (out[0,0].cpu().detach().numpy()[...,None],
                     out[0,1].cpu().detach().numpy()[...,None],
                     out[0,1].cpu().detach().numpy()[...,None]*0.), axis=-1)

#normalise with macro velocity [1,0] as imposed
#vfield0 = vfield0 / vfield0[:,:,0,0].mean()
#vfield0[:,:,0,0] = vfield0[:,:,0,0] / vfield0[:,:,0,0].mean()
#vfield0[:,:,0,1] = vfield0[:,:,0,1] - vfield0[:,:,0,1].mean()
# --> didn't help!

# microstructure
m0 = microstructure(          Ifn = Ifn,         #labeled image
                                L = [1,1,1/beta.shape[0]],     #physical dimension of RVE
                      label_fluid = 0,          #label for fluid region
                      label_solid = 1,          #label for solid(porous) region 
                       micro_beta = beta,       #local fluctuation field beta
                    )

# algorithm parameters
p0 = param_algo(  cv_criterion = 1e-6,           #convergence criterion
                reference_phi0 = phi0,
                reference_beta0 = beta0,
                          itMax = 10000,           #max number of iterations
                        cv_acc = True,
                      AA_depth = 4,
                      AA_increment = 2
                )


# load direction
J =[1,0,0]


# -load & fluid condition
l0 = load_fluid_condition( macro_load = J,     #gradient of pressure
                            viscosity = mu,    #fluid viscosity
                      viscosity_solid = mue,   #fluid viscosity in solid region
                          )

# solution

tracemalloc.start()
    
H, vfield, gmacro = brinkman_fft_solver_velocity(m0, l0, p0, freqType='modified', freqLaplacian='classical', vfield0=vfield0)
#H, vfield, gmacro = brinkman_fft_solver_velocity(m0, l0, p0, freqType='modified', freqLaplacian='classical', vfield0=vfield0)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
tracemalloc.stop()



# compare
from matplotlib import cm
cm_colo = cm.ScalarMappable(norm=None, cmap='viridis')

fig = plt.figure(figsize=(7, 7))

#
ax = fig.add_subplot(2, 4, 1)
im = ax.imshow(x, cmap='gray')
ax.set_title('Input x: beta field')
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(im, ax=ax)

#
ax = fig.add_subplot(2, 4, 2)
vmin = vfield[:,:,0,0].min()
vmax = vfield[:,:,0,0].max()
im = ax.imshow(y[0].squeeze(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
ax.set_title('Ground-truth y1')
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(im, ax=ax)

#
ax = fig.add_subplot(2, 4, 3)
im = ax.imshow(out[0,0].squeeze().detach().numpy(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
ax.set_title('Model prediction')
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(im, ax=ax)

#
ax = fig.add_subplot(2, 4, 4)
im = ax.imshow(vfield[:,:,0,0], cmap=cm_colo.get_cmap())
ax.set_title('FFT solver')
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(im, ax=ax)

########


#
ax = fig.add_subplot(2, 4, 5)
im = ax.imshow(x, cmap='gray')
ax.set_title('Input x: beta field')
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(im, ax=ax)

#
ax = fig.add_subplot(2, 4, 6)
vmin = vfield[:,:,0,1].min()
vmax = vfield[:,:,0,1].max()
im = ax.imshow(y[1].squeeze(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
ax.set_title('Ground-truth y2')
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(im, ax=ax)

#
ax = fig.add_subplot(2, 4, 7)
im = ax.imshow(out[0,1].squeeze().detach().numpy(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
ax.set_title('Model prediction')
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(im, ax=ax)

#
ax = fig.add_subplot(2, 4, 8)
im = ax.imshow(vfield[:,:,0,1], cmap=cm_colo.get_cmap())
ax.set_title('FFT solver')
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(im, ax=ax)

fig.suptitle('Inputs, ground-truth output and predictions.', y=0.98)
plt.tight_layout()
plt.show()