
############
# FNO
############
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
import matplotlib.pyplot as plt

device = 'cpu'

# load the data
root_dir = r"R64/"
train_loader, test_loaders, data_processor = load_stokesbrinkman(
                         root_dir=root_dir, 
                         n_train=7, 
                         n_tests=[5],
                         batch_size=4, 
                         test_batch_sizes=[4],
                         test_resolutions=[64],
                         train_resolution=64,
                         grid_boundaries=[[0,1],[0,1]],
                         positional_encoding=True,
                         encode_input=False,
                         encode_output=False,
                         encoding='channel-wise')
data_processor = data_processor.to(device)


print(test_loaders)


"""
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
trainer = Trainer(model=model, n_epochs=50,
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
 """
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

"""
# load the trained model
model.load_state_dict(torch.load(save_dir+"model_state_dict.pt"))
print("It is working")


# %%
# Plot the prediction, and compare with the ground-truth 

test_samples = test_loaders[64].dataset
model.to(device)
"""
#print(len(test_samples))
#print(test_samples[1])

"""
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
    #Ground truth
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
"""

############################
############################
############################
############################
### test on unseen data  ###
############################
############################
############################
############################
"""
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
    #Ground trueth
    y = data['y'].cpu()
    
    #
    ax = fig.add_subplot(3, 5, index*5 + 1)
    im = ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('log $\\beta$')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax)
    #
    ax = fig.add_subplot(3, 5, index*5 + 2)
    vmin = y[0].min()
    vmax = y[0].max()
    im = ax.imshow(y[0].squeeze(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Ground-truth $u_x$ (FFT)')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation = 'horizontal')
    #
    ax = fig.add_subplot(3, 5, index*5 + 3)
    im = ax.imshow(out[0,0].squeeze().detach().numpy(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Prediction $u_x$ (FNO)')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation = 'horizontal')
    #
    ax = fig.add_subplot(3, 5, index*5 + 4)
    vmin = y[1].min()
    vmax = y[1].max()
    im = ax.imshow(y[1].squeeze(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Ground-truth $u_y$ (FFT)')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation = 'horizontal')
    #
    ax = fig.add_subplot(3, 5, index*5 + 5)
    im = ax.imshow(out[0,1].squeeze().detach().numpy(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Prediction $u_y$ (FNO)')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation = 'horizontal')

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.show()


from calculateK import *

###############################
###############################
# Calulate Macro-permeability #
###############################
###############################
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

"""