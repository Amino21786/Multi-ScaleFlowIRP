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
root_dir = r"R64/"
train_loader, test_loaders, data_processor = load_stokesbrinkman(
                         root_dir=root_dir,
                         n_train=1, 
                         n_tests=[20],
                         batch_size=4, 
                         test_batch_sizes=[16],
                         test_resolutions=[64],
                         train_resolution=64,
                         grid_boundaries=[[0,1],[0,1]],
                         positional_encoding=True,
                         encode_input=False,
                         encode_output=False,
                         encoding='channel-wise')
data_processor = data_processor.to(device)

model = TFNO( n_modes=(64, 64), 
              in_channels=3, out_channels=2,
              hidden_channels=32, projection_channels=64, 
              factorization='tucker', rank=0.42)
model = model.to(device)
save_dir = r'model/bkm/'
model.load_state_dict(torch.load(save_dir+"model_state_dict.pt"))
test_samples = test_loaders[64].dataset
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