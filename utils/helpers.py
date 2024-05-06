import torch
import numpy as np
from utils.classes import microstructure, load_fluid_condition, param_algo, grid
from utils.brinkman_mod import *
from scipy.fft import fftfreq, fftn, ifftn

def recon_pressure_grad(phi, beta, vfield, L=[1,1,1], device='cpu'):
    #beta - shape (BxCxWxHxD)
    #vfield - shape (BxCxWxHxD)
    
    Lx, Ly, Lz = L
    beta = beta.numpy()
    #original
    print('its below')
    print(beta.shape)
    print(vfield.shape)
    print('its above')
    nx, ny, nz = beta.shape #, beta.shape[2:], beta.shape[2:]
    #ASK YANG here - not sure of right format
    #nx, ny, nz = 1, 1, 1
    dx, dy, dz = Lx/nx, Ly/nz, Lz/nz
        
    # laplacian of velocity
    ii = np.pi * fftfreq(nx,1./nx) / nx
    jj = np.pi * fftfreq(ny,1./ny) / ny
    kk = np.pi * fftfreq(nz,1./nz) / nz
    jj,ii,kk = np.meshgrid(jj, ii, kk)
    freq = np.zeros((nx, ny, nz, 3))
    freq[:,:,:,0] = 2./dx * np.sin(ii)*np.cos(jj)*np.cos(kk)
    freq[:,:,:,1] = 2./dy * np.cos(ii)*np.sin(jj)*np.cos(kk)
    freq[:,:,:,2] = 2./dz * np.cos(ii)*np.cos(jj)*np.sin(kk)
    freqSquare = freq[:,:,:,0]**2 + freq[:,:,:,1]**2 + freq[:,:,:,2]**2
    freqSquare = freqSquare[None,None,:,:,:]
    
    # term1: phi*laplacian(vfield)
    term1 = vfield * 0.
    for j0 in range(2):
        term1[:,j0:j0+1] = ifftn(-freqSquare* fftn(vfield[:,j0:j0+1])).real #*phi which is 1 everywhere
    
    # term2: beta*vfield
    term2 = vfield * 0.
    for j0 in range(2):
        term2[:,j0:j0+1] = beta*vfield[:,j0:j0+1]
    
    # grad pressure: term1 - term2, phi*laplacian(vfield) - beta*vfield
    return term1 - term2
 


def calc_pressure_grad(data, out=None, mu=1, mue=1, L=[1,1,1], device='cpu', inputEncoded=True, pressureProvided=False):
    if inputEncoded==False:    #out: BxCxWxH
        beta = data['x'][0]    #data['x']: CxWxH, data['y']: CxWxH
    elif inputEncoded==True:
        beta = 10**data['x'][0]
    phi = beta * 0 
    phi[data['x'][0]==0] = mu
    phi[data['x'][0]!=0] = mue
    
    phi = phi[...,None].detach().to(device)
    beta = beta[...,None].detach().to(device)
    
    if pressureProvided==True:
        return torch.moveaxis(data['y'][2:], 0, -1)[:,:,None,:].detach().to(device), \
               torch.moveaxis(out[0,2:], 0, -1)[:,:,None, :].detach().to(device)
    else:
        if out is None: #use the ground truth
            vfield = torch.moveaxis(data['y'][:2], 0, -1)[:,:,None,:].detach().to(device)
            return recon_pressure_grad(phi, beta, vfield, L=L, device=device)
        else:
            vfield_true = torch.moveaxis(data['y'][:2], 0, -1)[:,:,None,:].detach().to(device)
            vfield_pred = torch.moveaxis(out[0,:2], 0, -1)[:,:,None, :].detach().to(device)
            vfield_pred = vfield_pred / torch.sqrt(torch.mean(vfield_pred[...,0].flatten())**2 + \
                                                   torch.mean(vfield_pred[...,1].flatten())**2)
            return recon_pressure_grad(phi, beta, vfield_true, L=L, device=device), \
                   recon_pressure_grad(phi, beta, vfield_pred, L=L, device=device)
   


def preparation(data, out, J=[1,0,0], inputEncoded=True, mu=1, mue=1):
    #input
    x = torch.moveaxis(data['x'], 0,-1)[:,:,None,:][...,0].detach().cpu().numpy()
    if inputEncoded==True:
        x[x>0] = 10**x[x>0]
    
    #rescale beta with new mu
    if mu!=1:
        x = x*mu
    
    #output
    out = torch.moveaxis(out[0], 0,-1)[:,:,None,:].detach().cpu().numpy()
    
    Ifn = (x>0).astype(np.uint8)
    beta = np.array([x, x, x, x*0, x*0, x*0])
    beta = np.moveaxis(beta, 0, -1)
    
    beta0 = beta.max() / 2
    phi0 = (mu+mue) / 2
    
    #initial velocity field
    vfield0 = np.stack( (out[...,0], out[...,1], out[...,1]*0.), axis=-1)
    
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
    
    # -load & fluid condition
    l0 = load_fluid_condition( macro_load = J,     #gradient of pressure
                                viscosity = mu,    #fluid viscosity
                          viscosity_solid = mue,   #fluid viscosity in solid region
                              )
    
    return m0, l0, p0, vfield0





    
