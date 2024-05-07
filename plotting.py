import numpy as np 
import matplotlib.pyplot as plt

# Extracting the data from the folder
Kpred = np.load('dataCNN/Kspred.npy', allow_pickle=True)
Ktrue = np.load('dataCNN/Kstrue.npy', allow_pickle=True)
#Ktrue = Ktrue[700:]
print(Kpred.shape)
print(Ktrue.shape)
#Ktrue = (Ktrue - np.mean(Ktrue, axis =0))/np.std(Ktrue, axis =0)

# Mean and variance between the predictions and ground truth
print("Mean of predictions:", np.mean(Kpred, axis = 0))
print("Variance of predictions:", np.var(Kpred, axis = 0))
print("Mean of ground truth:", np.mean(Ktrue, axis = 0))
print("Variance of ground truth:", np.var(Ktrue, axis = 0))


quantity_names = ['$K_{11}$', '$K_{12}$', '$K_{21}$', '$K_{22}$']
min_val = min(np.min(Kpred), np.min(Ktrue))
max_val = max(np.max(Kpred), np.max(Ktrue))
# Plotting
plt.figure(figsize=(7, 7))
for j in range(4):
    plt.subplot(2, 2, j+1)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-')  
    plt.scatter(Ktrue[:,j], Kpred[:,j], color='blue', alpha=0.5, s = 30, label='Predicted vs Ground Truth')
    plt.xlabel('Ground Truth (FFT)')
    plt.ylabel('Predicted (FNO-CNN)')
    plt.title(f'Normalised permeability tensor component {quantity_names[j]}')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()
