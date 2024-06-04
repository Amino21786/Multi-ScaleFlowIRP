# IRP Multi-Scale flow: Neural networks strand 
This project was part of my 1st year MRes of statistical applied mathematics at the University of Bath (SAMBa) PhD programme. Part of a wider group project based on building off the work of [[1]](#1) on multi-scale flow through porous media.
# Multi-scale porous media flow
This folder consists of sample images of fibres in a composite material (64 x 64, 128 x 128, R64 and R128 respectively) with different Brinkman terms and model Python files to extract the velocity maps and permeability tensors (K).

Neural networks
- Fourier neural operator (FNO)
- Convolution neural network (CNN)




## Functionality of the files
A huge number of these files are accredited to Yang Chen and built off his existing FFT solver for this problem. His GitHub repository is https://github.com/yang-chen-2022/fftpermeability-StokesBrinkman.
Extra files on top of that include:
- dev_FNO_results.py - FNO solver using the FFT ground truth images and results for comparison to produce velocity maps and the permeability tensors. (Yang Chen's work)
- data.py - processing the FNO results into a format to use for the CNN and saves them as NumPy arrays in the dataCNN folder
- multiScaleFlowCNN.py - Main file I worked on (CNN folder has more python CNN files) and constructed to feed off the FNO results (e.g R64/resu folder with 800 velocity maps) to obtain K predictions.
- plotting.py - plotting the K tensor predictions versus ground truth from multiScaleCNN.py
- CNN folder including python files such as MNISTCNN.py and meanVelocityCNN.py, which test out the capability of the CNN and proof of concept for the MNIST digits and fashion datasets and random 28 x 28 velocity maps --> K. There is also a velocityDifferentiationCNN.py which models the 2 x 2 Hessian matrix for given 2 x 2 velocity matrix (x,y) inputs
Note the data folders such as R64, R128 and dataCNN are not uploaded here due to the size of the folders exceeding the GitHub size limit. They are available upon request.

## Libraries
In this project, a number of Python libraries were used for model construction, mathematical calculations, plotting and data manipulation (including the standard numpy, matplotlib, pandas etc) 
The non-trivial extensively used Python libraries include:
- neuralop -> construction of the FNO for the fibre images to learn the PDE operator 
- torch and torchvision -> CNN use, usual ML/DL tinkering with datasets, loss functions etc including example datasets
- jax, equinox, optax -> like torch, used for the CNN in particular multiScaleFlowCNN.py, as was swift and responsive to run compared to torch

Note there are more local libraries referenced in between coding files (for instance utils.brinkman_amitex in dev_FNO_mdev.py) these are linked accordingly

## Interface instructions
To create the conda environment one can install it via a dependencies file command:
```
conda env create -f dependencies.yml
```
This exports the relevant libraries into a new environment called my_env_project, which can then be activated by:
```
conda activate my_env_project
```

Then, the code can be run through the command line for example
```
python dev_FNO_mdev.py
```

or using coding editors such as VSC to run the individual files. (In the future there will be a construction of a main.py to run all the necessary models needed for results)

# Acknowledgements
Huge thanks to Dr Yang Chen from the Mechanical Engineering department at the University of Bath, who proposed this project and helped considerably with providing image folders, datasets and FNO code files to build from.
Also great thanks to the SAMBa IRP group team of Dr James Foster, Dr Eike Mueller, Dr Theresa Smith and more.
This project was part of my 1st year MRes of statistical applied mathematics at the University of Bath (SAMBa) PhD programme.


## References
<a id="1">[1]</a> 
Chen, Y. (2023). High‐performance computational homogenization of Stokes–Brinkman flow with an Anderson‐accelerated FFT method. International Journal for Numerical Methods in Fluids, 95(9), 1441-1467.



























