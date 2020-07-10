# Example script of WaveBlocks framework.
# This script creates a defocused image in front of the microscope and optimizes a Spatial Light Modulator
# in its optical path to bring the image back to focus, this by minimizing the MSE between the image generated with the GT defocus, vs the one
# with the current defocus. For this the file microLFPM uses a wave-propagation module
# and a Camera module for rendering.

# Josue Page Vizcaino
# pv.josue@gmail.com
# 02/23/2020, Bern Switzerland

import WaveBlocksPytorch as ob
import torch
import torch.nn as nn
import tables
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import math

from microscopes.microLFPM import Microscope as MicroscopeWithSLM
from microscopes.microWithPropagation import Microscope as MicroscopeDefocus

# Configuration parameters
lr = 1e3
nEpochs = 1000
# Defocus distance in front of the objective
GT_defocus = -20.0

# Fetch Device to use
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Enable plotting 
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
plt.ion()
plt.show() 

# Load image to use as our object in front of the microscope
obj_image = TF.to_tensor(Image.open('config_files/GT.tif')).unsqueeze(0).to(device)
obj_image/=obj_image.max()

# Create opticalConfig object with the information from the microscope
opticalConfig = ob.OpticConfig()

psf_size = 17*5
opticalConfig.PSF_config.NA = 0.45
opticalConfig.PSF_config.M = 20
opticalConfig.PSF_config.Ftl = 165000
opticalConfig.PSF_config.wvl = 0.63
opticalConfig.PSF_config.ni = 1
opticalConfig.PSF_config.fobj = opticalConfig.PSF_config.Ftl/opticalConfig.PSF_config.M

# Camera                                     
opticalConfig.sensor_pitch = 6.9

# Define PSF
PSF = ob.PSF(opticalConfig)
_,psfIn = PSF.forward(opticalConfig.sensor_pitch/opticalConfig.PSF_config.M, psf_size, np.array([0]))

# This variable controls the min posible defocus in image space, 
# as the sampling of the Fourier space depends on the propagation distance
opticalConfig.minDefocus = -5*opticalConfig.PSF_config.M**2

# Spatial Light Modulator and Relay optics configuration
opticalConfig.relay_focal_lenght = 150000   # 150mm
opticalConfig.relay_apperture_width = 50800 # 50.8mm diameter
opticalConfig.pm_shape = [128,128]
opticalConfig.max_phase_shift = 5.9*math.pi
opticalConfig.pm_sampling = 8               # Pixel size of SLM in micrometers

opticalConfig.useMLA = False
opticalConfig.useRelays = True

# Define names of variables to learn
vars_to_learn = ["phaseMask.function_img"]
learning_rates = [lr]

# Create a Microscope
WBMicro = MicroscopeWithSLM(psfIn, vars_to_learn, opticalConfig).to(device)
# Microscope for GT
WBMicroGT = MicroscopeDefocus(psfIn, [], opticalConfig).to(device)

# Fetch variables to learn from microscope
trainable_vars = WBMicro.get_trainable_variables()
# Create pairs for optimizer, in case of more than one parameter to optimize
trainable_vars_and_lr = len(vars_to_learn)*[0]
for varId,var in enumerate(trainable_vars):
    trainable_vars_and_lr[varId] = {'params':var,'lr':learning_rates[varId]}

# Define loss functions
crit = nn.MSELoss()
optimizer = optim.Adam(trainable_vars_and_lr, lr=lr)

# Generate GT defocused image
Gt_img = torch.nn.functional.interpolate(obj_image.transpose(2,3),obj_image.shape[-2:])

# Arrays for storing statistics
errors = []

# Optimize
for ep in range(nEpochs):
    plt.clf()
    optimizer.zero_grad()
    # Predict defocused image with current defocus
    currImg = WBMicro(obj_image)
    # Compute error
    diff = crit(Gt_img,currImg)
    curr_error = diff.detach().item()
    # Propagate gradients back
    diff.backward()
    # Update wave_prop
    optimizer.step()

    # Store errors
    errors.append(curr_error)

    # Store current prediction
    curr_phaseMask = WBMicro.phaseMask.function_img.squeeze().detach()

    print(str(ep)+' MSE: '+str(curr_error))

    # Display results       
    plt.subplot(2,2,2)
    plt.imshow(currImg[0,0,:,:].detach().cpu().numpy())
    plt.title('Current Guess')
    plt.gray()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.subplot(2,2,1)
    plt.imshow(Gt_img[0,0,:,:].detach().cpu().numpy())
    plt.title('GT image')
    plt.gray()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    ax = plt.subplot(2,2,3)
    ax.plot(errors, alpha=0.9, color="b" , label='Image')
    ax.hlines(0, 0, len(errors)+1, linewidth=1, color="k")
    ax.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.title("L2 loss image: "+ '{:06.3f}'.format(curr_error))

    ax = plt.subplot(2,3,6)
    ax.imshow(curr_phaseMask.detach().cpu().numpy())
    plt.title('Current Phase-mask pattern')

    plt.suptitle('Microscope refocusing with WaveBlocks')
    plt.pause(0.1)
    plt.savefig('out/frame_'+'{:3d}'.format(ep)+".png")
    plt.show()
psfFile.close()