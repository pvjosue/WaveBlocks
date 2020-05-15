# Example script of WaveBlocks framework.
# This script creates a defocused image in front of the microscope and predicts the defocusing
# distance, by minimizing the MSE between the image generated with the GT defocus, vs the one
# with the current defocus. For this the file microWithPropagation uses a wave-propagation module
# and a Camera module for rendering.

# Josue Page Vizcaino
# pv.josue@gmail.com
# 02/08/2020, Bern Switzerland

import WaveBlocksPytorch as ob
import torch
import torch.nn as nn
import tables
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

from microscopes.microLFPM import Microscope

# Configuration parameters
lr = 1e-2
lrNet = 1e-1
nEpochs = 200
# Defocus distance in front of the objective
GT_defocus = -20

# Fetch Device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enable plotting 
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
plt.ion()
plt.show() 

# Load PSF and optics information from file
psfFile = tables.open_file('config_files/psfLFGT.h5', "r", driver="H5FD_CORE")
# Load PSF and arrange it as [1,nDepths,x,y,2], the last dimension stores the complex data 
psfIn = torch.tensor(psfFile.root.PSFWaveStack, dtype=torch.float32, requires_grad=True).permute(1,3,2,0).unsqueeze(0)

# Load volume to use as our object in front of the microscope
GT_volume = torch.tensor(psfFile.root.volume).unsqueeze(0).to(device)
GT_volume/=GT_volume.max()
nDepths = GT_volume.shape[1]

# Create opticalConfig object with the information from the microscope
opticalConfig = ob.OpticConfig(psfFile.root.wavelenght[0], 1)
# Load info from PSF file 
# Camera                                     
opticalConfig.sensor_pitch = psfFile.root.sensorRes[0]
# Main Optics
opticalConfig.NA = psfFile.root.NA[0]
opticalConfig.M = psfFile.root.M[0]
opticalConfig.ftl = psfFile.root.ftl[0]
opticalConfig.fobj = opticalConfig.ftl/opticalConfig.M
opticalConfig.useRelays = False
# MLA
opticalConfig.useMLA = True
opticalConfig.Nnum = psfFile.root.Nnum
opticalConfig.mla2sensor = psfFile.root.mla2sensor
opticalConfig.fm = psfFile.root.fm

# Create a Microscope
WBMicro = Microscope(psfIn, [], opticalConfig).to(device)
WBMicro.eval()

# Load GT LF image
GT_LF_img = torch.tensor(psfFile.root.LFImage, dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0)

# Initial Volume
curr_volume = torch.autograd.Variable(torch.ones(GT_volume.shape, requires_grad=True), requires_grad=True)

# Refinement net
refine_net = torch.nn.Conv2d(nDepths, nDepths, kernel_size=5, padding=2, padding_mode='reflect')

params = [{'params' : refine_net.parameters(), 'lr' : lrNet },
            {'params' : curr_volume, 'lr' : lr}]

crit = nn.MSELoss()
optimizer = optim.Adam(params, lr=lr)

# Arrays for storing statistics
errors = []
predictions = []

# Optimize
for ep in range(nEpochs):
    plt.clf()
    optimizer.zero_grad()
    # Predict defocused image with current defocus
    regularization = refine_net(curr_volume)

    # ReLU as non negativity constraing
    currImg = WBMicro(curr_volume)
    

    diff = crit(GT_LF_img, currImg) + 1e-6*torch.abs(regularization).sum()
    
    # Compute error
    curr_error = diff.detach().item()
    # Propagate gradients back
    diff.backward()
    # Update wave_prop
    optimizer.step()

    # Store errors
    errors.append(curr_error)

    print(str(ep)+' MSE: '+str(curr_error))

    # Display results       
    plt.subplot(3,2,2)
    plt.imshow(currImg[0,0,:,:].detach().cpu().numpy())
    plt.title('Current Guess')
    plt.gray()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    
    plt.subplot(3,2,1)
    plt.imshow(GT_LF_img[0,0,:,:].detach().cpu().numpy())
    plt.title('GT image')
    plt.gray()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    ax = plt.subplot(3,2,3)
    ax.plot(errors, alpha=0.9, color="b" , label='Image')
    # ax.plot(errorsPM, alpha=0.9, color="r" , label='PM')
    ax.hlines(0, 0, len(errors)+1, linewidth=1, color="k")
    ax.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.title("L2 loss image: "+ '{:06.3f}'.format(curr_error))


    plt.subplot(3,2,4)
    plt.imshow(regularization[0,:,:,:].sum(0).detach().cpu().numpy())
    plt.title('refined')
    plt.gray()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    plt.subplot(3,2,5)
    plt.imshow(GT_volume[0,:,:,:].sum(0).detach().cpu().numpy())
    plt.title('GT volume')
    plt.gray()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    plt.subplot(3,2,6)
    plt.imshow(curr_volume[0,:,:,:].sum(0).detach().cpu().numpy())
    plt.title('current_volume')
    plt.gray()
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    plt.pause(0.1)
    plt.show()
psfFile.close()