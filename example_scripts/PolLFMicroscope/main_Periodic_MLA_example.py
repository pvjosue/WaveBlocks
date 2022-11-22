# Example script of WaveBlocks framework.
# This script creates a defocused image in front of the microscope and predicts the defocusing
# distance, by minimizing the MSE between the image generated with the GT defocus, vs the one
# with the current defocus. For this the file microWithPropagation uses a wave-propagation module
# and a Camera module for rendering.

# Josue Page Vizcaino
# pv.josue@gmail.com
# 02/08/2020, Bern Switzerland

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv
from datetime import datetime
import pathlib
from tifffile import imwrite

# Waveblocks imports
from waveblocks.microscopes.lightfield_micro import Microscope
from waveblocks.blocks.optic_config import OpticConfig
from waveblocks.blocks.microlens_arrays import MLAType
import waveblocks.blocks.point_spread_function as psf

# torch.set_num_threads(16)

# Optical Parameters
depth_step = 0.43
depth_range = [-depth_step*5, depth_step*5]
depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
nDepths = len(depths)
vol_xy_size = 501



# Configuration parameters
file_path = pathlib.Path(__file__).parent.absolute()
data_path = file_path.parent.joinpath("data")
plot = False
# Fetch Device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enable plotting
if plot:
    fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor="w", edgecolor="k")
    plt.ion()
    plt.show()

# Load volume to use as our object in front of the microscope
vol_file = h5py.File(
    data_path.joinpath("fish_phantom_251_251_51.h5"), "r"
)
GT_volume = (
    torch.tensor(vol_file['fish_phantom'])
    .permute(2, 1, 0)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)
GT_volume = torch.nn.functional.interpolate(
    GT_volume, [vol_xy_size, vol_xy_size, nDepths]
)
GT_volume = GT_volume[:, 0, ...].permute(0, 3, 1, 2).contiguous()
GT_volume /= GT_volume.max()

# Create opticalConfig object with the information from the microscope
opticalConfig = OpticConfig()

# Update optical config from input PSF
psf_size = 17 * 11
opticalConfig.PSF_config.NA = 1.2 
opticalConfig.PSF_config.M = 60
opticalConfig.PSF_config.Ftl = 200000
opticalConfig.PSF_config.wvl = 0.593
opticalConfig.PSF_config.ni = 1.35
opticalConfig.PSF_config.ni0 = 1.35
opticalConfig.PSF_config.fobj = (
    opticalConfig.PSF_config.Ftl / opticalConfig.PSF_config.M
)

# first zero found in the center at:
# depths = np.append(depths,-2*opticalConfig.PSF_config.wvl/(opticalConfig.PSF_config.NA**2))
# depths = np.append(depths,2*opticalConfig.PSF_config.wvl/(opticalConfig.PSF_config.NA**2))

# Camera
opticalConfig.sensor_pitch = 6.5
opticalConfig.use_relay = False

# MLA
opticalConfig.use_mla = True
opticalConfig.MLAPitch = 100
opticalConfig.Nnum = 2 * [opticalConfig.MLAPitch // opticalConfig.sensor_pitch]
opticalConfig.Nnum = [int(n + (1 if (n % 2 == 0) else 0)) for n in opticalConfig.Nnum]
opticalConfig.mla2sensor = 2500
opticalConfig.fm = 2500
opticalConfig.mla_type = MLAType.periodic

# Define PSF
PSF = psf.PSF(opticalConfig)
_, psf_in = PSF.forward(
    opticalConfig.sensor_pitch / opticalConfig.PSF_config.M, psf_size, depths
)

# Create a Microscope
WBMicro = Microscope(optic_config=opticalConfig, members_to_learn=[], psf_in=psf_in).to(
    device
)
# WBMicro.mla2sensor = WBMicro.mla2sensor.to("cuda:1")
WBMicro.eval()

with torch.no_grad():
    # Compute GT LF image
    GT_LF_img = WBMicro(
        GT_volume.detach()
    )
    
plt.imshow(GT_LF_img[0,0,...].detach().cpu().numpy())
plt.savefig('output_PolScopeSettings_ni135.png')
# LF_center = LF_psf[0,:,-1,-1,:,:].detach().numpy() ; 
# imwrite('PSF_center.tif', LF_center)
plt.show()


# for n in range(nDepths):
#     plt.subplot(1,3,1)
#     psf = (torch.abs(WBMicro.psf_at_sensor[0,n,0,-1,...])**2)
#     plt.imshow(psf.detach().cpu().numpy())
#     plt.title(str(psf.sum().item()))
#     plt.subplot(1,3,2)
#     psf = torch.abs(psf-(torch.abs(WBMicro.psf_at_sensor[0,n,-1,-1,...])**2))
#     plt.imshow(psf.detach().cpu().numpy())
#     plt.title(str(psf.sum().item()))
#     plt.subplot(1,3,3)
#     psf = (torch.abs(WBMicro.psf_at_sensor[0,n,-1,0,...])**2)
#     plt.imshow(psf.detach().cpu().numpy())
#     plt.title(str(psf.sum().item()))
#     # plt.title(str(n))
#     plt.show()