# Example script of WaveBlocks framework.
# This script uses a Fourier Light Field (FLF) microscope with a Microlens Array (MLA)
# The aim of this experiment is to look at the functionality of the FLF microscope with a MLA
# Therefore we forward project a volume of a fish, generating the GT_LF_image.

# Erik Riedel & Josef Kamysek
# erik.riedel@tum.de & josef@kamysek.com
# 15/10/2020, later by Josue Page 15/09/2020

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import pathlib
import logging

# Waveblocks imports
import waveblocks
from waveblocks.microscopes.fourier_lightfield_mla_micro import Microscope
import waveblocks.reconstruction.deconvolution.richardson_lucy as RL
from waveblocks.utils.misc_tools import volume_2_projections, load_volume
from waveblocks.blocks.optic_config import OpticConfig
from waveblocks.blocks.microlens_arrays import MLAType
import waveblocks.blocks.point_spread_function as psf
from waveblocks.utils.helper import get_free_gpu

torch.set_num_threads(8)

# Configure logging
logger = logging.getLogger("Waveblocks")
waveblocks.set_logging(debug_mla=False, debug_microscope=False, debug_richardson_lucy=False, debug_optimizer=True)

use_gpu = False

# Optical Parameters
depth_range = [-50, 50]
depth_step = 10
depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
n_depths = len(depths)

# Problem divisor (for faster performance)
problem_divisor = 2.5 #if use_gpu else 1

# Set volume size
vol_xy_size = int(301 / problem_divisor)
vol_xy_size = vol_xy_size + int(vol_xy_size%2==0)

# Reconstruction parameters
n_iterations = 100

# Configuration parameters
file_path = pathlib.Path(__file__).parent.absolute()
output_path = file_path.parent.joinpath("runs/")
data_path = file_path.parent.joinpath("data")
posfix = "_basic_example_"

# Which image to use?
# Default fish
volume_path = data_path.joinpath("fish_phantom_251_251_51.h5")
# or actual XLFM fish? 
volume_path = data_path.joinpath("XLFM_stack_001.tif")


# Fetch Device to use
device = torch.device(
    "cuda:1" if use_gpu and torch.cuda.is_available() else "cpu"
)

# Load volume to use as our object in front of the microscope
gt_volume = load_volume(str(volume_path), [vol_xy_size, vol_xy_size, n_depths], 0.15, device=device)

################### Create opticalConfig object with the information from the microscope
optic_config = OpticConfig()

# Update optical config from input PSF
# Lateral size of PSF in pixels
psf_size = int(2161 / problem_divisor)
psf_size = psf_size + int(psf_size%2==0)

# Microscope numerical aperture
optic_config.PSF_config.NA = 0.8
# Microscope magnification
optic_config.PSF_config.M = 16
# Microscope tube-lens focal length
optic_config.PSF_config.Ftl = 180000
# Objective focal length = Ftl/M
# optic_config.PSF_config.fobj = optic_config.PSF_config.Ftl / optic_config.PSF_config.M
# Emission wavelength
optic_config.PSF_config.wvl = 0.63
# Immersion refractive index
optic_config.PSF_config.ni = 1.344
optic_config.PSF_config.ni0 = 1.344

# Camera
optic_config.sensor_pitch = 6.45

# MLA
optic_config.use_mla = True
optic_config.mla_type = MLAType.coordinate
# Distance between micro lenses centers
optic_config.MLAPitch = 1000
optic_config.MLAPitch_pixels = optic_config.MLAPitch // optic_config.sensor_pitch
# Number of pixels behind a single lens
optic_config.Nnum = 2 * [optic_config.MLAPitch // optic_config.sensor_pitch]
optic_config.Nnum = [int(n + (1 if (n % 2 == 0) else 0)) for n in optic_config.Nnum]
# Distance between the mla and the sensor
optic_config.mla2sensor = 36100
# MLA focal length
optic_config.fm = 36100
# Define PSF
with torch.no_grad():
    PSF = psf.PSF(optic_config=optic_config, members_to_learn=[])
    _, psfIn = PSF.forward(
        optic_config.sensor_pitch / optic_config.PSF_config.M, psf_size, depths
    )

# Define phase_mask to initialize
optic_config.use_relay = True
optic_config.relay_focal_length = 125000
optic_config.relay_aperture_width = 38100

# Enable fourier conv
optic_config.use_fft_conv = True


####################### Create Microscope
mla_shape = 2*[psf_size]
# List of all coordinates in the MLA
mla_coordinates = [
    (187, 1052),
    (292, 1334),
    (301, 766),
    (368, 1636),
    (380, 469),
    (658, 1209),
    (662, 910),
    (667, 1858),
    (685, 252),
    (717, 1504),
    (723, 611),
    (1071, 1977),
    (1078, 1067),
    (1080, 1662),
    (1081, 1362),
    (1091, 152),
    (1092, 465),
    (1091, 768),
    (1437, 1519),
    (1455, 620),
    (1481, 1868),
    (1502, 266),
    (1505, 1219),
    (1510, 920),
    (1781, 1655),
    (1799, 493),
    (1866, 1351),
    (1867, 794),
    (1984, 1078)]

# Arrange the lenslet coordinates such that they fit in the image
mla_coordinates = waveblocks.blocks.microlens_arrays.coordinate_mla.fit_microlenses_into_image(mla_coordinates, optic_config.MLAPitch_pixels, mla_shape)

wb_micro = Microscope(
    optic_config=optic_config,
    members_to_learn=[],
    psf_in=psfIn,
    mla_coordinates=mla_coordinates,
    mla_shape=mla_shape,
).to(device)
wb_micro.eval()

####################### Create observed FLF image
with torch.no_grad():
    gt_flf_img = wb_micro(gt_volume.detach()).detach()
    plt.figure(figsize=(10,10))
    plt.imshow(gt_flf_img[0, 0, :, :].detach().cpu().numpy())
    plt.savefig(str(output_path)+'/FLFM_simulated_image.png')
    plt.show()
    # plt.imshow(wb_micro.psf.float()[0, 0, :, :].detach().cpu().numpy())
    # plt.show()

####################### Reconstruct 3D volume
    logger.setLevel(logging.INFO)
    currVol, img_est, losses = RL.RichardsonLucy(
        wb_micro, gt_flf_img, n_iterations, gt_volume.shape
    )
    logger.setLevel(logging.DEBUG)


####################### Show result
projected_volume = volume_2_projections(currVol.permute(0, 2, 3, 1).unsqueeze(1))
gt_projected_volume = volume_2_projections(gt_volume.permute(0, 2, 3, 1).unsqueeze(1))
error = volume_2_projections(
    (gt_volume - currVol).abs().permute(0, 2, 3, 1).unsqueeze(1)
)


####################### Save to file
plt.figure(figsize=(15,10))
plt.subplot(2, 3, 1)
plt.imshow(gt_projected_volume[0, 0, :, :].detach().cpu().numpy())
plt.title("GT")
plt.subplot(2, 3, 2)
plt.imshow(projected_volume[0, 0, :, :].detach().cpu().numpy())
plt.title("RL deconvolution")
plt.subplot(2, 3, 3)
plt.imshow(error[0, 0, :, :].detach().cpu().numpy())
plt.title("Max error: " + str(error.max().item()))
plt.subplot(2, 1, 2)
plt.plot(list(range(1, n_iterations)), losses[1:])
plt.grid()
# use log scale
plt.xscale("log")
plt.yscale("log", base=2)
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("MSE", fontsize=10)
plt.title("MSE")

plt.savefig(str(output_path)+'/RL_result'+posfix+'.png')
plt.show()
