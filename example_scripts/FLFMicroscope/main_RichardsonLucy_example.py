"""
Example script of WaveBlocks framework.
This script uses a Fourier Light Field (FLF) microscope with a Microlens Array (MLA)

The aim of this experiment is to look at the functionality of the Richardson Lucy deconvolution algorithm
Therefore we forward project a volume of a fish, generating the GT_LF_image and deconvolute it


Erik Riedel & Josef Kamysek
erik.riedel@tum.de & josef@kamysek.com
15/10/2020, Munich, Germany
"""

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import tables
import pathlib
import logging

# Waveblocks imports
import waveblocks
from waveblocks.microscopes.fourier_lightfield_mla_micro import Microscope
import waveblocks.reconstruction.deconvolution.richardson_lucy as RL
from waveblocks.utils.misc_tools import volume_2_projections
from waveblocks.blocks.optic_config import OpticConfig
from waveblocks.blocks.microlens_arrays import MLAType
import waveblocks.blocks.point_spread_function as psf
from waveblocks.utils.helper import get_free_gpu

torch.set_num_threads(8)

logger = logging.getLogger("Waveblocks")
waveblocks.set_logging(debug_mla=True, debug_microscope=True, debug_richardson_lucy=True, debug_optimizer=True)

# Optical Parameters
depth_range = [-50, 50]
depth_step = 10
depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
n_depths = len(depths)
# Change size of volume you are looking at
vol_xy_size = 151
n_iterations = 50

# Configuration parameters
lr = 5e-2
max_volume = 1
n_epochs = 1
file_path = pathlib.Path(__file__).parent.absolute()
output_path = file_path.parent.joinpath("runs/LF_PM/")
data_path = file_path.parent.joinpath("data")
prefix = "_basic_example_"
# Fetch Device to use
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load volume to use as our object in front of the microscope
vol_file = tables.open_file(
    data_path.joinpath("fish_phantom_251_251_51.h5"), "r", driver="H5FD_CORE"
)
gt_volume = (
    torch.tensor(vol_file.root.fish_phantom)
    .permute(2, 1, 0)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)
gt_volume = torch.nn.functional.interpolate(
    gt_volume, [vol_xy_size, vol_xy_size, n_depths]
)
gt_volume = gt_volume[:, 0, ...].permute(0, 3, 1, 2).contiguous()
gt_volume = torch.nn.functional.pad(gt_volume, [2, 2, 2, 2])
gt_volume /= gt_volume.max()
gt_volume *= max_volume
gt_volume[gt_volume < 0.15] = 0

# GT_volume = torch.zeros_like(GT_volume)
# GT_volume[0,5,75,75] = 1
# Create opticalConfig object with the information from the microscope
optic_config = OpticConfig()

# Update optical config from input PSF
# Lateral size of PSF in pixels
psf_size = 17 * 31
# Microscope numerical aperture
optic_config.PSF_config.NA = 0.45
# Microscope magnification
optic_config.PSF_config.M = 6
# Microscope tube-lens focal length
optic_config.PSF_config.Ftl = 165000
# Objective focal length = Ftl/M
optic_config.PSF_config.fobj = optic_config.PSF_config.Ftl / optic_config.PSF_config.M
# Emission wavelength
optic_config.PSF_config.wvl = 0.63
# Immersion refractive index
optic_config.PSF_config.ni = 1

# Camera
optic_config.sensor_pitch = 3.9
optic_config.useRelays = False

# MLA
optic_config.use_mla = True
optic_config.mla_type = MLAType.coordinate
# Distance between micro lenses centers
optic_config.MLAPitch = 250

# Number of pixels behind a single lens
optic_config.Nnum = 2 * [optic_config.MLAPitch // optic_config.sensor_pitch]
optic_config.Nnum = [int(n + (1 if (n % 2 == 0) else 0)) for n in optic_config.Nnum]

# Distance between the mla and the sensor
optic_config.mla2sensor = 2500

# MLA focal length
optic_config.fm = 2500

# Define PSF
PSF = psf.PSF(optic_config=optic_config, members_to_learn=[])
_, psfIn = PSF.forward(
    optic_config.sensor_pitch / optic_config.PSF_config.M, psf_size, depths
)

# Define phase_mask to initialize
optic_config.use_relay = True
optic_config.relay_focal_length = 150000
optic_config.relay_separation = optic_config.relay_focal_length * 2
optic_config.relay_aperture_width = 50800

# Enable fourier conv
optic_config.use_fft_conv = True

# Create Microscope
# List of all coordinates in the MLA
mla_coordinates = [
    (100, 250),
    (250, 250),
    (400, 250),
    (100, 100),
    (250, 100),
    (400, 100),
    (100, 400),
    (250, 400),
    (400, 400),
]
mla_shape = [500, 500]
wb_micro = Microscope(
    optic_config=optic_config,
    members_to_learn=[],
    psf_in=psfIn,
    mla_coordinates=mla_coordinates,
    mla_shape=mla_shape,
).to(device)
wb_micro.eval()

# Create observed FLF image
gt_flf_img = wb_micro(gt_volume.detach()).detach()
plt.imshow(gt_flf_img[0, 0, :, :].detach().cpu().numpy())
plt.show()
plt.imshow(wb_micro.psf.float()[0, 0, :, :].detach().cpu().numpy())
plt.show()

# psfAtSensor has shape(1,11,340,340)    ;   GT_FLF_img has shape (1,1,375,375)
logger.setLevel(logging.INFO)
currVol, img_est, losses = RL.RichardsonLucy(
    wb_micro, gt_flf_img, n_iterations, gt_volume.shape
)
logger.setLevel(logging.DEBUG)

# Crop to volume of interest
x_start = int((currVol.shape[2] / 2) - (vol_xy_size / 2)) + 20
y_start = int((currVol.shape[3] / 2) - (vol_xy_size / 2)) + 20


# Show volumes
projected_volume = volume_2_projections(currVol.permute(0, 2, 3, 1).unsqueeze(1))
gt_projected_volume = volume_2_projections(gt_volume.permute(0, 2, 3, 1).unsqueeze(1))
error = volume_2_projections(
    (gt_volume - currVol).abs().permute(0, 2, 3, 1).unsqueeze(1)
)

plt.subplot(2, 4, 1)
plt.imshow(gt_flf_img[0, 0, :, :].detach().cpu().numpy())
plt.title("LF img")
plt.subplot(2, 4, 2)
plt.imshow(gt_projected_volume[0, 0, :, :].detach().cpu().numpy())
plt.title("GT")
plt.subplot(2, 4, 3)
plt.imshow(projected_volume[0, 0, :, :].detach().cpu().numpy())
plt.title("RL deconvolution")
plt.subplot(2, 4, 4)
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

plt.savefig('output.png')
# plt.show()
