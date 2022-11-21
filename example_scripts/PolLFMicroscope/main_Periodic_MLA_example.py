"""
Example script of WaveBlocks framework.
This script uses a Fourier Light Field (FLF) microscope with a Microlens Array (MLA)

The aim of this experiment is to demonstrate using the FLF microscope with a standard periodic MLA.
Therefore we forward project a volume of a fish, generating the GT_LF_image.

# Erik Riedel & Josef Kamysek
# erik.riedel@tum.de & josef@kamysek.com
# 15/10/2020, Munich, Germany

"""

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import pathlib
import h5py

# Waveblocks imports
import waveblocks
from waveblocks.microscopes.lightfield_micro2 import Microscope
from waveblocks.blocks.optic_config import OpticConfig
from waveblocks.blocks.microlens_arrays import MLAType
import waveblocks.blocks.point_spread_function as psf
from waveblocks.utils.helper import get_free_gpu

waveblocks.set_logging(debug_mla=True, debug_microscope=True, debug_richardson_lucy=True, debug_optimizer=True)

torch.set_num_threads(8)

# Optical Parameters
depth_range = [-50, 50]
depth_step = 25
depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
n_depths = len(depths)
# Change size of volume you are looking at
vol_xy_size = 141

# Configuration parameters
lr = 5e-2
max_volume = 1
n_epochs = 1

file_path = pathlib.Path(__file__).parent.absolute()
output_path = file_path.parent.joinpath("runs/LF_PM/")
prefix = "_basic_example_"

# Fetch Device to use
device = torch.device(
    "cuda:" + str(get_free_gpu()) if torch.cuda.is_available() else "cpu"
)

# Load volume to use as our object in front of the microscope
vol_file = h5py.File(
    file_path.parent.joinpath("data/fish_phantom_251_251_51.h5"),
    "r")
gt_volume = (
    torch.tensor(vol_file['fish_phantom'])
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

# Create opticalConfig object with the information from the microscope
optic_config = OpticConfig()

# Update optical config from input PSF
# Lateral size of PSF in pixels
psf_size = 17 * 31
# Microscope numerical aperture
optic_config.PSF_config.NA = 0.9
# Microscope magnification
optic_config.PSF_config.M = 20
# Microscope tube-lens focal length
optic_config.PSF_config.Ftl = 200000
# Objective focal length = Ftl/M
optic_config.PSF_config.fobj = optic_config.PSF_config.Ftl / optic_config.PSF_config.M
# Emission wavelength
optic_config.PSF_config.wvl = 0.593
# Immersion refractive index
optic_config.PSF_config.ni = 1

# Camera
optic_config.sensor_pitch = 6.5

# MLA
optic_config.use_mla = True
optic_config.mla_type = MLAType.periodic
# Distance between micro lenses centers
optic_config.MLAPitch = 100

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

# Create Microscope
wb_micro = Microscope(
    optic_config=optic_config, members_to_learn=[], psf_in=psfIn
).to(device)
wb_micro.eval()

# Create observed FLF image
gt_flf_img = wb_micro(gt_volume.detach()).detach()

# Print observed image
plt.imshow(gt_flf_img[0, 0, :, :].detach().cpu().numpy())
plt.show()
plt.savefig('output.png')
print("Script finished!")
