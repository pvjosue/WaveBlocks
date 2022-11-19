"""
Example script of WaveBlocks framework.
This script uses a Fourier Light Field (FLF) microscope with a Microlens Array (MLA)

The aim of this experiment is to compare the fourier convolution with a conventional
Therefore we forward project a volume of a fish, generating the GT_LF_image.
We do this with each technique and compare results and speed.

Erik Riedel & Josef Kamysek
erik.riedel@tum.de & josef@kamysek.com
15/10/2020, Munich, Germany
"""

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import tables
import time
import pathlib
import os

# Waveblocks imports
import waveblocks
from waveblocks.microscopes.fourier_lightfield_mla_micro import Microscope
from waveblocks.blocks.optic_config import OpticConfig
import waveblocks.blocks.point_spread_function as psf
from waveblocks.blocks.microlens_arrays import MLAType
from waveblocks.utils.helper import get_free_gpu

waveblocks.set_logging(debug_mla=True, debug_microscope=True, debug_richardson_lucy=True, debug_optimizer=True)

torch.set_num_threads(8)


work_dir = os.getcwd()
save_img_path = f"{work_dir}/outputs/{os.path.basename(__file__)[:-3]}/"
os.makedirs(save_img_path, exist_ok=True)

# Optical Parameters
depth_range = [-50, 50]
depth_step = 25
depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
n_depths = len(depths)
# Change size of volume you are looking at
vol_xy_size = 151

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
    "cuda:" + str(get_free_gpu()) if torch.cuda.is_available() else "cpu"
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
# Force odd volume
if gt_volume.shape[2] % 2 == 0:
    gt_volume = torch.nn.functional.pad(gt_volume, [1, 0, 1, 0])
gt_volume /= gt_volume.max()
gt_volume *= max_volume

# GT_volume = torch.zeros_like(GT_volume)
# GT_volume[:,0,74:78,74:80] = 1


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


##################################################
# Test fourier conv
optic_config.use_fft_conv = True
wb_micro = Microscope(
    optic_config=optic_config,
    members_to_learn=[],
    psf_in=psfIn,
    mla_coordinates=mla_coordinates,
    mla_shape=mla_shape,
).to(device)
wb_micro.eval()

# Create observed FLF image
torch.cuda.synchronize()
torch.cuda.synchronize()
start = time.perf_counter()

gt_flf_img_fourier = wb_micro(gt_volume.detach()).detach()

torch.cuda.synchronize()
end = time.perf_counter()
time_fft_conv = end - start


optic_config.use_fft_conv = False
wb_micro = Microscope(
    optic_config=optic_config,
    members_to_learn=[],
    psf_in=psfIn,
    mla_coordinates=mla_coordinates,
    mla_shape=mla_shape,
).to(device)
wb_micro.eval()

torch.cuda.synchronize()
torch.cuda.synchronize()
start = time.perf_counter()

# Create observed FLF image
gt_flf_img = wb_micro(gt_volume.detach()).detach()

torch.cuda.synchronize()
end = time.perf_counter()
time_conv = end - start


error = (gt_flf_img - gt_flf_img_fourier).abs()


r = [0, -1]
plt.figure(figsize=(10,3))
plt.subplot(1, 3, 1)
plt.imshow(gt_flf_img[0, :, r[0] : r[1], r[0] : r[1]].sum(0).detach().cpu().numpy())
plt.title(f"Conv. time: {time_conv:,.4f}")
plt.subplot(1, 3, 2)
plt.imshow(
    gt_flf_img_fourier[0, :, r[0] : r[1], r[0] : r[1]].sum(0).detach().cpu().numpy()
)
plt.title(f"Four. Conv. time: {time_fft_conv:,.4f}")
plt.subplot(1, 3, 3)
plt.imshow(error[0, 0, r[0] : r[1], r[0] : r[1]].detach().cpu().numpy())
plt.title(f"Max error {error.max().detach().cpu().item():,.4f}")

plt.savefig(f'{save_img_path}/Fourier_vs_conv.png')
