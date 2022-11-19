"""
Example script of WaveBlocks framework.
This script uses a Fourier Light Field (FLF) microscope with a Microlens Array (MLA) and a phase mask (PM)

The aim of this experiment is to show the minimum configuration of a FLF microscope with a PM included
Therefore we forward project a volume of a fish.


# Erik Riedel & Josef Kamysek
# erik.riedel@tum.de & josef@kamysek.com
# 15/10/2020, Munich, Germany

"""

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import tables
import torch
import pathlib
import math

# Waveblocks imports
import waveblocks
from waveblocks.microscopes.fourier_lightfield_mla_micro import Microscope, preset1
from waveblocks.blocks.microlens_arrays import MLAType
from waveblocks.blocks.optic_config import OpticConfig
import waveblocks.blocks.point_spread_function as psf
from waveblocks.utils.helper import get_free_gpu
from waveblocks.utils import generate_phase_masks as pm

waveblocks.set_logging(debug_mla=True, debug_microscope=True, debug_richardson_lucy=True, debug_optimizer=True)

torch.set_num_threads(8)

# Optical Parameters
depth_range = [-50, 50]
depth_step = 10
depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
n_depths = len(depths)
# Change size of volume you are looking at
vol_xy_size = 51

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
gt_volume = torch.nn.functional.pad(gt_volume, [2, 2, 2, 2])
gt_volume /= gt_volume.max()
gt_volume *= max_volume

# Create opticalConfig object with the information from the microscope
optic_config = preset1()


# Update optical config from input PSF
# Lateral size of PSF in pixels
psf_size = 17 * 31

# Define PSF
PSF = psf.PSF(optic_config=optic_config, members_to_learn=[])
_, psf_in = PSF.forward(
    optic_config.sensor_pitch / optic_config.PSF_config.M, psf_size, depths
)


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

# Enable Fourier convolutions
optic_config.use_fft_conv = True

# Define phase mask
optic_config.use_pm = True
# Number of pixels in phase mask
# compute pixel size at fourier plane of first lens
Fs = 1.0 / optic_config.sensor_pitch
cycles_perum = Fs / psf_in.shape[-2]
# incoherent resolution limit
resolution_limit = optic_config.PSF_config.wvl / (
    optic_config.PSF_config.NA * optic_config.PSF_config.M
)
n_pix_in_fourier = resolution_limit / cycles_perum
# diameter of the objective back focal plane, which acts as the entrance pupil for our system
d_obj = optic_config.PSF_config.fobj * optic_config.PSF_config.NA
# sampling size on the fourier domain
fourier_metric_sampling = d_obj / n_pix_in_fourier

optic_config.pm_sampling = fourier_metric_sampling

optic_config.pm_pixel_size = 8

optic_config.pm_shape = [527, 527]

optic_config.pm_max_phase_shift = 5.4 * math.pi

pm_image = pm.create_phasemask(
    pm.PhaseMaskType.cubic,
    pm.PhaseMaskShape.square,
    {
        "x": optic_config.pm_shape[0],
        "y": optic_config.pm_shape[1],
        "ratio": 0.3,
        "offset": [0, 0],
    },
    information={"max_phase_shift": 1},
)

pm_image = pm_image.unsqueeze(0).unsqueeze(0).float()

wb_micro = Microscope(
    optic_config=optic_config,
    members_to_learn=[],
    psf_in=psf_in,
    mla_coordinates=mla_coordinates,
    mla_shape=mla_shape,
    pm_image=pm_image,
).to(device)

wb_micro.eval()

# Create observed FLF image
gt_flf_img = wb_micro(gt_volume.detach()).detach()

# Print observed image
plt.imshow(gt_flf_img[0, 0, :, :].detach().cpu().numpy())
plt.show()

print("Script finished!")
