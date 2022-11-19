# Example script of WaveBlocks framework.
# This script uses a Light Field microscope with a Phase-Mask in its optical path.
# The aim of this experiment is to recover a phase-mask based on observed images at the camera sensor, for this:
# - We set a PM and forward project a volume of a fish, generating the GT_LF_image.
# - Create a clean microscope with a random initial PM.
# - Forward project the same volume with the new PM and compute the error between the GT_LF_image and the current_image.
# - Backpropagate the error and update the PM

# Josue Page Vizcaino
# pv.josue@gmail.com
# 02/09/2020, München Germany

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import tables
import torch.optim as optim
import math
from datetime import datetime
import pathlib
from tqdm import tqdm

# Waveblocks imports
from waveblocks.microscopes.lightfield_pm_micro import Microscope
from waveblocks.blocks.optic_config import OpticConfig
from waveblocks.blocks.microlens_arrays import MLAType
import waveblocks.blocks.point_spread_function as psf
from waveblocks.utils.helper import get_free_gpu
import logging
import os 

logger = logging.getLogger("Waveblocks")
logger.setLevel(logging.INFO)
torch.set_num_threads(8)

work_dir = os.getcwd()
save_img_path = f"{work_dir}/outputs/{os.path.basename(__file__)[:-3]}/"
os.makedirs(save_img_path, exist_ok=True)

# Optical Parameters
depth_range = [-50, 50]
depth_step = 10
depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
nDepths = len(depths)
vol_xy_size = 251

# Configuration parameters
lr = 5e-2
maxVolume = 1
nEpochs = 2000

file_path = pathlib.Path(__file__).parent.absolute()
output_path = str(file_path.parent.joinpath("runs/LF_PM/")) + "/"
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
GT_volume = (
    torch.tensor(vol_file.root.fish_phantom)
    .permute(2, 1, 0)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)
GT_volume = torch.nn.functional.interpolate(
    GT_volume, [vol_xy_size, vol_xy_size, nDepths]
)
GT_volume = GT_volume[:, 0, ...].permute(0, 3, 1, 2).contiguous()
GT_volume = torch.nn.functional.pad(GT_volume, [2, 2, 2, 2])
GT_volume /= GT_volume.max()
GT_volume *= maxVolume

# Create opticalConfig object with the information from the microscope
opticalConfig = OpticConfig()

# Update optical config from input PSF
# Lateral size of PSF in pixels
psf_size = 17 * 5
# Microscope numerical aperture
opticalConfig.PSF_config.NA = 0.45
# Microscope magnification
opticalConfig.PSF_config.M = 20
# Microscope tube-lens focal length
opticalConfig.PSF_config.Ftl = 165000
# Objective focal length = Ftl/M
opticalConfig.PSF_config.fobj = (
    opticalConfig.PSF_config.Ftl / opticalConfig.PSF_config.M
)
# Emission wavelength
opticalConfig.PSF_config.wvl = 0.63
# Immersion refractive index
opticalConfig.PSF_config.ni = 1

# Camera
opticalConfig.sensor_pitch = 6.9

# MLA
opticalConfig.use_mla = False
opticalConfig.mla_type = MLAType.periodic
# Distance between micro lenses centers
opticalConfig.MLAPitch = 112
# Number of pixels behind a single lens
opticalConfig.Nnum = 2 * [opticalConfig.MLAPitch // opticalConfig.sensor_pitch]
opticalConfig.Nnum = [int(n + (1 if (n % 2 == 0) else 0)) for n in opticalConfig.Nnum]
# Distance between the mla and the sensor
opticalConfig.mla2sensor = 2500
# MLA focal length
opticalConfig.fm = 2500

# Define PSF
PSF = psf.PSF(opticalConfig, [])
_, psf_in = PSF.forward(
    opticalConfig.sensor_pitch / opticalConfig.PSF_config.M, psf_size, depths
)

# Define phase_mask to initialize
opticalConfig.use_relay = True
opticalConfig.relay_focal_length = 150000
opticalConfig.relay_separation = opticalConfig.relay_focal_length * 2
opticalConfig.relay_aperture_width = 50800

# Size of central circular window
pm_ratio = 0.25
opticalConfig.pm_pixel_size = 8  # size of PM pixels in micrometers
# Number of pixels in phase mask
# compute pixel size at fourier plane of first lens
Fs = 1.0 / opticalConfig.sensor_pitch
cycles_perum = Fs / psf_in.shape[-2]
# incoherent resolution limit
resolution_limit = opticalConfig.PSF_config.wvl / (
    opticalConfig.PSF_config.NA * opticalConfig.PSF_config.M
)
n_pix_in_fourier = resolution_limit / cycles_perum
# diameter of the objective back focal plane, which acts as the entrance pupil for our system
d_obj = opticalConfig.PSF_config.fobj * opticalConfig.PSF_config.NA
# sampling size on the fourier domain
fourier_metric_sampling = d_obj / n_pix_in_fourier

opticalConfig.pm_sampling = fourier_metric_sampling
opticalConfig.pm_shape = [
    int(opticalConfig.pm_pixel_size / fourier_metric_sampling * 1080),
    int(opticalConfig.pm_pixel_size / fourier_metric_sampling * 1080),
]
opticalConfig.max_phase_shift = 5.4 * math.pi

# Create a GT Phase mask
u = np.arange(start=-0.5, stop=0.5, step=1 / opticalConfig.pm_shape[0], dtype="float")
v = np.arange(start=-0.5, stop=0.5, step=1 / opticalConfig.pm_shape[1], dtype="float")
X, Y = np.meshgrid(v, u)
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()
if True:
    # round phase-mask
    pm_mask = ~(torch.sqrt(torch.mul(X, X) + torch.mul(Y, Y))).ge(
        torch.tensor([pm_ratio])
    )
else:
    # square phase-mask
    pm_mask = ~(
        torch.abs(X).ge(torch.tensor([pm_ratio / 2]))
        | torch.abs(Y).ge(torch.tensor([pm_ratio / 2]))
    )

X *= opticalConfig.pm_shape[0]
Y *= opticalConfig.pm_shape[1]

if True:
    # round phase mask
    pm_img = pm_mask.float() * torch.sqrt(torch.mul(X, X) + torch.mul(Y, Y))
else:
    # cubic phase mask
    pm_img = (
        0.0005
        * (torch.mul(torch.mul(X, X), X) + torch.mul(torch.mul(Y, Y), Y))
        * pm_mask.float()
    )

# Set Phase mask and store GT
GT_Phase_Mask = pm_img.unsqueeze(0).unsqueeze(0).float().to(device)
opticalConfig.pm_image = GT_Phase_Mask

# Create Ground Truth Microscope
WBMicro = Microscope(optic_config=opticalConfig, members_to_learn=[], psf_in=psf_in).to(
    device
)
WBMicro.eval()

# Create observed LF image
GT_LF_img = WBMicro(GT_volume.detach()).detach()
del WBMicro

# Reset Phase mask to random values
opticalConfig.pm_image = torch.ones_like(opticalConfig.pm_image)
# Create Microscope to optimize
WBMicro = Microscope(
    optic_config=opticalConfig,
    members_to_learn=["phaseMask.function_img"],
    psf_in=psf_in,
).to(device)

# Gather parameters
params = [{"params": WBMicro.get_trainable_variables(), "lr": lr}]

crit = torch.nn.MSELoss()
optimizer = optim.Adam(params, lr=lr)

save_folder = (
    output_path
    + datetime.now().strftime("%Y_%m_%d__%H:%M:%S")
    + "_"
    + str(nDepths)
    + "_nD__"
    + str(lr)
    + "_lr__"
    + prefix
)

# Create tensorboard summary writer to log stuff
# writer = SummaryWriter(log_dir=save_folder)

# Arrays for storing statistics
errors_imgs = []
errors_pm = []
# Optimize
for ep in tqdm(range(nEpochs)):
    plt.clf()
    optimizer.zero_grad()
    WBMicro.zero_grad()

    # Simulate image with current PhaseMask
    currImg = WBMicro(GT_volume)

    # Compute error between observed and GT images
    img_loss = crit(currImg, GT_LF_img)

    # Propagate gradients back
    img_loss.backward()
    # Update phase mask
    optimizer.step()

    # Store errors_imgs
    curr_error = img_loss.item()
    errors_imgs.append(curr_error)

    # Phase mask error
    curr_PM = WBMicro.phaseMask.function_img.detach()
    curr_pm_error = crit(GT_Phase_Mask, curr_PM).item()
    errors_pm.append(curr_pm_error)

    # logger.info(str(ep) + " MSE_img: " + str(curr_error))
    # logger.info(str(ep) + " MSE_PM: " + str(curr_pm_error))

    if ep%10==0:
        # plt.subplot(2,2,1)
        # plt.imshow(volume_2_projections(GT_volume).squeeze().cpu().numpy())
        plt.subplot(3,2,1)
        plt.imshow(GT_LF_img.sum(1).squeeze().cpu().numpy())
        plt.title('GT_LF')
        plt.subplot(3,2,2)
        plt.imshow(currImg.sum(1).detach().squeeze().cpu().numpy())
        plt.title('Pred_LF')
        plt.subplot(3,2,3)
        plt.imshow(GT_Phase_Mask[0, ...].detach().squeeze().cpu().numpy())
        plt.title('GT_PM')
        plt.subplot(3,2,4)
        plt.imshow(curr_PM[0, ...].detach().squeeze().cpu().numpy())
        plt.title('Pred_PM')
        plt.subplot(3,2,5)
        plt.plot(errors_pm)
        plt.title('Error PM')
        plt.subplot(3,2,6)
        plt.plot(errors_imgs)
        plt.title('Error img')
        plt.show()
        plt.savefig(f'{save_img_path}/output_PM_learning_ep{ep}.png')