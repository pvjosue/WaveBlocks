# Josue Page Vizcaino
# pv.josue@gmail.com
# 22/June/2020, Bern Switzerland

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import h5py
import pathlib

# Waveblocks imports
from waveblocks.microscopes.fourier_lightfield_mla_micro import Microscope
from waveblocks.blocks.optic_config import OpticConfig
from waveblocks.blocks.camera import Camera
import waveblocks.blocks.point_spread_function as psf
import waveblocks.utils.complex_operations as co

use_amp = False
if hasattr(torch.cuda, "amp"):
    use_amp = True
    from torch.cuda.amp import autocast, GradScaler

torch.set_num_threads(16)

# Optical Parameters
nDepths = 64
nSpheres = 1
mSphereSize = 1
nSimulations = 10

mlas_per_psf = 5
depth_step = 0.9
depth_range = [
    -depth_step * nDepths // 2,
    -depth_step * nDepths // 2 + depth_step * (nDepths - 1),
]
depths = -np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
nDepths = len(depths)

subsampling = 0.5
nGpus = min(1, torch.cuda.device_count())
# device_array = ["cuda:"+str(n) for n in range(nGpus)]
# device_array = ["cuda:"+str(n) for n in range(1,nGpus)]
device_array = ["cuda:1"]
print(device_array)
depths_per_Micro = nDepths // len(device_array)

# Configuration parameters
file_path = pathlib.Path(__file__).parent.absolute()
output_path = str(file_path.parent.joinpath("runs/simulations_single_beads/")) + "/"
data_path = file_path.parent.joinpath("data")
prefix = "single_bead_center"
plot = False
# Fetch Device to use
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Enable plotting
if plot:
    fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor="w", edgecolor="k")
    plt.ion()
    plt.show()

# Create opticalConfig object with the information from the microscope
opticalConfig = OpticConfig()

# Camera
opticalConfig.sensor_pitch = 3.45 / subsampling
opticalConfig.use_relay = False
# MLA
opticalConfig.use_mla = True
opticalConfig.MLAPitch = 112
opticalConfig.Nnum = 2 * [opticalConfig.MLAPitch // opticalConfig.sensor_pitch]
opticalConfig.Nnum = [int(n + (1 if (n % 2 == 0) else 0)) for n in opticalConfig.Nnum]
opticalConfig.mla2sensor = 2500
opticalConfig.fm = 2500

# Update optical config from input PSF
psf_size = opticalConfig.Nnum[0] * mlas_per_psf
opticalConfig.PSF_config.NA = 0.9
opticalConfig.PSF_config.M = 40
opticalConfig.PSF_config.Ftl = 165000
opticalConfig.PSF_config.wvl = 0.63
opticalConfig.PSF_config.ni = 1
opticalConfig.PSF_config.fobj = (
    opticalConfig.PSF_config.Ftl / opticalConfig.PSF_config.M
)

# Out Total mlas
mlas_behind_vol = 3
mlas_neight = 9
total_mlas = mlas_behind_vol + mlas_neight - 1

vol_shape = mlas_behind_vol * opticalConfig.Nnum[0]
full_vol_shape = total_mlas * opticalConfig.Nnum[0]
pad_vol = full_vol_shape - vol_shape
pad_vol_val = [
    pad_vol // 2,
    pad_vol - pad_vol // 2,
    pad_vol // 2,
    pad_vol - pad_vol // 2,
]

# Define PSF
PSF = psf.PSF(opticalConfig)
_, psf_in = PSF.forward(
    opticalConfig.sensor_pitch / opticalConfig.PSF_config.M, psf_size, depths
)

# Compute psfs in single microscope for correct normalization
WBMicro = Microscope(optic_config=opticalConfig, members_to_learn=[], psf_in=psf_in).to(
    device_array[0]
)

normalized_psf = Camera.normalize_psf(WBMicro.psf_at_sensor.abs())

del WBMicro

# Create a Microscopes
psf_indices_per_micro = [
    range(i, min(i + depths_per_Micro, nDepths - 1))
    for i in range(0, nDepths, depths_per_Micro)
]
WBMicro_array = []

for ix, curr_psf_indices in enumerate(psf_indices_per_micro):
    WBMicro = Microscope(
        optic_config=opticalConfig,
        members_to_learn=[],
        psf_in=normalized_psf[:, curr_psf_indices, ...],
        precompute=False,
    ).to(device_array[ix])
    WBMicro.eval()
    WBMicro_array.append(WBMicro)

save_folder = (
    output_path
    + datetime.now().strftime("%Y_%m_%d__%H:%M:%S")
    + "_"
    + str(nDepths)
    + "_nD__"
    + str("{:.2f}".format(depth_range[0]))
    + "_"
    + str(depth_step)
    + "_"
    + str("{:.2f}".format(depth_range[1]))
    + "_D__"
    + str(subsampling)
    + "__"
    + str(mlas_per_psf)
    + "_mlasPpsf_"
    + prefix
)

# Create summary writer to log stuff
writer = SummaryWriter(log_dir=save_folder)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Configure spheres shape
pixelSize = opticalConfig.sensor_pitch / opticalConfig.PSF_config.M
mSphereNpixels = np.ceil(mSphereSize / pixelSize) // 2 * 2 + 1
mSphereNpixelsHalf = int(mSphereNpixels // 2)
mSphereSize = mSphereNpixels * pixelSize

# Create microsphere image
y, x = np.ogrid[
    -mSphereNpixelsHalf : mSphereNpixelsHalf + 1,
    -mSphereNpixelsHalf : mSphereNpixelsHalf + 1,
]
mask = mSphereNpixelsHalf ** 2 - torch.from_numpy(x * x + y * y)
mask[mask < 0] = 0
mask = mask.float()
mask /= mask.sum()

# Create output file
LF_shape = [opticalConfig.Nnum[0], opticalConfig.Nnum[1], total_mlas, total_mlas]
h5file = h5py.File(
    save_folder
    + "/Simul_spheres_"
    + str(nSpheres)
    + "nS__"
    + str(nDepths)
    + "nD__"
    + str(subsampling)
    + "subS__"
    + str(nSimulations)
    + "nFiles.h5",
    "w",
)
h5file.create_dataset(
    "volData",
    shape=[full_vol_shape, full_vol_shape, nDepths, nSimulations],
    dtype="|u1",
)
h5file.create_dataset(
    "LFData",
    shape=[LF_shape[0], LF_shape[1], LF_shape[2], LF_shape[3], nSimulations],
    dtype="|u1",
)
h5file.create_dataset(
    "LFImageData",
    shape=[LF_shape[0] * LF_shape[2], LF_shape[1] * LF_shape[3], nSimulations],
    dtype="|u1",
)

# Optimize
with torch.no_grad():
    for ep in range(nSimulations):
        plt.clf()

        # Create empty volume
        curr_volume = torch.zeros(1, nDepths, vol_shape, vol_shape).to(device)
        # Fill with spheres
        x_coords = torch.LongTensor(nSpheres).random_(0, vol_shape)
        y_coords = torch.LongTensor(nSpheres).random_(0, vol_shape)
        z_coords = torch.LongTensor(nSpheres).random_(0, nDepths)

        x_coords = torch.LongTensor([vol_shape // 2])
        y_coords = torch.LongTensor([vol_shape // 2])
        z_coords = torch.LongTensor([ep])

        for nSph in range(nSpheres):
            curr_volume[
                0,
                z_coords[nSph].item(),
                x_coords[nSph].item()
                - mSphereNpixelsHalf : (x_coords[nSph].item() + mSphereNpixelsHalf + 1),
                y_coords[nSph].item()
                - mSphereNpixelsHalf : (y_coords[nSph].item() + mSphereNpixelsHalf + 1),
            ] = mask
        # pad for full image size
        curr_volume = F.pad(curr_volume, pad_vol_val)

        start.record()
        with autocast():
            vol_temp = curr_volume[:, psf_indices_per_micro[0], ...].to(device_array[0])
            curr_LF_img = WBMicro_array[0].forward(vol_temp)
            for wbm in range(1, len(WBMicro_array)):
                vol_temp = curr_volume[:, psf_indices_per_micro[wbm], ...].to(
                    device_array[wbm]
                )
                curr_LF_img += WBMicro_array[wbm].forward(vol_temp).to(device_array[0])

        end.record()
        torch.cuda.synchronize()
        print("time: " + str(end.elapsed_time(start)))

        # Store
        curr_LF_img = (255 * curr_LF_img / curr_LF_img.max()).type(torch.uint8)
        curr_volume = (255 * curr_volume / curr_volume.max()).type(torch.uint8)

        curr_LF = co.LF2Spatial(curr_LF_img, LF_shape)

        h5file["volData"][:, :, :, ep] = (
            curr_volume[0, ...].permute(1, 2, 0).cpu().numpy()
        )
        h5file["LFData"][:, :, :, :, ep] = curr_LF[0, 0, ...].cpu().numpy()
        h5file["LFImageData"][:, :, ep] = curr_LF_img[0, 0, ...].cpu().numpy()

        def img_to_tb(img):
            return img.detach().cpu().numpy()

        writer.add_image("LFImg", img_to_tb(curr_LF_img[0, :, :, :]), ep)
        writer.add_image(
            "volData", 255 * img_to_tb(curr_volume[0, :, :, :].sum(0).unsqueeze(0)), ep
        )
        writer.add_image(
            "volData_xz",
            255 * img_to_tb(curr_volume[0, :, :, :].sum(1).unsqueeze(0)),
            ep,
        )

        if plot:
            # Display results
            #

            plt.subplot(1, 3, 1)
            plt.imshow(curr_LF_img[0, 0, :, :].detach().cpu().numpy())
            plt.title("LF img")
            plt.gray()
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])

            plt.subplot(1, 3, 2)
            plt.imshow(255 * curr_volume[0, :, :, :].sum(0).detach().cpu().numpy())
            plt.title("LF volume")
            plt.gray()
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])

            plt.subplot(1, 3, 3)
            plt.imshow(255 * curr_volume[0, :, :, :].sum(1).detach().cpu().numpy())
            plt.title("LF volume xz")
            plt.gray()
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])

            plt.pause(0.1)
            plt.show()

h5file.close()
writer.close()
